"""Eval harness: task success, latency and cost for every task in a fixed eval set.

    python -m intake_agent.harness --mock      # no API calls, writes evals/RESULTS.md
    python -m intake_agent.harness             # real model (needs ANTHROPIC_API_KEY)
    python -m intake_agent.harness --mock --price-in 2 --price-out 10

The task set is evals/eval_set.jsonl. Two kinds of task:

- extract: run the extractor on one labeled PDF. Success means every field matches
  ground truth. "Safe" means every wrong field was sent to review, so nothing wrong
  could be released without a person looking at it.
- rag: retrieve, answer, and check citations. Success means the answer contains the
  expected text and cites the expected document (or, for questions the documents
  can't answer, that it says so instead of guessing).

Cost is input tokens x input price + output tokens x output price. Prices default to
the model's rates in config.py and can be overridden with --price-in/--price-out or
INTAKE_PRICE_IN/INTAKE_PRICE_OUT (USD per million tokens). In mock mode no tokens are
spent, so the harness estimates them from the prompt the model would have been sent;
those rows are marked as estimates.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pypdf import PdfReader

from .confidence import needs_review, score_all
from .config import PRICES_PER_MTOK, Settings, estimate_tokens
from .eval import values_match
from .extractor import SYSTEM_PROMPT, ExtractionError, Extractor, make_extractor
from .pdftext import collapse_fragments, extract_raw_text, has_usable_text
from .rag import NOT_FOUND, HashingEmbedder, RagIndex, load_corpus, make_answerer, make_store
from .rag.answer import Answerer
from .schema import FIELDS, ModelExtraction

# Rough input cost of one PDF page sent as a document block (text plus page image).
TOKENS_PER_SCANNED_PAGE = 1600
SCHEMA_JSON = json.dumps(ModelExtraction.model_json_schema())


@dataclass
class TaskResult:
    id: str
    kind: str
    success: bool
    safe: bool | None
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    estimated: bool
    note: str = ""


@dataclass
class RunInfo:
    mode: str
    model: str
    price_in: float
    price_out: float
    threshold: float
    store: str
    index_ms: float
    index_chunks: int


def load_tasks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def price(tokens_in: int, tokens_out: int, price_in: float, price_out: float) -> float:
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def _estimate_extraction(pdf: Path, fields: dict) -> tuple[int, int]:
    text = collapse_fragments(extract_raw_text(pdf))
    base = estimate_tokens(SYSTEM_PROMPT + SCHEMA_JSON)
    if has_usable_text(text):
        tokens_in = base + estimate_tokens(text)
    else:
        tokens_in = base + TOKENS_PER_SCANNED_PAGE * len(PdfReader(pdf).pages)
    out = {n: {"value": f.value, "confidence": f.model_confidence} for n, f in fields.items()}
    return tokens_in, estimate_tokens(json.dumps(out))


def run_extract(
    task: dict, extractor: Extractor, data_dir: Path, threshold: float, prices: tuple[float, float]
) -> TaskResult:
    doc_id = task["doc_id"]
    pdf = data_dir / f"{doc_id}.pdf"
    truth = json.loads((data_dir / f"{doc_id}.json").read_text())["fields"]
    t0 = time.perf_counter()
    try:
        result = extractor.extract(pdf)
    except ExtractionError as exc:
        ms = (time.perf_counter() - t0) * 1000
        return TaskResult(task["id"], "extract", False, True, ms, 0, 0, 0.0, False, str(exc))
    scored = score_all(result.fields)
    flagged = set(needs_review(scored, threshold))
    ms = (time.perf_counter() - t0) * 1000

    wrong = [n for n in FIELDS if not values_match(n, scored[n].value, truth[n])]
    unflagged = [n for n in wrong if n not in flagged]
    if result.input_tokens:
        t_in, t_out, est = result.input_tokens, result.output_tokens, False
    else:
        t_in, t_out = _estimate_extraction(pdf, result.fields)
        est = True
    note = ""
    if wrong:
        listed = ", ".join(wrong) if len(wrong) <= 3 else f"{len(wrong)} of {len(FIELDS)} fields"
        note = f"wrong: {listed}" + (
            f"; NOT FLAGGED: {', '.join(unflagged)}" if unflagged else " (all flagged)"
        )
    elif flagged:
        note = f"right, but {len(flagged)} field(s) sent to review"
    return TaskResult(
        task["id"],
        "extract",
        not wrong,
        not unflagged,
        ms,
        t_in,
        t_out,
        price(t_in, t_out, *prices),
        est,
        note,
    )


def run_rag(
    task: dict, index: RagIndex, answerer: Answerer, prices: tuple[float, float]
) -> TaskResult:
    t0 = time.perf_counter()
    prefix = "policy:" if task.get("scope") == "policy" else None
    hits = index.retrieve(task["question"], k=5, doc_id=task.get("doc_id"), prefix=prefix)
    ans = answerer.answer(task["question"], hits)
    ms = (time.perf_counter() - t0) * 1000

    if task.get("expect_not_found"):
        ok = ans.text == NOT_FOUND
        note = "" if ok else f"should have said not found, said {ans.text[:60]!r}"
    else:
        missing = [s for s in task["must_contain"] if s.casefold() not in ans.text.casefold()]
        cited = {c.doc_id for c in ans.citations}
        ok = ans.grounded and not missing and task["must_cite"] in cited
        if not ans.grounded:
            note = "said not found"
        elif missing:
            note = f"answer missing {missing}; got {ans.text[:60]!r}"
        elif task["must_cite"] not in cited:
            note = f"cited {sorted(cited)}, expected {task['must_cite']}"
        else:
            note = ""
    if ans.dropped_citations:
        note = (note + "; " if note else "") + f"{ans.dropped_citations} citation(s) dropped"
    return TaskResult(
        task["id"],
        "rag",
        ok,
        None,
        ms,
        ans.input_tokens,
        ans.output_tokens,
        price(ans.input_tokens, ans.output_tokens, *prices),
        ans.tokens_estimated,
        note,
    )


def run_harness(
    tasks: list[dict],
    mock: bool,
    settings: Settings,
    prices: tuple[float, float],
    data_dir: Path = Path("data/synthetic"),
    policy_dir: Path = Path("data/policies"),
) -> tuple[list[TaskResult], RunInfo]:
    extractor = make_extractor(mock, settings)
    answerer = make_answerer(mock, settings)
    emb = HashingEmbedder()
    store = make_store(emb.dim, settings.pg_dsn)
    store.clear()
    index = RagIndex(emb, store)
    t0 = time.perf_counter()
    n_chunks = index.build(load_corpus(data_dir, policy_dir)[0])
    index_ms = (time.perf_counter() - t0) * 1000

    results = []
    for task in tasks:
        if task["kind"] == "extract":
            results.append(run_extract(task, extractor, data_dir, settings.threshold, prices))
        elif task["kind"] == "rag":
            results.append(run_rag(task, index, answerer, prices))
        else:
            raise ValueError(f"unknown task kind {task['kind']!r} in {task['id']}")
    info = RunInfo(
        mode="mock" if mock else "model",
        model=settings.model,
        price_in=prices[0],
        price_out=prices[1],
        threshold=settings.threshold,
        store="pgvector" if settings.pg_dsn else "in-memory",
        index_ms=index_ms,
        index_chunks=n_chunks,
    )
    return results, info


def _pct(rows: list[TaskResult], q: float) -> float:
    xs = sorted(r.latency_ms for r in rows)
    if len(xs) == 1:
        return xs[0]
    return statistics.quantiles(xs, n=100, method="inclusive")[round(q * 100) - 1]


def _summary_row(label: str, rows: list[TaskResult]) -> str:
    n = len(rows)
    ok = sum(r.success for r in rows)
    safe_rows = [r for r in rows if r.safe is not None]
    safe = f"{sum(r.safe for r in safe_rows)}/{len(safe_rows)}" if safe_rows else "n/a"
    t_in, t_out = sum(r.input_tokens for r in rows), sum(r.output_tokens for r in rows)
    cost = sum(r.cost_usd for r in rows)
    return (
        f"| {label} | {n} | {ok}/{n} ({100 * ok / n:.0f}%) | {safe} | {_pct(rows, 0.5):.1f} | "
        f"{_pct(rows, 0.95):.1f} | {t_in:,} | {t_out:,} | ${cost:.5f} | ${cost / n:.6f} |"
    )


def render(results: list[TaskResult], info: RunInfo, run_date: date) -> str:
    est = any(r.estimated for r in results)
    cmd = "python -m intake_agent.harness" + (" --mock" if info.mode == "mock" else "")
    lines = [
        "# Eval results",
        "",
        f"Written by `{cmd}` on {run_date.isoformat()}. Task set: `evals/eval_set.jsonl` "
        f"({len(results)} tasks, synthetic data only).",
        "",
        f"- Mode: **{info.mode}**"
        + (
            " (rule-based extractor, extractive answerer, no API calls)"
            if info.mode == "mock"
            else ""
        ),
        f"- Priced as `{info.model}` at ${info.price_in:.2f} in / ${info.price_out:.2f} out "
        "per million tokens",
        f"- Review threshold {info.threshold}; vector store {info.store}; "
        f"index build {info.index_ms:.0f} ms for {info.index_chunks} chunks",
        "",
    ]
    if est:
        lines += [
            "Token counts marked `~` are estimates (about 4 characters per token) of the prompt "
            "the model would have been sent, so cost on those rows is a projection, not a bill. "
            "Mock latency is local compute only and says nothing about API latency.",
            "",
        ]
    lines += [
        "## Summary",
        "",
        "| Kind | Tasks | Success | Safe | p50 ms | p95 ms | Input tokens | Output tokens "
        "| Cost | Cost per task |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for kind in ("extract", "rag"):
        rows = [r for r in results if r.kind == kind]
        if rows:
            lines.append(_summary_row(kind, rows))
    lines.append(_summary_row("all", results))
    lines += [
        "",
        "## Per task",
        "",
        "| Task | Success | Safe | Latency ms | In tokens | Out tokens | Cost | Note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        mark = "~" if r.estimated else ""
        safe = "n/a" if r.safe is None else ("yes" if r.safe else "NO")
        lines.append(
            f"| {r.id} | {'yes' if r.success else 'no'} | {safe} | {r.latency_ms:.1f} | "
            f"{mark}{r.input_tokens:,} | {mark}{r.output_tokens:,} | ${r.cost_usd:.6f} | "
            f"{r.note.replace('|', '/')} |"
        )
    lines += [
        "",
        "Definitions: an extract task succeeds when all twelve fields match ground truth, and",
        "is safe when every wrong field was below the review threshold (so a person sees it",
        "before release). A rag task succeeds when the answer contains the expected text and",
        "cites the expected document with a quote that was found verbatim in the source, or,",
        "for questions the documents can't answer, when it says not found.",
        "",
    ]
    return "\n".join(lines)


def _prices(model: str, price_in: float | None, price_out: float | None) -> tuple[float, float]:
    default_in, default_out = PRICES_PER_MTOK.get(model, (0.0, 0.0))
    env_in, env_out = os.environ.get("INTAKE_PRICE_IN"), os.environ.get("INTAKE_PRICE_OUT")
    return (
        price_in if price_in is not None else float(env_in) if env_in else default_in,
        price_out if price_out is not None else float(env_out) if env_out else default_out,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Task success, latency and cost on the eval set.")
    ap.add_argument("--mock", action="store_true", help="no API calls")
    ap.add_argument("--tasks", type=Path, default=Path("evals/eval_set.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("evals/RESULTS.md"))
    ap.add_argument("--data", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--policies", type=Path, default=Path("data/policies"))
    ap.add_argument("--model", default=None, help="override INTAKE_MODEL")
    ap.add_argument("--price-in", type=float, default=None, help="USD per million input tokens")
    ap.add_argument("--price-out", type=float, default=None, help="USD per million output tokens")
    args = ap.parse_args(argv)

    base = Settings()
    settings = Settings(
        model=args.model or base.model,
        effort=base.effort,
        threshold=base.threshold,
        state_dir=base.state_dir,
        pg_dsn=base.pg_dsn,
    )
    if not args.mock and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Run with --mock, or export a key.", file=sys.stderr)
        return 2
    prices = _prices(settings.model, args.price_in, args.price_out)
    results, info = run_harness(
        load_tasks(args.tasks), args.mock, settings, prices, args.data, args.policies
    )
    md = render(results, info, date.today())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(md)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
