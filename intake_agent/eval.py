"""Evaluate the pipeline against the labeled synthetic set.

    python -m intake_agent.eval            # real model (needs ANTHROPIC_API_KEY)
    python -m intake_agent.eval --mock     # rule-based mock, no network

Runs the full pipeline (extract, validate, gate) in a throwaway state directory,
compares extracted values to ground truth, and writes results/eval-<date>[-mock].md.
Metrics are computed on the extraction before any human review, because that is what
the gate has to work with.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .config import Settings
from .extractor import make_extractor
from .pipeline import open_state, process_batch
from .schema import FIELDS, FieldValue
from .validators import normalize_value


def values_match(name: str, predicted: FieldValue, truth: FieldValue) -> bool:
    p, t = normalize_value(name, predicted), normalize_value(name, truth)
    if isinstance(p, str) and isinstance(t, str):
        return p.casefold() == t.casefold()
    return p == t


def _present(v: FieldValue) -> bool:
    return v not in (None, [], "")


@dataclass
class FieldStats:
    tp: int = 0  # predicted a value, and it is right
    fp: int = 0  # predicted a value, and it is wrong
    truth_present: int = 0
    correct: int = 0  # includes correctly predicting "absent"
    total: int = 0

    @property
    def precision(self) -> float | None:
        n = self.tp + self.fp
        return self.tp / n if n else None

    @property
    def recall(self) -> float | None:
        return self.tp / self.truth_present if self.truth_present else None

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class EvalResult:
    mode: str
    model: str
    threshold: float
    n_docs: int = 0
    fields: dict[str, FieldStats] = field(default_factory=lambda: defaultdict(FieldStats))
    exact_docs: int = 0
    queued_docs: int = 0
    failed_docs: int = 0
    auto_docs_with_error: int = 0
    wrong_fields: int = 0
    wrong_fields_flagged: int = 0
    total_cost: float = 0.0
    per_template: dict[str, list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0, 0])
    )
    errors: list[str] = field(default_factory=list)


def run_eval(
    data_dir: Path, mock: bool, settings: Settings, today: date | None = None
) -> EvalResult:
    truth_files = sorted(data_dir.glob("*.json"))
    if not truth_files:
        raise SystemExit(f"no ground truth in {data_dir}; run `python -m intake_agent.synth`")
    extractor = make_extractor(mock, settings)
    res = EvalResult(
        mode="mock" if mock else "model", model=extractor.model, threshold=settings.threshold
    )
    with tempfile.TemporaryDirectory() as tmp:
        run_settings = Settings(
            model=settings.model,
            effort=settings.effort,
            threshold=settings.threshold,
            state_dir=Path(tmp),
        )
        _, queue = open_state(run_settings)
        pdfs = [f.with_suffix(".pdf") for f in truth_files]
        report = process_batch(pdfs, extractor, queue, run_settings, today)
        res.failed_docs = len(report.failed)
        res.total_cost = report.cost_usd
        for tf in truth_files:
            truth = json.loads(tf.read_text())
            doc_id, template = truth["meta"]["doc_id"], truth["meta"]["template"]
            res.n_docs += 1
            tstats = res.per_template[template]
            tstats[0] += 1
            if doc_id in report.failed:
                res.errors.append(f"{doc_id}: extraction failed")
                tstats[2] += 1
                for name in FIELDS:
                    st = res.fields[name]
                    st.total += 1
                    st.truth_present += _present(truth["fields"][name])
                continue
            doc = queue.document(doc_id)
            queued = doc["status"] == "pending_review"
            res.queued_docs += queued
            tstats[2] += queued
            flagged = {i["field"] for i in queue.items(doc_id)}
            all_right = True
            for name in FIELDS:
                pred, gold = doc["fields"][name]["value"], truth["fields"][name]
                ok = values_match(name, pred, gold)
                st = res.fields[name]
                st.total += 1
                st.correct += ok
                st.truth_present += _present(gold)
                if _present(pred):
                    st.tp += ok
                    st.fp += not ok
                if not ok:
                    all_right = False
                    res.wrong_fields += 1
                    res.wrong_fields_flagged += name in flagged
                    res.errors.append(
                        f"{doc_id} ({template}) {name}: got {pred!r}, want {gold!r}"
                        + ("" if name in flagged else "  [NOT FLAGGED]")
                    )
            res.exact_docs += all_right
            tstats[1] += all_right
            if not all_right and not queued:
                res.auto_docs_with_error += 1
    return res


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render_markdown(res: EvalResult, run_date: date) -> str:
    n = res.n_docs or 1
    label = (
        "MOCK run: rule-based extractor, no model calls. These numbers measure the "
        "pipeline plumbing, not model quality."
        if res.mode == "mock"
        else f"Model run: `{res.model}`."
    )
    micro_tp = sum(s.tp for s in res.fields.values())
    micro_fp = sum(s.fp for s in res.fields.values())
    micro_present = sum(s.truth_present for s in res.fields.values())
    micro_correct = sum(s.correct for s in res.fields.values())
    micro_total = sum(s.total for s in res.fields.values())
    lines = [
        f"# Eval {run_date.isoformat()} ({res.mode})",
        "",
        label,
        "",
        f"- Documents: {res.n_docs} synthetic (data/synthetic)",
        f"- Extractor: `{res.model}`, review threshold {res.threshold}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Field precision (micro) | {_pct(_ratio(micro_tp, micro_tp + micro_fp))} |",
        f"| Field recall (micro) | {_pct(_ratio(micro_tp, micro_present))} |",
        f"| Field accuracy (micro) | {_pct(_ratio(micro_correct, micro_total))} |",
        f"| Document exact match | {_pct(res.exact_docs / n)} ({res.exact_docs}/{res.n_docs}) |",
        f"| Review-queue rate | {_pct(res.queued_docs / n)} ({res.queued_docs}/{res.n_docs}) |",
        f"| Auto-approved docs with any wrong field | {res.auto_docs_with_error} |",
        f"| Wrong fields caught by the gate | {res.wrong_fields_flagged}/{res.wrong_fields} |",
        f"| Extraction failures | {res.failed_docs} |",
        f"| Cost per document | ${res.total_cost / n:.5f} (total ${res.total_cost:.4f}) |",
        "",
        "## Per field",
        "",
        "| Field | Precision | Recall | Accuracy |",
        "|---|---|---|---|",
    ]
    for name in FIELDS:
        s = res.fields[name]
        lines.append(f"| {name} | {_pct(s.precision)} | {_pct(s.recall)} | {_pct(s.accuracy)} |")
    lines += [
        "",
        "## Per template",
        "",
        "| Template | Docs | Exact match | Sent to review |",
        "|---|---|---|---|",
    ]
    for template, (docs, exact, queued) in sorted(res.per_template.items()):
        lines.append(f"| {template} | {docs} | {exact} | {queued} |")
    lines += ["", "## Field errors (before human review)", ""]
    lines += [f"- {e}" for e in res.errors] or ["- none"]
    lines += [
        "",
        "Definitions: precision = correct non-empty predictions / non-empty predictions;",
        "recall = correct non-empty predictions / fields present in ground truth;",
        "accuracy counts a correctly empty field as right. Codes compare as sets; text",
        "compares case-insensitively after normalization.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate extraction on the labeled set.")
    ap.add_argument("--mock", action="store_true", help="use the rule-based mock extractor")
    ap.add_argument("--data", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--model", default=None, help="override INTAKE_MODEL")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args(argv)

    base = Settings()
    settings = Settings(
        model=args.model or base.model,
        effort=base.effort,
        threshold=args.threshold if args.threshold is not None else base.threshold,
        state_dir=base.state_dir,
    )
    if not args.mock and not _has_credentials():
        print("ANTHROPIC_API_KEY is not set. Run with --mock, or export a key.", file=sys.stderr)
        return 2
    run_date = date.today()
    res = run_eval(args.data, args.mock, settings, run_date)
    md = render_markdown(res, run_date)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"eval-{run_date.isoformat()}{'-mock' if args.mock else ''}.md"
    out.write_text(md)
    print(md)
    print(f"wrote {out}")
    return 0


def _has_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


if __name__ == "__main__":
    raise SystemExit(main())
