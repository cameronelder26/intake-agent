"""Compare embedders on retrieval and cited answers; writes a section of evals/RESULTS.md.

    python -m intake_agent.embed_eval

For each embedder it builds an in-memory index over the 40 documents and 5 payer
policies and measures, on the answerable tasks of evals/eval_set.jsonl plus the
vocabulary-miss cases in evals/synonym_set.jsonl:

- hit@k: some chunk in the top k contains every must_contain string and belongs to the
  must_cite source. Retrieval is NOT limited to the task's doc_id (only to policies for
  policy questions), so the embedder has to find the document itself.
- cited-answer accuracy: the extractive MockAnswerer on top of the same retrieval,
  scoped the way the harness scopes it. It is lexical, so it cannot bridge a synonym even
  when retrieval did; that is what the Claude answerer is for.
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import date
from pathlib import Path

from .config import Settings
from .harness import EMBED_END, EMBED_START, load_tasks, run_rag
from .rag import (
    Embedder,
    FastEmbedder,
    HashingEmbedder,
    InMemoryStore,
    RagIndex,
    load_corpus,
    make_answerer,
)

KS = (1, 3, 5)


def _rank(index: RagIndex, task: dict, k: int = 10) -> int | None:
    prefix = "policy:" if task.get("scope") == "policy" else None
    for i, h in enumerate(index.retrieve(task["question"], k=k, prefix=prefix), 1):
        body = h.chunk.text.casefold()
        if h.chunk.doc_id == task["must_cite"] and all(
            m.casefold() in body for m in task["must_contain"]
        ):
            return i
    return None


def evaluate(emb: Embedder, tasks: list[dict], syn: list[dict], data: Path, policies: Path):
    index = RagIndex(emb, InMemoryStore())
    t0 = time.perf_counter()
    n = index.build(load_corpus(data, policies)[0])
    build_s = time.perf_counter() - t0
    answerer = make_answerer(True, Settings())

    def stats(rows: list[dict]) -> dict:
        answerable = [t for t in rows if not t.get("expect_not_found")]
        ranks = [_rank(index, t) for t in answerable]
        ok = sum(run_rag(t, index, answerer, (0.0, 0.0)).success for t in rows)
        return {
            "n": len(rows),
            "n_ans": len(answerable),
            "hit": {k: sum(r is not None and r <= k for r in ranks) for k in KS},
            "answer_ok": ok,
            "ranks": dict(zip((t["id"] for t in answerable), ranks, strict=True)),
        }

    return {
        "name": emb.name,
        "chunks": n,
        "build_s": build_s,
        "main": stats(tasks),
        "syn": stats(syn),
    }


def render(res: list[dict], syn: list[dict], run_date: date) -> str:
    L = [EMBED_START, "## Embedder comparison: hashing vs real", ""]
    L += [
        f"Run on {run_date.isoformat()} with `python -m intake_agent.embed_eval` "
        "(in-memory store, mock extractive answerer, no API calls).",
        "",
        "hit@k: a top-k chunk holds the expected text and comes from the expected document, "
        "searching the whole index (not limited to the task's doc). Main-set questions such as "
        "'What is the member ID?' are written to be used with a doc_id filter and name no "
        "patient, so unscoped hit@k on them is capped by ambiguity, not by the embedder; the "
        "synonym cases name the patient or plan and are the fair test. Cited-answer accuracy is "
        "the extractive mock answerer over the same retrieval, scoped as the harness does.",
        "",
        "| Embedder | Set | Tasks | hit@1 | hit@3 | hit@5 | Cited-answer accuracy |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in res:
        for label, key in (("main eval_set (rag)", "main"), ("synonym cases", "syn")):
            s = r[key]
            h = [f"{s['hit'][k]}/{s['n_ans']}" for k in KS]
            L.append(
                f"| `{r['name']}` | {label} | {s['n']} | {h[0]} | {h[1]} | {h[2]} "
                f"| {s['answer_ok']}/{s['n']} |"
            )
    L += ["", "### Vocabulary-miss cases (rank of the right chunk, `-` = not in top 10)", ""]
    L += ["| Case | Question | Form wording | " + " | ".join(f"`{r['name']}`" for r in res) + " |"]
    L += ["|---|---|---|" + "---|" * len(res)]
    for t in syn:
        cells = [str(r["syn"]["ranks"][t["id"]] or "-") for r in res]
        L.append(
            f"| {t['id']} | {t['question']} | {t.get('note', '')} | " + " | ".join(cells) + " |"
        )
    L += [
        "",
        "Index sizes: "
        + ", ".join(
            f"`{r['name']}` {r['chunks']} chunks in {r['build_s'] * 1000:.0f} ms" for r in res
        )
        + ".",
        EMBED_END,
    ]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", type=Path, default=Path("evals/eval_set.jsonl"))
    ap.add_argument("--synonyms", type=Path, default=Path("evals/synonym_set.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("evals/RESULTS.md"))
    ap.add_argument("--data", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--policies", type=Path, default=Path("data/policies"))
    args = ap.parse_args(argv)
    tasks = [t for t in load_tasks(args.tasks) if t["kind"] == "rag"]
    syn = load_tasks(args.synonyms)
    res = [
        evaluate(e, tasks, syn, args.data, args.policies)
        for e in (HashingEmbedder(), FastEmbedder())
    ]
    block = render(res, syn, date.today())
    old = args.out.read_text() if args.out.exists() else ""
    pat = re.compile(re.escape(EMBED_START) + ".*?" + re.escape(EMBED_END), re.S)
    new = pat.sub(lambda _: block, old) if pat.search(old) else old.rstrip("\n") + "\n\n" + block
    args.out.write_text(new.rstrip("\n") + "\n")
    print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
