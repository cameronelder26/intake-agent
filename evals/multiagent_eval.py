"""Human-queue load and error rate, with and without the reviewer agent.

    python evals/multiagent_eval.py [--out evals/RESULTS-multiagent.md]

Runs the 12 extract tasks of the 32-task set (evals/eval_set.jsonl) through the graph in
mock mode, no API key. The other 20 tasks are RAG questions and do not touch the gate.

Two scenarios:
  as-is   the mock extractor's real output
  seeded  the same, with deterministic faults injected into every document (below)

Definitions, per field (12 documents x 12 fields = 144):
  queue load    fields below the confidence threshold after the gate, i.e. what a human is asked
  silent error  a wrong value that is NOT in the queue, so nobody looked at it
  caught error  a wrong value that IS in the queue, so a human sees it with the reviewer's note
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import date
from pathlib import Path

from intake_agent.config import Settings
from intake_agent.eval import values_match
from intake_agent.extractor import MockExtractor
from intake_agent.graph import make_checkpointer, run_document
from intake_agent.harness import load_tasks
from intake_agent.multiagent import MockReviewer
from intake_agent.pipeline import open_state
from intake_agent.schema import FIELDS

DATA = Path("data/synthetic")
SEED_CONF = 0.7  # below the 0.85 threshold, so every seeded fault is a candidate for the reviewer


class Seeded(MockExtractor):
    """Injects three kinds of fault, chosen by document position so runs are repeatable."""

    def __init__(self):
        self.n = 0

    def extract(self, path):
        res = super().extract(path)
        f = res.fields
        i = self.n
        self.n += 1
        if not f["patient_name"].value:  # scanned page: nothing to corrupt
            return res
        # 1. payer replaced by a plausible plan that is not in the document
        f["payer"].value = "Zenith Health Partners"
        f["payer"].model_confidence = SEED_CONF
        # 2. NPI with a wrong last digit (fails the checksum)
        npi = f["referring_npi"].value
        if npi:
            f["referring_npi"].value = npi[:-1] + str((int(npi[-1]) + 1) % 10)
            f["referring_npi"].model_confidence = SEED_CONF
        # 3. on every other document, patient name replaced by the referring provider's name:
        #    valid, and written in the document, but wrong. The reviewer cannot see this.
        if i % 2 == 0 and f["referring_provider"].value:
            f["patient_name"].value = f["referring_provider"].value
            f["patient_name"].model_confidence = SEED_CONF
        return res


def run(extractor_cls, with_reviewer: bool) -> dict:
    tasks = [t for t in load_tasks(Path("evals/eval_set.jsonl")) if t["kind"] == "extract"]
    totals = dict(docs=0, fields=0, wrong=0, queue_fields=0, queue_docs=0, silent=0, caught=0)
    cleared_wrong = 0
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            model="claude-haiku-5-5", effort="low", threshold=0.85, state_dir=Path(tmp), pg_dsn=""
        )
        _, queue = open_state(settings)
        ext = extractor_cls()
        for t in tasks:
            pdf = DATA / f"{t['doc_id']}.pdf"
            truth = json.loads((DATA / f"{t['doc_id']}.json").read_text())["fields"]
            st = run_document(
                pdf,
                ext,
                queue,
                settings,
                doc_id=t["doc_id"],
                reviewer=MockReviewer() if with_reviewer else None,
                checkpointer=make_checkpointer(None),
            )
            wrong = {n for n in FIELDS if not values_match(n, st["scored"][n].value, truth[n])}
            flagged = set(st["flagged"])
            totals["docs"] += 1
            totals["fields"] += len(FIELDS)
            totals["wrong"] += len(wrong)
            totals["queue_fields"] += len(flagged)
            totals["queue_docs"] += bool(flagged)
            totals["silent"] += len(wrong - flagged)
            totals["caught"] += len(wrong & flagged)
            cleared_wrong += len(
                [
                    v
                    for v in st.get("verdicts", [])
                    if v["decision"] == "clear" and v["field"] in wrong
                ]
            )
    totals["cleared_wrong"] = cleared_wrong
    return totals


def _row(label: str, a: dict, b: dict) -> str:
    return (
        f"| {label} | {a['queue_docs']} | {a['queue_fields']} | {a['silent']} "
        f"({a['silent'] / a['fields']:.1%}) | {a['caught']} |\n"
        f"| {label} + reviewer | {b['queue_docs']} | {b['queue_fields']} | {b['silent']} "
        f"({b['silent'] / b['fields']:.1%}) | {b['caught']} |"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("evals/RESULTS-multiagent.md"))
    args = ap.parse_args(argv)
    res = {}
    for name, cls in (("as-is", MockExtractor), ("seeded", Seeded)):
        res[name] = (run(cls, False), run(cls, True))
    asis, seeded = res["as-is"], res["seeded"]
    md = f"""# Multi-agent eval: reviewer on and off

Written by `python evals/multiagent_eval.py` on {date.today()}. Mock mode: rule-based extractor,
rule-based reviewer, no API calls. Task set: the 12 extract tasks of `evals/eval_set.jsonl`
(12 documents, 144 fields); the other 20 tasks of the 32 are RAG questions and never reach the gate.
Threshold 0.85. Synthetic data only.

| Scenario | Docs queued | Fields queued | Silent errors | Errors a human sees |
|---|---|---|---|---|
{_row("as-is", *asis)}
{_row("seeded faults", *seeded)}

- Silent error: a wrong value that nobody reviews and that would be released.
  Percentage is of all 144 fields.
- Errors a human sees: wrong values that are in the queue, so a person can fix them.

## What the numbers say

**As-is.** The mock extractor's own output has {asis[0]["wrong"]} wrong fields of 144, and they
are all on the two scanned documents, where every field is empty and flagged. Without the reviewer
{asis[0]["queue_fields"]} fields from {asis[0]["queue_docs"]} documents go to a person. With it
{asis[1]["queue_fields"]} fields from {asis[1]["queue_docs"]} documents do: the reviewer clears the
low-confidence fields that pass validation and are written in the source, and escalates the rest
(a bad NPI checksum, missing values). Silent errors stay at {asis[1]["silent"]}.

**Seeded faults.** Every text document gets a payer that is not in the document and an NPI with a
broken checksum. On every other document the patient name is swapped for the referring provider's
name. The first two are written to be caught by the reviewer's two tools. The third is a
valid-looking value that is in the source text, so the reviewer cannot tell it from a right one.
Without the reviewer {seeded[0]["queue_fields"]} fields go to a person and {seeded[0]["silent"]}
errors are silent. With it {seeded[1]["queue_fields"]} fields go to a person and
{seeded[1]["silent"]} are silent, all of them swapped names the reviewer cleared
({seeded[1]["cleared_wrong"]} wrong fields cleared). So the reviewer cuts the queue by checking
what can be checked, and it adds a silent-error risk of its own for faults that pass both of its
tools. A human still sees every escalated field with the reviewer's reason beside it.

Limits: the faults are injected by this script, not found in the wild, and the reviewer here is the
deterministic mock. A model reviewer with the same two tools may behave differently. The
as-is numbers are the real extractor output.

Re-run: `python evals/multiagent_eval.py`.
"""
    args.out.write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
