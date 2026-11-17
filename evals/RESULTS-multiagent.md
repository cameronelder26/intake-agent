# Multi-agent eval: reviewer on and off

Written by `python evals/multiagent_eval.py` on 2026-11-17. Mock mode: rule-based extractor,
rule-based reviewer, no API calls. Task set: the 12 extract tasks of `evals/eval_set.jsonl`
(12 documents, 144 fields); the other 20 tasks of the 32 are RAG questions and never reach the gate.
Threshold 0.85. Synthetic data only.

| Scenario | Docs queued | Fields queued | Silent errors | Errors a human sees |
|---|---|---|---|---|
| as-is | 9 | 76 | 0 (0.0%) | 23 |
| as-is + reviewer | 9 | 35 | 0 (0.0%) | 23 |
| seeded faults | 12 | 90 | 0 (0.0%) | 47 |
| seeded faults + reviewer | 12 | 53 | 5 (3.5%) | 42 |

- Silent error: a wrong value that nobody reviews and that would be released.
  Percentage is of all 144 fields.
- Errors a human sees: wrong values that are in the queue, so a person can fix them.

## What the numbers say

**As-is.** The mock extractor's own output has 23 wrong fields of 144, and they
are all on the two scanned documents, where every field is empty and flagged. Without the reviewer
76 fields from 9 documents go to a person. With it
35 fields from 9 documents do: the reviewer clears the
low-confidence fields that pass validation and are written in the source, and escalates the rest
(a bad NPI checksum, missing values). Silent errors stay at 0.

**Seeded faults.** Every text document gets a payer that is not in the document and an NPI with a
broken checksum. On every other document the patient name is swapped for the referring provider's
name. The first two are written to be caught by the reviewer's two tools. The third is a
valid-looking value that is in the source text, so the reviewer cannot tell it from a right one.
Without the reviewer 90 fields go to a person and 0
errors are silent. With it 53 fields go to a person and
5 are silent, all of them swapped names the reviewer cleared
(5 wrong fields cleared). So the reviewer cuts the queue by checking
what can be checked, and it adds a silent-error risk of its own for faults that pass both of its
tools. A human still sees every escalated field with the reviewer's reason beside it.

Limits: the faults are injected by this script, not found in the wild, and the reviewer here is the
deterministic mock. A model reviewer with the same two tools may behave differently. The
as-is numbers are the real extractor output.

Re-run: `python evals/multiagent_eval.py`.
