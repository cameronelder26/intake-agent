# Eval results

Written by `python -m intake_agent.harness --mock` on 2026-11-17. Task set: `evals/eval_set.jsonl` (32 tasks, synthetic data only).

- Mode: **mock** (rule-based extractor, extractive answerer, no API calls)
- Priced as `claude-haiku-5-5` at $0.10 in / $0.50 out per million tokens
- Review threshold 0.85; vector store in-memory; embedder `hashing-v1-384`; index build 118 ms for 86 chunks

Token counts marked `~` are estimates (about 4 characters per token) of the prompt the model would have been sent, so cost on those rows is a projection, not a bill. Mock latency is local compute only and says nothing about API latency.

## Summary

| Kind | Tasks | Success | Safe | p50 ms | p95 ms | Input tokens | Output tokens | Cost | Cost per task |
|---|---|---|---|---|---|---|---|---|---|
| extract | 12 | 10/12 (83%) | 12/12 | 1.9 | 6.3 | 13,997 | 2,205 | $0.00250 | $0.000209 |
| rag | 20 | 17/20 (85%) | n/a | 0.3 | 0.4 | 9,882 | 793 | $0.00138 | $0.000069 |
| all | 32 | 27/32 (84%) | 12/12 | 0.4 | 4.7 | 23,879 | 2,998 | $0.00389 | $0.000121 |

## Per task

| Task | Success | Safe | Latency ms | In tokens | Out tokens | Cost | Note |
|---|---|---|---|---|---|---|---|
| extract-doc-001 | yes | yes | 3.7 | ~940 | ~198 | $0.000193 |  |
| extract-doc-012 | yes | yes | 1.9 | ~934 | ~190 | $0.000188 | right, but 1 field(s) sent to review |
| extract-doc-034 | yes | yes | 1.9 | ~934 | ~188 | $0.000187 | right, but 1 field(s) sent to review |
| extract-doc-003 | yes | yes | 2.4 | ~908 | ~191 | $0.000186 |  |
| extract-doc-005 | yes | yes | 1.9 | ~894 | ~183 | $0.000181 | right, but 2 field(s) sent to review |
| extract-doc-010 | yes | yes | 1.9 | ~900 | ~190 | $0.000185 |  |
| extract-doc-008 | yes | yes | 1.9 | ~957 | ~194 | $0.000193 | right, but 12 field(s) sent to review |
| extract-doc-017 | yes | yes | 1.5 | ~947 | ~184 | $0.000187 | right, but 12 field(s) sent to review |
| extract-doc-007 | yes | yes | 6.7 | ~901 | ~192 | $0.000186 | right, but 12 field(s) sent to review |
| extract-doc-013 | yes | yes | 5.9 | ~898 | ~189 | $0.000184 | right, but 12 field(s) sent to review |
| extract-doc-002 | no | yes | 0.6 | ~2,392 | ~153 | $0.000316 | wrong: 11 of 12 fields (all flagged) |
| extract-doc-011 | no | yes | 0.5 | ~2,392 | ~153 | $0.000316 | wrong: 12 of 12 fields (all flagged) |
| rag-01 | yes | n/a | 0.2 | ~318 | ~31 | $0.000047 |  |
| rag-02 | yes | n/a | 0.1 | ~311 | ~30 | $0.000046 |  |
| rag-03 | yes | n/a | 0.1 | ~310 | ~41 | $0.000051 |  |
| rag-04 | yes | n/a | 0.1 | ~341 | ~26 | $0.000047 |  |
| rag-05 | yes | n/a | 0.1 | ~298 | ~31 | $0.000045 |  |
| rag-06 | no | n/a | 0.1 | ~294 | ~33 | $0.000046 | answer missing ['CCP604194172']; got 'MEMBER NAME: Quinn Mockridge' |
| rag-07 | no | n/a | 0.1 | ~344 | ~10 | $0.000039 | said not found |
| rag-08 | yes | n/a | 0.1 | ~331 | ~10 | $0.000038 |  |
| rag-09 | no | n/a | 0.1 | ~272 | ~26 | $0.000040 | should have said not found, said 'DX CODES: J30.9' |
| rag-10 | yes | n/a | 0.1 | ~320 | ~10 | $0.000037 |  |
| rag-11 | yes | n/a | 0.4 | ~740 | ~87 | $0.000117 |  |
| rag-12 | yes | n/a | 0.4 | ~744 | ~55 | $0.000102 |  |
| rag-13 | yes | n/a | 0.4 | ~727 | ~54 | $0.000100 |  |
| rag-14 | yes | n/a | 0.4 | ~750 | ~61 | $0.000106 |  |
| rag-15 | yes | n/a | 0.4 | ~742 | ~56 | $0.000102 |  |
| rag-16 | yes | n/a | 0.3 | ~555 | ~44 | $0.000077 |  |
| rag-17 | yes | n/a | 0.3 | ~562 | ~50 | $0.000081 |  |
| rag-18 | yes | n/a | 0.3 | ~647 | ~41 | $0.000085 |  |
| rag-19 | yes | n/a | 0.3 | ~602 | ~10 | $0.000065 |  |
| rag-20 | yes | n/a | 0.3 | ~674 | ~87 | $0.000111 |  |

Definitions: an extract task succeeds when all twelve fields match ground truth, and
is safe when every wrong field was below the review threshold (so a person sees it
before release). A rag task succeeds when the answer contains the expected text and
cites the expected document with a quote that was found verbatim in the source, or,
for questions the documents can't answer, when it says not found.

<!-- embeddings:start -->
## Embedder comparison: hashing vs real

Run on 2026-11-17 with `python -m intake_agent.embed_eval` (in-memory store, mock extractive answerer, no API calls).

hit@k: a top-k chunk holds the expected text and comes from the expected document, searching the whole index (not limited to the task's doc). Main-set questions such as 'What is the member ID?' are written to be used with a doc_id filter and name no patient, so unscoped hit@k on them is capped by ambiguity, not by the embedder; the synonym cases name the patient or plan and are the fair test. Cited-answer accuracy is the extractive mock answerer over the same retrieval, scoped as the harness does.

| Embedder | Set | Tasks | hit@1 | hit@3 | hit@5 | Cited-answer accuracy |
|---|---|---|---|---|---|---|
| `hashing-v1-384` | main eval_set (rag) | 20 | 7/16 | 10/16 | 10/16 | 17/20 |
| `hashing-v1-384` | synonym cases | 5 | 1/5 | 2/5 | 2/5 | 1/5 |
| `fastembed-bge-small-en-v1.5` | main eval_set (rag) | 20 | 6/16 | 11/16 | 11/16 | 17/20 |
| `fastembed-bge-small-en-v1.5` | synonym cases | 5 | 1/5 | 4/5 | 5/5 | 1/5 |

### Vocabulary-miss cases (rank of the right chunk, `-` = not in top 10)

| Case | Question | Form wording | `hashing-v1-384` | `fastembed-bge-small-en-v1.5` |
|---|---|---|---|---|
| syn-01 | What is the member ID for Quinn Mockridge? | form says SUBSCRIBER ID | - | 2 |
| syn-02 | What is the subscriber ID for Jordan Stubbins? | form says Member ID | 7 | 2 |
| syn-03 | Which doctor referred Quinn Mockridge? | form says ORDERING PHYSICIAN | - | 2 |
| syn-04 | How quickly does Contoso Care decide a rush request? | policy says Expedited review | 2 | 4 |
| syn-05 | Does Contoso Care need pre-approval for a home sleep test? | policy says prior authorization, home sleep apnea test | 1 | 1 |

Index sizes: `hashing-v1-384` 86 chunks in 93 ms, `fastembed-bge-small-en-v1.5` 86 chunks in 1398 ms.
<!-- embeddings:end -->
