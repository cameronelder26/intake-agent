"""Build a LangSmith dataset from the 32 eval tasks and run an evaluation on it.

    LANGSMITH_API_KEY=... python evals/langsmith_dataset.py [--mock] [--reviewer]

Runs only when LANGSMITH_API_KEY is set; with no key it prints why and exits 0, so CI and
anyone without an account are unaffected. Nothing is created or sent otherwise.

Dataset "intake-agent-eval-32": one example per task. Extract tasks have the ground truth
fields as the reference output. RAG tasks carry the question and the strings the answer must
contain. The evaluation target runs the same code as `python -m intake_agent.harness`
(mock by default) and three evaluators score the run: field accuracy, silent errors, and
RAG pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

DATA = Path("data/synthetic")
POLICIES = Path("data/policies")
TASKS = Path("evals/eval_set.jsonl")
DATASET = "intake-agent-eval-32"


def load_tasks() -> list[dict]:
    return [json.loads(line) for line in TASKS.read_text().splitlines() if line.strip()]


def example_for(task: dict) -> dict:
    """One LangSmith example: inputs, reference outputs, metadata."""
    if task["kind"] == "extract":
        truth = json.loads((DATA / f"{task['doc_id']}.json").read_text())["fields"]
        return {
            "inputs": {"kind": "extract", "doc_id": task["doc_id"]},
            "outputs": {"fields": truth},
            "metadata": {"task_id": task["id"]},
        }
    return {
        "inputs": {
            "kind": "rag",
            "question": task["question"],
            "doc_id": task.get("doc_id"),
            "scope": task.get("scope"),
        },
        "outputs": {
            "must_contain": task.get("must_contain", []),
            "must_cite": task.get("must_cite"),
            "expect_not_found": bool(task.get("expect_not_found")),
        },
        "metadata": {"task_id": task["id"]},
    }


def build_dataset(client, tasks: list[dict]):
    if client.has_dataset(dataset_name=DATASET):
        return client.read_dataset(dataset_name=DATASET)
    ds = client.create_dataset(DATASET, description="intake-agent 32-task eval set, synthetic data")
    ex = [example_for(t) for t in tasks]
    client.create_examples(
        dataset_id=ds.id,
        inputs=[e["inputs"] for e in ex],
        outputs=[e["outputs"] for e in ex],
        metadata=[e["metadata"] for e in ex],
    )
    return ds


def make_target(mock: bool, reviewer: bool):
    from intake_agent.config import Settings
    from intake_agent.extractor import make_extractor
    from intake_agent.graph import make_checkpointer, run_document
    from intake_agent.multiagent import make_reviewer
    from intake_agent.pipeline import open_state
    from intake_agent.rag import HashingEmbedder, RagIndex, load_corpus, make_answerer, make_store

    tmp = Path(tempfile.mkdtemp())
    settings = Settings(state_dir=tmp, pg_dsn="")
    _, queue = open_state(settings)
    extractor = make_extractor(mock, settings)
    rv = make_reviewer(mock, settings) if reviewer else None
    emb = HashingEmbedder()
    index = RagIndex(emb, make_store(emb.dim, ""))
    index.build(load_corpus(DATA, POLICIES)[0])
    answerer = make_answerer(mock, settings)

    def target(inputs: dict) -> dict:
        if inputs["kind"] == "extract":
            st = run_document(
                DATA / f"{inputs['doc_id']}.pdf",
                extractor,
                queue,
                settings,
                doc_id=f"{inputs['doc_id']}-{os.urandom(3).hex()}",
                reviewer=rv,
                checkpointer=make_checkpointer(None),
            )
            return {
                "fields": {n: f.value for n, f in st["scored"].items()},
                "flagged": st["flagged"],
                "route": st["route"],
            }
        prefix = "policy:" if inputs.get("scope") == "policy" else None
        hits = index.retrieve(inputs["question"], k=5, doc_id=inputs.get("doc_id"), prefix=prefix)
        ans = answerer.answer(inputs["question"], hits)
        return {
            "answer": ans.text,
            "grounded": ans.grounded,
            "cited": [c.doc_id for c in ans.citations],
        }

    return target


def field_accuracy(outputs: dict, reference_outputs: dict) -> dict:
    from intake_agent.eval import values_match

    if "fields" not in reference_outputs:
        return {"key": "field_accuracy", "score": None}
    truth = reference_outputs["fields"]
    ok = sum(values_match(n, outputs["fields"].get(n), v) for n, v in truth.items())
    return {"key": "field_accuracy", "score": ok / len(truth)}


def silent_errors(outputs: dict, reference_outputs: dict) -> dict:
    from intake_agent.eval import values_match

    if "fields" not in reference_outputs:
        return {"key": "silent_errors", "score": None}
    wrong = [
        n
        for n, v in reference_outputs["fields"].items()
        if not values_match(n, outputs["fields"].get(n), v)
    ]
    return {"key": "silent_errors", "score": len([n for n in wrong if n not in outputs["flagged"]])}


def rag_pass(outputs: dict, reference_outputs: dict) -> dict:
    if "must_contain" not in reference_outputs:
        return {"key": "rag_pass", "score": None}
    if reference_outputs.get("expect_not_found"):
        return {"key": "rag_pass", "score": float(not outputs["grounded"])}
    text = outputs["answer"].lower()
    ok = outputs["grounded"] and all(s.lower() in text for s in reference_outputs["must_contain"])
    cite = reference_outputs.get("must_cite")
    ok = ok and (cite is None or cite in outputs["cited"])
    return {"key": "rag_pass", "score": float(ok)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="rule-based extractor and answerer")
    ap.add_argument("--reviewer", action="store_true", help="put the reviewer agent in the graph")
    args = ap.parse_args(argv)
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY is not set: nothing to do. See LANGSMITH-STEPS.md.")
        return 0
    from langsmith import Client, evaluate

    mock = args.mock or not os.environ.get("ANTHROPIC_API_KEY")
    client = Client()
    build_dataset(client, load_tasks())
    result = evaluate(
        make_target(mock, args.reviewer),
        data=DATASET,
        evaluators=[field_accuracy, silent_errors, rag_pass],
        experiment_prefix="intake-"
        + ("mock" if mock else "model")
        + ("-reviewer" if args.reviewer else ""),
        max_concurrency=1,
    )
    print(f"experiment: {result.experiment_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
