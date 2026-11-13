"""Command line: run the pipeline, work the review queue, release, kill switch."""

from __future__ import annotations

import getpass
import json
import os
import re
from pathlib import Path
from typing import Annotated

import typer

from . import killswitch
from .config import Settings
from .extractor import make_extractor
from .pdftext import collapse_fragments, extract_raw_text, has_usable_text
from .pipeline import open_state, process_batch
from .review_queue import QueueError, parse_correction
from .schema import FIELDS
from .synth import generate

app = typer.Typer(no_args_is_help=True, help="intake-agent: referral and prior-auth intake.")
queue_app = typer.Typer(no_args_is_help=True, help="Review queue.")
kill_app = typer.Typer(no_args_is_help=True, help="Kill switch for new extractions.")
audit_app = typer.Typer(no_args_is_help=True, help="Audit log.")
rag_app = typer.Typer(no_args_is_help=True, help="Ask the documents and payer policies.")
app.add_typer(queue_app, name="queue")
app.add_typer(kill_app, name="kill")
app.add_typer(audit_app, name="audit")
app.add_typer(rag_app, name="rag")

Reviewer = Annotated[
    str,
    typer.Option(
        "--reviewer",
        "-r",
        help="Name recorded in the audit log.",
        default_factory=lambda: os.environ.get("INTAKE_REVIEWER") or getpass.getuser(),
    ),
]


def _fail(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


@app.command("generate")
def generate_cmd(out: Path = Path("data/synthetic"), n: int = 40, seed: int = 7) -> None:
    """Generate synthetic PDFs and ground truth."""
    paths = generate(out, n, seed)
    typer.echo(f"wrote {len(paths)} documents to {out}")


@app.command()
def run(
    data: Path = Path("data/synthetic"),
    mock: Annotated[bool, typer.Option(help="Use the rule-based mock, no API calls.")] = False,
    limit: int = 0,
) -> None:
    """Extract every PDF in DATA, gate on confidence, and queue anything uncertain."""
    settings = Settings()
    if not mock and not os.environ.get("ANTHROPIC_API_KEY"):
        _fail("ANTHROPIC_API_KEY is not set. Use --mock to run without the API.")
    _, queue = open_state(settings)
    paths = sorted(data.glob("*.pdf"))
    if limit:
        paths = paths[:limit]
    report = process_batch(paths, make_extractor(mock, settings), queue, settings)
    typer.echo(
        f"processed {len(report.processed)}  auto-approved {len(report.auto_approved)}  "
        f"queued {len(report.pending_review)}  skipped {len(report.skipped)}  "
        f"failed {len(report.failed)}  cost ${report.cost_usd:.4f}"
    )
    if report.halted_reason:
        typer.secho(f"HALTED by kill switch ({report.halted_reason})", fg=typer.colors.YELLOW)


@queue_app.command("list")
def queue_list() -> None:
    """Documents waiting for a human."""
    _, queue = open_state(Settings())
    rows = queue.pending()
    if not rows:
        typer.echo("queue is empty")
        return
    typer.echo(f"{'doc':<10} {'open fields':>11} {'min conf':>9}")
    for r in rows:
        typer.echo(f"{r['doc_id']:<10} {r['open_fields']:>11} {r['min_confidence']:>9.2f}")


def _context(text: str, needles: list[str], width: int = 2) -> list[str]:
    lines = text.split("\n")
    hits = set()
    for i, line in enumerate(lines):
        if any(n and n.lower() in line.lower() for n in needles):
            hits.update(range(max(0, i - width), min(len(lines), i + width + 1)))
    return [lines[i] for i in sorted(hits)]


@queue_app.command("show")
def queue_show(
    doc_id: str, text: bool = typer.Option(False, help="Print the full source text.")
) -> None:
    """Every field with its confidence, and source text around each flagged field."""
    _, queue = open_state(Settings())
    try:
        doc = queue.document(doc_id)
    except QueueError as exc:
        _fail(str(exc))
    items = {i["field"]: i for i in queue.items(doc_id)}
    source = collapse_fragments(extract_raw_text(Path(doc["source_path"])))
    typer.echo(
        f"{doc_id}  status={doc['status']}  model={doc['model']}  source={doc['source_path']}\n"
    )
    for name in FIELDS:
        f = doc["fields"][name]
        item = items.get(name)
        mark = f"[{item['status'].upper()}]" if item else ""
        issues = f"  ({'; '.join(f['issues'])})" if f["issues"] else ""
        typer.echo(f"{mark:<12} {name:<19} {f['confidence']:.2f}  {json.dumps(f['value'])}{issues}")
    if not has_usable_text(source):
        typer.echo("\nNo text layer (scanned). Open the PDF to review: " + doc["source_path"])
        return
    typer.echo("")
    for name, item in items.items():
        if item["status"] != "open":
            continue
        value = item["value"]
        needles = value if isinstance(value, list) else [value or ""]
        needles += re.split(r"[_ ]", name)[:1]
        typer.secho(f"--- source near {name}", bold=True)
        for line in _context(source, [str(n) for n in needles])[:8]:
            typer.echo(f"    {line}")
    if text:
        typer.echo("\n--- full source text\n" + source)


@queue_app.command("approve")
def queue_approve(
    doc_id: str,
    reviewer: Reviewer,
    field: Annotated[str | None, typer.Option(help="Approve one field only.")] = None,
) -> None:
    """Approve one field, or every open field on the document, as extracted."""
    _, queue = open_state(Settings())
    try:
        if field:
            queue.approve_field(doc_id, field, reviewer)
        else:
            queue.approve_document(doc_id, reviewer)
    except QueueError as exc:
        _fail(str(exc))
    typer.echo(f"{doc_id}: {queue.document(doc_id)['status']}")


@queue_app.command("correct")
def queue_correct(doc_id: str, field: str, value: str, reviewer: Reviewer) -> None:
    """Replace a field value. Codes are comma separated; 'null' clears the field."""
    _, queue = open_state(Settings())
    try:
        queue.correct_field(doc_id, field, parse_correction(field, value), reviewer)
    except QueueError as exc:
        _fail(str(exc))
    typer.echo(f"{doc_id}: {queue.document(doc_id)['status']}")


@queue_app.command("reject")
def queue_reject(
    doc_id: str, reason: Annotated[str, typer.Option(help="Why.")], reviewer: Reviewer
) -> None:
    """Reject the document. It will never be released."""
    _, queue = open_state(Settings())
    try:
        queue.reject_document(doc_id, reviewer, reason)
    except QueueError as exc:
        _fail(str(exc))
    typer.echo(f"{doc_id}: rejected")


@app.command()
def release(out: Path = Path("out/released")) -> None:
    """Write every approved record downstream (here: one JSON file per document)."""
    _, queue = open_state(Settings())
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for status in ("auto_approved", "approved"):
        for row in queue.documents(status):
            record = queue.release(row["doc_id"])
            (out / f"{row['doc_id']}.json").write_text(json.dumps(record, indent=2) + "\n")
            count += 1
    typer.echo(f"released {count} documents to {out}")


@app.command()
def status() -> None:
    """Document counts by status, and the kill switch state."""
    settings = Settings()
    _, queue = open_state(settings)
    counts: dict[str, int] = {}
    for row in queue.documents():
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    for name, n in sorted(counts.items()):
        typer.echo(f"{name:<15} {n}")
    typer.echo(f"kill switch     {killswitch.engaged(settings) or 'off'}")


@kill_app.command("on")
def kill_on(reviewer: Reviewer, reason: str = "manual stop") -> None:
    """Stop new extractions. Review and release keep working."""
    settings = Settings()
    audit, _ = open_state(settings)
    killswitch.engage(settings, reason, reviewer)
    audit.append(actor=reviewer, action="kill_switch_on", detail={"reason": reason})
    typer.echo(f"kill switch ON ({settings.kill_file})")


@kill_app.command("off")
def kill_off(reviewer: Reviewer) -> None:
    """Allow extractions again (the env var, if set, still wins)."""
    settings = Settings()
    audit, _ = open_state(settings)
    if killswitch.disengage(settings):
        audit.append(actor=reviewer, action="kill_switch_off")
    typer.echo(f"kill switch: {killswitch.engaged(settings) or 'off'}")


@kill_app.command("status")
def kill_status() -> None:
    typer.echo(killswitch.engaged(Settings()) or "off")


@audit_app.command("verify")
def audit_verify() -> None:
    """Check the hash chain."""
    audit, _ = open_state(Settings())
    ok, bad = audit.verify()
    n = len(audit.records())
    if ok:
        typer.echo(f"ok: {n} records, chain intact")
    else:
        _fail(f"chain broken at seq {bad}")


@audit_app.command("tail")
def audit_tail(n: Annotated[int, typer.Option("-n", "--lines")] = 10) -> None:
    """Print the last N audit records."""
    audit, _ = open_state(Settings())
    for rec in audit.records()[-n:]:
        typer.echo(
            json.dumps(
                {
                    k: rec[k]
                    for k in ("seq", "ts", "actor", "action", "doc_id", "field", "before", "after")
                }
            )
        )


def _rag_index(settings: Settings, data: Path, policies: Path, rebuild: bool):
    from .rag import HashingEmbedder, RagIndex, load_corpus, make_store

    emb = HashingEmbedder()
    idx = RagIndex(emb, make_store(emb.dim, settings.pg_dsn))
    if rebuild or idx.store.count() == 0:
        idx.store.clear()
        sources, skipped = load_corpus(data, policies)
        n = idx.build(sources)
        backend = "pgvector" if settings.pg_dsn else "memory"
        typer.echo(
            f"indexed {n} chunks from {len(sources)} sources into {backend}; "
            f"skipped {len(skipped)} scans with no text layer",
            err=True,
        )
    return idx


@rag_app.command("index")
def rag_index(data: Path = Path("data/synthetic"), policies: Path = Path("data/policies")) -> None:
    """(Re)build the index. Persists only when INTAKE_PG_DSN points at Postgres + pgvector."""
    settings = Settings()
    _rag_index(settings, data, policies, rebuild=True)
    if not settings.pg_dsn:
        typer.echo("INTAKE_PG_DSN is not set, so the in-memory index was not kept.", err=True)


@rag_app.command("ask")
def rag_ask(
    question: str,
    reviewer: Reviewer,
    doc: Annotated[str | None, typer.Option(help="Only search this document id.")] = None,
    policies_only: Annotated[
        bool, typer.Option("--policy-only", help="Only search payer policies.")
    ] = False,
    k: int = 5,
    mock: Annotated[bool, typer.Option(help="Extractive answerer, no API calls.")] = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
    data: Path = Path("data/synthetic"),
    policies: Path = Path("data/policies"),
) -> None:
    """Answer QUESTION from the indexed text, citing doc id and character span."""
    from .rag import make_answerer

    settings = Settings()
    if not mock and not os.environ.get("ANTHROPIC_API_KEY"):
        _fail("ANTHROPIC_API_KEY is not set. Use --mock to run without the API.")
    idx = _rag_index(settings, data, policies, rebuild=False)
    hits = idx.retrieve(question, k=k, doc_id=doc, prefix="policy:" if policies_only else None)
    ans = make_answerer(mock, settings).answer(question, hits)
    audit, _ = open_state(settings)
    audit.append(
        actor=reviewer,
        action="rag_answer",
        doc_id=doc,
        model=ans.model,
        detail={
            "question": question,
            "grounded": ans.grounded,
            "citations": [c.model_dump() for c in ans.citations],
            "dropped_citations": ans.dropped_citations,
        },
    )
    if as_json:
        typer.echo(ans.model_dump_json(indent=2))
        return
    typer.echo(ans.text)
    for c in ans.citations:
        typer.echo(f"  [{c.doc_id} {c.start}-{c.end}] {c.quote!r}")


if __name__ == "__main__":
    app()
