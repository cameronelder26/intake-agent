from datetime import date
from pathlib import Path

from intake_agent import killswitch
from intake_agent.extractor import MockExtractor
from intake_agent.pipeline import process_batch

TODAY = date(2026, 11, 10)


class CountingExtractor(MockExtractor):
    """Mock extractor that can pull the kill switch after N documents."""

    def __init__(self, settings, stop_after: int | None = None):
        self.settings = settings
        self.stop_after = stop_after
        self.calls = 0

    def extract(self, path: Path):
        self.calls += 1
        if self.stop_after is not None and self.calls == self.stop_after:
            killswitch.engage(self.settings, "operator stop", "test")
        return super().extract(path)


def pdfs(dataset: Path) -> list[Path]:
    return sorted(dataset.glob("*.pdf"))


def test_env_var_blocks_all_extraction(dataset, settings, state, monkeypatch):
    monkeypatch.setenv("INTAKE_KILL_SWITCH", "1")
    _, queue = state
    ex = CountingExtractor(settings)
    report = process_batch(pdfs(dataset), ex, queue, settings, TODAY)
    assert ex.calls == 0
    assert report.processed == []
    assert "INTAKE_KILL_SWITCH" in report.halted_reason


def test_kill_file_stops_a_running_batch_at_the_next_document(dataset, settings, state):
    audit, queue = state
    ex = CountingExtractor(settings, stop_after=2)
    report = process_batch(pdfs(dataset), ex, queue, settings, TODAY)
    assert ex.calls == 2
    assert len(report.processed) == 2
    assert report.halted_reason and "operator stop" in report.halted_reason
    assert audit.records()[-1]["action"] == "halted_by_kill_switch"


def test_queue_drains_while_engaged(dataset, settings, state):
    _, queue = state
    process_batch(pdfs(dataset), MockExtractor(), queue, settings, TODAY)
    killswitch.engage(settings, "incident", "test")
    assert killswitch.engaged(settings)

    # New extractions are blocked...
    ex = CountingExtractor(settings)
    process_batch(pdfs(dataset), ex, queue, settings, TODAY)
    assert ex.calls == 0

    # ...but review and release still work.
    for row in queue.pending():
        queue.approve_document(row["doc_id"], reviewer="alice")
    released = [
        queue.release(r["doc_id"])
        for status in ("auto_approved", "approved")
        for r in queue.documents(status)
    ]
    assert released
    assert queue.pending() == []


def test_disengage(settings):
    killswitch.engage(settings, "x", "test")
    assert killswitch.disengage(settings)
    assert killswitch.engaged(settings) is None
