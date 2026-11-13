"""Eval harness: runs the fixed task set in mock mode and checks the numbers add up."""

from datetime import date
from pathlib import Path

import pytest

from intake_agent.harness import load_tasks, main, price, render, run_harness

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "evals" / "eval_set.jsonl"


@pytest.fixture(scope="module")
def tasks():
    return load_tasks(TASKS)


def run(tasks, settings, prices=(0.10, 0.50)):
    return run_harness(
        tasks, True, settings, prices, ROOT / "data" / "synthetic", ROOT / "data" / "policies"
    )


def test_eval_set_is_fixed_and_well_formed(tasks):
    assert len(tasks) == 32
    assert len({t["id"] for t in tasks}) == len(tasks)
    assert {t["kind"] for t in tasks} == {"extract", "rag"}
    for t in tasks:
        if t["kind"] == "extract":
            assert (ROOT / "data" / "synthetic" / f"{t['doc_id']}.pdf").exists()
        elif not t.get("expect_not_found"):
            assert t["must_contain"] and t["must_cite"]


def test_mock_run_measures_every_task(tasks, settings):
    results, info = run(tasks, settings)
    assert [r.id for r in results] == [t["id"] for t in tasks]
    for r in results:
        assert r.latency_ms > 0
        assert r.estimated and r.input_tokens > 0 and r.output_tokens > 0
        assert r.cost_usd == pytest.approx(price(r.input_tokens, r.output_tokens, 0.10, 0.50))
    # The safety property: nothing the mock got wrong escaped the review gate.
    assert all(r.safe for r in results if r.kind == "extract")
    assert info.index_chunks > 0 and info.store == "in-memory"


def test_price_is_configurable(tasks, settings):
    cheap, _ = run(tasks[:3], settings, (0.10, 0.50))
    dear, _ = run(tasks[:3], settings, (1.00, 5.00))
    for a, b in zip(cheap, dear, strict=True):
        assert b.cost_usd == pytest.approx(10 * a.cost_usd)


def test_render_has_summary_and_rows(tasks, settings):
    results, info = run(tasks, settings)
    md = render(results, info, date(2026, 11, 11))
    assert "| all | 32 |" in md
    assert md.count("\n| rag-") == 20 and md.count("\n| extract-") == 12
    assert "projection, not a bill" in md


def test_cli_writes_results(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("INTAKE_PG_DSN", raising=False)
    out = tmp_path / "RESULTS.md"
    assert main(["--mock", "--out", str(out), "--price-in", "2", "--price-out", "10"]) == 0
    assert "$2.00 in / $10.00 out" in out.read_text()


def test_model_mode_needs_a_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["--out", str(tmp_path / "x.md")]) == 2
