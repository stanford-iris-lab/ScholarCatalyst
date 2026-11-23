import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("BENCH_DIR", "unused")  # config.py requires it at import
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "evaluation"))
import llm_rerank  # noqa: E402

CANDIDATES = [{"id": f"d{i}", "title": f"t{i}"} for i in range(5)]


@pytest.mark.parametrize("reply", ['[{"id": "d2"}, "d1"]', "5", "null"])
def test_tournament_falls_back_on_malformed_reply(monkeypatch, reply):
    monkeypatch.setattr(llm_rerank, "chat", lambda *a, **k: reply)
    trace = []
    ranked = llm_rerank.rerank_tournament("q", CANDIDATES, client=None, trace=trace)
    assert sorted(d["id"] for d in ranked) == sorted(d["id"] for d in CANDIDATES)
    assert trace and not any(t["parsed_ok"] for t in trace)


def test_tournament_uses_a_valid_reply(monkeypatch):
    monkeypatch.setattr(llm_rerank, "chat", lambda *a, **k: '["d3", "d1"]')
    ranked = llm_rerank.rerank_tournament("q", CANDIDATES, client=None)
    assert [d["id"] for d in ranked][:2] == ["d3", "d1"]
