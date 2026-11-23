import json
import os
import sys
from pathlib import Path

os.environ.setdefault("BENCH_DIR", "unused")  # config.py requires it at import
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "evaluation"))
from evaluate import write_results  # noqa: E402


def _metrics(qt, recall):
    return {"model": "bm25", qt: {"gold": {"n_queries": 1, "recall@20": recall}}}


def test_single_query_type_keeps_other_stored_results(tmp_path):
    write_results("bm25", tmp_path, ["core_query", "subfield_query"],
                  {"model": "bm25",
                   "core_query": {"gold": {"n_queries": 1, "recall@20": 0.5}},
                   "subfield_query": {"gold": {"n_queries": 1, "recall@20": 0.25}}},
                  partial=False)
    # re-evaluate only subfield_query, as `evaluate.py --query-type subfield_query` does
    write_results("bm25", tmp_path, ["subfield_query"], _metrics("subfield_query", 0.75), partial=False)

    saved = json.loads((tmp_path / "results" / "bm25.json").read_text())
    assert saved["core_query"]["gold"]["recall@20"] == 0.5
    assert saved["subfield_query"]["gold"]["recall@20"] == 0.75

    summary = json.loads((tmp_path / "results" / "summary.json").read_text())["models"]["bm25"]
    assert summary["core_query"]["gold"]["recall@20"] == 0.5
    assert summary["subfield_query"]["gold"]["recall@20"] == 0.75
