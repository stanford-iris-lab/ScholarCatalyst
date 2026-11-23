import argparse
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "evaluation"))
import retrieve_lateon  # noqa: E402

SELF_ID = "arxiv_2507.00001"


def _corpus():
    # 60 docs published after the query's paper rank first, then the query's own paper, then 140 older docs
    docs = [{"id": f"new{i}", "title": "t", "text": "x", "published": "2025-12-01"} for i in range(60)]
    docs.append({"id": SELF_ID, "title": "self", "text": "x", "published": "2025-07-09"})
    docs += [{"id": f"old{i}", "title": "t", "text": "x", "published": "2024-01-01"} for i in range(140)]
    return docs


def _install_fake_pylate(monkeypatch, order):
    class FakeModel:
        def __init__(self, **kw):
            pass

        def encode(self, texts, **kw):
            return list(texts)

    class FakeRetriever:
        """Returns the corpus in a fixed order, top k only, like a PLAID search."""

        def __init__(self, index):
            pass

        def retrieve(self, queries_embeddings, k):
            return [[{"id": d, "score": 1.0 - i / 1000} for i, d in enumerate(order[:k])]
                    for _ in queries_embeddings]

    pylate = types.ModuleType("pylate")
    pylate.models = types.SimpleNamespace(ColBERT=FakeModel)
    pylate.indexes = types.SimpleNamespace(PLAID=lambda **kw: None)
    pylate.retrieve = types.SimpleNamespace(ColBERT=FakeRetriever)
    monkeypatch.setitem(sys.modules, "pylate", pylate)


def test_rankings_are_refilled_and_drop_the_query_paper(tmp_path, monkeypatch):
    corpus = _corpus()
    (tmp_path / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in corpus))
    q = {"id": "q0", "paper_id": SELF_ID, "question": "q", "type": "core_query", "paper_published": "2025-07-09"}
    (tmp_path / "queries.jsonl").write_text(
        json.dumps(q) + "\n" + json.dumps({**q, "id": "q1", "type": "subfield_query"}))
    _install_fake_pylate(monkeypatch, [d["id"] for d in corpus])

    args = argparse.Namespace(set_name="", tag="lateon", model_path="m", query_length=32,
                              document_length=300, top_k=100, search_chunk=50, batch_size=64)
    retrieve_lateon.stage_retrieve(args, tmp_path)

    for qt in ("core_query", "subfield_query"):
        run = json.loads((tmp_path / "runs" / "multi_vector" / "lateon" / f"{qt}.jsonl").read_text())
        ids = [r["doc_id"] for r in run["ranking"]]
        assert SELF_ID not in ids
        assert ids == [f"old{i}" for i in range(100)]
