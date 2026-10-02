"""Assemble one depth 100 ranking: the agent's own ids, then up to TRAJECTORY_CAP ids it read, then a retriever backfill.

Ids outside the corpus, the query paper, papers after the query date and duplicates are dropped; each item records its segment.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieve import temporal_filter

DEPTH = 100
TRAJECTORY_CAP = 25  # ids the agent merely saw; capped so the back-fill keeps most of the tail
_TOKEN = re.compile(r"[A-Za-z0-9_.:/-]+")


def parse_ranked_ids(text: str, known_ids) -> list[str]:
    """Corpus ids mentioned in an agent's answer, in order of first mention, deduped.
    Works on JSON arrays, bullet lists and prose; anything that is not a corpus id is ignored."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in _TOKEN.findall(text or ""):
        tok = tok.rstrip(".,:")
        if tok in known_ids and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def as_ids(paper_id) -> set[str]:
    """The query's own paper as a set of corpus ids (a bare id or an iterable of ids; see utils.source_ids)."""
    ids = {paper_id} if isinstance(paper_id, str) else set(paper_id)
    return ids - {""}


def withhold(retrieve, paper_id):
    """A search tool that never returns the query's own paper (any of its corpus ids): the month-granular temporal
    filter lets a paper of the query's month through, and the assembly below only drops it from the final ranking.
    The search agents wrap their retriever with this and refuse the paper in their read tools too."""
    withheld = as_ids(paper_id)
    def retrieve_without(query: str, top_k: int, query_date: str = ""):
        docs = retrieve(query, top_k + len(withheld), query_date)
        return [d for d in docs if d["id"] not in withheld][:top_k]
    return retrieve_without


def assemble(agent_ids, seen_ids, backfill_ids, *, known_ids, paper_id,
             query_date: str, corpus_dates: dict[str, str], depth: int = DEPTH) -> list[dict]:
    """Merge the three segments into one ranking of at most `depth` items; `paper_id` is the query's own paper,
    a bare id or every corpus id of it."""
    used = as_ids(paper_id)
    out: list[dict] = []
    for source, ids, cap in (("agent", agent_ids, None), ("trajectory", seen_ids, TRAJECTORY_CAP), ("backfill", backfill_ids, None)):
        taken = 0
        for doc_id in ids:
            if doc_id in used or doc_id not in known_ids or (cap is not None and taken >= cap):
                continue
            used.add(doc_id)
            taken += 1
            out.append({"doc_id": doc_id, "source": source})
    return temporal_filter(out, query_date, corpus_dates)[:depth]


def backfill_ids(retrieve_fn, question: str, query_date: str, want: int = DEPTH) -> list[str]:
    """Ids from a retrieve.build_retrieve_fn retriever for the original question.
    The retriever applies the date filter after cutting to top_k, so ask for more."""
    if not question.strip():
        return []
    return [d["id"] for d in retrieve_fn(question, want * 3, query_date)][:want]
