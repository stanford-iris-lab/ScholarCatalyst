from __future__ import annotations

import json
import re
from pathlib import Path


def load_corpus(bench_dir: Path) -> dict[str, dict]:
    """Return {doc_id: {title, abstract, text, ...}} from corpus.jsonl."""
    corpus = {}
    for line in (bench_dir / "corpus.jsonl").read_text().splitlines():
        d = json.loads(line)
        corpus[d["id"]] = d
    return corpus


_TITLE_NORM = re.compile(r"[^a-z0-9]+")


def normalize_title(title: str) -> str:
    return _TITLE_NORM.sub(" ", (title or "").lower()).strip()


def title_index(corpus: dict[str, dict]) -> dict[str, list[str]]:
    """{normalized title: [corpus ids]} for a corpus, built once per run."""
    index: dict[str, list[str]] = {}
    for doc_id, doc in corpus.items():
        index.setdefault(normalize_title(doc.get("title", "")), []).append(doc_id)
    return index


def source_ids(paper_id: str, paper_title: str, index: dict[str, list[str]]) -> set[str]:
    """Every corpus id that is the query's own paper: its id, plus any other corpus entry with the same normalized
    title (a Semantic Scholar record of the paper next to, or instead of, its arXiv record). A title under four words
    is not matched by title, so a generic title cannot withhold an unrelated paper."""
    ids = {paper_id} if paper_id else set()
    key = normalize_title(paper_title)
    if len(key.split()) >= 4:
        ids.update(index.get(key, []))
    return ids


def source_alias_map(bench_dir: Path, queries: list[dict]) -> dict[str, set[str]]:
    """{query_id: source_ids} for scoring: the corpus titles are read once from corpus.jsonl."""
    index: dict[str, list[str]] = {}
    for line in (bench_dir / "corpus.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            index.setdefault(normalize_title(d.get("title", "")), []).append(d["id"])
    return {q.get("query_id") or q.get("id"): source_ids(q.get("paper_id", ""), q.get("paper_title", ""), index) for q in queries}


def load_queries(bench_dir: Path, query_type: str) -> list[dict]:
    """Return list of query records from queries.jsonl filtered by type, with
    positive_docs joined in from rels/{query_type}.jsonl -- queries.jsonl itself
    doesn't store them, rels/ is the single source of truth for answers."""
    rels_path = bench_dir / "rels" / f"{query_type}.jsonl"
    pos_docs_by_qid: dict[str, list] = {}
    if rels_path.exists():
        for line in rels_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            pos_docs_by_qid[r["query_id"]] = r.get("positive_docs") or []

    path = bench_dir / "queries.jsonl"
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("type") != query_type:
            continue
        d["query_id"] = d["id"]  # alias for evaluate.py compatibility
        d["positive_docs"] = pos_docs_by_qid.get(d["id"], [])
        records.append(d)
    return records


def load_run(bench_dir: Path, model_id: str, query_type: str) -> dict[str, list[dict]]:
    """Return {query_id: ranking} from runs/{category}/{model_id}/{query_type}.jsonl
    (category resolved via config.runs_subpath; agentic-style ids that already
    carry their own relative path are used as-is)."""
    from config import runs_subpath
    path = bench_dir / "runs" / runs_subpath(model_id) / f"{query_type}.jsonl"
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        d = json.loads(line)
        result[d["query_id"]] = d["ranking"]
    return result


def load_seen(bench_dir: Path, model_id: str, query_type: str) -> dict[str, list[str]]:
    """Return {query_id: seen_ids} for agentic runs that record the corpus ids the agent
    touched (agentic/runner.py writes them as "seen"); empty for every other run."""
    from config import runs_subpath
    path = bench_dir / "runs" / runs_subpath(model_id) / f"{query_type}.jsonl"
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if "seen" in d:
            result[d["query_id"]] = d["seen"]
    return result
