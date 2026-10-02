from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import BENCH_DIR, MODEL_REGISTRY, QUERY_TYPES, TOP_K, runs_subpath
from encoder import get_encoder
from run_report import run_report


def parse_ym(date_str: str, doc_id: str = "") -> tuple[int, int] | None:
    """Parse (year, month) from various date formats. Returns None if unparseable."""
    if not date_str:
        # fallback: extract from arxiv ID (YYMM.NNNNN)
        m = re.match(r"^arxiv_(\d{2})(\d{2})\.\d+", doc_id)
        if m:
            yy, mm = int(m.group(1)), int(m.group(2))
            year = 2000 + yy if yy <= 30 else 1900 + yy
            return year, mm
        return None
    # ISO datetime: 2023-10-12T...
    m = re.match(r"^(\d{4})-(\d{2})", date_str)
    if m:
        return int(m.group(1)), int(m.group(2))
    # year only: "2023"
    m = re.match(r"^(\d{4})$", date_str)
    if m:
        return int(m.group(1)), 0
    return None


def load_queries(bench_dir: Path, query_type: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """Returns (qids, texts, paper_published_dates, paper_ids)."""
    path = bench_dir / "queries.jsonl"
    qids, texts, dates, papers = [], [], [], []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        if d.get("type") != query_type:
            continue
        qids.append(d["id"])
        texts.append(d.get("question") or "")
        dates.append(d.get("paper_published", ""))
        papers.append(d.get("paper_id", ""))
    return qids, texts, dates, papers


def _paper_ids(paper_id) -> set[str]:
    return ({paper_id} if isinstance(paper_id, str) else set(paper_id)) - {""}


def drop_paper(ranking: list[dict], paper_id) -> list[dict]:
    """The ranking without the query's own paper (a bare id or every corpus id of it). The temporal filter is
    month-granular, so a source paper published in the query's month passes it; it is never a gold paper and
    must not be retrievable, so every ranking written or served here drops it by id."""
    ids = _paper_ids(paper_id)
    return [item for item in ranking if item["doc_id"] not in ids] if ids else ranking


def without_paper(retrieve_fn, paper_id):
    """A retriever bound to one query that never returns the query's own paper (extra candidates are requested
    so the top-k stays full). For the reranking and query-augmentation pipelines, which call the shared
    retriever per query."""
    ids = _paper_ids(paper_id)
    if not ids:
        return retrieve_fn

    def retrieve(query: str, top_k: int, query_date: str = "") -> list[dict]:
        return [d for d in retrieve_fn(query, top_k + len(ids), query_date) if d["id"] not in ids][:top_k]

    return retrieve


def load_corpus_dates(bench_dir: Path) -> dict[str, str]:
    """Load doc_id → published date from corpus.jsonl."""
    path = bench_dir / "corpus.jsonl"
    dates: dict[str, str] = {}
    for line in path.read_text().splitlines():
        d = json.loads(line)
        pub = d.get("published", "")
        if pub:
            dates[d["id"]] = pub
    return dates


def temporal_filter(
    ranking: list[dict],
    query_date: str,
    corpus_dates: dict[str, str],
) -> list[dict]:
    """Remove docs published after (strictly later month than) query_date."""
    q_ym = parse_ym(query_date)
    if q_ym is None:
        return ranking
    q_year, q_month = q_ym
    filtered = []
    for item in ranking:
        doc_id = item["doc_id"]
        d_ym = parse_ym(corpus_dates.get(doc_id, ""), doc_id)
        if d_ym is None:
            filtered.append(item)  # unknown date → pass-through
            continue
        d_year, d_month = d_ym
        if d_year < q_year:
            filtered.append(item)
        elif d_year == q_year:
            if d_month == 0 or q_month == 0 or d_month <= q_month:
                filtered.append(item)  # same month = pass-through
    return filtered


def faiss_search(index, query_embs: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    import faiss  # noqa: F401
    scores, indices = index.search(query_embs.astype(np.float32), top_k)
    return scores, indices


def build_retrieve_fn(corpus: list[dict], corpus_map: dict[str, dict],
                       bench_dir: Path, retriever: str):
    """Build a (query, top_k, query_date="") -> list[doc] retriever (bm25 or dense faiss) for the LLM pipelines.

    A query_date applies temporal_filter, so docs published after the query's own paper are never returned."""
    corpus_dates = load_corpus_dates(bench_dir)

    def apply_date_filter(docs: list[dict], query_date: str) -> list[dict]:
        if not query_date:
            return docs
        ranking = [{"doc_id": d["id"]} for d in docs]
        keep = {item["doc_id"] for item in temporal_filter(ranking, query_date, corpus_dates)}
        return [d for d in docs if d["id"] in keep]

    if retriever == "bm25":
        # same bm25s index run_bm25() builds and searches, so reranking
        # pipelines see exactly the candidates the bm25 baseline run produced
        import bm25s
        import threading
        idx_dir = bench_dir / "embeddings" / "bm25s"
        if not (idx_dir / "doc_ids.json").exists():
            raise FileNotFoundError(
                f"bm25s index missing at {idx_dir}; run retrieve.py --model bm25 once to build it")
        bm = bm25s.BM25.load(str(idx_dir), mmap=True)
        bm_doc_ids = json.loads((idx_dir / "doc_ids.json").read_text())
        _bm25_lock = threading.Lock()

        def search(query: str, k: int) -> list[dict]:
            with _bm25_lock:
                idx_mat, _ = bm.retrieve(bm25s.tokenize([query], show_progress=False),
                                         k=k, show_progress=False)
            return [corpus_map[bm_doc_ids[int(j)]] for j in idx_mat[0]
                    if bm_doc_ids[int(j)] in corpus_map]

        def retrieve(query: str, top_k: int, query_date: str = "") -> list[dict]:
            if not query.strip():
                return []
            max_k = len(bm_doc_ids)
            k = min(top_k, max_k)  # bm25s rejects k above the index size
            if not query_date:
                return search(query, k)
            # widen the search depth until enough hits survive the date filter
            while True:
                filtered = apply_date_filter(search(query, k), query_date)
                if len(filtered) >= top_k or k >= max_k:
                    return filtered[:top_k]
                k = min(k * 4, max_k)

        return retrieve

    import faiss
    import threading
    from encoder import get_encoder
    emb_dir     = bench_dir / "embeddings" / retriever
    index       = faiss.read_index(str(emb_dir / "index.faiss"))
    doc_ids     = json.loads((emb_dir / "doc_ids.json").read_text())
    enc         = get_encoder(retriever)
    _dense_lock = threading.Lock()

    def search(q_emb: np.ndarray, k: int) -> list[dict]:
        with _dense_lock:
            scores, indices = index.search(q_emb, k)
        return [corpus_map[doc_ids[j]] for j in indices[0] if j >= 0 and doc_ids[j] in corpus_map]

    def retrieve(query: str, top_k: int, query_date: str = "") -> list[dict]:
        if not query.strip():
            return []
        q_emb = enc.encode_queries([query]).astype(np.float32)
        if not query_date:
            return search(q_emb, top_k)
        # widen the search depth until enough hits survive the date filter
        max_k = index.ntotal
        k = top_k
        while True:
            filtered = apply_date_filter(search(q_emb, k), query_date)
            if len(filtered) >= top_k or k >= max_k:
                return filtered[:top_k]
            k = min(k * 4, max_k)

    return retrieve


def run_bm25(emb_dir: Path, out_dir: Path, data_dir: Path, top_k: int, corpus_dates: dict[str, str]) -> int:
    # bm25s (sparse matrix scoring) replaces rank_bm25, whose pure python
    # get_scores took seconds per query at this corpus size
    import bm25s

    bench_dir = emb_dir.parent.parent
    idx_dir   = emb_dir.parent / "bm25s"
    if (idx_dir / "doc_ids.json").exists():
        retriever = bm25s.BM25.load(str(idx_dir), mmap=True)
        doc_ids   = json.loads((idx_dir / "doc_ids.json").read_text())
    else:
        print("  [bm25] building bm25s index from corpus.jsonl (one time)")
        doc_ids, corpus_texts = [], []
        for line in (bench_dir / "corpus.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            doc_ids.append(d["id"])
            corpus_texts.append(d.get("text") or "")
        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize(corpus_texts))
        idx_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(idx_dir))
        (idx_dir / "doc_ids.json").write_text(json.dumps(doc_ids))
        print(f"  [bm25] index saved → {idx_dir}")

    n_total = 0
    max_k = len(doc_ids)
    for qt in QUERY_TYPES:
        qids, texts, qdates, qpapers = load_queries(data_dir, qt)
        # bm25s cannot score an all-empty token list, and empty questions
        # scored zero under rank_bm25 anyway, so give them a no-match token
        safe_texts = [t if t.strip() else "emptyquestionplaceholder" for t in texts]

        # Date filtering can drop hits, so widen the search depth for short queries until each has top_k or the index is exhausted.
        final_rankings: list[list[dict] | None] = [None] * len(qids)
        pending = list(range(len(qids)))
        k = top_k
        while pending:
            pending_texts = [safe_texts[i] for i in pending]
            idx_mat, score_mat = retriever.retrieve(
                bm25s.tokenize(pending_texts, show_progress=False), k=k)
            still_pending = []
            for local_i, global_i in enumerate(pending):
                ranking = [{"doc_id": doc_ids[int(j)], "score": float(s)}
                           for j, s in zip(idx_mat[local_i], score_mat[local_i])]
                filtered = drop_paper(temporal_filter(ranking, qdates[global_i], corpus_dates), qpapers[global_i])
                if len(filtered) >= top_k or k >= max_k:
                    final_rankings[global_i] = filtered[:top_k]
                else:
                    still_pending.append(global_i)
            pending = still_pending
            k = min(k * 4, max_k)

        results = [{"query_id": qid, "ranking": final_rankings[i]} for i, qid in enumerate(qids)]
        run_file = out_dir / f"{qt}.jsonl"
        run_file.write_text("\n".join(json.dumps(r) for r in results))
        print(f"  [{qt}] {len(results)} queries → {run_file}")
        n_total += len(results)
    return n_total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",              required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--bench-dir",          type=Path, default=BENCH_DIR)
    ap.add_argument("--set",                default="", dest="set_name",
                    help="query set subdir under bench-dir (e.g. llm_set, author_set,"
                         " author_set/full_text); queries/rels are read from it and"
                         " runs are written into it, while corpus.jsonl and"
                         " embeddings/ stay shared at bench-dir root")
    ap.add_argument("--top-k",              type=int, default=TOP_K)
    args = ap.parse_args()

    if MODEL_REGISTRY[args.model].get("type") == "multi_vector":
        ap.error(f"{args.model} is a multi-vector model; use retrieve_lateon.py --stage retrieve")

    bench_dir    = args.bench_dir.resolve()
    data_dir     = (bench_dir / args.set_name) if args.set_name else bench_dir
    emb_dir      = bench_dir / "embeddings" / args.model
    out_dir      = data_dir / "runs" / runs_subpath(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_dates = load_corpus_dates(bench_dir)
    print(f"[retrieve] model={args.model}  top_k={args.top_k}  corpus dates loaded: {len(corpus_dates)} entries")

    t0 = time.monotonic()

    if args.model == "bm25":
        n_total = run_bm25(emb_dir, out_dir, data_dir, args.top_k, corpus_dates)
        run_report(args.model, n_total, time.monotonic() - t0)
        return

    import faiss
    index   = faiss.read_index(str(emb_dir / "index.faiss"))
    doc_ids = json.loads((emb_dir / "doc_ids.json").read_text())
    enc     = get_encoder(args.model)

    n_total = 0
    max_k = index.ntotal
    for qt in QUERY_TYPES:
        qids, texts, qdates, qpapers = load_queries(data_dir, qt)
        query_embs = enc.encode_queries(texts)

        # Date filtering can drop hits, so widen the search depth for short queries until each has top_k or the index is exhausted.
        final_rankings: list[list[dict] | None] = [None] * len(qids)
        pending = list(range(len(qids)))
        k = args.top_k
        while pending:
            scores, indices = faiss_search(index, query_embs[pending], k)
            still_pending = []
            for local_i, global_i in enumerate(pending):
                ranking = [
                    {"doc_id": doc_ids[j], "score": float(scores[local_i, kk])}
                    for kk, j in enumerate(indices[local_i]) if j >= 0
                ]
                filtered = drop_paper(temporal_filter(ranking, qdates[global_i], corpus_dates), qpapers[global_i])
                if len(filtered) >= args.top_k or k >= max_k:
                    final_rankings[global_i] = filtered[:args.top_k]
                else:
                    still_pending.append(global_i)
            pending = still_pending
            k = min(k * 4, max_k)

        results = [{"query_id": qid, "ranking": final_rankings[i]} for i, qid in enumerate(qids)]

        run_file = out_dir / f"{qt}.jsonl"
        run_file.write_text("\n".join(json.dumps(r) for r in results))
        print(f"  [{qt}] {len(results)} queries → {run_file}")
        n_total += len(results)

    run_report(args.model, n_total, time.monotonic() - t0)


if __name__ == "__main__":
    main()
