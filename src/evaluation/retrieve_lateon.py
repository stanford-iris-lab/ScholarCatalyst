"""Late interaction baselines (LateOn, ColBERTv2) over the corpus, writing run files evaluate.py scores unchanged.

Standalone on purpose (it runs in its own conda env), so it duplicates the small helpers from retrieve.py.

    python retrieve_lateon.py --stage index    --bench-dir .../evaluation
    python retrieve_lateon.py --stage retrieve --bench-dir .../evaluation --set llm_set
    python retrieve_lateon.py --stage all --model-path colbert-ir/colbertv2.0 --tag colbertv2 --bench-dir .../evaluation --set author_set
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

QUERY_TYPES = ["core_query", "subfield_query"]


def parse_ym(date_str: str, doc_id: str = "") -> tuple[int, int] | None:
    if not date_str:
        m = re.match(r"^arxiv_(\d{2})(\d{2})\.\d+", doc_id)
        if m:
            yy, mm = int(m.group(1)), int(m.group(2))
            return (2000 + yy if yy <= 30 else 1900 + yy), mm
        return None
    m = re.match(r"^(\d{4})-(\d{2})", date_str)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d{4})$", date_str)
    if m:
        return int(m.group(1)), 0
    return None


def temporal_filter(ranking: list[dict], query_date: str, corpus_dates: dict[str, str]) -> list[dict]:
    q_ym = parse_ym(query_date)
    if q_ym is None:
        return ranking
    q_year, q_month = q_ym
    filtered = []
    for item in ranking:
        d_ym = parse_ym(corpus_dates.get(item["doc_id"], ""), item["doc_id"])
        if d_ym is None:
            filtered.append(item)
            continue
        d_year, d_month = d_ym
        if d_year < q_year:
            filtered.append(item)
        elif d_year == q_year and (d_month == 0 or q_month == 0 or d_month <= q_month):
            filtered.append(item)
    return filtered


def load_queries(data_dir: Path, query_type: str) -> tuple[list[str], list[str], list[str]]:
    qids, texts, dates = [], [], []
    for line in (data_dir / "queries.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("type") != query_type:
            continue
        qids.append(d["id"])
        texts.append(d.get("question") or "")
        dates.append(d.get("paper_published", ""))
    return qids, texts, dates


def load_corpus(bench_dir: Path) -> tuple[list[str], list[str], dict[str, str]]:
    doc_ids, texts, dates = [], [], {}
    with (bench_dir / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            doc_ids.append(d["id"])
            texts.append(f"{d.get('title', '')} {d.get('text', d.get('abstract', ''))}".strip())
            if d.get("published"):
                dates[d["id"]] = d["published"]
    return doc_ids, texts, dates


def build_model(args):
    from pylate import models
    return models.ColBERT(
        model_name_or_path=args.model_path,
        query_length=args.query_length,
        document_length=args.document_length,
    )


def stage_index(args, bench_dir: Path) -> None:
    import numpy as np
    import torch
    from pylate import indexes

    doc_ids, texts, dates = load_corpus(bench_dir)
    print(f"[index] corpus docs: {len(doc_ids)}")
    model = build_model(args)

    embeddings = []
    chunk = args.encode_chunk
    for start in range(0, len(texts), chunk):
        batch = texts[start:start + chunk]
        embs = model.encode(batch, batch_size=args.batch_size, is_query=False,
                            show_progress_bar=(start == 0))
        embeddings.extend(np.asarray(e, dtype=np.float16) for e in embs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[index] encoded {min(start + chunk, len(texts))}/{len(texts)}")

    index_dir = bench_dir / "embeddings" / args.tag
    index_dir.mkdir(parents=True, exist_ok=True)
    index = indexes.PLAID(index_folder=str(index_dir), index_name="plaid", override=True)
    index.add_documents(documents_ids=doc_ids, documents_embeddings=embeddings)
    print(f"[index] PLAID saved -> {index_dir}")


def stage_retrieve(args, bench_dir: Path) -> None:
    from pylate import indexes, retrieve

    data_dir = (bench_dir / args.set_name) if args.set_name else bench_dir
    _, _, corpus_dates = load_corpus(bench_dir)
    model = build_model(args)
    index = indexes.PLAID(index_folder=str(bench_dir / "embeddings" / args.tag),
                          index_name="plaid", override=False)
    retriever = retrieve.ColBERT(index=index)

    out_dir = data_dir / "runs" / "multi_vector" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for qt in QUERY_TYPES:
        qids, texts, qdates = load_queries(data_dir, qt)
        q_embs = model.encode(texts, batch_size=args.batch_size, is_query=True,
                              show_progress_bar=True)
        # search in chunks: fast_plaid moves the whole query batch to the GPU
        # at once, and long queries (256 tokens) blow past GPU memory when all
        # queries of a set go in a single call
        hits = []
        for start in range(0, len(q_embs), args.search_chunk):
            hits.extend(retriever.retrieve(
                queries_embeddings=q_embs[start:start + args.search_chunk],
                k=args.top_k))
            print(f"  [{qt}] searched {min(start + args.search_chunk, len(q_embs))}/{len(q_embs)}")
        results = []
        for qid, qdate, docs in zip(qids, qdates, hits):
            ranking = [{"doc_id": d["id"], "score": float(d["score"])} for d in docs]
            ranking = temporal_filter(ranking, qdate, corpus_dates)
            results.append({"query_id": qid, "ranking": ranking})
        run_file = out_dir / f"{qt}.jsonl"
        run_file.write_text("\n".join(json.dumps(r) for r in results))
        print(f"  [{qt}] {len(results)} queries -> {run_file}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["index", "retrieve", "all"], required=True)
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--set", default="", dest="set_name",
                    help="query set subdir (llm_set, author_set, ...), retrieve stage only")
    ap.add_argument("--model-path", default="lightonai/LateOn")
    ap.add_argument("--tag", default="lateon",
                    help="name for the index dir (embeddings/<tag>) and run dir"
                         " (runs/multi_vector/<tag>); use colbertv2 for"
                         " colbert-ir/colbertv2.0 so it matches config.py")
    ap.add_argument("--query-length", type=int, default=32,
                    help="model was trained with 32; raise to feed longer queries")
    ap.add_argument("--document-length", type=int, default=300)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--search-chunk", type=int, default=50,
                    help="queries per PLAID search call; keeps GPU memory bounded"
                         " for long query lengths")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--encode-chunk", type=int, default=5000)
    args = ap.parse_args()

    bench_dir = args.bench_dir.resolve()
    if args.stage in ("index", "all"):
        stage_index(args, bench_dir)
    if args.stage in ("retrieve", "all"):
        stage_retrieve(args, bench_dir)


if __name__ == "__main__":
    main()
