from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import BENCH_DIR, MODEL_REGISTRY
from encoder import get_encoder


def load_corpus(bench_dir: Path) -> tuple[list[str], list[str]]:
    doc_ids, texts = [], []
    with (bench_dir / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            doc_ids.append(d["id"])
            title = d.get("title", "")
            text  = d.get("text", d.get("abstract", ""))
            texts.append(f"{title} {text}".strip())
    return doc_ids, texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",     required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--bench-dir", type=Path, default=BENCH_DIR)
    ap.add_argument("--workers",   type=int, default=8, help="concurrent requests for API embedding models")
    args = ap.parse_args()

    if MODEL_REGISTRY[args.model].get("type") == "multi_vector":
        ap.error(f"{args.model} is a multi-vector model; use retrieve_lateon.py --stage index")

    bench_dir = args.bench_dir.resolve()
    out_dir   = bench_dir / "embeddings" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_ids, texts = load_corpus(bench_dir)
    print(f"[encode_corpus] model={args.model}  docs={len(doc_ids)}")

    enc = get_encoder(args.model)
    if hasattr(enc, "workers"):
        enc.workers = args.workers

    if args.model == "bm25":
        enc.encode_corpus(texts, doc_ids)
        # BM25 index is in-memory; save doc_ids only for retrieve.py
        (out_dir / "doc_ids.json").write_text(json.dumps(doc_ids))
        # Persist tokenized corpus for retrieval
        import pickle
        (out_dir / "bm25.pkl").write_bytes(pickle.dumps(enc.bm25))
        print(f"[encode_corpus] saved bm25.pkl + doc_ids.json → {out_dir}")
    else:
        import faiss
        embs = enc.encode_corpus(texts).astype(np.float32)
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        faiss.write_index(index, str(out_dir / "index.faiss"))
        (out_dir / "doc_ids.json").write_text(json.dumps(doc_ids))
        print(f"[encode_corpus] saved index.faiss {embs.shape} → {out_dir}")


if __name__ == "__main__":
    main()
