"""Per query corpus views for agents that search files: hard links to the month buckets a query may see.

Visibility follows retrieve.temporal_filter, and the query paper is never visible.

    python agentic/views.py --bench-dir $BENCH_DIR --query-type core_query --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieve import parse_ym
from utils import load_queries

UNKNOWN = "unknown"  # bucket for papers with no parseable date; always visible
KEEP_FIELDS = ("id", "title", "text", "published")


def bucket_name(doc: dict) -> str:
    ym = parse_ym(doc.get("published", ""), doc["id"])
    if ym is None:
        return UNKNOWN
    year, month = ym
    return f"{year:04d}-{month:02d}"  # month 00 = year-only date


def build_buckets(corpus_path: Path, store_dir: Path) -> dict:
    """Split corpus.jsonl into store_dir/<bucket>.jsonl. Reuses an existing store built
    from the same corpus file (size and mtime). Returns the manifest."""
    manifest_path = store_dir / "manifest.json"
    stat = corpus_path.stat()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("corpus_size") == stat.st_size and manifest.get("corpus_mtime") == stat.st_mtime:
            return manifest
    if store_dir.exists():
        shutil.rmtree(store_dir)
    store_dir.mkdir(parents=True)
    counts: dict[str, int] = {}
    handles: dict[str, object] = {}
    try:
        for line in corpus_path.read_text().splitlines():
            if not line.strip():
                continue
            doc = json.loads(line)
            name = bucket_name(doc)
            if name not in handles:
                handles[name] = (store_dir / f"{name}.jsonl").open("w")
            handles[name].write(json.dumps({k: doc[k] for k in KEEP_FIELDS if k in doc}) + "\n")
            counts[name] = counts.get(name, 0) + 1
    finally:
        for h in handles.values():
            h.close()
    manifest = {"corpus_size": stat.st_size, "corpus_mtime": stat.st_mtime,
                "n_docs": sum(counts.values()), "buckets": counts}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def visible_buckets(buckets: list[str], query_date: str) -> list[str]:
    """Bucket names a query may see, by the temporal_filter rules."""
    q = parse_ym(query_date)
    if q is None:
        return list(buckets)  # no query date: nothing can be excluded
    q_year, q_month = q
    keep = []
    for name in buckets:
        if name == UNKNOWN:
            keep.append(name)
            continue
        year, month = int(name[:4]), int(name[5:7])
        if year < q_year or (year == q_year and (month == 0 or q_month == 0 or month <= q_month)):
            keep.append(name)
    return keep


def build_view(store_dir: Path, view_dir: Path, query_date: str, paper_id="") -> dict:
    """Create view_dir with one file per visible bucket and a view.json manifest; `paper_id` is the query's own
    paper, a bare id or every corpus id of it. An existing view built for the same query date and ids is reused."""
    ids = sorted({paper_id} if isinstance(paper_id, str) else set(paper_id)); ids = [i for i in ids if i]
    manifest = json.loads((store_dir / "manifest.json").read_text())
    names = visible_buckets(sorted(manifest["buckets"]), query_date)
    key = {"query_date": query_date, "paper_id": ids[0] if len(ids) == 1 else ids, "buckets": names}
    view_json = view_dir / "view.json"
    if view_json.exists():
        old = json.loads(view_json.read_text())
        if {k: old.get(k) for k in key} == key:
            return old
    if view_dir.exists():
        shutil.rmtree(view_dir)
    view_dir.mkdir(parents=True)
    n_docs = 0
    for name in names:
        src, dst = store_dir / f"{name}.jsonl", view_dir / f"{name}.jsonl"
        n_docs += manifest["buckets"][name]
        if ids and any(f'"id": "{i}"' in src.read_text() for i in ids):
            n_docs -= _copy_without(src, dst, ids)
        else:
            _link_or_copy(src, dst)
    info = {**key, "n_docs": n_docs}
    view_json.write_text(json.dumps(info, indent=2))
    return info


def _copy_without(src: Path, dst: Path, ids: list[str]) -> int:
    """Copy a bucket minus the query paper (every id of it). Returns how many lines were dropped."""
    dropped = 0; withheld = set(ids)
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            if json.loads(line)["id"] in withheld:
                dropped += 1
                continue
            fout.write(line)
    return dropped


_warned_copy = False


def _link_or_copy(src: Path, dst: Path) -> None:
    global _warned_copy
    try:
        os.link(src, dst)
    except OSError:  # store and views on different filesystems
        if not _warned_copy:
            print(f"[views] hard link failed ({src} -> {dst}); copying buckets instead, which costs disk")
            _warned_copy = True
        shutil.copy2(src, dst)


def main() -> None:
    from config import BENCH_DIR, QUERY_TYPES
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir",  type=Path, default=BENCH_DIR)
    ap.add_argument("--query-type", default="all", choices=QUERY_TYPES + ["all"])
    ap.add_argument("--limit",      type=int, default=None, help="first N queries per type")
    args = ap.parse_args()

    bench_dir = args.bench_dir.resolve()
    store = build_buckets(bench_dir / "corpus.jsonl", bench_dir / "agentic" / "corpus_by_month")
    print(f"[views] {store['n_docs']} docs in {len(store['buckets'])} buckets under {bench_dir / 'agentic' / 'corpus_by_month'}")
    for qt in QUERY_TYPES if args.query_type == "all" else [args.query_type]:
        queries = load_queries(bench_dir, qt)[: args.limit]
        for q in queries:
            info = build_view(bench_dir / "agentic" / "corpus_by_month",
                              bench_dir / "agentic" / "views" / qt / q["query_id"],
                              q.get("paper_published", ""), q.get("paper_id", ""))
            print(f"  [{qt}] {q['query_id']}: {info['n_docs']} docs visible, {len(info['buckets'])} buckets")


if __name__ == "__main__":
    main()
