"""Run one agent over one query type and write runs/agentic/<pipeline>/<query_type>.jsonl for evaluate.py.

Each finished query is appended as it completes, so a killed run resumes; a query whose agent raises is
recorded in <query_type>.failed.jsonl and rerun on resume.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieve import build_retrieve_fn, load_corpus_dates
from utils import load_corpus, load_queries, load_run, source_ids, title_index
from agentic import requests_log
from agentic.ranking import DEPTH, assemble, backfill_ids, parse_ranked_ids
from agentic.trajectory import Trajectory
from agentic.views import build_buckets, build_view
from agentic.fulltext import FullText, write_helper
import anthropic_native


def load_citations(bench_dir: Path) -> dict[str, list[str]]:
    """{doc_id: [cited doc_ids]} from <bench>/citations.jsonl ({"id", "cites": [...]} per line),
    the optional corpus-restricted citation graph the expand tool needs. Empty when absent."""
    path = bench_dir / "citations.jsonl"
    if not path.exists():
        return {}
    graph = {}
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            graph[d["id"]] = list(d.get("cites") or [])
    return graph


def write_manifest(out_dir: Path, **fields) -> None:
    """run.json next to the run files: the launch parameters, library versions, the code commit and the native
    route's settings, so a run can be told from another after the fact."""
    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = None
    versions = {}
    for name in ("anthropic", "openai"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            versions[name] = None
    manifest = {**fields, "commit": commit, "versions": versions, "native_replay_thinking": anthropic_native.REPLAY_THINKING,
                "native_max_tokens": anthropic_native.MAX_TOKENS, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    path = out_dir / "run.json"
    runs = json.loads(path.read_text()) if path.exists() else []
    runs.append(manifest)
    path.write_text(json.dumps(runs, indent=1, default=str))


def run_queries(bench_dir: Path, pipeline: str, query_type: str, agent_fn, *,
                backfill: str = "bm25", model: str = "gpt-4.1", limit: int | None = None,
                query_ids: list[str] | None = None, views: bool = False, depth: int = DEPTH,
                extra_tools: dict | None = None, workers: int = 4,
                full_text: Path | None = None, read_policy: str = "full") -> dict:
    corpus_map = load_corpus(bench_dir)
    corpus = list(corpus_map.values())
    corpus_dates = load_corpus_dates(bench_dir)
    titles = title_index(corpus_map)  # the query paper may sit in the corpus under another id (a Semantic Scholar record)
    retrieve_fn = build_retrieve_fn(corpus, corpus_map, bench_dir, backfill)
    fulltext = FullText(full_text, read_policy) if full_text else None
    tools = {"retrieve": retrieve_fn, "corpus": corpus_map, "corpus_dates": corpus_dates, "bench_dir": bench_dir,
             "citations": load_citations(bench_dir),
             "model": model, "fulltext": fulltext, **(extra_tools or {})}  # the LLM agents share one client through this dict

    queries = load_queries(bench_dir, query_type)
    if query_ids:
        wanted = set(query_ids)
        queries = [q for q in queries if q["query_id"] in wanted]
    if limit:
        queries = queries[:limit]

    out_dir = bench_dir / "runs" / "agentic" / pipeline
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / f"{query_type}.jsonl"
    failed_path = out_dir / f"{query_type}.failed.jsonl"  # queries whose agent raised: not results, rerun on resume
    done = set()
    if run_path.exists():
        done = {json.loads(l)["query_id"] for l in run_path.read_text().splitlines() if l.strip()}
    write_manifest(out_dir, pipeline=pipeline, model=model, backfill=backfill, query_type=query_type, workers=workers,
                   full_text=str(full_text) if full_text else None, read_policy=read_policy if full_text else None, extra=extra_tools)

    # Back-fill from the retriever's own scored run when it exists (same ranking as its
    # Table row, already date-filtered); otherwise retrieve live.
    fill_run = load_run(bench_dir, backfill, query_type)
    if fill_run:
        print(f"  back-fill from runs/{backfill} ({len(fill_run)} queries)")

    store = None
    if views:
        store = bench_dir / "agentic" / "corpus_by_month"
        build_buckets(bench_dir / "corpus.jsonl", store)

    pending = [q for q in queries if q["query_id"] not in done]
    totals = {"queries": 0, "skipped": len(queries) - len(pending), "failed": 0,
              "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    lock = threading.Lock()
    t0 = time.monotonic()

    def one(q: dict) -> None:
        qid = q["query_id"]
        withheld = source_ids(q.get("paper_id", ""), q.get("paper_title", ""), titles)
        q = {**q, "source_ids": sorted(withheld)}  # the agents withhold every id of the query paper
        view_dir = None
        if views:
            view_dir = bench_dir / "agentic" / "views" / query_type / qid
            build_view(store, view_dir, q.get("paper_published", ""), sorted(withheld))
            if fulltext:
                write_helper(fulltext, view_dir)  # bash-driven agents read papers through read_paper.py
        failed = False
        with Trajectory(out_dir / "trajectories" / query_type / f"{qid}.jsonl", qid, model) as traj:
            requests_log.set_current(traj.requests)  # chat_create records this thread's calls here
            try:
                answer = agent_fn(q, view_dir, traj, tools)
            except Exception as e:  # noqa: BLE001 - one bad query must not stop the run
                failed = True
                traj.event("error", error=repr(e))
                print(f"  [{query_type}] {qid}: agent failed ({e!r}); recorded in {failed_path.name}, rerun on resume")
                answer = ""
            finally:
                requests_log.set_current(None)
            if failed:
                with lock:
                    with failed_path.open("a") as f:
                        f.write(json.dumps({"query_id": qid, "error": traj.last_error}) + "\n")
                    totals["failed"] += 1
                return
            agent_ids = parse_ranked_ids(answer, corpus_map)
            fill = [r["doc_id"] for r in fill_run[qid]] if qid in fill_run else \
                   backfill_ids(retrieve_fn, q.get("question", ""), q.get("paper_published", ""), depth)
            ranking = assemble(agent_ids, traj.seen, fill, known_ids=corpus_map, paper_id=withheld,
                               query_date=q.get("paper_published", ""), corpus_dates=corpus_dates, depth=depth)
            summary = traj.finish(answer, ranking)
        segments = {s: sum(1 for r in ranking if r["source"] == s) for s in ("agent", "trajectory", "backfill")}
        line = json.dumps({"query_id": qid, "ranking": ranking, "seen": list(traj.seen), "segments": segments,
                           "summary": {**summary, "full_text": bool(fulltext), "read_policy": read_policy if fulltext else None,
                                       "reasoning_effort": (extra_tools or {}).get("reasoning_effort")}})
        with lock:
            out.write(line + "\n")
            out.flush()
            totals["queries"] += 1
            totals["input_tokens"] += summary["input_tokens"]
            totals["output_tokens"] += summary["output_tokens"]
            totals["cost_usd"] += summary["cost_usd"]
            print(f"  [{query_type}] {qid}: agent={segments['agent']} trajectory={segments['trajectory']} "
                  f"backfill={segments['backfill']} tokens={summary['input_tokens']}/{summary['output_tokens']} "
                  f"cost=${summary['cost_usd']:.4f} ({totals['queries']}/{len(pending)})", flush=True)

    with run_path.open("a") as out, ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(one, pending))  # list() re-raises anything the worker itself could not catch
    minutes = (time.monotonic() - t0) / 60
    totals["queries_per_minute"] = round(totals["queries"] / minutes, 2) if minutes > 0 and totals["queries"] else 0.0
    print(f"[{pipeline}/{query_type}] {totals['queries']} queries run, {totals['skipped']} already done, "
          f"{totals['failed']} failed, tokens {totals['input_tokens']}/{totals['output_tokens']}, cost ${totals['cost_usd']:.2f}, "
          f"{totals['queries_per_minute']} queries/min with {workers} workers -> {run_path}")
    return totals
