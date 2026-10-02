"""Agentic search baselines: the agent searches the corpus itself and its run becomes a ranking.

    python agentic_search.py --agent grep --model gpt-4.1 --bench-dir $BENCH_DIR
    python agentic_search.py --agent toolcall --model gpt-4.1 --bench-dir $BENCH_DIR
    python agentic_search.py --agent deepresearch --model o3 --full-text corpus_full_text.jsonl --bench-dir $BENCH_DIR
    python evaluate.py --model agentic/toolcall-bm25 --bench-dir $BENCH_DIR
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import BENCH_DIR, QUERY_TYPES
from agentic.ranking import DEPTH
import llm_rerank
from agentic.fulltext import POLICIES
from agentic.runner import run_queries

AGENTS = {
    "stub": "agentic.agents.stub",
    "toolcall": "agentic.agents.toolcall",
    "deepresearch": "agentic.agents.deepresearch",
    "grep": "agentic.agents.grep",
}  # name -> module with run()

NEEDS_VIEWS = {"grep"}
DEFAULT_READ_POLICY = {"deepresearch": "abs+intro+refs"}  # --read-policy when not given; "full" otherwise
BACKFILL_CHOICES = ["bm25", "qwen3-8b", "qwen3-4b", "gemini-2", "text-embedding-3-large",
                    "bge-large", "openscholar", "scincl", "specter2"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent",      required=True, choices=list(AGENTS))
    ap.add_argument("--pipeline",   default=None, help="run name under runs/agentic/ (default: <agent>-<backfill>)")
    ap.add_argument("--model",      default="gpt-4.1", help="LLM backbone id passed to the agent")
    ap.add_argument("--backfill",   default="bm25", choices=BACKFILL_CHOICES,
                    help="retriever that fills the ranking to --depth after the agent's own ids")
    ap.add_argument("--depth",      type=int, default=DEPTH)
    ap.add_argument("--query-type", default="all", choices=QUERY_TYPES + ["all"])
    ap.add_argument("--limit",      type=int, default=None, help="first N queries per type (pilots)")
    ap.add_argument("--query-ids",  type=Path, default=None, help="file with one query id per line")
    ap.add_argument("--views",      action="store_true", help="build per-query corpus views for file-searching agents")
    ap.add_argument("--workers",    type=int, default=4, help="queries run concurrently (each agent may add its own threads)")
    ap.add_argument("--full-text",  type=Path, default=None, help="corpus_full_text.jsonl; adds a read tool (search stays on abstracts)")
    ap.add_argument("--read-policy", default=None, choices=list(POLICIES), help="what the read tool may show (default: full; deepresearch: abs+intro+refs)")
    ap.add_argument("--reasoning-effort", default=None, choices=list(llm_rerank.REASONING_EFFORTS),
                    help="reasoning effort sent with every model call; default: the model's own")
    ap.add_argument("--bench-dir",  type=Path, default=BENCH_DIR)
    args = ap.parse_args()

    bench_dir = args.bench_dir.resolve()
    pipeline = args.pipeline or f"{args.agent}-{args.backfill}"
    if args.agent in NEEDS_VIEWS and not args.views:
        print(f"[{args.agent}] needs per-query corpus views; enabling --views")
        args.views = True
    agent_fn = importlib.import_module(AGENTS[args.agent]).run
    query_ids = [l.strip() for l in args.query_ids.read_text().splitlines() if l.strip()] if args.query_ids else None
    read_policy = args.read_policy or DEFAULT_READ_POLICY.get(args.agent, "full")
    llm_rerank.REASONING_EFFORT = args.reasoning_effort
    extra_tools = {"reasoning_effort": args.reasoning_effort}

    for qt in QUERY_TYPES if args.query_type == "all" else [args.query_type]:
        run_queries(bench_dir, pipeline, qt, agent_fn, backfill=args.backfill, model=args.model,
                    limit=args.limit, query_ids=query_ids, views=args.views, depth=args.depth, workers=args.workers,
                    full_text=args.full_text, read_policy=read_policy, extra_tools=extra_tools)
    print(f"\nscore it with:\n  python evaluate.py --model agentic/{pipeline} --bench-dir {bench_dir}")


if __name__ == "__main__":
    main()
