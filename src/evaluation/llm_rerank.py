"""LLM reranking pipelines: leave the query untouched, rerank a fixed
retriever's candidates with an LLM instead.

See llm_query_augment.py for the other half of this split: pipelines that
rewrite or expand the query text before handing it to a fixed retriever.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

import openai

sys.path.insert(0, str(Path(__file__).parent))
from config import BENCH_DIR, QUERY_TYPES
from evaluate import quick_scores
from run_report import run_report
from retrieve import build_retrieve_fn, without_paper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm_client"))
from llm_client import GEMINI_NATIVE_BASE

from agentic import requests_log
import anthropic_native

OPENAI_BASE     = "https://api.openai.com/v1"
DEFAULT_MODEL   = "gpt-5.4"

ANTHROPIC_MAX_TOKENS = 4096
# OpenAI reasoning families reject a temperature parameter with HTTP 400
REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5", "gpt-6")
NO_FORCED_TOOL_CHOICE = ("claude-fable",)  # rejects tool_choice=required ('type "tool" and "any" are not supported for this model')



def build_run_descriptor(
    pipeline: str,
    retriever: str,
    oracle_names: list[str],
    model: str,
    top_k: int,
    rerank_k: int,
    tournament_k: int,
    limit: int | None = None,
    query_ids_stem: str | None = None,
) -> str:
    """The run's runs/agentic/<descriptor>/ folder name -- kept as a pure
    function (rather than inlined in main()) so a sweep driver can predict a
    run's path without invoking the CLI."""
    llm_tag = model.replace("/", "-")
    if pipeline == "oracle_tournament":
        retriever_tag = oracle_names[0] if len(oracle_names) == 1 else f"union{len(oracle_names)}"
    else:
        retriever_tag = retriever
    descriptor = f"{pipeline}-{retriever_tag}-{llm_tag}"
    if pipeline in ("rerank_tournament", "oracle_tournament"):
        descriptor += f"-pool{top_k}-tk{tournament_k}"
    elif pipeline == "rerank" and rerank_k != 50:
        descriptor += f"-k{rerank_k}"
    if limit:
        descriptor += f"-test{limit}"
    if query_ids_stem:
        descriptor += f"-{query_ids_stem}"
    return descriptor


def strip_json_fence(raw: str) -> str:
    """Some models wrap JSON responses in a ```json ... ``` fence despite
    being told to output raw JSON; strip that before parsing."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


RetrieveFn = Callable[[str, int, str], list[dict]]


def make_client(model: str):
    """Pick a client and endpoint for the model's own family, each through its own direct API key."""
    if model.startswith("google/"):
        return openai.OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_NATIVE_BASE)
    if model.startswith("anthropic/"):
        import anthropic
        return anthropic.Anthropic(api_key=_anthropic_key(), max_retries=5)  # the agents run on it through anthropic_native
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return openai.OpenAI(api_key=key, base_url=OPENAI_BASE)


def _anthropic_key() -> str:
    """The direct Anthropic key: the project's grant key (ANTHROPIC_API_KEY_1) before the lab key."""
    return os.environ.get("ANTHROPIC_API_KEY_1") or os.environ.get("ANTHROPIC_API_KEY") or ""


def api_model_id(model: str) -> str:
    """The id sent to the endpoint make_client(model) picks: the vendor prefix is dropped."""
    if model.startswith("google/"):
        return model.removeprefix("google/")
    if model.startswith("anthropic/"):
        return model.removeprefix("anthropic/")
    return model


REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
REASONING_EFFORT: str | None = None  # set once per process by the CLI (--reasoning-effort); None sends the model default


def reasoning_kwargs(client) -> dict:
    """Request kwargs that set the reasoning effort; nothing when no effort is set."""
    if not REASONING_EFFORT:
        return {}
    if anthropic_native.is_native(client):  # the Claude runs use the model's default thinking; an effort setting is not wired
        raise RuntimeError("--reasoning-effort is not supported on the direct Anthropic route; drop the flag")
    return {"reasoning_effort": REASONING_EFFORT}


def sampling_kwargs(model_api: str) -> dict:
    """Sampling parameters for chat.completions.create: none for reasoning models, which reject temperature."""
    bare = model_api.rsplit("/", 1)[-1]
    return {} if bare.startswith(REASONING_PREFIXES) else {"temperature": 0.0}


NATIVE_RETRY_SLEEPS = (20, 40, 60)  # seconds; after the SDK's own five fast retries, for bursts of 429 / 529 / connection errors


def _with_native_retry(native: bool, fn):
    """fn() with slow retries on the direct Anthropic route: the SDK retries within seconds; an overload lasting longer
    would otherwise fail the query. Every attempt is recorded by the request log inside fn."""
    if not native:
        return fn()
    import anthropic
    for i, pause in enumerate(NATIVE_RETRY_SLEEPS + (None,)):
        try:
            return fn()
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            if pause is None:
                raise
            print(f"[info] anthropic {type(e).__name__}, retrying in {pause}s ({i + 1}/{len(NATIVE_RETRY_SLEEPS)})")
            time.sleep(pause)


def chat_create(client, model_api: str, messages: list[dict], agent: str = "main", **kwargs):
    """chat.completions.create with the model's sampling kwargs. When the
    endpoint rejects temperature the call is retried once without it, so a
    reasoning id missing from REASONING_PREFIXES still works. Every request
    (the retry too) is recorded on the thread's RequestLog when one is active,
    under the role label agent."""
    native = anthropic_native.is_native(client)
    sampling = {} if native else sampling_kwargs(model_api)  # the native route sends no temperature (the SDK has none; Claude Fable rejects it)
    kwargs = {**reasoning_kwargs(client), **kwargs}
    if native and kwargs.get("tool_choice") == "required" and model_api.rsplit("/", 1)[-1].startswith(NO_FORCED_TOOL_CHOICE):
        kwargs["tool_choice"] = "auto"  # Claude Fable rejects a forced tool choice on every route; skip the failing request the fallback below would retry
    log = requests_log.current()
    raw_create = anthropic_native.creator(client) if native else client.chat.completions.create

    def create(**kw):
        if log is None:
            return _with_native_retry(native, lambda: raw_create(model=model_api, messages=messages, **kw))
        return _with_native_retry(native, lambda: log.call(raw_create, model_api, messages, kw, agent=agent))

    try:
        return create(**sampling, **kwargs)
    except Exception as e:
        if kwargs.get("tool_choice") == "required" and "tool_choice" in str(e):
            # Claude Fable rejects a forced tool choice ('type "tool" and "any" are not supported for this model'); the agents that
            # force their first searches fall back to letting the model choose, and their nudge handles a prose reply
            print("[info] model rejects tool_choice=required, retrying with auto")
            return create(**sampling, **{**kwargs, "tool_choice": "auto"})
        if not (sampling and "temperature" in str(e)):
            raise
        print("[info] model rejects temperature, retrying with model default")
        return create(**kwargs)


def chat(client, model: str, messages: list[dict], max_retries: int = 3) -> str:
    is_anthropic = hasattr(client, "messages") and not hasattr(client, "chat")
    for attempt in range(max_retries):
        try:
            if is_anthropic:
                system = next((m["content"] for m in messages if m["role"] == "system"), "")
                user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
                resp = client.messages.create(
                    model=model, max_tokens=ANTHROPIC_MAX_TOKENS, system=system,
                    messages=[{"role": "user", "content": user}],
                )
                content = "".join(blk.text for blk in resp.content if getattr(blk, "type", None) == "text").strip()
                if content:
                    return content
                print(f"[warn] chat empty response (attempt {attempt+1}/{max_retries})")
            else:
                resp = chat_create(client, model, messages)
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    return content
                reason = resp.choices[0].finish_reason if resp.choices else "unknown"
                print(f"[warn] chat empty response (attempt {attempt+1}/{max_retries}, finish_reason={reason})")
        except Exception as e:
            print(f"[warn] chat error (attempt {attempt+1}/{max_retries}): {e}")
        if attempt < max_retries - 1:
            time.sleep(5 * (attempt + 1))
    return ""


# Simple Listwise Reranker (single LLM call, top-K ≤ ~50)

RERANK_SYSTEM = (
    "You are an expert at scientific literature retrieval.\n"
    "Your goal is to identify prior papers that could help a researcher pursue the given research question.\n"
    "Given a research question and a numbered list of candidate papers, "
    "rank ALL papers from most to least likely to be useful for advancing this research question.\n"
    "Rank a paper higher when its scientific contribution would meaningfully "
    "inform how a researcher thinks about or approaches this question. "
    "Rank it lower when it is merely topically related and does not provide "
    "insight for this research.\n"
    "Output only a JSON array of integer IDs (1-indexed) in ranked order. "
    "Always include every id from the input list exactly once.\n"
    "Example (20 papers): [14, 3, 20, 7, 1, ..., 18, 2, 9]"
)


def doc_line_indexed(i: int, d: dict, title_only: bool) -> str:
    line = f'[{i+1}] title={d.get("title", "")}'
    if not title_only:
        line += f'\nabstract={d.get("abstract", d.get("text", ""))}'
    return line


def doc_line(d: dict, title_only: bool) -> str:
    line = f'id={d["id"]} title={d.get("title","")}'
    if not title_only:
        line += f'\nabstract={d.get("abstract", d.get("text",""))}'
    return line


def rerank(
    query: str,
    candidates: list[dict],
    client: openai.OpenAI,
    model: str = DEFAULT_MODEL,
    title_only: bool = False,
    trace: list[dict] | None = None,
    query_id: str | None = None,
) -> list[dict]:
    """Rerank candidates with one LLM call (best for top K up to about 50).

    Candidates are shuffled first so retriever order cannot bias the output; trace, if given, gets one record per call."""
    if not candidates:
        return []

    shuffled = candidates[:]
    random.shuffle(shuffled)

    doc_list = "\n".join(
        doc_line_indexed(i, d, title_only) for i, d in enumerate(shuffled)
    )
    raw = chat(client, model, [
        {"role": "system", "content": RERANK_SYSTEM},
        {"role": "user",   "content": f"Query: {query}\n\nPapers:\n{doc_list}"},
    ])
    n_returned = None
    parsed_ok = False
    try:
        ranked_indices = json.loads(strip_json_fence(raw))
        seen = set()
        ranked: list[dict] = []
        for idx in ranked_indices:
            i = int(idx) - 1
            if 0 <= i < len(shuffled) and i not in seen:
                ranked.append(shuffled[i])
                seen.add(i)
        n_returned = len(ranked)
        parsed_ok = True
        # append any candidates the LLM missed
        for i, d in enumerate(shuffled):
            if i not in seen:
                ranked.append(d)
        result = ranked
    except (json.JSONDecodeError, ValueError, TypeError):
        result = shuffled

    if trace is not None:
        trace.append({
            "query_id": query_id,
            "stage": "single",
            "model": model,
            "candidates": [{"doc_id": d["id"], "title": d.get("title", "")} for d in shuffled],
            "raw_response": raw,
            "parsed_ok": parsed_ok,
            "n_returned_by_llm": n_returned,
            "output_order_doc_ids": [d["id"] for d in result],
        })
    return result


# Tournament Listwise Reranker

RERANK_TOURNAMENT_SYSTEM = (
    "You are an expert at scientific literature retrieval.\n"
    "Your goal is to identify prior papers that could help a researcher pursue the given research question.\n"
    "Given a research question and a list of candidate papers, "
    "rank ALL papers from most to least likely to be useful for advancing this research question.\n"
    "Rank a paper higher when its scientific contribution would meaningfully "
    "inform how a researcher thinks about or approaches this question. "
    "Rank it lower when it is merely topically related and does not provide "
    "insight for this research.\n"
    "Output only a JSON array of paper IDs in ranked order, most relevant first.\n"
    'Example: ["id_a", "id_b", "id_c"]'
)

def rerank_tournament(
    query: str,
    candidates: list[dict],
    client: openai.OpenAI,
    model: str = DEFAULT_MODEL,
    b: int = 20,
    k: int = 4,
    trace: list[dict] | None = None,
    query_id: str | None = None,
) -> list[dict]:
    """trace, when given a list, gets one record appended per batch call
    (round/batch_index identify where in the elimination tree it happened) --
    see --trace on the CLI."""
    def rank_batch(batch: list[dict], stage: str, round_idx: int, batch_idx: int) -> list[dict]:
        batch_map = {d["id"]: d for d in batch}
        doc_list = "\n".join(
            f'id={d["id"]} title={d.get("title","")}\nabstract={d.get("abstract", d.get("text",""))}'
            for d in batch
        )
        raw = chat(client, model, [
            {"role": "system", "content": RERANK_TOURNAMENT_SYSTEM},
            {"role": "user",   "content": f"Query: {query}\n\nPapers:\n{doc_list}"},
        ])
        n_returned = None
        parsed_ok = False
        try:
            ranked_ids = json.loads(strip_json_fence(raw))
            seen: set[str] = set()
            ranked = []
            for i in ranked_ids:
                if i in batch_map and i not in seen:
                    ranked.append(batch_map[i])
                    seen.add(i)
            n_returned = len(ranked)
            parsed_ok = True
            remaining = [d for d in batch if d["id"] not in seen]
            result = ranked + remaining
        except (json.JSONDecodeError, KeyError):
            result = batch

        if trace is not None:
            trace.append({
                "query_id": query_id,
                "stage": stage,
                "round": round_idx,
                "batch_index": batch_idx,
                "model": model,
                "candidates": [{"doc_id": d["id"], "title": d.get("title", "")} for d in batch],
                "raw_response": raw,
                "parsed_ok": parsed_ok,
                "n_returned_by_llm": n_returned,
                "output_order_doc_ids": [d["id"] for d in result],
            })
        return result

    pool = candidates[:]
    random.shuffle(pool)
    tails: list[list[dict]] = []

    round_idx = 0
    while len(pool) > b:
        promoted, tail = [], []
        for batch_idx, i in enumerate(range(0, len(pool), b)):
            batch = pool[i : i + b]
            ranked_batch = rank_batch(batch, "tournament_round", round_idx, batch_idx)
            promoted.extend(ranked_batch[:k])
            tail.extend(ranked_batch[k:])
        tails.append(tail)
        pool = promoted
        round_idx += 1

    final = rank_batch(pool, "tournament_final", round_idx, 0)

    result = final[:]
    for tail in reversed(tails):
        result.extend(tail)
    return result


# Oracle tournament baseline: reranks a candidate pool that already
# contains the gold docs, an upper bound on the reranker alone.

def oracle_tournament(
    query: str,
    gold_ids: list[str],
    retrieve_fns: list[RetrieveFn],
    corpus_id_map: dict[str, dict],
    client: openai.OpenAI,
    model: str = DEFAULT_MODEL,
    top_k_per_retriever: int = 300,
    b: int = 20,
    k: int = 4,
    trace: list[dict] | None = None,
    query_id: str | None = None,
) -> list[dict]:
    ordered_ids: list[str] = []
    seen_ids: set[str] = set()
    for fn in retrieve_fns:
        for r in fn(query, top_k_per_retriever):
            if r["id"] not in seen_ids:
                ordered_ids.append(r["id"])
                seen_ids.add(r["id"])

    if len(retrieve_fns) == 1:
        # single fixed retriever: gold injected, pool size held at
        # top_k_per_retriever by dropping the lowest-ranked non-gold ids to
        # make room, so |pool| == K exactly (the retriever-verifier gap setup).
        gold_set = set(gold_ids)
        missing_gold = [g for g in gold_ids if g not in seen_ids]
        if missing_gold:
            n_to_drop = len(missing_gold)
            kept, dropped = [], 0
            for doc_id in reversed(ordered_ids):
                if dropped < n_to_drop and doc_id not in gold_set:
                    dropped += 1
                    continue
                kept.append(doc_id)
            kept.reverse()
            ordered_ids = kept + missing_gold
    else:
        # multi-retriever union oracle: keep the old additive behavior (pool
        # can exceed top_k_per_retriever), unchanged for existing callers.
        for gid in gold_ids:
            if gid not in seen_ids:
                ordered_ids.append(gid)
                seen_ids.add(gid)

    if trace is not None:
        trace.append({
            "query_id": query_id,
            "stage": "oracle_pool",
            "gold_ids": gold_ids,
            "gold_missing_from_retrieval": [g for g in gold_ids if g not in seen_ids],
            "pool_size": len(ordered_ids),
        })

    candidates = [corpus_id_map[i] for i in ordered_ids if i in corpus_id_map]
    return rerank_tournament(query, candidates, client, model, b=b, k=k, trace=trace, query_id=query_id)


# CLI

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir",   type=Path, default=BENCH_DIR)
    ap.add_argument("--set",         default="", dest="set_name",
                    help="query set subdir under bench-dir (e.g. llm_set, author_set);"
                         " queries/rels are read from it and runs are written into it,"
                         " while corpus.jsonl and embeddings/ stay at bench-dir root,"
                         " matching retrieve.py --set")
    ap.add_argument("--query-type",  choices=["core_query", "subfield_query", "all"],
                    default="all")
    ap.add_argument("--pipeline",    choices=["rerank", "rerank_tournament", "oracle_tournament"],
                    default="rerank")
    ap.add_argument("--retriever",   default="gemini-2",
                    help="retriever model id (from MODEL_REGISTRY) or 'bm25'")
    ap.add_argument("--model",       default=DEFAULT_MODEL)
    ap.add_argument("--top-k",       type=int, default=1000)
    ap.add_argument("--rerank-k",    type=int, default=50,
                    help="candidates passed to rerank (single LLM call); keep ≤ 50")
    ap.add_argument("--tournament-k", type=int, default=5,
                    help="candidates promoted per batch in tournament rounds (batch size fixed at 20)")
    ap.add_argument("--oracle-retrievers", default="gemini-2",
                    help="comma separated retrievers whose top-k union forms the"
                         " oracle_tournament candidate pool (gold docs always added);"
                         " e.g. bm25,gemini-2,qwen3-8b for the union oracle")
    ap.add_argument("--max-workers", type=int, default=8,
                    help="parallel query workers (LLM calls)")
    ap.add_argument("--limit",       type=int, default=None,
                    help="only run the first N queries per query type (for quick testing)")
    ap.add_argument("--query-ids-file", type=Path, default=None,
                    help="JSON file of {query_type: [query_id, ...]} restricting which"
                         " queries to run, e.g. a fixed random subset written by"
                         " make_gap_subset.py; applied before --limit")
    ap.add_argument("--out",         type=Path, default=None)
    ap.add_argument("--trace", action="store_true",
                    help="also write {query_type}.trace.jsonl next to the run: one record per"
                         " LLM call (exact candidates shown, raw response, parsed ranking) for"
                         " auditing how a specific paper's ranking was decided")
    args = ap.parse_args()

    bench_dir = args.bench_dir.resolve()
    data_dir  = (bench_dir / args.set_name) if args.set_name else bench_dir
    qtypes = QUERY_TYPES if args.query_type == "all" else [args.query_type]
    client = make_client(args.model)
    llm_model = api_model_id(args.model)

    query_ids_by_qt: dict[str, set[str]] | None = None
    if args.query_ids_file:
        raw_ids = json.loads(args.query_ids_file.read_text())
        query_ids_by_qt = {qt: set(ids) for qt, ids in raw_ids.items()}

    corpus = [json.loads(l) for l in (bench_dir / "corpus.jsonl").read_text().splitlines() if l.strip()]
    corpus_map = {d["id"]: d for d in corpus}

    retrieve = build_retrieve_fn(corpus, corpus_map, bench_dir, args.retriever)

    # oracle pool spec: union of top-k results from the --oracle-retrievers
    # list, plus the gold documents, tournament ranked (paper section on the
    # approximate quality bound)
    oracle_names = [r.strip() for r in args.oracle_retrievers.split(",") if r.strip()]
    oracle_retrieves = []
    if args.pipeline == "oracle_tournament":
        oracle_retrieves = [
            retrieve if r == args.retriever else build_retrieve_fn(corpus, corpus_map, bench_dir, r)
            for r in oracle_names
        ]

    print(f"[{args.pipeline}+{args.retriever}] corpus={len(corpus)}  qtypes={qtypes}  workers={args.max_workers}")

    descriptor = build_run_descriptor(
        args.pipeline, args.retriever, oracle_names, args.model,
        args.top_k, args.rerank_k, args.tournament_k,
        limit=args.limit,
        query_ids_stem=args.query_ids_file.stem if args.query_ids_file else None,
    )
    run_dir = data_dir / "runs" / "agentic" / descriptor  # reruns overwrite, same as retrieve.py's baselines

    t0 = time.monotonic()
    n_done = 0

    for qt in qtypes:
        rels_by_qid: dict[str, list[str]] = {}
        for line in (data_dir / "rels" / f"{qt}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            qid = r["query_id"]
            rels_by_qid[qid] = [p["id"] if isinstance(p, dict) else p
                                 for p in (r.get("positive_docs") or [])]

        allowed = query_ids_by_qt.get(qt, set()) if query_ids_by_qt is not None else None
        queries, queries_text = [], {}
        for line in (data_dir / "queries.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            q = json.loads(line)
            if q.get("type") == qt and q["id"] in rels_by_qid and (allowed is None or q["id"] in allowed):
                queries.append(q)
                queries_text[q["id"]] = q.get("question", "")
        if args.limit:
            queries = queries[:args.limit]

        print(f"\n--- {qt}: {len(queries)} queries ---")

        def process_query(q: dict) -> tuple[str, list[str], list[dict] | None]:
            qid   = q["id"]
            qtext = queries_text[qid]
            qdate = q.get("paper_published", "")
            retrieve_q = without_paper(retrieve, q.get("paper_id", ""))  # the query's own paper is never a candidate
            trace: list[dict] | None = [] if args.trace else None

            if args.pipeline == "rerank":
                candidates = retrieve_q(qtext, args.rerank_k, qdate)
                ranked = rerank(qtext, candidates, client, llm_model, trace=trace, query_id=qid)
            elif args.pipeline == "oracle_tournament":
                # Gold docs are injected so the run measures ranking alone; at top_k <= 20 one listwise call ranks the whole pool.
                gold_ids = rels_by_qid.get(qid, [])
                date_bound = [(lambda q, n, f=without_paper(f, q_paper), d=qdate: f(q, n, d))
                              for f in oracle_retrieves for q_paper in [q.get("paper_id", "")]]
                batch_size = 999 if args.top_k <= 20 else 20
                ranked = oracle_tournament(
                    qtext, gold_ids, date_bound,
                    corpus_map, client, llm_model,
                    top_k_per_retriever=args.top_k, b=batch_size, k=args.tournament_k,
                    trace=trace, query_id=qid)
            else:  # rerank_tournament
                candidates = retrieve_q(qtext, args.top_k, qdate)
                ranked = rerank_tournament(qtext, candidates, client, llm_model, k=args.tournament_k,
                                            trace=trace, query_id=qid)
            return qid, [r["id"] for r in ranked], trace

        run_dir.mkdir(parents=True, exist_ok=True)
        out = args.out or run_dir / f"{qt}.jsonl"

        # Results append per completed query so a crash loses
        # nothing, and a rerun resumes from what the file already holds.
        # Delete the run file first when the prompt or pipeline changed.
        done_qids: set[str] = set()
        if out.exists():
            for line in out.read_text().splitlines():
                if line.strip():
                    done_qids.add(json.loads(line)["query_id"])
            if done_qids:
                print(f"resuming: {len(done_qids)} queries already in {out}")
        todo = [q for q in queries if q["id"] not in done_qids]

        trace_path = out.parent / f"{out.stem}.trace.jsonl" if args.trace else None

        n_written = 0
        from tqdm import tqdm
        with out.open("a", encoding="utf-8") as sink, \
             (trace_path.open("a", encoding="utf-8") if trace_path else nullcontext()) as trace_sink, \
             ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(process_query, q): q for q in todo}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{qt[:3]}"):
                qid, ranked_ids, trace = fut.result()
                sink.write(json.dumps({
                    "query_id": qid,
                    "ranking": [{"doc_id": d} for d in ranked_ids],
                }, ensure_ascii=False) + "\n")
                sink.flush()
                if trace_sink is not None:
                    for record in trace:
                        trace_sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                    trace_sink.flush()
                n_written += 1
        n_done += n_written
        print(f"Results → {out} ({len(done_qids) + n_written}/{len(queries)} queries)")
        if trace_path:
            print(f"Trace   → {trace_path}")
        pool_k = args.rerank_k if args.pipeline == "rerank" else args.top_k
        quick_scores(qt, out, queries, rels_by_qid, max_k=pool_k)

    run_report(f"{args.pipeline}+{args.retriever} ({args.model})", n_done, time.monotonic() - t0)
    print(f"\nScore with: python evaluate.py --model agentic/{descriptor}")


if __name__ == "__main__":
    main()
