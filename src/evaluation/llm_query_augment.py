from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

OPENAI_BASE     = "https://api.openai.com/v1"
DEFAULT_MODEL   = "gpt-5.4"

ANTHROPIC_MAX_TOKENS = 4096



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
    """Pick a client and endpoint for the model's own family."""
    if model.startswith("google/"):
        return openai.OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_NATIVE_BASE)
    if model.startswith("anthropic/"):
        import anthropic
        return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return openai.OpenAI(api_key=key, base_url=OPENAI_BASE)


def chat(client, model: str, messages: list[dict], max_retries: int = 3) -> str:
    is_anthropic = hasattr(client, "messages") and not hasattr(client, "chat")
    # reasoning models (gpt-6 family, gpt-5 pro tiers) only accept the default
    # temperature, so send none for them; the on the fly fallback below covers
    # any other model that refuses it
    no_temperature = model.startswith(("gpt-5", "gpt-6"))
    sampling: dict = {} if no_temperature else {"temperature": 0.0}
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
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **sampling,
                )
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    return content
                reason = resp.choices[0].finish_reason if resp.choices else "unknown"
                print(f"[warn] chat empty response (attempt {attempt+1}/{max_retries}, finish_reason={reason})")
        except Exception as e:
            if sampling and "temperature" in str(e):
                sampling = {}
                print("[info] model rejects temperature, retrying with model default")
                continue
            print(f"[warn] chat error (attempt {attempt+1}/{max_retries}): {e}")
        if attempt < max_retries - 1:
            time.sleep(5 * (attempt + 1))
    return ""


# Single-hop Query Rewriter

SINGLE_HOP_SYSTEM = (
    "You are helping a researcher find prior scientific papers that could help "
    "them pursue an early-stage research question.\n"
    "Strengthen the researcher's question so that it retrieves the prior work "
    "this research needs more effectively.\n"
    "Preserve the context and intent of the original question while strengthening it.\n"
    "Do not introduce a different research direction or a solution.\n"
    "Do not use Boolean operators, search syntax, or a keyword list.\n"
    "Output only the strengthened question in natural language."
)


def single_hop_retrieve(
    query: str,
    retrieve_fn: RetrieveFn,
    client: openai.OpenAI,
    model: str = DEFAULT_MODEL,
    top_k: int = 1000,
    query_date: str = "",
) -> tuple[list[dict], str]:
    rewritten = chat(client, model, [
        {"role": "system", "content": SINGLE_HOP_SYSTEM},
        {"role": "user",   "content": query},
    ]) or query
    return retrieve_fn(rewritten, top_k, query_date), rewritten


# Multi-Query Retrieval

MULTI_QUERY_SYSTEM = (
    "You are helping a researcher find prior scientific papers that could help "
    "them pursue an early-stage research question.\n"
    "Generate up to 5 augmented versions of the researcher's question.\n"
    "Each version must keep the original question text unchanged and add a "
    "short continuation that makes the search more specific.\n"
    "Make the continuations diverse: each should point toward a distinct "
    "plausible line of prior work that could inform the same research situation.\n"
    "Do not introduce a different research direction or a solution.\n"
    "Write natural prose only, no keyword lists.\n"
    "Output one augmented question per line and nothing else."
)

_STRIP_PREFIX = re.compile(r"^\s*(\d+[.)]\s*|[-*•]\s*)")


def parse_queries(raw: str) -> list[str]:
    queries = []
    for line in raw.splitlines():
        line = _STRIP_PREFIX.sub("", line).strip()
        if line:
            queries.append(line)
    return queries[:5]


def rrf_merge(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    id_to_doc: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            did = doc["id"]
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
            id_to_doc[did] = doc
    return [id_to_doc[did] for did in sorted(scores, key=lambda d: scores[d], reverse=True)]


def multi_query_retrieve(
    query: str,
    retrieve_fn: RetrieveFn,
    client: openai.OpenAI,
    model: str = DEFAULT_MODEL,
    top_k: int = 1000,
    query_date: str = "",
) -> tuple[list[dict], list[str]]:
    raw = chat(client, model, [
        {"role": "system", "content": MULTI_QUERY_SYSTEM},
        {"role": "user",   "content": query},
    ])
    sub_queries = parse_queries(raw)
    if not sub_queries:
        sub_queries = [query]
    ranked_lists = [retrieve_fn(q, top_k, query_date) for q in sub_queries]
    merged = rrf_merge(ranked_lists)
    return merged, sub_queries


# HyDE for inspiration retrieval: hypothetical PRIOR WORK abstracts (not a
# hypothetical answer document), searched in place of the question so the
# query vector moves from question space into abstract space

HYDE_SYSTEM_TEMPLATE = (
    "You are helping a researcher find the prior work that inspired a new research idea.\n\n"
    "Below is a research question written by the authors of a paper before they solved it. "
    "Your job is NOT to answer the question. Your job is to imagine the earlier papers that "
    "the authors might have read and drawn on when forming this question.\n\n"
    "Write {n} hypothetical abstracts of such prior papers. Each abstract should:\n"
    "- Read like a real paper abstract (title + 120-180 words): problem, method, key finding.\n"
    "- Describe a paper that already existed before the question was written, so it must not "
    "solve the question itself. It contributes one idea, tool, observation, or framing that "
    "the authors could have borrowed.\n"
    "- Come from a different angle than the others. Cover at least one adjacent field or task "
    "where the same underlying idea appears under different terminology.\n"
    "- Use the vocabulary that paper's own community would use, not the vocabulary of the question.\n\n"
    "Do not name real papers, authors, or years. Do not mention the research question or the "
    "target paper. Output only the abstracts.\n\n"
    'Output format (JSON):\n[\n  {{"title": "...", "abstract": "..."}},\n  ...\n]'
)


def hyde_retrieve(
    query: str,
    retrieve_fn: RetrieveFn,
    client: openai.OpenAI,
    model: str = DEFAULT_MODEL,
    top_k: int = 1000,
    query_date: str = "",
    n_abstracts: int = 5,
) -> tuple[list[dict], list[str]]:
    raw = chat(client, model, [
        {"role": "system", "content": HYDE_SYSTEM_TEMPLATE.format(n=n_abstracts)},
        {"role": "user",   "content": f"Research question:\n{query}"},
    ])
    docs: list[str] = []
    try:
        parsed = json.loads(strip_json_fence(raw))
        for item in parsed:
            title = (item.get("title") or "").strip()
            abstract = (item.get("abstract") or "").strip()
            if abstract:
                docs.append(f"{title} {abstract}".strip())
    except (json.JSONDecodeError, AttributeError, TypeError):
        docs = []
    if not docs:
        docs = [query]
    ranked_lists = [retrieve_fn(d, top_k, query_date) for d in docs[:n_abstracts]]
    merged = rrf_merge(ranked_lists)
    return merged, docs


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="LLM query augmentation pipelines")
    ap.add_argument("--bench-dir",   type=Path, default=BENCH_DIR)
    ap.add_argument("--set",         default="", dest="set_name",
                    help="query set subdir under bench-dir (e.g. llm_set, author_set);"
                         " queries/rels are read from it and runs are written into it,"
                         " while corpus.jsonl and embeddings/ stay at bench-dir root,"
                         " matching retrieve.py --set")
    ap.add_argument("--query-type",  choices=["core_query", "subfield_query", "all"],
                    default="all")
    ap.add_argument("--pipeline",    choices=["single_hop", "multi_query", "hyde"],
                    default="single_hop")
    ap.add_argument("--hyde-n",      type=int, default=5,
                    help="number of hypothetical prior work abstracts for the hyde pipeline")
    ap.add_argument("--retriever",   default="gemini-2",
                    help="retriever model id (from MODEL_REGISTRY) or 'bm25'")
    ap.add_argument("--model",       default=DEFAULT_MODEL)
    ap.add_argument("--top-k",       type=int, default=1000)
    ap.add_argument("--max-workers", type=int, default=8,
                    help="parallel query workers (LLM calls)")
    ap.add_argument("--limit",       type=int, default=None,
                    help="only run the first N queries per query type (for quick testing)")
    ap.add_argument("--out",         type=Path, default=None)
    args = ap.parse_args()

    bench_dir = args.bench_dir.resolve()
    data_dir  = (bench_dir / args.set_name) if args.set_name else bench_dir
    qtypes = QUERY_TYPES if args.query_type == "all" else [args.query_type]
    client = make_client(args.model)
    llm_model = args.model.removeprefix("google/").removeprefix("anthropic/")

    corpus = [json.loads(l) for l in (bench_dir / "corpus.jsonl").read_text().splitlines() if l.strip()]
    corpus_map = {d["id"]: d for d in corpus}

    retrieve = build_retrieve_fn(corpus, corpus_map, bench_dir, args.retriever)

    print(f"[{args.pipeline}+{args.retriever}] corpus={len(corpus)}  qtypes={qtypes}  workers={args.max_workers}")

    llm_tag    = args.model.replace("/", "-")
    descriptor = f"{args.pipeline}-{args.retriever}-{llm_tag}"
    if args.limit:
        descriptor += f"-test{args.limit}"
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

        queries, queries_text = [], {}
        for line in (data_dir / "queries.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            q = json.loads(line)
            if q.get("type") == qt and q["id"] in rels_by_qid:
                queries.append(q)
                queries_text[q["id"]] = q.get("question", "")
        if args.limit:
            queries = queries[:args.limit]

        print(f"\n--- {qt}: {len(queries)} queries ---")

        def process_query(q: dict) -> tuple[str, list[str], str | list[str] | None]:
            qid   = q["id"]
            qtext = queries_text[qid]
            qdate = q.get("paper_published", "")
            retrieve_q = without_paper(retrieve, q.get("paper_id", ""))  # the query's own paper is never a candidate

            if args.pipeline == "single_hop":
                ranked, generated_query = single_hop_retrieve(qtext, retrieve_q, client, llm_model, args.top_k, qdate)
            elif args.pipeline == "multi_query":
                ranked, generated_query = multi_query_retrieve(qtext, retrieve_q, client, llm_model, args.top_k, qdate)
            else:  # hyde
                ranked, generated_query = hyde_retrieve(qtext, retrieve_q, client, llm_model, args.top_k, qdate,
                                                        n_abstracts=args.hyde_n)
            return qid, [r["id"] for r in ranked], generated_query

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

        n_written = 0
        from tqdm import tqdm
        with out.open("a", encoding="utf-8") as sink, \
             ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(process_query, q): q for q in todo}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{qt[:3]}"):
                qid, ranked_ids, gen_q = fut.result()
                sink.write(json.dumps({
                    "query_id": qid,
                    **({"generated_query": gen_q} if gen_q is not None else {}),
                    "ranking": [{"doc_id": d} for d in ranked_ids],
                }, ensure_ascii=False) + "\n")
                sink.flush()
                n_written += 1
        n_done += n_written
        print(f"Results → {out} ({len(done_qids) + n_written}/{len(queries)} queries)")
        quick_scores(qt, out, queries, rels_by_qid, max_k=args.top_k)

    run_report(f"{args.pipeline}+{args.retriever} ({args.model})", n_done, time.monotonic() - t0)
    print(f"\nScore with: python evaluate.py --model agentic/{descriptor}")


if __name__ == "__main__":
    main()
