"""Tool calling search agent over the local corpus, in the style of PaperScout.

Tools: search, expand and optionally read. Pooled papers are judged for usefulness and ranked by score.
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_rerank import api_model_id, chat_create, make_client
from retrieve import temporal_filter
from agentic.ranking import withhold
from agentic import requests_log

MAX_STEPS = 5       # PaperScout base.yaml max_steps
SEARCH_K = 10       # PaperScout / PaSa search_papers
POOL_CAP = 60       # papers judged per query; bounds the judge cost
JUDGE_WORKERS = 8

SYSTEM = (
    "You are a research agent. Given an early-stage research question, find prior papers whose "
    "ideas could genuinely help a researcher pursue it, not papers that are merely on the same topic. "
    "You have at most {max_steps} rounds of tool calls. Call search with different angles of the "
    "question{expand_hint}. Do not assume or reveal a solution to the question. "
    "When you have enough candidates, reply with the single word DONE."
)
EXPAND_HINT = ", and call expand on a promising paper to follow its references"

SEARCH_TOOL = {"type": "function", "function": {
    "name": "search",
    "description": "Search the paper corpus. Returns up to 10 papers with id, title, year and abstract.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "natural-language query or keywords"}},
                   "required": ["query"]}}}
EXPAND_TOOL = {"type": "function", "function": {
    "name": "expand",
    "description": "List the corpus papers cited by a paper already in your results, to broaden coverage around it.",
    "parameters": {"type": "object", "properties": {"doc_id": {"type": "string", "description": "id of a paper returned earlier"}},
                   "required": ["doc_id"]}}}

READ_TOOL = {"type": "function", "function": {
    "name": "read",
    "description": "Read the full text of a paper from your results (what you may see depends on the run's read policy). Optional section title to read one section.",
    "parameters": {"type": "object", "properties": {"doc_id": {"type": "string"}, "section": {"type": "string", "description": "a section title from sections(), optional"}},
                   "required": ["doc_id"]}}}
SECTIONS_TOOL = {"type": "function", "function": {
    "name": "sections",
    "description": "List the section titles of a paper's full text, so you can pick which to read.",
    "parameters": {"type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"]}}}
READ_HINT = ", and call read on a paper to see more than its abstract"

JUDGE = (
    "Research question:\n{question}\n\nCandidate paper:\nTitle: {title}\nAbstract: {abstract}\n\n"
    "Would this paper's ideas meaningfully help a researcher pursue the research question, beyond being "
    "topically related? Reply with one integer from 0 (no) to 10 (certainly), nothing else."
)


def tool_call_dict(tc) -> dict:
    """The tool call as the API expects it back. Gemini 3.x attaches a thought signature
    under extra_content and rejects the next request without it."""
    d = {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
    extra = getattr(tc, "extra_content", None) or (getattr(tc, "model_extra", None) or {}).get("extra_content")
    if extra:
        d["extra_content"] = extra
    return d


def _doc_line(doc: dict) -> str:
    year = (doc.get("published") or "")[:4]
    return f"[{doc['id']}] {doc.get('title', '')} ({year})\n{(doc.get('text') or '')[:400]}"


def _cache_tokens(usage) -> dict:
    """The prompt-cache read and write counts of a reply (Anthropic's usage, folded into prompt_tokens), for the llm line."""
    d = getattr(usage, "prompt_tokens_details", None)
    return {"cache_read_tokens": getattr(d, "cached_tokens", 0) or 0, "cache_write_tokens": getattr(d, "cache_write_tokens", 0) or 0}


def _call(client, model_api: str, model: str, messages: list[dict], traj, tools=None, tool_choice="auto", agent="main"):
    """One chat call: logs tokens, returns the message.
    tool_choice="required" makes the API return a tool call (deepresearch's first searches);
    agent is the role label of the request record (main, judge, plan, reflect, synthesis)."""
    kwargs = {"tools": tools, "tool_choice": tool_choice} if tools else {}
    resp = chat_create(client, model_api, messages, agent=agent, **kwargs)
    usage = getattr(resp, "usage", None)
    n_in = getattr(usage, "prompt_tokens", 0) or 0
    n_out = getattr(usage, "completion_tokens", 0) or 0
    finish = getattr(resp.choices[0], "finish_reason", None)
    traj.llm_call(n_in, n_out, cost_usd=getattr(usage, "cost", None), **_cache_tokens(usage), tools=bool(tools), finish_reason=finish)
    traj.last_finish_reason = finish
    if finish == "content_filter":  # the model's classifier refused (Claude: stop_reason "refusal"); the caller sees an empty reply
        details = getattr(resp, "stop_details", None) or {}
        traj.refusal(agent, details.get("category") if isinstance(details, dict) else None)
    return resp.choices[0].message


def _judge_one(client, model_api, model, question, doc, traj) -> int | None:
    """The 0 to 10 usefulness score, or None when the judge call was refused or answered without a number."""
    msg = _call(client, model_api, model,
                [{"role": "user", "content": JUDGE.format(question=question, title=doc.get("title", ""),
                                                          abstract=(doc.get("text") or "")[:1500])}], traj, agent="judge")
    m = re.search(r"\d+", msg.content or "")
    return max(0, min(10, int(m.group()))) if m else None


def run(query: dict, view_dir, traj, tools) -> str:
    model = tools["model"]
    client = tools.get("client") or tools.setdefault("client", make_client(model))  # one client per run, shared by the workers
    model_api = api_model_id(model)
    corpus, corpus_dates = tools["corpus"], tools["corpus_dates"]
    citations = tools.get("citations") or {}
    withheld = set(query.get("source_ids") or [query.get("paper_id", "")]) - {""}  # every corpus id of the query paper: withheld from the search tool, refused by every read tool
    retrieve = withhold(tools["retrieve"], withheld)
    question, qdate = query.get("question", ""), query.get("paper_published", "")
    pool: dict[str, dict] = {}  # doc_id -> doc, in discovery order

    def do_search(args: dict) -> tuple[str, list[str]]:
        docs = retrieve(str(args.get("query", "")), SEARCH_K, qdate)
        for d in docs:
            pool.setdefault(d["id"], d)
        return "\n\n".join(_doc_line(d) for d in docs) or "no results", [d["id"] for d in docs]

    def do_expand(args: dict) -> tuple[str, list[str]]:
        if str(args.get("doc_id", "")) in withheld:
            return f"unknown id {args.get('doc_id', '')}", []
        cited = [{"doc_id": c} for c in citations.get(str(args.get("doc_id", "")), []) if c in corpus]
        docs = [corpus[c["doc_id"]] for c in temporal_filter(cited, qdate, corpus_dates)][:SEARCH_K]
        for d in docs:
            pool.setdefault(d["id"], d)
        return "\n\n".join(_doc_line(d) for d in docs) or "no citations known for this paper", [d["id"] for d in docs]

    # Seed: the raw question as a search, so the retriever's own view of it is pooled before any rewrite.
    seed_text, seed_ids = do_search({"query": question})
    traj.tool_call("search", {"query": question, "seed": True}, seed_text, doc_ids=seed_ids)

    fulltext = tools.get("fulltext")  # agentic.fulltext.FullText or None

    def do_read(args: dict) -> tuple[str, list[str], dict]:
        doc_id = str(args.get("doc_id", ""))
        if doc_id in withheld:
            return f"unknown id {doc_id}", [], {"has_refs": False}
        # has_refs on read events: the share of read papers with a parsable bibliography
        return fulltext.read(doc_id, args.get("section") or None), ([doc_id] if doc_id in corpus else []), \
               {"has_refs": fulltext.has_refs(doc_id)}

    def do_sections(args: dict) -> tuple[str, list[str]]:
        doc_id = str(args.get("doc_id", ""))
        if doc_id in withheld:
            return f"unknown id {doc_id}", []
        return json.dumps(fulltext.sections(doc_id)), ([doc_id] if doc_id in corpus else [])

    schemas = [SEARCH_TOOL] + ([EXPAND_TOOL] if citations else []) \
              + ([READ_TOOL] + ([SECTIONS_TOOL] if fulltext.policy == "choose" else []) if fulltext else [])
    handlers = {"search": do_search, "expand": do_expand, "read": do_read, "sections": do_sections}
    hints = (EXPAND_HINT if citations else "") + (READ_HINT if fulltext else "")
    messages = [{"role": "system", "content": SYSTEM.format(max_steps=MAX_STEPS, expand_hint=hints)},
                {"role": "user", "content": f"Research question:\n{question}\n\nAlready in your results (a search of the question as written):\n{seed_text}"}]
    for _ in range(MAX_STEPS):
        msg = _call(client, model_api, model, messages, traj, tools=schemas)
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            break
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [tool_call_dict(tc) for tc in calls]})
        for tc in calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            fn = handlers.get(tc.function.name)
            result, ids, *meta = fn(args) if fn else (f"unknown tool {tc.function.name}", [])
            traj.tool_call(tc.function.name, args, result, doc_ids=ids, **(meta[0] if meta else {}))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    docs = list(pool.values())[:POOL_CAP]
    if not docs:
        return "[]"
    judge = requests_log.inherit(lambda d: _judge_one(client, model_api, model, question, d, traj))
    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
        scores = list(ex.map(judge, docs))
    order = sorted(range(len(docs)), key=lambda i: (scores[i] is None, -(scores[i] or 0), i))  # judge score, then discovery order; unjudged papers (refused or no number) after every scored one
    traj.event("judge", scores={docs[i]["id"]: scores[i] for i in order})
    return json.dumps([docs[i]["id"] for i in order])
