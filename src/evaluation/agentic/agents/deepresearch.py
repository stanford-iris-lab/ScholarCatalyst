"""Deep research agent over the local corpus: plan, search with reflection, read, then synthesize.

Same search tool as toolcall.py with a different procedure; one synthesis call orders the pool (no per paper
judge). Without --full-text, read shows the paper's title and full abstract.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_rerank import api_model_id, make_client
from agentic.jsonarray import first_json_array
from agentic.agents.toolcall import SEARCH_TOOL, _call, _doc_line, tool_call_dict
from agentic.ranking import withhold

MAX_SEARCHES = 20   # search calls per query, the seed included (toolcall: 5 rounds)
MIN_SEARCHES = 3    # searches (seed included) before the model may answer without a tool call
NUDGE_MAX = 2       # prose replies pushed back to the tools per query
MAX_READS = 10      # read calls per query
POOL_CAP = 150      # papers the synthesis may rank (toolcall: 60)
SEARCH_K = 10
REFLECT_EVERY = 5   # tool calls between reflections
SYNTH_MAX = 50      # ids the synthesis may return
PLAN_MAX = 6        # sub-questions

SYSTEM = (
    "You are a research agent. Given an early-stage research question, find prior papers whose ideas "
    "could genuinely help a researcher pursue it, not papers that are merely on the same topic. Work "
    "through the sub-questions of your plan: call search with different angles ({max_searches} searches "
    "at most) and read on promising papers to see more than the abstract, including their bibliography, "
    "then search for the titles worth following ({max_reads} reads at most). Do not assume or reveal a "
    "solution to the question. When the plan is covered, reply without calling a tool."
)
SYSTEM_ABS = (  # no full text configured: read returns the full abstract, nothing beyond it
    "You are a research agent. Given an early-stage research question, find prior papers whose ideas "
    "could genuinely help a researcher pursue it, not papers that are merely on the same topic. Work "
    "through the sub-questions of your plan: call search with different angles ({max_searches} searches "
    "at most) and read on promising papers to see their full abstract ({max_reads} reads at most). Do not "
    "assume or reveal a solution to the question. When the plan is covered, reply without calling a tool."
)
PLAN = (
    "Research question:\n{question}\n\nSplit this question into at most {n} sub-questions a "
    "literature search should answer to find prior work whose ideas could help pursue it. Reply with a "
    "JSON array of strings and nothing else."
)
REFLECT = (
    "Reflection. The plan was:\n{plan}\n\nWhich sub-questions are still uncovered by the papers found so "
    "far, and what will you search or read next? Reply in a few sentences, without calling a tool."
)
CONTINUE = "Continue with the searches and reads your reflection names."
NUDGE = "Use the search tool before answering."
SYNTH = (
    "Papers found ({n_pool}):\n{pool}\n\nResearch question:\n{question}\n\nFrom these papers only, pick the "
    "ones whose ideas would meaningfully help a researcher pursue the question, most useful first, at most "
    "{n}. Reply with a JSON array of their ids and nothing else."
)
BUDGET = "budget exhausted: no more {name} calls for this query"

READ_TOOL = {"type": "function", "function": {
    "name": "read",
    "description": "Read a paper from your results: its full abstract, or more under a full-text read policy.",
    "parameters": {"type": "object", "properties": {"doc_id": {"type": "string", "description": "id of a paper returned earlier"}},
                   "required": ["doc_id"]}}}
TOOLS = [SEARCH_TOOL, READ_TOOL]


def _text_turn(msg) -> dict:
    """A text-only assistant turn for the history; the reply blocks ride along (extra_content) so the native
    Anthropic route can replay the thinking behind it."""
    turn = {"role": "assistant", "content": msg.content or ""}
    extra = getattr(msg, "extra_content", None)
    if isinstance(extra, dict) and extra.get("anthropic_blocks"):
        turn["extra_content"] = extra
    return turn


def _json_array(text: str) -> list:
    """The first JSON array in a reply (see agentic.jsonarray); a reply without one is a
    failure of the model."""
    value = first_json_array(text)
    if value is None:
        raise RuntimeError(f"no JSON array in reply: {(text or '')[:200]!r}")
    return value


def run(query: dict, view_dir, traj, tools) -> str:
    model = tools["model"]
    client = tools.get("client") or tools.setdefault("client", make_client(model))  # one client per run, shared by the workers
    model_api = api_model_id(model)
    corpus, fulltext = tools["corpus"], tools.get("fulltext")
    question, qdate, paper_id = query.get("question", ""), query.get("paper_published", ""), query.get("paper_id", "")
    withheld = set(query.get("source_ids") or [paper_id]) - {""}  # every corpus id of the query paper
    retrieve = withhold(tools["retrieve"], withheld)  # the query's own paper never reaches the agent
    pool: dict[str, dict] = {}  # doc_id -> doc, in discovery order, at most POOL_CAP
    caps = {"search": MAX_SEARCHES, "read": MAX_READS}
    used = {"search": 0, "read": 0}

    def pooled(docs: list[dict]) -> list[str]:
        """Admit docs while the pool has room; returns the ids of docs now in the pool."""
        for d in docs:
            if len(pool) < POOL_CAP:
                pool.setdefault(d["id"], d)
        return [d["id"] for d in docs if d["id"] in pool]

    def do_search(args: dict) -> tuple[str, list[str]]:
        docs = retrieve(str(args.get("query", "")), SEARCH_K, qdate)
        return "\n\n".join(_doc_line(d) for d in docs) or "no results", pooled(docs)

    def do_read(args: dict) -> tuple[str, list[str]]:
        doc_id = str(args.get("doc_id", ""))
        if doc_id not in corpus or doc_id in withheld:  # the query's own paper is withheld
            return f"unknown id {doc_id}", []
        d = corpus[doc_id]
        text = fulltext.read(doc_id) if fulltext else f"{d.get('title', '')}\n\n{d.get('text') or ''}"
        return text, pooled([corpus[doc_id]])

    handlers = {"search": do_search, "read": do_read}

    def call_tool(name: str, args: dict) -> tuple[str, list[str]]:
        if name not in handlers:
            return f"unknown tool {name}", []
        if used[name] >= caps[name]:
            return BUDGET.format(name=name), []
        used[name] += 1
        return handlers[name](args)

    # Step 1: plan
    system = {"role": "system", "content": (SYSTEM if fulltext else SYSTEM_ABS).format(max_searches=MAX_SEARCHES, max_reads=MAX_READS)}
    msg = _call(client, model_api, model, [system, {"role": "user", "content": PLAN.format(question=question, n=PLAN_MAX)}],
                traj, agent="plan")
    try:
        plan = [str(s) for s in _json_array(msg.content)][:PLAN_MAX]
    except RuntimeError as e:
        # a malformed plan is not worth losing the query: search from the question itself
        plan = [question]
        traj.event("plan_fallback", reason=str(e)[:200])
    if not plan:
        raise RuntimeError("empty plan")
    traj.event("plan", sub_questions=plan)
    plan_text = "\n".join(f"- {s}" for s in plan)

    # Seed: the raw question as a search, so the pool is never empty and the model sees the
    # retriever's own view of the question up front (as in toolcall.py)
    seed_text, seed_ids = call_tool("search", {"query": question})
    traj.tool_call("search", {"query": question, "seed": True}, seed_text, doc_ids=seed_ids)

    # Step 2: iterate, with a reflection after every REFLECT_EVERY tool calls
    messages = [system, {"role": "user", "content": f"Research question:\n{question}\n\nPlan:\n{plan_text}\n\n"
                                                    f"Already in your results (a search of the question as written):\n{seed_text}"}]
    max_turns = MAX_SEARCHES + MAX_READS + 2 * (MAX_SEARCHES + MAX_READS) // REFLECT_EVERY + 2
    turns = n_calls = n_reflect = n_nudge = 0
    while turns < max_turns:
        choice = "required" if used["search"] < MIN_SEARCHES else "auto"
        msg = _call(client, model_api, model, messages, traj, tools=TOOLS, tool_choice=choice)
        turns += 1
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            messages.append(_text_turn(msg))
            # a reasoning backbone may answer in prose even with tools offered: push it back to search; a refused or cut reply is not pushed
            if traj.last_finish_reason not in ("content_filter", "length") and n_nudge < NUDGE_MAX and used["search"] < MAX_SEARCHES and (not pool or used["search"] < MIN_SEARCHES):
                n_nudge += 1
                traj.event("nudge", n=n_nudge, text=(msg.content or "")[:2000])
                messages.append({"role": "user", "content": NUDGE})
                continue
            break
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [tool_call_dict(tc) for tc in calls]})
        for tc in calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result, ids = call_tool(tc.function.name, args)
            # has_refs on read events, as in toolcall.py: the share of read papers with a parsable bibliography
            meta = {"has_refs": bool(fulltext) and fulltext.has_refs(str(args.get("doc_id", "")))} if tc.function.name == "read" else {}
            traj.tool_call(tc.function.name, args, result, doc_ids=ids, **meta)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        n_calls += len(calls)
        if all(used[n] >= caps[n] for n in caps):
            break
        if n_calls // REFLECT_EVERY > n_reflect:
            n_reflect += 1
            messages.append({"role": "user", "content": REFLECT.format(plan=plan_text)})
            reflect_msg = _call(client, model_api, model, messages, traj, agent="reflect"); reply = reflect_msg.content or ""
            turns += 1
            traj.event("reflect", text=reply)
            messages += [_text_turn(reflect_msg), {"role": "user", "content": CONTINUE}]

    # Step 3: synthesize; the ordered ids are the ranking, the pool is what the agent saw
    listing = "\n".join(f"[{d['id']}] {d.get('title', '')}" for d in pool.values()) or "none"
    messages.append({"role": "user", "content": SYNTH.format(n_pool=len(pool), pool=listing, question=question, n=SYNTH_MAX)})
    synth = _call(client, model_api, model, messages, traj, agent="synthesis")
    if traj.last_finish_reason == "content_filter":  # a refused synthesis: the agent ranks nothing, the row is what it saw plus the back-fill
        traj.event("synthesis", ids=[], dropped=0, refused=True)
        return "[]"
    wanted = [str(i) for i in _json_array(synth.content)]
    kept = list(dict.fromkeys(i for i in wanted if i in pool and i not in withheld))[:SYNTH_MAX]
    traj.event("synthesis", ids=kept, dropped=len(wanted) - len(kept))
    return json.dumps(kept)
