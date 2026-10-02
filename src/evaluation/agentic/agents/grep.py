"""Grep agent in the style of DCI-Agent-Lite: one bash tool run inside the query's corpus view.

Outputs are truncated, commands time out and the subprocess gets no API keys. This is not a sandbox,
so run it on a throwaway machine or user.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_rerank import api_model_id, chat_create, make_client
from agentic.agents.toolcall import _cache_tokens
from agentic.ranking import parse_ranked_ids

MAX_TURNS = 20         # tool rounds; the context grows with every output, so this bounds cost
OUTPUT_BYTES = 4000    # per command output shown to the model
COMMAND_TIMEOUT = 30   # seconds

SYSTEM = (
    "You are a research agent searching a local paper corpus for prior work whose ideas could genuinely "
    "help a researcher pursue an early-stage research question, not papers that are merely on the same topic.\n"
    "The current directory holds the corpus as files named YYYY-MM.jsonl, one paper per line as JSON with the "
    "fields id, title, text (the abstract) and published; only papers the researcher could have read are present. "
    "Search the abstracts, not only the titles. Use bash with ripgrep, for example\n"
    "  rg -i 'phrase one|phrase two' *.jsonl | head -c 4000        # matching papers with id, title and abstract\n"
    "  rg -i -c 'phrase' *.jsonl | sort -t: -k2 -nr | head          # which months have the most matches\n"
    "  rg -i -o '\"id\": \"[^\"]+\", \"title\": \"[^\"]{{0,120}}' 2023-*.jsonl | rg -i 'phrase' | head -30   # compact id + title list\n"
    "Try 5 to 10 different angles, including specific technical terms, method names and problem formulations "
    "that a relevant paper would use even if the question does not. Read the abstracts of at least 20 candidates "
    "before answering; a paper counts only if you have seen its abstract. You have at most {max_turns} commands. "
    "Do not assume or reveal a solution to the question. When done, reply with a JSON array of paper ids, most "
    "useful first, and nothing else."
)
BASH_TOOL = {"type": "function", "function": {
    "name": "bash",
    "description": "Run a bash command in the corpus directory and return its output (truncated).",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
WRAP_UP = "You are out of commands. Reply now with the JSON array of paper ids, most useful first, and nothing else."


def _tool_call_dict(tc) -> dict:
    """The tool call as the API expects it back; Gemini 3.x needs its thought signature echoed."""
    d = {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
    extra = getattr(tc, "extra_content", None) or (getattr(tc, "model_extra", None) or {}).get("extra_content")
    if extra:
        d["extra_content"] = extra
    return d


def _scrubbed_env() -> dict:
    return {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}


def run_command(command: str, cwd: Path) -> str:
    try:
        r = subprocess.run(["bash", "-c", command], cwd=cwd, env=_scrubbed_env(), capture_output=True,
                           text=True, errors="replace", timeout=COMMAND_TIMEOUT)
        out = (r.stdout + ("\n[stderr] " + r.stderr if r.stderr.strip() else "")).strip()
        if r.returncode:
            out += f"\n[exit {r.returncode}]"
    except subprocess.TimeoutExpired:
        out = f"[timed out after {COMMAND_TIMEOUT}s]"
    if len(out) > OUTPUT_BYTES:
        out = out[:OUTPUT_BYTES] + f"\n[truncated to {OUTPUT_BYTES} bytes]"
    return out or "[no output]"


def _call(client, model_api: str, model: str, messages: list[dict], traj, tools=None):
    kwargs = {"tools": tools, "tool_choice": "auto"} if tools else {}
    resp = chat_create(client, model_api, messages, agent="main", **kwargs)
    usage = getattr(resp, "usage", None)
    n_in = getattr(usage, "prompt_tokens", 0) or 0
    n_out = getattr(usage, "completion_tokens", 0) or 0
    finish = getattr(resp.choices[0], "finish_reason", None)
    traj.llm_call(n_in, n_out, cost_usd=getattr(usage, "cost", None), **_cache_tokens(usage), tools=bool(tools), finish_reason=finish)
    traj.last_finish_reason = finish
    if finish == "content_filter":
        details = getattr(resp, "stop_details", None) or {}
        traj.refusal("main", details.get("category") if isinstance(details, dict) else None)
    return resp.choices[0].message


def run(query: dict, view_dir, traj, tools) -> str:
    if view_dir is None:
        raise ValueError("the grep agent needs a per-query corpus view; run with --views")
    model = tools["model"]
    client = tools.get("client") or tools.setdefault("client", make_client(model))  # one client per run, shared by the workers
    model_api = api_model_id(model)
    corpus = tools["corpus"]
    view_dir = Path(view_dir)

    messages = [{"role": "system", "content": SYSTEM.format(max_turns=MAX_TURNS)},
                {"role": "user", "content": f"Research question:\n{query.get('question', '')}"}]
    for turn in range(MAX_TURNS):
        msg = _call(client, model_api, model, messages, traj, tools=[BASH_TOOL])
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            return msg.content or ""
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [_tool_call_dict(tc) for tc in calls]})
        for tc in calls:
            try:
                command = str(json.loads(tc.function.arguments or "{}").get("command", ""))
            except json.JSONDecodeError:
                command = ""
            out = run_command(command, view_dir) if command else "[empty command]"
            traj.tool_call("bash", {"command": command}, out, doc_ids=parse_ranked_ids(out, corpus))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    messages.append({"role": "user", "content": WRAP_UP})
    return _call(client, model_api, model, messages, traj).content or ""
