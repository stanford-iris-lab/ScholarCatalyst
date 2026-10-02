"""Run the agents' OpenAI style chat.completions calls on Anthropic's native Messages API.

Thinking blocks must be sent back verbatim with tool results: a reply's full block list rides in the first
tool call's extra_content["anthropic_blocks"], and to_messages replays it.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

MAX_TOKENS = 16384  # the reply cap; thinking counts against it
PROMPT_CACHE = os.environ.get("ANTHROPIC_PROMPT_CACHE", "1") != "0"  # top-level cache_control on multi-turn calls: the vendor's automatic prefix caching, which leaves the reply unchanged
REPLAY_THINKING = os.environ.get("ANTHROPIC_REPLAY_THINKING", "1") != "0"  # "0": send tool-use turns back without their thinking blocks
STOP_REASONS = {"tool_use": "tool_calls", "end_turn": "stop", "max_tokens": "length", "model_context_window_exceeded": "length", "refusal": "content_filter", "stop_sequence": "stop"}


def is_native(client) -> bool:
    return hasattr(client, "messages") and not hasattr(client, "chat")


def to_tools(tools: list[dict] | None) -> list[dict]:
    return [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
             "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}} for t in tools or []]


def to_tool_choice(choice) -> dict | None:
    if choice is None:
        return None
    if isinstance(choice, dict):  # {"type": "function", "function": {"name": ...}}
        return {"type": "tool", "name": choice["function"]["name"]}
    return {"auto": {"type": "auto"}, "required": {"type": "any"}, "none": {"type": "none"}}[choice]


def _stored_blocks(m: dict) -> list | None:
    """The reply blocks an assistant turn was made from: on the message's own extra_content (text-only replies the
    deep research agent keeps) or on its first tool call's (tool-calling replies)."""
    candidates = [m.get("extra_content")] + [tc.get("extra_content") for tc in m.get("tool_calls") or []]
    for extra in candidates:
        if isinstance(extra, dict) and extra.get("anthropic_blocks"):
            blocks = extra["anthropic_blocks"]
            return blocks if REPLAY_THINKING else [x for x in blocks if x.get("type") != "thinking"]
    return None


def to_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """(system, messages) for messages.create from an OpenAI-format history: system messages become the system
    string; tool results become tool_result blocks in one user message right after the assistant turn that called
    them, with any following user text in the same message; consecutive user turns are merged; an assistant turn
    is replayed from its stored blocks (thinking included) when it has them, on the message or on its first tool call."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
    out: list[dict] = []

    def push_user(blocks: list[dict]) -> None:
        if out and out[-1]["role"] == "user":
            out[-1]["content"] = out[-1]["content"] + blocks
        else:
            out.append({"role": "user", "content": blocks})

    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        if role == "user":
            push_user([{"type": "text", "text": m["content"]}])
        elif role == "tool":
            push_user([{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m.get("content") or ""}])
        elif role == "assistant":
            blocks = _stored_blocks(m)
            if blocks is None:
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                for tc in m.get("tool_calls") or []:
                    fn = tc["function"]
                    try:
                        inp = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        inp = {"raw": fn.get("arguments")}
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": fn["name"], "input": inp})
            if not blocks:
                continue  # an empty reply has nothing the API accepts; the turn is dropped
            out.append({"role": "assistant", "content": blocks})
    return system, out


class Completion:
    """The reply in the shape the agents read (choices[0].message, usage)."""

    def __init__(self, resp, request: dict | None = None, cost_fn=None):
        blocks = [b.model_dump() if hasattr(b, "model_dump") else dict(b) for b in resp.content]
        self.stop_reason = resp.stop_reason
        refused = self.stop_reason == "refusal"
        text = "" if refused else "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        details = getattr(resp, "stop_details", None)
        self.stop_details = details.model_dump() if hasattr(details, "model_dump") else (dict(details) if isinstance(details, dict) else None)
        truncated = self.stop_reason in ("max_tokens", "model_context_window_exceeded")
        calls = []
        for b in blocks:
            if b.get("type") != "tool_use" or truncated or refused:
                continue
            tc = SimpleNamespace(id=b["id"], type="function", function=SimpleNamespace(name=b["name"], arguments=json.dumps(b.get("input") or {})),
                                 extra_content={"anthropic_blocks": blocks} if not calls else None)
            calls.append(tc)
        u = resp.usage
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        prompt = (u.input_tokens or 0) + cache_read + cache_write
        out_tokens = u.output_tokens or 0
        unbilled = self.stop_reason == "refusal" and out_tokens == 0
        cost = None
        if unbilled:
            cost = 0.0
        elif cost_fn is not None:
            try:
                cost = cost_fn(u.input_tokens or 0, out_tokens, cache_read, cache_write)  # cache reads and writes at their own prices
            except (KeyError, ValueError):
                cost = None
        self.id, self.model = getattr(resp, "id", None), getattr(resp, "model", None)
        self.request_id = getattr(resp, "_request_id", None)
        self.native_request = request
        message = SimpleNamespace(content=text, tool_calls=calls or None, role="assistant", extra_content={"anthropic_blocks": blocks} if blocks and not refused else None)
        self.choices = [SimpleNamespace(message=message, finish_reason=STOP_REASONS.get(self.stop_reason, self.stop_reason))]
        self.usage = SimpleNamespace(prompt_tokens=0 if unbilled else prompt, completion_tokens=0 if unbilled else out_tokens, cost=cost,
                                     prompt_tokens_details=SimpleNamespace(cached_tokens=cache_read, cache_write_tokens=cache_write))
        self._native = resp.model_dump() if hasattr(resp, "model_dump") else str(resp)

    def model_dump(self) -> dict:
        m = self.choices[0].message
        return {"id": self.id, "model": self.model, "request_id": self.request_id, "stop_reason": self.stop_reason, "stop_details": self.stop_details,
                "choices": [{"finish_reason": self.choices[0].finish_reason, "message": {
                    "content": m.content, "tool_calls": [{"id": t.id, "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in m.tool_calls or []]}}],
                "usage": {"prompt_tokens": self.usage.prompt_tokens, "completion_tokens": self.usage.completion_tokens}, "anthropic": self._native,
                "anthropic_request": self.native_request}


def creator(client, cost_fn=None):
    """A create(model=..., messages=..., **kwargs) with the OpenAI signature that calls client.messages.create.
    cost_fn(input_tokens, output_tokens, cache_read_tokens, cache_write_tokens), when given, prices each reply.
    Multi-turn calls (tools offered, or a history with assistant or tool turns) carry the top-level
    cache_control, so every turn of an agent loop reads the previous turn's prefix from the cache; a single-shot
    call (judge, plan, closed book) does not, since a cache write it would never read back costs more than the call."""
    def create(model: str, messages: list[dict], tools=None, tool_choice=None, max_tokens: int = MAX_TOKENS, **kwargs):
        system, msgs = to_messages(messages)
        kw = {"model": model, "max_tokens": max_tokens, "messages": msgs}
        if PROMPT_CACHE and (tools or any(m["role"] in ("assistant", "tool") for m in messages)):
            kw["cache_control"] = {"type": "ephemeral"}
        if system:
            kw["system"] = system
        if tools:
            kw["tools"] = to_tools(tools)
            choice = to_tool_choice(tool_choice)
            if choice:
                kw["tool_choice"] = choice
        kwargs.pop("temperature", None)  # the SDK has no temperature argument and Claude Fable rejects one
        kw.update(kwargs)
        return Completion(client.messages.create(**kw), request=kw, cost_fn=cost_fn)
    return create
