"""Unified LLM call layer: one call_llm_async() for every provider, and batch submit/poll/fetch.

call_llm_async() sends one request now; submit_batch() / poll_batch() / fetch_batch_results() handle batches
(up to 24h turnaround, about 50% cheaper), resumable across runs through the pending_batches helpers.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=False)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import provider_of

GEMINI_NATIVE_BASE   = "https://generativelanguage.googleapis.com/v1beta"

_openai_client_cache: dict[str, object] = {}
_anthropic_client_cache: dict[str, object] = {}


@dataclass
class LlmResult:
    raw: str
    parsed: object
    input_tokens: int
    output_tokens: int
    elapsed_sec: float
    cost: float | None = None
    error: str | None = None


_CODE_FENCE_PREFIX = re.compile(r"^```(?:json)?\s*\n?", re.IGNORECASE)
_CODE_FENCE_SUFFIX = re.compile(r"\n?```\s*$")


def raise_for_status_with_body(resp) -> None:
    """resp.raise_for_status() alone drops the response body. Every batch
    provider puts the actual reason (validation error, quota, permission)
    there, not in the generic "4xx Client Error" message, so every httpx call
    in this module uses this instead to keep that reason in the traceback."""
    try:
        resp.raise_for_status()
    except Exception as e:
        raise type(e)(f"{e}, body: {resp.text[:1000]}") from e


def parse_json_robust(raw: str) -> object | None:
    """Extract JSON from LLM response text: strips markdown fences, falls
    back to the outermost matching bracket pair if the fenced parse fails."""
    s = (raw or "").strip()
    s = _CODE_FENCE_PREFIX.sub("", s)
    s = _CODE_FENCE_SUFFIX.sub("", s).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        i = s.find(open_ch)
        if i < 0:
            continue
        j = s.rfind(close_ch)
        if j > i:
            try:
                return json.loads(s[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


# ── realtime clients ────────────────────────────────────────────────────────

def get_openai_compatible_client(api_key: str, base_url: str | None) -> object:
    """Cached openai.AsyncOpenAI client."""
    import openai
    cache_key = f"{base_url}:{api_key}"
    if cache_key not in _openai_client_cache:
        _openai_client_cache[cache_key] = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    return _openai_client_cache[cache_key]


def get_anthropic_client(api_key: str) -> object:
    import anthropic
    if api_key not in _anthropic_client_cache:
        _anthropic_client_cache[api_key] = anthropic.AsyncAnthropic(api_key=api_key)
    return _anthropic_client_cache[api_key]




async def call_llm_async(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    sema: Optional[asyncio.Semaphore] = None,
    max_tokens: int = 8192,
) -> LlmResult:
    """Single async LLM call, any provider. Retries once on a 429/rate-limit
    error."""
    provider = provider_of(model)

    async def _call() -> LlmResult:
        t0 = time.monotonic()
        if provider == "anthropic":
            client = get_anthropic_client(os.environ.get("ANTHROPIC_API_KEY"))
            resp = await client.messages.create(
                model=model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
            )
            raw = "".join(blk.text for blk in resp.content if getattr(blk, "type", None) == "text")
            in_tok, out_tok = resp.usage.input_tokens, resp.usage.output_tokens
            real_cost = None

        elif provider == "openai":
            client = get_openai_compatible_client(os.environ.get("OPENAI_API_KEY"), None)
            kwargs = dict(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
            )
            if seed is not None:
                kwargs["seed"] = seed
            resp = await client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
            in_tok, out_tok = resp.usage.prompt_tokens, resp.usage.completion_tokens
            real_cost = None

        elif provider == "google":
            import httpx
            bare_model = model.removeprefix("google/")
            api_key = os.environ.get("GEMINI_API_KEY", "")
            url = f"{GEMINI_NATIVE_BASE}/models/{bare_model}:generateContent"
            generation_config = {"temperature": temperature, "maxOutputTokens": max_tokens}
            if seed is not None:
                generation_config["seed"] = seed
            body = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": generation_config,
            }
            async with httpx.AsyncClient() as http_client:
                resp = await http_client.post(
                    url, headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json=body, timeout=120,
                )
            raise_for_status_with_body(resp)
            data = resp.json()
            parts = data["candidates"][0]["content"]["parts"]
            raw = "".join(p.get("text", "") for p in parts if "text" in p)
            usage = data.get("usageMetadata") or {}
            in_tok, out_tok = usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)
            real_cost = None

        else:
            raise ValueError(f"unknown provider {provider!r} for model {model!r}")

        elapsed = time.monotonic() - t0
        return LlmResult(raw=raw, parsed=parse_json_robust(raw), input_tokens=in_tok,
                          output_tokens=out_tok, elapsed_sec=elapsed, cost=real_cost)

    for attempt in range(2):
        try:
            if sema:
                async with sema:
                    result = await _call()
            else:
                result = await _call()
            if result.parsed is None:
                result.error = "json parse failed"
            return result
        except Exception as e:
            if attempt == 0 and ("429" in str(e) or "rate" in str(e).lower()):
                await asyncio.sleep(5)
                continue
            return LlmResult(raw="", parsed=None, input_tokens=0, output_tokens=0,
                              elapsed_sec=0.0, error=str(e)[:300])
    return LlmResult(raw="", parsed=None, input_tokens=0, output_tokens=0,
                      elapsed_sec=0.0, error="retries exhausted")


# ── batch: submit/poll/fetch are separate so a caller can persist the handle and collect results in a later run ──

@dataclass
class BatchRequest:
    custom_id: str
    model: str
    system: str
    user: str
    temperature: float = 0.0
    seed: Optional[int] = None
    max_tokens: int = 8192


@dataclass
class BatchHandle:
    provider: str  # "openai" | "anthropic" | "google"
    batch_id: str
    model: str  # google batches are single-model; informational for openai/anthropic


def _chat_body(req: BatchRequest) -> dict:
    body = {
        "model": req.model,
        "messages": [{"role": "system", "content": req.system}, {"role": "user", "content": req.user}],
        "temperature": req.temperature,
    }
    if req.seed is not None:
        body["seed"] = req.seed
    return body


def submit_batch(provider: str, requests: list[BatchRequest]) -> BatchHandle:
    """Submits one batch for `provider`. All `requests` must share the same provider."""
    if provider == "openai":
        return _submit_batch_openai(requests)
    if provider == "anthropic":
        return _submit_batch_anthropic(requests)
    if provider == "google":
        return _submit_batch_google(requests)
    raise ValueError(f"batch not supported for provider {provider!r}")


def _submit_batch_openai(requests: list[BatchRequest]) -> BatchHandle:
    import io
    import openai
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    lines = [
        json.dumps({
            "custom_id": r.custom_id, "method": "POST", "url": "/v1/chat/completions",
            "body": _chat_body(r),
        })
        for r in requests
    ]
    buf = io.BytesIO(("\n".join(lines) + "\n").encode("utf-8"))
    buf.name = "batch_input.jsonl"
    uploaded = client.files.create(file=buf, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h",
    )
    return BatchHandle(provider="openai", batch_id=batch.id, model="")


def _submit_batch_anthropic(requests: list[BatchRequest]) -> BatchHandle:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    reqs = [
        {
            "custom_id": r.custom_id,
            "params": {
                "model": r.model, "max_tokens": r.max_tokens, "system": r.system,
                "messages": [{"role": "user", "content": r.user}],
            },
        }
        for r in requests
    ]
    batch = client.messages.batches.create(requests=reqs)
    return BatchHandle(provider="anthropic", batch_id=batch.id, model="")


def _submit_batch_google(requests: list[BatchRequest]) -> BatchHandle:
    import httpx
    models = {r.model for r in requests}
    if len(models) != 1:
        raise ValueError(f"Gemini batch is single-model; got {models}")
    model = requests[0].model
    bare_model = model.removeprefix("google/")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    url = f"{GEMINI_NATIVE_BASE}/models/{bare_model}:batchGenerateContent"
    body = {
        "batch": {
            "display_name": f"llm_client-{bare_model}",
            "input_config": {
                "requests": {
                    "requests": [
                        {
                            "request": {
                                "system_instruction": {"parts": [{"text": r.system}]},
                                "contents": [{"role": "user", "parts": [{"text": r.user}]}],
                                "generationConfig": {"temperature": r.temperature},
                            },
                            "metadata": {"key": r.custom_id},
                        }
                        for r in requests
                    ]
                }
            },
        }
    }
    for attempt in range(3):
        try:
            resp = httpx.post(url, headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                               json=body, timeout=120)
            break
        except (httpx.ReadTimeout, httpx.ConnectError, httpx.ConnectTimeout):
            if attempt == 2:
                raise
            time.sleep(5)
    raise_for_status_with_body(resp)
    data = resp.json()
    return BatchHandle(provider="google", batch_id=data["name"], model=model)


def poll_batch(handle: BatchHandle) -> str:
    """Returns a normalized status: 'pending' or 'done' or 'failed'."""
    if handle.provider == "openai":
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        batch = client.batches.retrieve(handle.batch_id)
        if batch.status == "completed":
            return "done"
        if batch.status in ("failed", "expired", "cancelled"):
            return "failed"
        return "pending"

    if handle.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        batch = client.messages.batches.retrieve(handle.batch_id)
        return "done" if batch.processing_status == "ended" else "pending"

    if handle.provider == "google":
        import httpx
        api_key = os.environ.get("GEMINI_API_KEY", "")
        resp = httpx.get(f"{GEMINI_NATIVE_BASE}/{handle.batch_id}",
                          headers={"x-goog-api-key": api_key}, timeout=30)
        raise_for_status_with_body(resp)
        state = resp.json().get("metadata", {}).get("state")
        if state == "BATCH_STATE_SUCCEEDED":
            return "done"
        if state in ("BATCH_STATE_FAILED", "BATCH_STATE_CANCELLED", "BATCH_STATE_EXPIRED"):
            return "failed"
        return "pending"

    raise ValueError(f"unknown provider {handle.provider!r}")


def fetch_batch_results(handle: BatchHandle) -> dict[str, LlmResult]:
    """Only call once poll_batch() returns 'done'. Returns {custom_id: LlmResult}."""
    if handle.provider == "openai":
        return _fetch_batch_results_openai(handle)
    if handle.provider == "anthropic":
        return _fetch_batch_results_anthropic(handle)
    if handle.provider == "google":
        return _fetch_batch_results_google(handle)
    raise ValueError(f"unknown provider {handle.provider!r}")


def _fetch_batch_results_openai(handle: BatchHandle) -> dict[str, LlmResult]:
    import openai
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    batch = client.batches.retrieve(handle.batch_id)
    out: dict[str, LlmResult] = {}
    if batch.output_file_id:
        content = client.files.content(batch.output_file_id).text
        for line in content.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            custom_id = item["custom_id"]
            if item.get("error"):
                out[custom_id] = LlmResult(raw="", parsed=None, input_tokens=0, output_tokens=0,
                                            elapsed_sec=0.0, error=str(item["error"])[:300])
                continue
            body = item["response"]["body"]
            raw = body["choices"][0]["message"]["content"] or ""
            usage = body.get("usage") or {}
            in_tok, out_tok = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            cost = None
            out[custom_id] = LlmResult(raw=raw, parsed=parse_json_robust(raw), input_tokens=in_tok,
                                        output_tokens=out_tok, elapsed_sec=0.0, cost=cost)
    if batch.error_file_id:
        content = client.files.content(batch.error_file_id).text
        for line in content.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            out.setdefault(item["custom_id"], LlmResult(
                raw="", parsed=None, input_tokens=0, output_tokens=0, elapsed_sec=0.0,
                error=str(item.get("error"))[:300],
            ))
    return out


def _fetch_batch_results_anthropic(handle: BatchHandle) -> dict[str, LlmResult]:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    out: dict[str, LlmResult] = {}
    for result in client.messages.batches.results(handle.batch_id):
        custom_id = result.custom_id
        if result.result.type != "succeeded":
            out[custom_id] = LlmResult(raw="", parsed=None, input_tokens=0, output_tokens=0,
                                        elapsed_sec=0.0, error=f"batch result: {result.result.type}")
            continue
        message = result.result.message
        raw = "".join(blk.text for blk in message.content if getattr(blk, "type", None) == "text")
        in_tok, out_tok = message.usage.input_tokens, message.usage.output_tokens
        cost = None
        out[custom_id] = LlmResult(raw=raw, parsed=parse_json_robust(raw), input_tokens=in_tok,
                                    output_tokens=out_tok, elapsed_sec=0.0, cost=cost)
    return out


def _fetch_batch_results_google(handle: BatchHandle) -> dict[str, LlmResult]:
    import httpx
    api_key = os.environ.get("GEMINI_API_KEY", "")
    resp = httpx.get(f"{GEMINI_NATIVE_BASE}/{handle.batch_id}",
                      headers={"x-goog-api-key": api_key}, timeout=60)
    raise_for_status_with_body(resp)
    data = resp.json()
    items = data.get("response", {}).get("inlinedResponses", {}).get("inlinedResponses", [])
    out: dict[str, LlmResult] = {}
    for item in items:
        custom_id = (item.get("metadata") or {}).get("key", "")
        body = item.get("response")
        candidates = (body or {}).get("candidates")
        if not candidates:
            err = item.get("error") or {"message": "no candidates in response"}
            out[custom_id] = LlmResult(raw="", parsed=None, input_tokens=0, output_tokens=0,
                                        elapsed_sec=0.0, error=str(err)[:300])
            continue
        parts = candidates[0].get("content", {}).get("parts", [])
        raw = "".join(p.get("text", "") for p in parts if "text" in p)
        usage = body.get("usageMetadata") or {}
        in_tok, out_tok = usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)
        cost = None
        out[custom_id] = LlmResult(raw=raw, parsed=parse_json_robust(raw), input_tokens=in_tok,
                                    output_tokens=out_tok, elapsed_sec=0.0, cost=cost)
    return out


# ── resumable pending-batch state, so a killed/re-run process picks up
# polling instead of re-submitting. One JSON file per caller-chosen path
# (e.g. report_dir/pending_batches.json). ───────────────────────────────────

def load_pending_batches(state_path: Path) -> list[dict]:
    if not state_path.is_file():
        return []
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_pending_batches(state_path: Path, entries: list[dict]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def collect_batches_sync(entries: list[dict], state_path: Path, poll_interval: float = 60.0,
                         max_interval: float = 300.0) -> dict[str, LlmResult]:
    """Block until every batch entry settles; returns {custom_id: LlmResult}."""
    import time
    remaining = list(entries)
    out: dict[str, LlmResult] = {}
    interval = poll_interval
    while remaining:
        still_pending = []
        for entry in remaining:
            handle = BatchHandle(provider=entry["provider"], batch_id=entry["batch_id"], model=entry["model"])
            status = poll_batch(handle)
            if status == "pending":
                still_pending.append(entry)
                continue
            if status == "failed":
                for item in entry["items"]:
                    out[item["custom_id"]] = LlmResult(raw="", parsed=None, input_tokens=0, output_tokens=0,
                                                        elapsed_sec=0.0, error="batch failed/expired/cancelled")
                continue
            results = fetch_batch_results(handle)
            out.update(results)
        remaining = still_pending
        save_pending_batches(state_path, remaining)
        if remaining:
            print(f"[batch] {len(remaining)} batch(es) still pending, checking again in {int(interval)}s ...")
            time.sleep(interval)
            interval = min(interval * 1.5, max_interval)
    state_path.unlink(missing_ok=True)
    return out
