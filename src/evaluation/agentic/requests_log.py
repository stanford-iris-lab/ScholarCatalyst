"""Per request log of an agent's model traffic: <qid>.requests.jsonl with FIELDS, raw bodies under <qid>.bodies/.

chat_create records to the current thread's log (set_current), so wrap pool workers with inherit().
A record that cannot be written is counted as dropped and never fails the query.
"""
from __future__ import annotations

import gzip
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ("ts", "duration_ms", "model", "input_tokens", "output_tokens", "cache_read_tokens",
          "cache_creation_tokens", "cost_metered_usd", "agent", "request_id", "body_ref", "error")

_current = threading.local()


def set_current(log: "RequestLog | None") -> None:
    _current.log = log


def current() -> "RequestLog | None":
    return getattr(_current, "log", None)


def inherit(fn):
    """fn wrapped so that a pool worker thread records to the calling thread's log."""
    log = current()

    def wrapped(*args, **kwargs):
        set_current(log)
        try:
            return fn(*args, **kwargs)
        finally:
            set_current(None)
    return wrapped


def utc_iso(t: float | None = None) -> str:
    """ISO 8601 UTC of a unix time, now by default."""
    dt = datetime.now(timezone.utc) if t is None else datetime.fromtimestamp(t, timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_default(obj):
    """json.dump fallback for a body: pydantic models dump, anything else is kept as text."""
    return obj.model_dump() if hasattr(obj, "model_dump") else str(obj)


class RequestLog:
    def __init__(self, path_base: Path):
        self.path = Path(f"{path_base}.requests.jsonl")
        self.bodies = Path(f"{path_base}.bodies")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.rmtree(self.bodies, ignore_errors=True)  # a killed attempt's bodies would otherwise outlive the new sidecar
        self.n = 0  # lines written
        self.dropped = 0  # records the recorder could not write
        self._seq = 0  # body numbering
        self._lock = threading.Lock()
        self._f = self.path.open("w")

    def close(self) -> None:
        self._f.close()

    def append(self, *, ts: str, duration_ms: float | None, model: str, input_tokens: int = 0, output_tokens: int = 0,
               cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
               cost_metered_usd: float | None = None, agent: str = "main", request_id: str | None = None,
               body: dict | None = None, error: str | None = None) -> dict | None:
        """One line; body, when given, is gzipped under the bodies dir and referenced by body_ref.
        A record that cannot be written is printed, counted in dropped and skipped (None)."""
        try:
            return self._write(ts, duration_ms, model, input_tokens, output_tokens, cache_read_tokens,
                               cache_creation_tokens, cost_metered_usd, agent, request_id, body, error)
        except Exception as e:
            with self._lock:
                self.dropped += 1
            print(f"[warn] {self.path.name}: request record dropped: {e!r}", file=sys.stderr)
            return None

    def _write(self, ts, duration_ms, model, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens,
               cost_metered_usd, agent, request_id, body, error) -> dict:
        body_ref = None
        if body is not None:
            with self._lock:
                self._seq += 1
                n = self._seq
            self.bodies.mkdir(exist_ok=True)
            with gzip.open(self.bodies / f"{n}.json.gz", "wt", compresslevel=6) as f:
                json.dump(body, f, ensure_ascii=False, default=_json_default)
            body_ref = f"{self.bodies.name}/{n}.json.gz"
        rec = {"ts": ts, "duration_ms": None if duration_ms is None else round(duration_ms, 1), "model": model,
               "input_tokens": input_tokens, "output_tokens": output_tokens, "cache_read_tokens": cache_read_tokens,
               "cache_creation_tokens": cache_creation_tokens,
               "cost_metered_usd": cost_metered_usd, "agent": agent, "request_id": request_id,
               "body_ref": body_ref, "error": error}
        with self._lock:
            self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._f.flush()
            self.n += 1
        return rec

    def call(self, create, model: str, messages: list[dict], kwargs: dict, *, agent: str):
        """create(model=..., messages=..., **kwargs) with the request recorded: the response, or the
        exception (re-raised) with its request id when the SDK attached one. append never raises,
        so the exception the caller sees is always create's own."""
        ts, t0 = utc_iso(), time.monotonic()
        tools = kwargs.get("tools")
        request = {"model": model, "messages": messages, "tools": tools,
                   "kwargs": {k: v for k, v in kwargs.items() if k != "tools"}}
        try:
            resp = create(model=model, messages=messages, **kwargs)
        except Exception as e:
            self.append(ts=ts, duration_ms=(time.monotonic() - t0) * 1000, model=model, agent=agent,
                        request_id=getattr(e, "request_id", None), body={"request": request, "response": None},
                        error=repr(e))
            raise
        usage = getattr(resp, "usage", None)
        details = getattr(usage, "prompt_tokens_details", None)
        n_in = getattr(usage, "prompt_tokens", 0) or 0
        n_out = getattr(usage, "completion_tokens", 0) or 0
        self.append(ts=ts, duration_ms=(time.monotonic() - t0) * 1000, model=getattr(resp, "model", None) or model,
                    input_tokens=n_in, output_tokens=n_out,
                    cache_read_tokens=getattr(details, "cached_tokens", 0) or 0,
                    cache_creation_tokens=getattr(details, "cache_write_tokens", 0) or 0,
                    cost_metered_usd=getattr(usage, "cost", None),
                    agent=agent, request_id=getattr(resp, "request_id", None) or getattr(resp, "id", None),
                    body={"request": request, "response": resp})  # serialized inside append's guard; a native reply dumps its wire request too
        return resp
