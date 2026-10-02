"""One JSONL log per query: tool calls, LLM calls and the final answer, plus touched ids and token and cost totals.

Use it as a context manager; an exception inside the block is logged as an "error" line and re-raised.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from agentic.requests_log import RequestLog


class Trajectory:
    def __init__(self, path: Path, query_id: str, model: str = ""):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.query_id, self.model = path, query_id, model
        self.requests = RequestLog(path.with_suffix(""))
        self.seen: dict[str, None] = {}  # ordered set of corpus ids the agent touched
        self.input_tokens = self.output_tokens = 0
        self.n_llm_calls = self.n_tool_calls = 0
        self.cost_usd = 0.0
        self.tags: dict = {}  # run conditions an agent declares via tag(); carried onto the summary
        self.n_refusals = 0  # replies the model's classifier refused (empty, HTTP 200); one "refusal" line each
        self.last_finish_reason: str | None = None
        self.last_error: str | None = None
        self._t0 = time.monotonic()
        self._step = 0
        self._f = path.open("w")
        self._lock = threading.Lock()  # agents may log from worker threads
        self.event("start", query_id=query_id, model=model)

    def __enter__(self) -> "Trajectory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.event("error", error=repr(exc))
        self._f.close()
        self.requests.close()

    def event(self, kind: str, **fields) -> None:
        if kind == "error":
            self.last_error = str(fields.get("error"))
        with self._lock:
            self._step += 1
            rec = {"step": self._step, "t": round(time.monotonic() - self._t0, 3), "kind": kind, **fields}
            self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._f.flush()

    def tool_call(self, name: str, args, result: str = "", doc_ids=(), seconds: float | None = None, **fields) -> None:
        self.n_tool_calls += 1
        for doc_id in doc_ids:
            self.seen.setdefault(doc_id)
        self.event("tool", name=name, args=args, result=result[:2000], doc_ids=list(doc_ids), seconds=seconds, **fields)

    def llm_call(self, input_tokens: int, output_tokens: int, cost_usd: float | None = None, **fields) -> None:
        with self._lock:
            self.n_llm_calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            if cost_usd is not None:
                self.cost_usd += cost_usd
        self.event("llm", input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd, **fields)

    def refusal(self, agent: str, category: str | None = None) -> None:
        """A refused reply: counted on the summary and logged, so a run row and the paper can state how many."""
        with self._lock:
            self.n_refusals += 1
        self.event("refusal", agent=agent, category=category)

    def tag(self, **fields) -> None:
        """Declare a run condition: logged now as a "condition" line and carried onto finish."""
        self.tags.update(fields)
        self.event("condition", **fields)

    def summary(self) -> dict:
        return {**self.tags, "query_id": self.query_id, "model": self.model,
                "seconds": round(time.monotonic() - self._t0, 3),
                "n_llm_calls": self.n_llm_calls, "n_tool_calls": self.n_tool_calls,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost_usd, 6),
                "n_seen": len(self.seen), "n_refusals": self.n_refusals,
                "n_requests": self.requests.n, "n_requests_dropped": self.requests.dropped,
                "requests_path": str(self.requests.path)}

    def finish(self, answer: str, ranking: list[dict]) -> dict:
        summary = {**self.summary(), "n_ranked": len(ranking)}
        self.event("finish", answer=answer[:20000], ranking=ranking, **summary)
        return summary
