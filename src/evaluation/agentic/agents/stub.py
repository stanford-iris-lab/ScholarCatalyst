"""No-API agent for smoke tests: reads the retriever's top hits and lists the first few.

Its ranking equals the back-fill retriever's, so scores should match that retriever's
own run; a gap means the runner, the view, or the ranking assembly is broken.
"""
from __future__ import annotations

import json

LIST_N = 5   # ids the agent puts in its answer
READ_N = 20  # docs it "reads", which count as trajectory ids


def run(query: dict, view_dir, traj, tools) -> str:
    docs = tools["retrieve"](query["question"], READ_N, query.get("paper_published", ""))
    traj.tool_call("retrieve", {"query": query["question"], "top_k": READ_N}, doc_ids=[d["id"] for d in docs])
    traj.llm_call(input_tokens=0, output_tokens=0, note="stub agent, no LLM call")
    return json.dumps([d["id"] for d in docs[:LIST_N]])
