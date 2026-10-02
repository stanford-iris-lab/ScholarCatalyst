"""Shared terminal report for retrieve.py / llm_query_augment.py / llm_rerank.py:
measured time (and cost, for LLM-based pipelines) for the queries actually
run, plus a linear projection to larger query counts."""
from __future__ import annotations


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def run_report(
    label: str,
    n_queries: int,
    elapsed_sec: float,
    total_cost: float | None = None,
    project_to: tuple[int, ...] = (500, 1000),
) -> None:
    if n_queries == 0:
        return
    per_query_sec  = elapsed_sec / n_queries
    per_query_cost = (total_cost / n_queries) if total_cost is not None else None

    print(f"\n=== Run report: {label} ===")
    cost_part = f"   cost=${total_cost:.2f}   (${per_query_cost:.4f}/query)" if total_cost is not None else ""
    print(f"  measured    n={n_queries:<6} elapsed={format_duration(elapsed_sec):<10}"
          f"({per_query_sec:.4f}s/query){cost_part}")

    for n in project_to:
        if n <= n_queries:
            continue
        proj_sec = per_query_sec * n
        cost_part = f"   cost=~${per_query_cost * n:.2f}" if per_query_cost is not None else ""
        print(f"  projected   n={n:<6} elapsed=~{format_duration(proj_sec):<9}{cost_part}")
