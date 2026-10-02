"""Core retrieval metrics operating on ranked string ID lists."""
from __future__ import annotations

import math


def recall_at_k(ranked_ids: list[str], relevant_ids: list[str] | set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = set(ranked_ids[:k])
    return len(set(relevant_ids) & top) / len(relevant_ids)


def precision_at_k(ranked_ids: list[str], relevant_ids: list[str] | set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top = ranked_ids[:k]
    rel = set(relevant_ids)
    return sum(1 for rid in top if rid in rel) / k


def ndcg_at_k(ranked_ids: list[str], relevant_ids: list[str] | set[str], k: int) -> float:
    rel = set(relevant_ids)
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, rid in enumerate(ranked_ids[:k])
        if rid in rel
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked_ids: list[str], relevant_ids: list[str] | set[str]) -> float:
    rel = set(relevant_ids)
    for i, rid in enumerate(ranked_ids):
        if rid in rel:
            return 1.0 / (i + 1)
    return 0.0


_DEFAULT_RECALL_LEVELS = [i / 10 for i in range(11)]  # 0.0, 0.1, ..., 1.0


def interpolated_precision_recall_curve(
    ranked_ids: list[str],
    relevant_ids: list[str] | set[str],
    recall_levels: list[float] | None = None,
) -> list[tuple[float, float]]:
    """Return interpolated (recall, precision) pairs at each recall level.

    Interpolated precision at recall r = max precision at all ranks where recall >= r.
    """
    if recall_levels is None:
        recall_levels = _DEFAULT_RECALL_LEVELS
    rel = set(relevant_ids)
    n_rel = len(rel)
    if n_rel == 0:
        return [(r, 0.0) for r in recall_levels]

    # Build (recall, precision) at each rank where a relevant doc appears
    rp_points: list[tuple[float, float]] = []
    n_found = 0
    for i, rid in enumerate(ranked_ids):
        if rid in rel:
            n_found += 1
            rp_points.append((n_found / n_rel, n_found / (i + 1)))
    rp_points.append((0.0, 1.0))  # sentinel for interpolation at recall=0

    result = []
    for r in recall_levels:
        # max precision among points with recall >= r
        prec = max((p for rc, p in rp_points if rc >= r), default=0.0)
        result.append((r, prec))
    return result


def average_interpolated_precision(
    ranked_ids: list[str],
    relevant_ids: list[str] | set[str],
    recall_levels: list[float] | None = None,
) -> float:
    """Mean of interpolated precision at each recall level (e.g. 11-point AP)."""
    curve = interpolated_precision_recall_curve(ranked_ids, relevant_ids, recall_levels)
    return sum(p for _, p in curve) / len(curve) if curve else 0.0
