"""Evaluate retrieval runs against the benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import BENCH_DIR, MODEL_GROUPS, MODEL_REGISTRY, QUERY_TYPES, runs_subpath
from utils import load_queries, load_run, load_seen, source_alias_map
from calculate_metrics import recall_at_k, precision_at_k, ndcg_at_k

EVAL_K_VALUES          = [5, 10, 15, 20, 25]
EVAL_K_VALUES_EXTENDED = EVAL_K_VALUES + [50, 100]


def k_values_for(model: str) -> list[int]:
    """Every run reports k up to 100. Agentic runs written by agentic/runner.py are
    filled to depth 100 with a retriever ranking; llm_query_augment.py / llm_rerank.py
    runs rank a small pool, so their recall@100 equals recall at the pool size."""
    return EVAL_K_VALUES_EXTENDED


# gold (annotation-based) evals

def gold_central_eval(queries: list[dict], run: dict, k_values: list[int]) -> dict:
    """Precision/recall/ndcg for core_query using annotated positive_docs."""
    prec_scores = {k: [] for k in k_values}
    rec_scores  = {k: [] for k in k_values}
    ndcg_scores = {k: [] for k in k_values}
    count = 0
    for q in queries:
        pos = q.get("positive_docs") or []
        if not pos:
            continue
        pos_ids    = [p["id"] if isinstance(p, dict) else p for p in pos]
        ranked_ids = [r["doc_id"] for r in run.get(q["query_id"], [])]
        for k in k_values:
            prec_scores[k].append(precision_at_k(ranked_ids, pos_ids, k))
            rec_scores[k].append(recall_at_k(ranked_ids, pos_ids, k))
            ndcg_scores[k].append(ndcg_at_k(ranked_ids, pos_ids, k))
        count += 1
    if not count:
        null = {f"{m}@{k}": None for k in k_values for m in ("precision", "recall", "ndcg")}
        return {"n_queries": 0, **null}
    out: dict = {"n_queries": count}
    for k in k_values:
        out[f"precision@{k}"] = round(sum(prec_scores[k]) / count, 4)
        out[f"recall@{k}"]    = round(sum(rec_scores[k])  / count, 4)
        out[f"ndcg@{k}"]      = round(sum(ndcg_scores[k]) / count, 4)
    return out


def gold_thread_eval(queries: list[dict], run: dict, k_values: list[int]) -> dict:
    """Precision/recall/ndcg for subfield_query using annotated positive_docs."""
    prec_scores = {k: [] for k in k_values}
    rec_scores  = {k: [] for k in k_values}
    ndcg_scores = {k: [] for k in k_values}
    count = 0
    for q in queries:
        pos = q.get("positive_docs") or []
        if not pos:
            continue
        pos_ids    = [p["id"] if isinstance(p, dict) else p for p in pos]
        ranked_ids = [r["doc_id"] for r in run.get(q["query_id"], [])]
        for k in k_values:
            prec_scores[k].append(precision_at_k(ranked_ids, pos_ids, k))
            rec_scores[k].append(recall_at_k(ranked_ids, pos_ids, k))
            ndcg_scores[k].append(ndcg_at_k(ranked_ids, pos_ids, k))
        count += 1
    if not count:
        null = {f"{m}@{k}": None for k in k_values for m in ("precision", "recall", "ndcg")}
        return {"n_queries": 0, **null}
    out: dict = {"n_queries": count}
    for k in k_values:
        out[f"precision@{k}"] = round(sum(prec_scores[k]) / count, 4)
        out[f"recall@{k}"]    = round(sum(rec_scores[k])  / count, 4)
        out[f"ndcg@{k}"]      = round(sum(ndcg_scores[k]) / count, 4)
    return out


def trajectory_recall(queries: list[dict], seen: dict[str, list[str]], aliases: dict[str, set[str]] | None = None) -> float | None:
    """Share of a query's positives the agent touched anywhere in its run, averaged over
    queries with positives. The ceiling any ranking built from that run can reach."""
    scores = []
    for q in queries:
        pos = q.get("positive_docs") or []
        if not pos or q["query_id"] not in seen:
            continue
        pos_ids = {p["id"] if isinstance(p, dict) else p for p in pos}
        touched = set(seen[q["query_id"]]) - (aliases or {}).get(q["query_id"], {q.get("paper_id", "")})
        scores.append(len(pos_ids & touched) / len(pos_ids))
    return round(sum(scores) / len(scores), 4) if scores else None


def filter_self_from_run(run: dict, queries: list[dict], aliases: dict[str, set[str]] | None = None) -> dict:
    """Rankings without the query's own paper: its id, or every corpus id of it when `aliases` is given."""
    drop = {q["query_id"]: (aliases or {}).get(q["query_id"], {q.get("paper_id", "")}) for q in queries}
    return {
        qid: [r for r in ranking if r["doc_id"] not in drop.get(qid, set())]
        for qid, ranking in run.items()
    }


def write_results(model: str, bench_dir: Path, qtypes: list[str], metrics: dict, partial: bool) -> None:
    results_dir = bench_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / f"{model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)  # model may contain "/" (agentic/{descriptor})
    to_write = dict(metrics)
    to_write["partial"] = partial
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        for qt in qtypes:
            if qt in to_write and qt in existing and isinstance(existing[qt], dict):
                for sub, sub_d in existing[qt].items():
                    if sub not in to_write[qt]:
                        to_write[qt][sub] = sub_d
    out_path.write_text(json.dumps(to_write, indent=2, ensure_ascii=False))

    summary_path = results_dir / "summary.json"
    summary: dict = {"benchmark": bench_dir.name, "models": {}}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.setdefault("models", {})[model] = {
        qt: {
            sub: {k: v for k, v in sub_d.items() if k != "per_query"}
            for sub, sub_d in to_write[qt].items()
            if isinstance(sub_d, dict)
        }
        for qt in qtypes if qt in to_write
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if not partial:
        print(f"\nsaved → {out_path}")
        print(f"summary → {summary_path}")


def quick_scores(qt: str, run_path: Path, queries: list[dict],
                 rels_by_qid: dict[str, list[str]], max_k: int | None = None) -> None:
    """Compact main-metric line printed right after a pipeline run finishes.

    Main metrics are recall at 20/50/100 and nDCG at the largest reachable k.
    Pipelines that rank a bounded candidate pool pass max_k so k stops at the
    pool size. queries use the run-script schema (id, paper_id)."""
    ks = [k for k in (20, 50, 100) if max_k is None or k <= max_k]
    if not ks and max_k:
        ks = [max_k]
    run: dict[str, list[str]] = {}
    for line in run_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            run[r["query_id"]] = [d["doc_id"] for d in r["ranking"]]
    rows = []
    for q in queries:
        qid = q["id"]
        pos = rels_by_qid.get(qid) or []
        if qid not in run or not pos:
            continue
        ranked = [d for d in run[qid] if d != q.get("paper_id", "")]
        rows.append((ranked, pos))
    if not rows:
        print(f"[{qt}] quick scores: no scored queries")
        return
    cells = [f"R@{k} {sum(recall_at_k(r, p, k) for r, p in rows) / len(rows):.2f}"
             for k in ks]
    ndcg_k = ks[-1]
    cells.append(f"nDCG@{ndcg_k} {sum(ndcg_at_k(r, p, ndcg_k) for r, p in rows) / len(rows):.2f}")
    print(f"[{qt}] n={len(rows)}  " + "  ".join(cells))


def print_metric_table(qt: str, gold: dict, k_values: list[int]) -> None:
    print(f"[{qt}] n={gold['n_queries']}")
    if not gold["n_queries"]:
        return
    print(f"  {'k':>4}  {'precision':>9}  {'recall':>9}  {'ndcg':>9}")
    for k in k_values:
        p = gold.get(f"precision@{k}")
        r = gold.get(f"recall@{k}")
        n = gold.get(f"ndcg@{k}")
        if p is None:
            continue
        print(f"  {k:>4}  {p:>9.4f}  {r:>9.4f}  {n:>9.4f}")


def print_summary_table(all_metrics: list[dict], qtypes: list[str]) -> None:
    """Cross model comparison, one row per model."""
    columns = [("R@20", "recall@20"), ("R@50", "recall@50"),
               ("R@100", "recall@100"), ("N@100", "ndcg@100")]
    name_width = max([len("model")] + [len(m["model"]) for m in all_metrics])
    group_width = len("  ".join(f"{label:>7}" for label, _ in columns))
    line1 = " " * name_width + "".join(f" | {qt:<{group_width}}" for qt in qtypes)
    line2 = "model".ljust(name_width) + "".join(
        " | " + "  ".join(f"{label:>7}" for label, _ in columns) for _ in qtypes)
    print("\n" + line1)
    print(line2)
    print("-" * len(line2))
    for m in all_metrics:
        row = m["model"].ljust(name_width)
        for qt in qtypes:
            gold = (m.get(qt) or {}).get("gold") or {}
            cells = []
            for label, key in columns:
                v = gold.get(key)
                cells.append(f"{v:>7.2f}" if v is not None else f"{'-':>7}")
            row += " | " + "  ".join(cells)
        print(row)


def eval_model(model: str, bench_dir: Path, qtypes: list[str], only_run_queries: bool = False) -> dict:
    metrics: dict = {
        "model":     model,
        "benchmark": bench_dir.name,
        "eval_date": date.today().isoformat(),
    }
    k_values = k_values_for(model)

    for qt in qtypes:
        queries = load_queries(bench_dir, qt)
        run     = load_run(bench_dir, model, qt)
        if not run:
            print(f"[{qt}] no run file found — skipping")
            continue
        aliases = source_alias_map(bench_dir, queries)
        run = filter_self_from_run(run, queries, aliases)
        if only_run_queries:  # pilots: score only the queries the run actually covers
            n_all = len(queries)
            queries = [q for q in queries if q["query_id"] in run]
            print(f"[{qt}] scoring {len(queries)} of {n_all} queries present in the run")

        gold = gold_central_eval(queries, run, k_values) if qt == "core_query" \
               else gold_thread_eval(queries, run, k_values)
        seen = load_seen(bench_dir, model, qt)
        if seen:
            gold["trajectory_recall"] = trajectory_recall(queries, seen, aliases)
        metrics[qt] = {"gold": dict(gold)}
        print_metric_table(qt, gold, k_values)

    write_results(model, bench_dir, qtypes, metrics, partial=False)
    return metrics


def resolve_models(model_arg: str | None, bench_dir: Path) -> list[str]:
    """Resolve --model value to a concrete list. None → interactive selection."""
    runs_dir = bench_dir / "runs"

    def available(names: list[str]) -> list[str]:
        return [m for m in names if (runs_dir / runs_subpath(m)).exists()]

    def extra_agentic() -> list[str]:
        """Every agentic run found under runs/agentic/, as a path relative to runs/."""
        agentic_dir = runs_dir / "agentic"
        if not agentic_dir.exists():
            return []
        leaf_dirs = {p.parent for p in agentic_dir.rglob("*.jsonl")}
        return sorted(str(d.relative_to(runs_dir)) for d in leaf_dirs)

    # named group
    if model_arg in MODEL_GROUPS:
        return available(MODEL_GROUPS[model_arg]) + (
            extra_agentic() if model_arg == "agentic" else []
        )

    # all
    if model_arg == "all":
        registry = available(list(MODEL_REGISTRY))
        return registry + extra_agentic()

    # comma-separated explicit list
    if model_arg is not None and "," in model_arg:
        return [m.strip() for m in model_arg.split(",") if m.strip()]

    # single model
    if model_arg is not None:
        return [model_arg]

    # ── interactive ───────────────────────────────────────────────────────────
    extra_ag = extra_agentic()
    groups: dict[str, list[str]] = {
        g: available(ms) + (extra_ag if g == "agentic" else [])
        for g, ms in MODEL_GROUPS.items()
    }
    all_models = available(list(MODEL_REGISTRY)) + extra_ag

    tty_in  = open("/dev/tty")
    tty_out = open("/dev/tty", "w")

    def ask(prompt: str) -> str:
        tty_out.write(prompt)
        tty_out.flush()
        return tty_in.readline().strip()

    tty_out.write("\n=== Model Selection ===\n")
    tty_out.write("  0) all\n")
    group_keys = list(groups.keys())
    for i, g in enumerate(group_keys, 1):
        avail = groups[g]
        tty_out.write(f"  {i}) {g:<8}  {', '.join(avail) if avail else '(none available)'}\n")
    tty_out.write(f"  {len(group_keys)+1}) custom\n")
    tty_out.flush()

    choice = ask(f"\nSelect [0-{len(group_keys)+1}]: ")

    if choice == "0":
        tty_in.close(); tty_out.close()
        return all_models

    if choice.isdigit() and 1 <= int(choice) <= len(group_keys):
        g_name   = group_keys[int(choice) - 1]
        g_models = groups[g_name]
        if not g_models:
            tty_out.write("  No models available in this group.\n"); tty_out.flush()
            tty_in.close(); tty_out.close()
            return []
        tty_out.write(f"\n  [{g_name}]\n")
        for i, m in enumerate(g_models, 1):
            tty_out.write(f"    {i}) {m}\n")
        tty_out.flush()
        sub = ask("  Select (indices comma-separated, or Enter for all): ")
        tty_in.close(); tty_out.close()
        if not sub:
            return g_models
        idxs = [int(x.strip()) for x in sub.split(",") if x.strip().isdigit()]
        return [g_models[i - 1] for i in idxs if 0 < i <= len(g_models)]

    if choice == str(len(group_keys) + 1):
        tty_out.write("\n  All available models:\n")
        for i, m in enumerate(all_models, 1):
            tty_out.write(f"    {i}) {m}\n")
        tty_out.flush()
        sel = ask("  Select (indices comma-separated): ")
        tty_in.close(); tty_out.close()
        idxs = [int(x.strip()) for x in sel.split(",") if x.strip().isdigit()]
        return [all_models[i - 1] for i in idxs if 0 < i <= len(all_models)]

    tty_in.close(); tty_out.close()
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model",      default=None,
                    help="model/group name, comma-separated model list, 'all',"
                         " or omit for interactive selection"
                         f" (groups: {', '.join(MODEL_GROUPS)})")
    ap.add_argument("--bench-dir",  type=Path, default=BENCH_DIR)
    ap.add_argument("--set",        default="", dest="set_name",
                    help="query set subdir under bench-dir (e.g. llm_set, author_set,"
                         " author_set/full_text); queries/rels/runs/results all live"
                         " under it, matching retrieve.py --set")
    ap.add_argument("--query-type", default=None,
                    help="one of: " + ", ".join(QUERY_TYPES) + " (default: all)")
    ap.add_argument("--only-run-queries", action="store_true",
                    help="score only queries present in the run file (for --limit pilots); the default scores every query")
    args = ap.parse_args()

    bench_dir = args.bench_dir.resolve()
    if args.set_name:
        bench_dir = bench_dir / args.set_name
    models = resolve_models(args.model, bench_dir)
    if not models:
        print("No models selected.")
        return

    qtypes = [args.query_type] if args.query_type else QUERY_TYPES

    if len(models) > 1:
        print(f"Running {len(models)} models: {', '.join(models)}")

    all_metrics = []
    for model in models:
        if len(models) > 1:
            print(f"\n{'='*60}\n  model: {model}\n{'='*60}")
        all_metrics.append(eval_model(model, bench_dir, qtypes,
                                      only_run_queries=args.only_run_queries))

    if len(all_metrics) > 1:
        print_summary_table(all_metrics, qtypes)


if __name__ == "__main__":
    main()
