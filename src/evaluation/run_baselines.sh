#!/usr/bin/env bash
# Run full baseline pipeline (encode → retrieve → evaluate) for a benchmark.
# Usage: bash run_baselines.sh [BENCH_DIR]
#   BENCH_DIR  : path to benchmark (default: $BENCH_DIR env var)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCH_DIR="${1:-${BENCH_DIR:?BENCH_DIR not set — export BENCH_DIR=/path/to/benchmark or pass it as \$1}}"
BENCH_DIR="$(cd "$BENCH_DIR" && pwd)"

BASELINES=(bm25 bge-large openscholar scincl specter2 qwen3-4b qwen3-8b gemini-2)

log() { echo "[$(date '+%H:%M:%S')] $*"; }
sep() { echo "────────────────────────────────────────"; }

log "bench_dir  = $BENCH_DIR"
log "baselines  = ${BASELINES[*]}"
sep

cd "$SCRIPT_DIR"

for MODEL in "${BASELINES[@]}"; do
    sep
    log "▶ $MODEL"

    EMB_DIR="$BENCH_DIR/embeddings/$MODEL"

    # ── encode ──────────────────────────────────────────────────────────────
    if [[ "$MODEL" == "bm25" ]]; then
        SENTINEL="$EMB_DIR/bm25.pkl"
    else
        SENTINEL="$EMB_DIR/index.faiss"
    fi

    if [[ -f "$SENTINEL" ]]; then
        log "  encode  SKIP (${SENTINEL##*/} exists)"
    else
        log "  encode  START"
        python encode_corpus.py --model "$MODEL" --bench-dir "$BENCH_DIR"
        log "  encode  DONE"
    fi

    # ── retrieve ─────────────────────────────────────────────────────────────
    RUN_SUBPATH=$(python -c "from config import runs_subpath; print(runs_subpath('$MODEL'))")
    RUN_DIR="$BENCH_DIR/runs/$RUN_SUBPATH"
    RUN_OK=true
    for QT in core_query subfield_query; do
        [[ -f "$RUN_DIR/${QT}.jsonl" ]] || { RUN_OK=false; break; }
    done

    if $RUN_OK; then
        log "  retrieve SKIP (run files exist)"
    else
        log "  retrieve START"
        python retrieve.py --model "$MODEL" --bench-dir "$BENCH_DIR"
        log "  retrieve DONE"
    fi

    # ── evaluate ──────────────────────────────────────────────────────────────
    log "  evaluate START"
    python evaluate.py --model "$MODEL" --bench-dir "$BENCH_DIR"
    log "  evaluate DONE"
done

sep
log "All baselines complete → $BENCH_DIR/results/"
