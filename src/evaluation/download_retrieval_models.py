#!/usr/bin/env python3
"""Download local retrieval models into retrieval_baseline/, skipping ones already present.

bm25, gemini-2 and text-embedding-3-large need no download.

    python download_retrieval_models.py --models qwen3-8b   # or: --models all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

BASELINE_DIR = Path(__file__).resolve().parents[2] / "retrieval_baseline"

MODELS: dict[str, dict] = {
    # ── general dense ─────────────────────────────────────────────────────────
    "bge-large": {
        "repo_id": "BAAI/bge-large-en-v1.5",
        "local_dir": BASELINE_DIR / "dense" / "bge-large-en-v1.5",
    },
    "qwen3-4b": {
        "repo_id": "Qwen/Qwen3-Embedding-4B",
        "local_dir": BASELINE_DIR / "dense" / "Qwen3-Embedding-4B",
    },
    "qwen3-8b": {
        "repo_id": "Qwen/Qwen3-Embedding-8B",
        "local_dir": BASELINE_DIR / "dense" / "Qwen3-Embedding-8B",
    },
    # ── paper-domain ──────────────────────────────────────────────────────────
    "openscholar": {
        "repo_id": "allenai/OpenScholar_Retriever",
        "local_dir": BASELINE_DIR / "paper" / "OpenScholar_Retriever",
    },
    "scincl": {
        "repo_id": "malteos/scincl",
        "local_dir": BASELINE_DIR / "paper" / "scincl",
    },
    "specter2": {
        "repo_id": "allenai/specter2_base",
        "local_dir": BASELINE_DIR / "paper" / "specter2_base",
        "note": "adapter allenai/specter2 is fetched automatically at runtime by the adapters library",
    },
}


def is_downloaded(local_dir: Path) -> bool:
    return (local_dir / "config.json").exists()


def download(key: str, cfg: dict) -> None:
    local_dir: Path = cfg["local_dir"]
    repo_id: str = cfg["repo_id"]

    if is_downloaded(local_dir):
        print(f"[SKIP] {key} — already present at {local_dir}")
        return

    print(f"[DOWN] {key}  ({repo_id})  →  {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"[DONE] {key}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=[*MODELS.keys(), "all"],
        metavar="MODEL",
        help=f"models to download (default: all).  choices: {', '.join(MODELS)} all",
    )
    args = parser.parse_args()

    targets = list(MODELS.keys()) if "all" in args.models else args.models
    for key in targets:
        download(key, MODELS[key])

    print("\nAll requested models ready.")


if __name__ == "__main__":
    main()
