from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=False)
BASELINE_DIR = ROOT / "retrieval_baseline"
BENCH_DIR    = Path(os.environ["BENCH_DIR"])

QUERY_TYPES = ["core_query", "subfield_query"]
TOP_K       = 100

QWEN_INSTRUCTION = (
    "Instruct: Given a scientific research question, retrieve papers that could inspire or contribute to answering the question, including relevant background knowledge, methods, and ideas.\nQuery: "
)

MODEL_REGISTRY: dict[str, dict] = {
    "bm25": {
        "type": "bm25",
    },
    "bge-large": {
        "type": "dense",
        "loader": "st",
        "path": BASELINE_DIR / "dense" / "bge-large-en-v1.5",
    },
    "qwen3-4b": {
        "type": "dense",
        "loader": "st",
        "path": BASELINE_DIR / "dense" / "Qwen3-Embedding-4B",
        "query_instruction": QWEN_INSTRUCTION,
        "batch_size": 4,
        "max_seq_length": 512,
    },
    "qwen3-8b": {
        "type": "dense",
        "loader": "st",
        "path": BASELINE_DIR / "dense" / "Qwen3-Embedding-8B",
        "query_instruction": QWEN_INSTRUCTION,
        "batch_size": 2,
        "max_seq_length": 512,
    },
    "openscholar": {
        "type": "dense",
        "loader": "contriever",
        "path": BASELINE_DIR / "paper" / "OpenScholar_Retriever",
    },
    "scincl": {
        "type": "dense",
        "loader": "st",
        "path": BASELINE_DIR / "paper" / "scincl",
    },
    "specter2": {
        "type": "dense",
        "loader": "adapter",
        "path": BASELINE_DIR / "paper" / "specter2_base",
    },
    "gemini-2": {
        "type": "api",
        "model": "google/gemini-embedding-2",
    },
    "text-embedding-3-large": {
        "type": "api",
        "model": "openai/text-embedding-3-large",
    },
    # multi-vector late interaction; encoded and retrieved by
    # retrieve_lateon.py (PLAID index), not encode_corpus.py/retrieve.py
    "lateon": {
        "type": "multi_vector",
        "model": "lightonai/LateOn",
    },
    "colbertv2": {
        "type": "multi_vector",
        "model": "colbert-ir/colbertv2.0",
    },
}

MODEL_GROUPS: dict[str, list[str]] = {
    "sparse":       ["bm25"],
    "dense":        ["bge-large", "qwen3-4b", "qwen3-8b", "gemini-2", "text-embedding-3-large"],
    "science":      ["openscholar", "scincl", "specter2"],
    "multi_vector": [k for k, v in MODEL_REGISTRY.items() if v.get("type") == "multi_vector"],
    "agentic":      [k for k, v in MODEL_REGISTRY.items() if v.get("type") == "agentic"],
}

# reverse lookup: model id -> category (sparse/dense/science/multi_vector).
# Agentic runs carry their own relative path under runs/agentic/... instead
# (see runs_subpath), so they are intentionally not included here.
MODEL_CATEGORY: dict[str, str] = {
    m: category for category, models in MODEL_GROUPS.items() if category != "agentic"
    for m in models
}


def runs_subpath(model: str) -> str:
    """Relative path under runs/ for a given model id."""
    category = MODEL_CATEGORY.get(model)
    return f"{category}/{model}" if category else model
