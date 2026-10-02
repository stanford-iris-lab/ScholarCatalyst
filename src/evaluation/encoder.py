from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


class _BM25sWrapper:
    """Picklable wrapper around bm25s retriever with rank_bm25-compatible get_scores()."""

    def __init__(self, retriever: Any) -> None:
        self._r = retriever

    def get_scores(self, tokens: list[str]) -> np.ndarray:
        scores = self._r.get_scores(tokens)
        return scores[0] if scores.ndim == 2 else scores


class BM25Encoder:
    def __init__(self) -> None:
        self.bm25 = None
        self.doc_ids: list[str] = []

    def index(self, texts: list[str], doc_ids: list[str]) -> None:
        import bm25s
        tokenized = bm25s.tokenize(texts, stopwords=None, stemmer=None, lower=True)
        retriever = bm25s.BM25()
        retriever.index(tokenized)
        self.bm25 = _BM25sWrapper(retriever)
        self.doc_ids = doc_ids

    def encode_corpus(self, texts: list[str], doc_ids: list[str]) -> None:
        self.index(texts, doc_ids)

    def score(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(query.lower().split())

    def encode_queries(self, texts: list[str]) -> list[np.ndarray]:
        return [self.score(t) for t in texts]


class DenseEncoder:
    """Covers ST, SPECTER2 adapter, and ReasonIR."""

    def __init__(self, cfg: dict) -> None:
        self.cfg    = cfg
        self.loader = cfg["loader"]
        self.model  = self.load(cfg)
        self.query_instruction = cfg.get("query_instruction", "")

    def load(self, cfg: dict) -> Any:
        path = str(cfg["path"])
        if cfg["loader"] == "st":
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(path)
            if "max_seq_length" in cfg:
                model.max_seq_length = cfg["max_seq_length"]
            return model

        if cfg["loader"] == "adapter":
            # adapters is incompatible with this repo's transformers 5.x /
            # huggingface_hub 1.x; run specter2 from the dedicated "specter2"
            # conda env, which pins the versions adapters supports.
            import torch
            from adapters import AutoAdapterModel
            from transformers import AutoTokenizer
            model     = AutoAdapterModel.from_pretrained(path)
            tokenizer = AutoTokenizer.from_pretrained(path)
            model.load_adapter("allenai/specter2", source="hf", set_active=True)
            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()
            self._tokenizer = tokenizer
            return model

        if cfg["loader"] == "reasonir":
            from transformers import AutoModel
            model = AutoModel.from_pretrained(path, torch_dtype="auto", trust_remote_code=True)
            model = model.cuda()
            model.eval()
            return model

        if cfg["loader"] == "contriever":
            import torch
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModel.from_pretrained(path)
            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()
            self._tokenizer = tokenizer
            return model

        raise ValueError(f"unknown loader: {cfg['loader']}")

    def encode_corpus(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        bs = self.cfg.get("batch_size", batch_size)
        if self.loader == "st":
            return self.model.encode(texts, batch_size=bs, show_progress_bar=True, normalize_embeddings=True)
        if self.loader == "adapter":
            return self.encode_adapter(texts, bs)
        if self.loader == "reasonir":
            return np.array([self.model.encode(t, instruction="") for t in texts])
        if self.loader == "contriever":
            return self.encode_contriever(texts, bs)
        raise ValueError(self.loader)

    def encode_queries(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        bs = self.cfg.get("batch_size", batch_size)
        if self.loader == "st":
            prefixed = [self.query_instruction + t for t in texts]
            return self.model.encode(prefixed, batch_size=bs, show_progress_bar=True, normalize_embeddings=True)
        if self.loader == "adapter":
            return self.encode_adapter(texts, batch_size)
        if self.loader == "reasonir":
            instr = self.query_instruction
            return np.array([self.model.encode(t, instruction=instr) for t in texts])
        if self.loader == "contriever":
            return self.encode_contriever(texts, bs)
        raise ValueError(self.loader)

    def encode_contriever(self, texts: list[str], batch_size: int) -> np.ndarray:
        import torch
        device = next(self.model.parameters()).device
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self._tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
            # mean pooling over non-padding tokens
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0)

    def encode_adapter(self, texts: list[str], batch_size: int) -> np.ndarray:
        import torch
        device = next(self.model.parameters()).device
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc   = self._tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            enc   = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
            emb = out.last_hidden_state[:, 0, :]
            emb = torch.nn.functional.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0)


class APIEncoder:
    """OpenAI-compatible embedding API."""

    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        import openai
        self.model  = model
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.workers = 8  # concurrent embedding requests; encode_corpus.py --workers sets it

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """One request with the retry rules: null data -> wait 10 s, 429 -> wait 5 s, 400 -> raise."""
        while True:
            try:
                resp = self.client.embeddings.create(model=self.model, input=batch, encoding_format="float")
                if resp.data is None:
                    err  = getattr(resp, "error", None) or {}
                    code = err.get("code", 0) if isinstance(err, dict) else 0
                    msg  = err.get("message", str(resp)) if isinstance(err, dict) else str(resp)
                    if code == 400:
                        raise ValueError(f"API 400 (invalid input): {msg}")
                    print(f"\n[warn] API returned null data, retrying in 10s… resp={resp}")
                    time.sleep(10)
                    continue
                return [d.embedding for d in resp.data]
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep(5)
                else:
                    raise

    def batch_encode(self, texts: list[str], batch_size: int = 100, workers: int = 8) -> np.ndarray:
        """Embed texts in batches with `workers` requests in flight; order is preserved."""
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        all_embs: list[list[float]] = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for embs in tqdm(ex.map(self._embed_batch, batches), total=len(batches),
                             desc=f"embed/{self.model.split('/')[-1]}", unit="batch"):
                all_embs.extend(embs)
        embs = np.array(all_embs, dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.maximum(norms, 1e-9)

    def encode_corpus(self, texts: list[str]) -> np.ndarray:
        return self.batch_encode(texts, workers=self.workers)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.batch_encode(texts)


OPENAI_BASE = "https://api.openai.com/v1"


def get_encoder(model_id: str):
    from config import MODEL_REGISTRY
    cfg = MODEL_REGISTRY[model_id]
    t   = cfg["type"]
    if t == "bm25":
        return BM25Encoder()
    if t == "dense":
        return DenseEncoder(cfg)
    if t == "api":
        model = cfg["model"]
        if model.startswith("google/"):
            return APIEncoder(model.removeprefix("google/"), os.environ["GEMINI_API_KEY"], GEMINI_OPENAI_BASE)
        if model.startswith("openai/"):
            return APIEncoder(model.removeprefix("openai/"), os.environ["OPENAI_API_KEY"], OPENAI_BASE)
        raise ValueError(f"unknown api model: {model!r}")
    raise ValueError(f"unknown encoder type: {t}")
