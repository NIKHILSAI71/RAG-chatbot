from __future__ import annotations
import json
import os
from typing import Any, Iterable
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from ragchat.core.config import settings
from ragchat.observability.metrics import init_metrics, VECTOR_SEARCH_SECONDS, VECTOR_SEARCH_ERRORS, with_circuit_breaker

META_PATH = "data/meta.json"

class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._ensure_dir()
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY not set for Pinecone vector store.")
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        existing = {i.name: i for i in self.pc.list_indexes()}
        if settings.pinecone_index not in existing:
            self.pc.create_index(
                name=settings.pinecone_index,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
        self.index = self.pc.Index(settings.pinecone_index)
        self.meta: list[dict[str, Any]] = []

    @staticmethod
    def _ensure_dir():
        os.makedirs("data", exist_ok=True)

    def add(self, vectors: list[np.ndarray], metas: list[dict[str, Any]]):
        if not vectors:
            return
        arr = np.vstack(vectors).astype("float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms
        upserts = []
        for vec, meta in zip(arr, metas):
            base_id = f"{meta.get('table','unknown')}::{meta.get('pk','')}::{len(self.meta)}"
            vid = meta.get("id") or base_id
            upserts.append({
                "id": vid,
                "values": vec.tolist(),
                "metadata": {k: v for k, v in meta.items() if k not in {"id"}},
            })
            self.meta.append(meta)
        for i in range(0, len(upserts), 100):
            self.index.upsert(upserts[i:i+100])

    def delete_ids(self, ids: Iterable[str]):
        ids_list = list(ids)
        if not ids_list:
            return
        for i in range(0, len(ids_list), 1000):
            self.index.delete(ids=ids_list[i:i+1000])

    def delete_row(self, table: str, pk: Any):
        try:
            self.index.delete(filter={"table": table, "pk": pk})
        except Exception:
            return

    def search(self, query_vec: np.ndarray, k: int = 20):
        if query_vec is None:
            return []
        q = query_vec.astype("float32")
        q /= (np.linalg.norm(q) + 1e-12)
        import time
        def call():
            return self.index.query(vector=q.tolist(), top_k=k, include_metadata=True)
        start = time.time()
        try:
            res = with_circuit_breaker("vector_search", call) if settings.metrics_enabled else call()
        except Exception as e:
            if settings.metrics_enabled:
                init_metrics(); VECTOR_SEARCH_ERRORS.inc()
            raise
        dur = time.time() - start
        if settings.metrics_enabled:
            init_metrics(); VECTOR_SEARCH_SECONDS.observe(dur)
        out = []
        for match in getattr(res, 'matches', []) or []:
            meta = dict(match.metadata or {})
            meta["score_sem"] = float(match.score or 0.0)
            out.append(meta)
        return out

    def save(self):
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f)

    @classmethod
    def load(cls) -> "VectorStore | None":
        if not settings.pinecone_api_key:
            return None
        meta = []
        if os.path.exists(META_PATH):
            try:
                with open(META_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = []
        dim = settings.embed_dim or (len(meta[0].get("vector", [])) if meta and isinstance(meta[0], dict) and meta[0].get("vector") else None)
        if dim is None:
            return None
        vs = cls(dim)
        import time
        now_ts = int(time.time())
        cleaned = []
        for m in meta:
            if not isinstance(m, dict):
                continue
            if "indexed_at" not in m:
                m["indexed_at"] = now_ts
            cleaned.append(m)
        vs.meta = cleaned
        return vs
