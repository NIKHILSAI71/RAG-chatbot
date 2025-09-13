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
        # Lightweight flag to avoid repeated expensive stats calls
        self._checked_consistency = False

    @staticmethod
    def _ensure_dir():
        os.makedirs("data", exist_ok=True)

    def add(self, vectors: list[np.ndarray], metas: list[dict[str, Any]]):
        if not vectors:
            return
        arr = np.vstack(vectors).astype("float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms
        def _sanitize(val):
            from datetime import datetime, date, time as dtime, timedelta
            if isinstance(val, (datetime, date, dtime)):
                return val.isoformat()
            if isinstance(val, timedelta):
                total_seconds = int(val.total_seconds())
                h = total_seconds // 3600
                m = (total_seconds % 3600) // 60
                s = total_seconds % 60
                return f"{h:02d}:{m:02d}:{s:02d}"
            if isinstance(val, dict):
                return {k: _sanitize(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_sanitize(v) for v in val]
            return val
        upserts = []
        for vec, meta in zip(arr, metas):
            base_id = f"{meta.get('table','unknown')}::{meta.get('pk','')}::{len(self.meta)}"
            vid = meta.get("id") or base_id
            sanitized_meta = {k: _sanitize(v) for k, v in meta.items() if k not in {"id"}}
            # Flatten & coerce metadata recursively so every final value is one of:
            # str | int | float | bool | list[str]
            from collections.abc import Mapping

            def _flatten(prefix: str, value: Any, out: dict[str, Any]):
                # Expand mappings (dict / pydantic BaseModel .dict())
                if isinstance(value, Mapping):
                    for sk, sv in value.items():
                        new_key = f"{prefix}_{sk}" if prefix else str(sk)
                        _flatten(new_key, sv, out)
                elif isinstance(value, list):
                    # Coerce list elements to strings (Pinecone only supports list[str])
                    out[prefix] = [e if isinstance(e, str) else str(e) for e in value]
                else:
                    # Primitive or unsupported -> coerce to primitive
                    if isinstance(value, (str, int, float, bool)):
                        out[prefix] = value
                    else:
                        out[prefix] = str(value)

            flattened: dict[str, Any] = {}
            for k, v in sanitized_meta.items():
                _flatten(k, v, flattened)

            # Final safety pass: ensure constraints
            safe_meta: dict[str, Any] = {}
            for k, v in flattened.items():
                if isinstance(v, list):
                    safe_meta[k] = [e if isinstance(e, str) else str(e) for e in v]
                elif isinstance(v, (str, int, float, bool)):
                    safe_meta[k] = v
                else:
                    safe_meta[k] = str(v)
            flattened = safe_meta
            upserts.append({"id": vid, "values": vec.tolist(), "metadata": flattened})
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
        def _sanitize(val):
            from datetime import datetime, date, time as dtime, timedelta
            if isinstance(val, (datetime, date, dtime)):
                # ISO 8601 string representation
                return val.isoformat()
            if isinstance(val, timedelta):
                # Represent as HH:MM:SS (floor to seconds)
                total_seconds = int(val.total_seconds())
                h = total_seconds // 3600
                m = (total_seconds % 3600) // 60
                s = total_seconds % 60
                return f"{h:02d}:{m:02d}:{s:02d}"
            if isinstance(val, dict):
                return {k: _sanitize(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_sanitize(v) for v in val]
            return val
        cleaned = [_sanitize(m) for m in self.meta]
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(cleaned, f)

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
        # Consistency check: if we have metadata but the remote index is empty (e.g. index wiped)
        # clear meta so that the next incremental_update / build triggers full re-embed.
        try:
            if vs.meta:
                stats = vs.index.describe_index_stats()
                # Pinecone serverless returns dict with total vector count nested; support common shapes.
                total = None
                if isinstance(stats, dict):
                    # Newer SDK: {'namespaces': {...}, 'dimension': 1536, 'index_fullness': 0.0, 'total_vector_count': 0}
                    total = stats.get("total_vector_count")
                    if total is None and "namespaces" in stats:
                        # Sum namespace counts if present
                        ns = stats.get("namespaces") or {}
                        total = sum((ns.get(k, {}).get("vector_count", 0) for k in ns))
                if total == 0:
                    print("[vector_store] Detected metadata without vectors in Pinecone; scheduling full re-embed.")
                    vs.meta = []
        except Exception as e:  # pragma: no cover - defensive
            print(f"[vector_store] Consistency check failed: {e}")
        return vs
