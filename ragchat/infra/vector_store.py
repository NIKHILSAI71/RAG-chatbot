from __future__ import annotations
import json
import os
import time
import logging
from typing import Any, Iterable
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from ragchat.core.config import settings
from ragchat.observability.metrics import (
    init_metrics,
    VECTOR_SEARCH_SECONDS,
    VECTOR_SEARCH_ERRORS,
    with_circuit_breaker,
)

META_PATH = "data/meta.json"

logger = logging.getLogger(__name__)


class VectorStore:
    """Thin wrapper around Pinecone providing add/search/delete & local metadata cache.

    Notes
    -----
    - Metadata is stored locally in `data/meta.json` for quick incremental rebuild logic.
    - Pinecone serverless index creation is asynchronous; we poll for readiness on init when we
      create a fresh index (bounded wait) to reduce early query failures.
    - Duplicate sanitize logic consolidated into shared helpers.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._ensure_dir()
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY not set for Pinecone vector store.")
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        existing = {i.name: i for i in self.pc.list_indexes()}
        created_new = False
        if settings.pinecone_index not in existing:
            logger.info("[vector_store] Creating Pinecone index '%s' (dim=%d)", settings.pinecone_index, dim)
            self.pc.create_index(
                name=settings.pinecone_index,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
            created_new = True
        self.index = self.pc.Index(settings.pinecone_index)
        if created_new:
            self._wait_for_index_ready(timeout=90)
        self.meta: list[dict[str, Any]] = []
        self._checked_consistency = False  # reserved for future use

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _ensure_dir():
        os.makedirs("data", exist_ok=True)

    def add(self, vectors: list[np.ndarray], metas: list[dict[str, Any]]):
        """Add vectors with associated metadata.

        Parameters
        ----------
        vectors : list[np.ndarray]
            Embedding vectors; each must have dimension == self.dim
        metas : list[dict[str, Any]]
            Metadata objects aligned with vectors.
        """
        if not vectors:
            return
        if len(vectors) != len(metas):
            raise ValueError("vectors and metas must be same length")
        arr = np.vstack(vectors).astype("float32")
        if arr.shape[1] != self.dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.dim} got {arr.shape[1]}")
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms
        upserts = []
        for vec, meta in zip(arr, metas):
            base_id = f"{meta.get('table','unknown')}::{meta.get('pk','')}::{len(self.meta)}"
            vid = meta.get("id") or base_id
            sanitized_meta = {k: self._sanitize_value(v) for k, v in meta.items() if k not in {"id"}}
            flattened = self._flatten_metadata(sanitized_meta)
            upserts.append({"id": vid, "values": vec.tolist(), "metadata": flattened})
            self.meta.append(meta)
        for i in range(0, len(upserts), 100):
            batch = upserts[i:i+100]
            try:
                self.index.upsert(batch)
            except Exception as e:
                logger.error("[vector_store] Upsert batch failed (%d items): %s", len(batch), e)
                raise

    def delete_ids(self, ids: Iterable[str]):
        ids_list = list(ids)
        if not ids_list:
            return
        for i in range(0, len(ids_list), 1000):
            self.index.delete(ids=ids_list[i:i+1000])

    def delete_row(self, table: str, pk: Any):
        try:
            self.index.delete(filter={"table": table, "pk": pk})
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("[vector_store] delete_row failed table=%s pk=%s: %s", table, pk, e)
            return

    def search(self, query_vec: np.ndarray, k: int = 20):
        """Semantic vector search.

        Returns list of metadata dicts enriched with score under 'score_sem'.
        """
        if query_vec is None:
            return []
        if query_vec.shape[0] != self.dim:
            raise ValueError(f"Query vector dim mismatch: expected {self.dim} got {query_vec.shape[0]}")
        q = query_vec.astype("float32")
        q /= (np.linalg.norm(q) + 1e-12)

        def call():
            return self.index.query(vector=q.tolist(), top_k=k, include_metadata=True)

        start = time.time()
        try:
            res = with_circuit_breaker("vector_search", call) if settings.metrics_enabled else call()
        except Exception as e:
            if settings.metrics_enabled:
                init_metrics(); VECTOR_SEARCH_ERRORS.inc()
            logger.error("[vector_store] Search error: %s", e)
            raise
        dur = time.time() - start
        if settings.metrics_enabled:
            init_metrics(); VECTOR_SEARCH_SECONDS.observe(dur)
        out = []
        matches = getattr(res, 'matches', []) or []
        for match in matches:
            meta = dict(getattr(match, 'metadata', {}) or {})
            meta["score_sem"] = float(getattr(match, 'score', 0.0) or 0.0)
            meta["id"] = getattr(match, 'id', None)
            out.append(meta)
        return out

    def save(self):
        cleaned = [self._sanitize_value(m) for m in self.meta]
        try:
            with open(META_PATH, "w", encoding="utf-8") as f:
                json.dump(cleaned, f)
        except Exception as e:  # pragma: no cover - defensive
            logger.error("[vector_store] Failed to save metadata: %s", e)

    @classmethod
    def load(cls) -> "VectorStore | None":
        if not settings.pinecone_api_key:
            return None
        meta: list[Any] = []
        if os.path.exists(META_PATH):
            try:
                with open(META_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("[vector_store] Failed to load metadata file: %s", e)
                meta = []
        dim = settings.embed_dim or (
            len(meta[0].get("vector", []))
            if meta and isinstance(meta[0], dict) and meta[0].get("vector")
            else None
        )
        if dim is None:
            return None
        vs = cls(dim)
        now_ts = int(time.time())
        cleaned: list[dict[str, Any]] = []
        for m in meta:
            if not isinstance(m, dict):
                continue
            if "indexed_at" not in m:
                m["indexed_at"] = now_ts
            cleaned.append(m)
        vs.meta = cleaned
        try:
            if vs.meta:
                stats = vs.index.describe_index_stats()
                total = None
                if isinstance(stats, dict):
                    total = stats.get("total_vector_count")
                    if total is None and "namespaces" in stats:
                        ns = stats.get("namespaces") or {}
                        total = sum((ns.get(k, {}).get("vector_count", 0) for k in ns))
                if total == 0:
                    logger.info("[vector_store] Metadata present but index empty; clearing local cache for re-embed.")
                    vs.meta = []
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("[vector_store] Consistency check failed: %s", e)
        return vs

    # ------------------------------------------------------------------
    # New helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_value(val: Any) -> Any:
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
            return {k: VectorStore._sanitize_value(v) for k, v in val.items()}
        if isinstance(val, list):
            return [VectorStore._sanitize_value(v) for v in val]
        return val

    @staticmethod
    def _flatten_metadata(meta: dict[str, Any]) -> dict[str, Any]:
        from collections.abc import Mapping
        out: dict[str, Any] = {}

        def _flatten(prefix: str, value: Any):
            if isinstance(value, Mapping):
                for sk, sv in value.items():
                    new_key = f"{prefix}_{sk}" if prefix else str(sk)
                    _flatten(new_key, sv)
            elif isinstance(value, list):
                out[prefix] = [e if isinstance(e, str) else str(e) for e in value]
            else:
                if isinstance(value, (str, int, float, bool)):
                    out[prefix] = value
                else:
                    out[prefix] = str(value)

        for k, v in meta.items():
            _flatten(k, v)
        return out

    def _wait_for_index_ready(self, timeout: int = 90, interval: float = 2.0):
        """Poll Pinecone until index is ready or timeout."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                desc = self.pc.describe_index(name=settings.pinecone_index)
                status = desc.get("status", {}) if isinstance(desc, dict) else {}
                if status.get("ready") is True:
                    logger.info("[vector_store] Index '%s' is ready", settings.pinecone_index)
                    return
            except Exception as e:  # pragma: no cover - network variability
                logger.debug("[vector_store] Waiting for index readiness: %s", e)
            time.sleep(interval)
        logger.warning("[vector_store] Timed out waiting for index readiness.")

    def health_check(self) -> bool:
        """Lightweight health check for vector store connectivity."""
        try:
            self.index.describe_index_stats()
            return True
        except Exception as e:  # pragma: no cover - defensive
            logger.error("[vector_store] Health check failed: %s", e)
            return False
