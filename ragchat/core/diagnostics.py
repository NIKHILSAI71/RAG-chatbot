from __future__ import annotations
"""Diagnostics utilities for verifying the RAG pipeline end-to-end.

Usage (interactive shell):

    from ragchat.core.diagnostics import pipeline_diagnostics
    print(pipeline_diagnostics("hello world"))
"""
from typing import Any
import numpy as np
from ragchat.core.embeddings import embed_query
from ragchat.core.retrieval import hybrid
from ragchat.infra.vector_store import VectorStore
from ragchat.core.config import settings


def pipeline_diagnostics(query: str) -> dict[str, Any]:
    out: dict[str, Any] = {"query": query}
    try:
        vec = embed_query(query)
        out["embed_dim"] = int(vec.shape[0])
        out["embed_norm"] = float(np.linalg.norm(vec))
    except Exception as e:
        out["embedding_error"] = str(e)
        return out
    try:
        results = hybrid(query, k_vec=5, k_ft=5)
        out["retrieval_count"] = len(results)
        if results:
            sample = results[0]
            out["sample_keys"] = list(sample.keys())
            out["sample_score_sem"] = sample.get("score_sem")
    except Exception as e:
        out["retrieval_error"] = str(e)
    try:
        vs = VectorStore.load()
        if vs:
            out["vector_health"] = vs.health_check()
            out["stored_meta"] = len(vs.meta)
    except Exception as e:
        out["vector_error"] = str(e)
    return out
