from __future__ import annotations
from typing import List, Deque, Tuple
import time
from collections import deque
import threading
import numpy as np
from google import genai
from google.genai import types
from ragchat.core.config import settings
from ragchat.observability.metrics import init_metrics, EMBED_CACHE_HIT, EMBED_CACHE_MISS, EMBED_BATCH_SECONDS, EMBED_ERRORS, with_circuit_breaker

EMBED_MODEL = "gemini-embedding-001"
_client: genai.Client | None = None
_q_cache: dict[str, np.ndarray] = {}
_q_order: Deque[str] = deque()
_q_lock = threading.Lock()

def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client

def batch(iterable: List[str], size: int):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

def embed_texts(texts: list[str]) -> list[np.ndarray]:
    if not texts:
        return []
    client = get_client()
    vectors: list[np.ndarray] = [None] * len(texts)  # type: ignore
    indexed = list(enumerate(texts))
    batches = [indexed[i:i+64] for i in range(0, len(indexed), 64)]
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process(batch_items: List[Tuple[int, str]]):
        contents = [
            types.Content(parts=[types.Part.from_text(text=t)]) if hasattr(types.Part, 'from_text')
            else types.Content(parts=[types.Part(text=t)])
            for _, t in batch_items
        ]
        start = time.time()
        def call():
            return client.models.embed_content(
                model=EMBED_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(
                    output_dimensionality=settings.embed_dim if settings.embed_dim else None
                )
            )
        try:
            resp = with_circuit_breaker("embeddings", call) if settings.metrics_enabled else call()
        except Exception as e:
            if settings.metrics_enabled:
                EMBED_ERRORS.inc()
            raise
        duration = time.time() - start
        if settings.metrics_enabled:
            EMBED_BATCH_SECONDS.observe(duration)
        out_vecs: list[np.ndarray] = []
        for emb in getattr(resp, 'embeddings', []):
            vals = getattr(emb, 'values', emb)
            out_vecs.append(np.array(vals, dtype="float32"))
        if len(out_vecs) != len(batch_items):
            raise RuntimeError("Embedding response size mismatch")
        return batch_items, out_vecs

    max_workers = max(1, min(settings.embed_concurrency, len(batches)))
    if max_workers > 1 and len(batches) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(process, b) for b in batches]
            for fut in as_completed(futures):
                batch_items, out_vecs = fut.result()
                for (idx, _), vec in zip(batch_items, out_vecs):
                    vectors[idx] = vec
    else:
        for b in batches:
            batch_items, out_vecs = process(b)
            for (idx, _), vec in zip(batch_items, out_vecs):
                vectors[idx] = vec
    return vectors  # type: ignore

def embed_query(q: str) -> np.ndarray:
    key = q.strip()
    if not key:
        return np.zeros(settings.embed_dim or 0, dtype="float32")
    with _q_lock:
        if key in _q_cache:
            if settings.metrics_enabled:
                init_metrics(); EMBED_CACHE_HIT.inc()
            return _q_cache[key]
    if settings.metrics_enabled:
        init_metrics(); EMBED_CACHE_MISS.inc()
    vec = embed_texts([key])[0]
    with _q_lock:
        _q_cache[key] = vec
        _q_order.append(key)
        limit = max(8, settings.embed_query_cache_size)
        while len(_q_order) > limit:
            old = _q_order.popleft()
            _q_cache.pop(old, None)
    return vec
