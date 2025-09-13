from __future__ import annotations
import time
from typing import Callable, Any
try:
    from prometheus_client import Counter, Gauge, Histogram  # type: ignore
except Exception:  # pragma: no cover
    class _Noop:
        def __init__(self, *a, **k):
            pass
        def labels(self, *a, **k):
            return self
        def set(self, *a, **k):
            return self
        def inc(self, *a, **k):
            return self
        def observe(self, *a, **k):
            return self
    Counter = Gauge = Histogram = _Noop  # type: ignore
import json
import os
from ragchat.core.config import settings

__all__ = [
    "init_metrics",
    "EMBED_CACHE_HIT", "EMBED_CACHE_MISS", "EMBED_BATCH_SECONDS", "EMBED_ERRORS",
    "VECTOR_SEARCH_SECONDS", "VECTOR_SEARCH_ERRORS", "CB_OPEN", "with_circuit_breaker",
    "CHAT_LATENCY_SECONDS", "INDEX_UPDATE_SECONDS"
]

_metrics_inited = False

EMBED_CACHE_HIT: Counter
EMBED_CACHE_MISS: Counter
EMBED_BATCH_SECONDS: Histogram
EMBED_ERRORS: Counter
VECTOR_SEARCH_SECONDS: Histogram
VECTOR_SEARCH_ERRORS: Counter
CB_OPEN: Gauge
CHAT_LATENCY_SECONDS: Histogram
INDEX_UPDATE_SECONDS: Histogram

# Provide placeholder assignments so symbols exist for from-import callers
EMBED_CACHE_HIT = None  # type: ignore
EMBED_CACHE_MISS = None  # type: ignore
EMBED_BATCH_SECONDS = None  # type: ignore
EMBED_ERRORS = None  # type: ignore
VECTOR_SEARCH_SECONDS = None  # type: ignore
VECTOR_SEARCH_ERRORS = None  # type: ignore
CB_OPEN = None  # type: ignore
CHAT_LATENCY_SECONDS = None  # type: ignore
INDEX_UPDATE_SECONDS = None  # type: ignore

# Circuit breaker state (shared for embeddings and search but keyed by name)
_cb_state: dict[str, dict[str, Any]] = {}


def init_metrics():
    global _metrics_inited, EMBED_CACHE_HIT, EMBED_CACHE_MISS, EMBED_BATCH_SECONDS, EMBED_ERRORS
    global VECTOR_SEARCH_SECONDS, VECTOR_SEARCH_ERRORS, CB_OPEN, CHAT_LATENCY_SECONDS, INDEX_UPDATE_SECONDS
    if _metrics_inited or not settings.metrics_enabled:
        return
    EMBED_CACHE_HIT = Counter("embed_cache_hit_total", "Embedding query cache hits")
    EMBED_CACHE_MISS = Counter("embed_cache_miss_total", "Embedding query cache misses")
    EMBED_BATCH_SECONDS = Histogram(
        "embed_batch_seconds", "Embedding batch latency seconds",
        buckets=_default_buckets()
    )
    EMBED_ERRORS = Counter("embed_errors_total", "Embedding errors")
    VECTOR_SEARCH_SECONDS = Histogram(
        "vector_search_seconds", "Vector search latency seconds",
        buckets=_default_buckets()
    )
    VECTOR_SEARCH_ERRORS = Counter("vector_search_errors_total", "Vector search errors")
    CB_OPEN = Gauge("circuit_breaker_open", "Circuit breaker open (1=open)", ["name"])
    CHAT_LATENCY_SECONDS = Histogram(
        "chat_latency_seconds", "End-to-end /chat request latency seconds",
        buckets=_default_buckets()
    )
    INDEX_UPDATE_SECONDS = Histogram(
        "index_update_seconds", "Incremental index/update job duration seconds",
        buckets=_job_buckets()
    )
    _metrics_inited = True


def _default_buckets():
    return (
        0.01, 0.025, 0.05, 0.075, 0.1,
        0.15, 0.25, 0.35, 0.5,
        0.75, 1.0, 1.5, 2.0,
        3.0, 5.0, 8.0, 13.0, 21.0, 34.0
    )


def _job_buckets():
    return (0.5, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144)


def with_circuit_breaker(name: str, fn: Callable[[], Any]):
    if not settings.metrics_enabled:
        return fn()
    st = _load_cb_state(name)
    now = time.time()
    if st["opened_at"] and (now - st["opened_at"]) < settings.cb_open_seconds:
        CB_OPEN.labels(name=name).set(1)
        raise RuntimeError(f"circuit_open:{name}")
    if (now - st["window_start"]) > settings.cb_fail_window:
        st["failures"] = 0
        st["window_start"] = now
    try:
        result = fn()
        st["failures"] = 0
        st["opened_at"] = 0.0
        CB_OPEN.labels(name=name).set(0)
        _persist_cb_state(name, st)
        return result
    except Exception:
        st["failures"] += 1
        if st["failures"] >= settings.cb_max_failures:
            st["opened_at"] = now
            CB_OPEN.labels(name=name).set(1)
        _persist_cb_state(name, st)
        raise


def _persist_cb_state(name: str, st: dict[str, Any]):
    if not settings.cb_persist:
        return
    try:
        path = settings.cb_state_file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        all_state = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    all_state = json.load(f)
                except Exception:
                    all_state = {}
        all_state[name] = st
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(all_state, f)
        os.replace(tmp, path)
    except Exception:
        pass


def _load_cb_state(name: str):
    st = _cb_state.get(name)
    if st is not None:
        return st
    st = {"failures": 0, "opened_at": 0.0, "window_start": time.time()}
    if settings.cb_persist and os.path.exists(settings.cb_state_file):
        try:
            with open(settings.cb_state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if name in data:
                saved = data[name]
                if isinstance(saved, dict) and {"failures", "opened_at", "window_start"}.issubset(saved.keys()):
                    st.update(saved)
        except Exception:
            pass
    _cb_state[name] = st
    return st
