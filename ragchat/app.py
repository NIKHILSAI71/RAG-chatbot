from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from threading import Event, Thread
import time
import logging
import sys
from ragchat.core.db import init_pool, fetch_schema, schema_cache_age_seconds
from ragchat.infra.vector_store import VectorStore
from ragchat.core.indexer import build_index, incremental_update
from ragchat.core.retrieval import set_vector_store
from ragchat.core.retrieval import _get_text_columns  # type: ignore
from ragchat.core.chat import chat_once
from ragchat.core.config import settings
from ragchat.observability.metrics import init_metrics, CHAT_LATENCY_SECONDS, INDEX_UPDATE_SECONDS
from ragchat.observability.tracing import init_tracing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ragchat.log', mode='a')
    ]
)

# Set specific loggers
logger = logging.getLogger(__name__)
logging.getLogger("ragchat.core.chat").setLevel(logging.INFO)
logging.getLogger("ragchat.core.retrieval").setLevel(logging.INFO)
logging.getLogger("ragchat.infra.vector_store").setLevel(logging.INFO)

# Reduce noise from other libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
try:
    # Import Prometheus client only if metrics are enabled to avoid hard dependency
    if settings.metrics_enabled:  # type: ignore
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST  # type: ignore
    else:
        generate_latest = None  # type: ignore
        CONTENT_TYPE_LATEST = None  # type: ignore
except Exception:  # pragma: no cover
    # If not installed, degrade gracefully; metrics endpoint will be skipped.
    generate_latest = None  # type: ignore
    CONTENT_TYPE_LATEST = None  # type: ignore
    if settings.metrics_enabled:
        logger.warning("prometheus_client not installed. Install 'prometheus-client' to enable metrics.")

app = FastAPI(title="RAG - Chatbot")
_vs: VectorStore | None = None
_index_worker_stop: Event | None = None
_index_worker_thread: Thread | None = None
_last_index_stats: dict | None = None


class ChatRequest(BaseModel):
    query: str


## Removed other request models; only ChatRequest retained.


class IndexWorker:
    """Background worker that periodically runs incremental updates against the VectorStore."""
    def __init__(self, vs: VectorStore, interval: int = 60):
        self.vs = vs
        self.interval = interval
        self._stop = Event()

    def run(self):
        while not self._stop.is_set():
            try:
                start = time.time()
                stats = incremental_update(self.vs)
                dur = time.time() - start
                if settings.metrics_enabled:
                    INDEX_UPDATE_SECONDS.observe(dur)
                global _last_index_stats
                _last_index_stats = stats | {"timestamp": time.time()}
                if stats.get("added_vectors", 0) or stats.get("reembedded_rows", 0):
                    # ensure retrieval layer sees latest metadata
                    set_vector_store(self.vs)
                # Log incremental update results
                logger.info(f"[index_worker] incremental_update: {stats}")
            except Exception as e:
                logger.error(f"[index_worker] error during incremental update: {e}")
            # wait with early exit
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


@app.on_event("startup")
async def startup_event():
    global _vs, _index_worker_stop, _index_worker_thread
    logger.info("Starting RAG chatbot application...")
    init_pool()
    existing = VectorStore.load()
    if existing:
        _vs = existing
        set_vector_store(_vs)
        logger.info("Loaded existing vector store with %d vectors", len(_vs.meta))
    else:
        # Build index on first run if enabled
        if settings.auto_index and settings.auto_index_initial_build:
            try:
                logger.info("Building initial index...")
                _vs = build_index(None)
                set_vector_store(_vs)
                logger.info("Initial index built successfully")
            except Exception as e:
                logger.error(f"[startup] Initial build failed: {e}")
    # Start background worker if enabled and vector store is available
    if settings.auto_index and _vs is not None:
        worker = IndexWorker(_vs, interval=settings.auto_index_interval)
        _index_worker_thread = Thread(target=worker.run, daemon=True)
        _index_worker_stop = worker
        _index_worker_thread.start()
        logger.info("Started background index worker")
    # Pre-warm schema & text column caches to reduce first-request latency
    try:
        # Tracing init
        init_tracing(app)
        fetch_schema()
        _get_text_columns()  # type: ignore
        if settings.metrics_enabled:
            init_metrics()
        # Prewarm embedding cache with configured queries (best-effort)
        if settings.prewarm_queries:
            from ragchat.core.embeddings import embed_query
            for q in settings.prewarm_queries:
                try:
                    embed_query(q)
                except Exception as e:
                    logger.warning(f"[startup] Prewarm failed for '{q}': {e}")
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"[startup] Cache pre-warm failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    global _index_worker_stop, _index_worker_thread
    logger.info("Shutting down RAG chatbot application...")
    if _index_worker_stop:
        try:
            _index_worker_stop.stop()
            logger.info("Stopped background index worker")
        except Exception as e:
            logger.error(f"Error stopping index worker: {e}")
    if _index_worker_thread:
        # give thread a moment to exit
        _index_worker_thread.join(timeout=2)
        logger.info("Index worker thread shutdown completed")
    logger.info("Application shutdown completed")


@app.get("/health")
async def health():
    from math import floor
    cache_age = schema_cache_age_seconds()
    age_ms = floor(cache_age * 1000) if cache_age is not None else None
    return {
        "status": "ok",
        "indexed_vectors": len(_vs.meta) if _vs else 0,
        "schema_cache_ms": age_ms,
        "last_index": _last_index_stats or {},
    }

# Manual reindex endpoint removed: indexing is automatic via background worker.

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.query.strip():
        logger.warning("Received empty chat query")
        raise HTTPException(400, "Query required")
    
    logger.info(f"Processing chat query: {req.query[:100]}...")  # Log first 100 chars
    start = time.time()
    answer = chat_once(req.query)
    duration = time.time() - start
    
    if settings.metrics_enabled:
        CHAT_LATENCY_SECONDS.observe(duration)
    
    logger.info(f"Chat query completed in {duration:.2f}s")
    return {"answer": answer}

if settings.metrics_enabled and generate_latest:  # type: ignore
    from fastapi.responses import Response

    @app.get("/metrics")
    async def metrics():  # type: ignore
        data = generate_latest()  # type: ignore
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)  # type: ignore


# To run: uvicorn ragchat.app:app --reload
