# RAG Chat (Gemini + MySQL + Pinecone)

Lean retrieval‑augmented chat service:
- Gemini for generation + embeddings
- Hybrid retrieval (MySQL LIKE/FULLTEXT + Pinecone semantic vectors)
- Automatic background incremental indexing
- Minimal API surface for speed (`/chat`, `/health`)

## Features
- Gemini embeddings (`gemini-embedding-001`) stored in a Pinecone serverless index
- Chunked ingestion of configured table.column list
- Hybrid ranking with blended lexical + semantic scores
- Optional row enrichment (pulls full row for top results)
- In‑process schema + column cache (TTL 60s)
- Query embedding LRU cache (configurable)
- Concurrent embedding batch processing

## Requirements
Python 3.11+

## Install
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Environment
Copy `.env.example` to `.env` and fill values:
```
GEMINI_API_KEY=...
MYSQL_HOST=127.0.0.1
MYSQL_USER=root
MYSQL_PASSWORD=pass
MYSQL_DB=mydb

# Indexing configuration (pick ONE approach):
# 1. Explicit columns (comma separated table.column)
INDEX_COLUMNS=content.title,content.description
# 2. Wildcard + auto-discovery (set to * or leave blank + enable auto discover)
# INDEX_COLUMNS=*
INDEX_AUTO_DISCOVER=true

# Optional tuning
CHUNK_CHARS=1200          # Max characters per chunk
MIN_CHUNK_CHARS=40         # Skip very small fragments
INDEX_ROW_LIMIT=50000      # Per table.column cap
TEXT_TYPES=char,varchar,text,mediumtext,longtext,json  # Data types considered textual
# Include date/time fields & enable row-wise concatenation of textual columns
TEXT_TYPES=char,varchar,text,mediumtext,longtext,json,date,datetime,time,timestamp,year
COMBINE_ROW_COLUMNS=true       # Concatenate all textual cols per row (captures place + time)
FULLTEXT_REQUIRED=false    # If true uses MATCH(); else fallback LIKE scan
EMBED_DIM=                 # Optionally reduce embedding dimensionality

# Pinecone
PINECONE_API_KEY=...
PINECONE_INDEX=ragchat-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```
If `FULLTEXT_REQUIRED=true`, ensure FULLTEXT indexes exist for the columns you want lexical retrieval on, e.g.:
```sql
ALTER TABLE content ADD FULLTEXT ft_content (title, description);
```

## Indexing
Startup attempts to load existing Pinecone index metadata from `data/meta.json`.
If missing and `AUTO_INDEX=true`, it performs an initial build. A lightweight background worker then runs `incremental_update` every `AUTO_INDEX_INTERVAL` seconds, only re‑embedding changed rows and persisting when deltas exist.

Discovery logic (when `INDEX_COLUMNS=*` and `INDEX_AUTO_DISCOVER=true`):
1. Read `INFORMATION_SCHEMA.COLUMNS`
2. Keep columns whose `DATA_TYPE` ∈ `TEXT_TYPES`
3. Concatenate row textual columns if `COMBINE_ROW_COLUMNS=true`

## Chat
```powershell
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"query":"Explain the recent events"}'
```
The system performs hybrid retrieval -> trims / enriches context -> single Gemini generation.

## Safety / Constraints
- Only `SELECT` queries permitted; enforced LIMIT (default 200)
- Function calling restricted to whitelisted tools
- Hybrid retrieval returns table + pk for provenance

## Project Structure
```
ragchat/
  app.py          # FastAPI app (chat + health + background worker)
  chat.py         # Simple RAG answer construction
  config.py       # Environment / settings
  db.py           # MySQL pool + schema/PK caching
  embeddings.py   # Gemini embeddings (LRU + concurrent batching)
  vector_store.py # Pinecone abstraction + local meta
  indexer.py      # Index build + incremental update (conditional persist)
  retrieval.py    # Hybrid retrieval (LIKE/FULLTEXT + semantic) + enrichment
```

## Extending
- Add reranking (cross-encoder) before final context selection.
- Add HTML/PDF ingestion -> convert to text -> index.
- Swap Pinecone for MySQL HeatWave Vector Store or other managed vector DB when available.

## New Environment Variables
```
EMBED_QUERY_CACHE_SIZE=256   # Max cached distinct query embeddings
EMBED_CONCURRENCY=4          # Parallel embedding batch workers
AUTO_INDEX=true              # Enable background incremental indexing
AUTO_INDEX_INTERVAL=60       # Seconds between update checks
ROW_ENRICH=true              # Pull & concatenate row values for top hits
ROW_ENRICH_CHARS=600         # Truncation for enriched text
METRICS_ENABLED=false        # Enable Prometheus metrics & /metrics endpoint
PREWARM_QUERIES=foo,bar      # Comma separated queries to warm embedding cache
CB_FAIL_WINDOW=60            # Circuit breaker failure rolling window seconds
CB_MAX_FAILURES=5            # Failures in window before opening
CB_OPEN_SECONDS=30           # Time breaker stays open before half-open
CB_PERSIST=true              # Persist breaker state across restarts
CB_STATE_FILE=./cb_state.json
TRACING_ENABLED=false        # Enable OpenTelemetry tracing
OTLP_ENDPOINT=               # Collector endpoint (http(s)://host:4318)
SERVICE_NAME=ragchat         # Resource service.name
TRACE_SAMPLE_RATIO=0.1       # Sampling ratio (0-1)
```

## Health Endpoint
## Metrics & Tracing
If `METRICS_ENABLED=true`, `/metrics` exposes Prometheus format including:
- `embed_cache_hit_total`, `embed_cache_miss_total`
- `embed_batch_seconds_bucket|count|sum` (Histogram)
- `vector_search_seconds_bucket|count|sum` (Histogram)
- `chat_latency_seconds_bucket|count|sum` (Histogram)
- `index_update_seconds_bucket|count|sum` (Histogram)
- `embed_errors_total`, `vector_search_errors_total`
- `circuit_breaker_open{name="embeddings|vector_search"}` (1=open)

If `TRACING_ENABLED=true`, OpenTelemetry spans are emitted (FastAPI + outbound requests). Set `OTLP_ENDPOINT` to export to a collector. Sample ratio via `TRACE_SAMPLE_RATIO`.

Circuit breaker raises `circuit_open:<name>` runtime error while open; system will auto retry after `CB_OPEN_SECONDS`.
`GET /health` returns:
```
{
  "status": "ok",
  "indexed_vectors": 1234,
  "schema_cache_ms": 842,
  "last_index": {"added_vectors":0,...,"saved":false,"timestamp":1712345678.12}
}
```

## Notes
Production hardening suggestions:
- Add metrics (Prometheus) & structured logging
- Backoff & retry around embedding / Pinecone calls
- Row-level ACL / query authorization
- Request rate limiting

## License
Internal / example use only (add license if distributing).
