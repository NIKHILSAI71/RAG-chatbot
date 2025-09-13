import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(override=False)

@dataclass
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    mysql_host: str = os.getenv("MYSQL_HOST", "127.0.0.1")
    mysql_user: str = os.getenv("MYSQL_USER", "root")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")
    mysql_db: str = os.getenv("MYSQL_DB", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "ragchat-index")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    embed_dim: int | None = int(os.getenv("EMBED_DIM")) if os.getenv("EMBED_DIM") else None
    index_row_limit: int = int(os.getenv("INDEX_ROW_LIMIT", "50000"))
    index_columns: list[str] = field(default_factory=lambda: [c for c in os.getenv(
        "INDEX_COLUMNS", "*"
    ).split(',') if c])
    index_auto_discover: bool = os.getenv("INDEX_AUTO_DISCOVER", "true").lower() in {"1","true","yes"}
    min_chunk_chars: int = int(os.getenv("MIN_CHUNK_CHARS", "8"))
    chunk_chars: int = int(os.getenv("CHUNK_CHARS", "1200"))
    text_types: list[str] = field(default_factory=lambda: [t.strip().lower() for t in os.getenv(
        "TEXT_TYPES", "char,varchar,text,mediumtext,longtext,json,date,datetime,time,timestamp,year"
    ).split(',') if t.strip()])
    combine_row_columns: bool = os.getenv("COMBINE_ROW_COLUMNS", "true").lower() in {"1","true","yes"}
    include_column_names: bool = os.getenv("INCLUDE_COLUMN_NAMES", "false").lower() in {"1","true","yes"}
    always_include_patterns: list[str] = field(default_factory=lambda: [p.strip() for p in os.getenv(
        "ALWAYS_INCLUDE_PATTERNS", ""
    ).split(',') if p.strip()])
    fulltext_required: bool = os.getenv("FULLTEXT_REQUIRED", "false").lower() in {"1","true","yes"}
    random_snippet_limit: int = int(os.getenv("RANDOM_SNIPPET_LIMIT", "0"))
    random_style: str = os.getenv("RANDOM_STYLE", "narrative").lower()
    simple_rag: bool = os.getenv("SIMPLE_RAG", "true").lower() in {"1","true","yes"}
    context_snippets: int = int(os.getenv("CONTEXT_SNIPPETS", "5"))
    context_chars: int = int(os.getenv("CONTEXT_CHARS", "220"))
    essential_terms: list[str] = field(default_factory=lambda: [t.strip().lower() for t in os.getenv(
        "ESSENTIAL_TERMS", "time,date,place,location,venue,address"
    ).split(',') if t.strip()])
    row_enrich: bool = os.getenv("ROW_ENRICH", "true").lower() in {"1","true","yes"}
    row_enrich_chars: int = int(os.getenv("ROW_ENRICH_CHARS", "600"))
    auto_index: bool = os.getenv("AUTO_INDEX", "true").lower() in {"1","true","yes"}
    auto_index_interval: int = int(os.getenv("AUTO_INDEX_INTERVAL", "60"))
    auto_index_initial_build: bool = os.getenv("AUTO_INDEX_INITIAL_BUILD", "true").lower() in {"1","true","yes"}
    embed_query_cache_size: int = int(os.getenv("EMBED_QUERY_CACHE_SIZE", "256"))
    embed_concurrency: int = int(os.getenv("EMBED_CONCURRENCY", "4"))
    metrics_enabled: bool = os.getenv("METRICS_ENABLED", "false").lower() in {"1","true","yes"}
    prewarm_queries: list[str] = field(default_factory=lambda: [q.strip() for q in os.getenv(
        "PREWARM_QUERIES", ""
    ).split(',') if q.strip()])
    cb_fail_window: int = int(os.getenv("CB_FAIL_WINDOW", "60"))
    cb_max_failures: int = int(os.getenv("CB_MAX_FAILURES", "5"))
    cb_open_seconds: int = int(os.getenv("CB_OPEN_SECONDS", "30"))
    cb_persist: bool = os.getenv("CB_PERSIST", "true").lower() in {"1","true","yes"}
    cb_state_file: str = os.getenv("CB_STATE_FILE", os.path.join(os.getcwd(), "cb_state.json"))
    tracing_enabled: bool = os.getenv("TRACING_ENABLED", "false").lower() in {"1","true","yes"}
    otlp_endpoint: str = os.getenv("OTLP_ENDPOINT", "")
    service_name: str = os.getenv("SERVICE_NAME", "ragchat")
    trace_sample_ratio: float = float(os.getenv("TRACE_SAMPLE_RATIO", "0.1"))
    # Time zone for temporal status inference (IANA name, e.g. 'UTC', 'America/New_York')
    status_timezone: str = os.getenv("STATUS_TIMEZONE", "UTC")
    # Return verbose errors to user (for debugging only; do not enable in prod)
    debug_errors: bool = os.getenv("DEBUG_ERRORS", "false").lower() in {"1","true","yes"}

settings = Settings()

if not settings.gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY not set. Provide it via environment or .env file.")

if not settings.mysql_db:
    raise RuntimeError("MYSQL_DB not set.")
