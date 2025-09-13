from __future__ import annotations
import threading
from contextlib import contextmanager
from typing import Iterator, Any
from mysql.connector import pooling, Error
from ragchat.core.config import settings

_pool_lock = threading.Lock()
_pool: pooling.MySQLConnectionPool | None = None
_pk_cache: dict[str, str] | None = None
_schema_cache: dict | None = None
_schema_lock = threading.Lock()
_schema_last_load: float | None = None
_SCHEMA_TTL_SECONDS = 60.0


def init_pool(min_size: int = 1, max_size: int = 8):
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = pooling.MySQLConnectionPool(
                pool_name="rag_pool",
                pool_size=max_size,
                host=settings.mysql_host,
                user=settings.mysql_user,
                password=settings.mysql_password,
                database=settings.mysql_db,
                autocommit=True,
            )
    return _pool


def get_conn():
    if _pool is None:
        raise RuntimeError("Connection pool not initialized. Call init_pool() on startup.")
    return _pool.get_connection()


@contextmanager
def connection() -> Iterator[Any]:
    conn = get_conn()
    try:
        yield conn
    finally:
        conn.close()


def fetch_schema(force: bool = False) -> dict:
    global _schema_cache, _schema_last_load
    import time
    with _schema_lock:
        now = time.time()
        if (not force and _schema_cache is not None and _schema_last_load is not None
                and (now - _schema_last_load) < _SCHEMA_TTL_SECONDS):
            return _schema_cache
        schema: dict[str, list[dict[str, str]]] = {}
        with connection() as conn:
            cur = conn.cursor(dictionary=True)
            cur.execute(
                """
                SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = %s
                ORDER BY TABLE_NAME, ORDINAL_POSITION
                """,
                (settings.mysql_db,),
            )
            for row in cur:
                schema.setdefault(row["TABLE_NAME"], []).append(
                    {"name": row["COLUMN_NAME"], "type": row["DATA_TYPE"]}
                )
        _schema_cache = schema
        _schema_last_load = now
        return schema


def schema_cache_age_seconds() -> float | None:
    import time
    if _schema_last_load is None:
        return None
    return time.time() - _schema_last_load


def fetch_primary_keys(force: bool = False) -> dict[str, str]:
    global _pk_cache
    if _pk_cache is not None and not force:
        return _pk_cache
    pks: dict[str, str] = {}
    with connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT k.TABLE_NAME, k.COLUMN_NAME
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS t
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE k
              ON t.CONSTRAINT_NAME = k.CONSTRAINT_NAME
             AND t.TABLE_SCHEMA = k.TABLE_SCHEMA
            WHERE t.TABLE_SCHEMA = %s
              AND t.CONSTRAINT_TYPE = 'PRIMARY KEY'
            ORDER BY k.TABLE_NAME, k.ORDINAL_POSITION
            """,
            (settings.mysql_db,),
        )
        for row in cur:
            pks.setdefault(row["TABLE_NAME"], row["COLUMN_NAME"])
    _pk_cache = pks
    return pks


def get_primary_key(table: str) -> str:
    pks = fetch_primary_keys()
    return pks.get(table, "id")


def safe_execute_select(query: str, params: tuple | None = None, limit_default: int = 200):
    q = query.strip().rstrip(";")
    q_upper = q.upper()
    if not q_upper.startswith("SELECT "):
        return {"error": "Only SELECT statements allowed"}
    if " LIMIT " not in q_upper:
        q = f"{q} LIMIT {limit_default}"
    try:
        with connection() as conn:
            cur = conn.cursor()
            cur.execute(q, params or ())
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return {"columns": cols, "rows": rows}
    except Error as e:
        return {"error": str(e), "sqlstate": getattr(e, 'sqlstate', None), "errno": getattr(e, 'errno', None)}
    except Exception as e:
        return {"error": str(e), "type": e.__class__.__name__}
