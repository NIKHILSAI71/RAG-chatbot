from __future__ import annotations
import threading
from contextlib import contextmanager
from typing import Iterator, Any, Protocol, runtime_checkable, Callable
from ragchat.core.config import settings

try:
    from mysql.connector import pooling, Error  # type: ignore
except Exception:  # pragma: no cover - optional dependency when not using mysql
    pooling = None  # type: ignore
    class Error(Exception):
        pass


@runtime_checkable
class DataSource(Protocol):
    def init(self): ...
    @contextmanager
    def connection(self) -> Iterator[Any]: ...
    def fetch_schema(self, force: bool = False) -> dict: ...
    def schema_cache_age_seconds(self) -> float | None: ...
    def get_primary_key(self, table: str) -> str: ...
    def safe_execute_select(self, query: str, params: tuple | None = None, limit_default: int = 200): ...


############################################################
# MySQL Implementation (default)
############################################################

class MySQLDataSource:
    def __init__(self):
        self._pool_lock = threading.Lock()
        self._pool = None
        self._pk_cache: dict[str, str] | None = None
        self._schema_cache: dict | None = None
        self._schema_lock = threading.Lock()
        self._schema_last_load: float | None = None
        self._SCHEMA_TTL_SECONDS = 60.0

    def init(self, min_size: int = 1, max_size: int = 8):
        if pooling is None:
            raise RuntimeError("mysql-connector not installed; required for MySQL datasource")
        with self._pool_lock:
            if self._pool is None:
                self._pool = pooling.MySQLConnectionPool(
                    pool_name="rag_pool",
                    pool_size=max_size,
                    host=settings.mysql_host,
                    user=settings.mysql_user,
                    password=settings.mysql_password,
                    database=settings.mysql_db,
                    autocommit=True,
                )
        return self._pool

    def _get_conn(self):
        if self._pool is None:
            raise RuntimeError("Connection pool not initialized. Call init() on startup.")
        return self._pool.get_connection()

    @contextmanager
    def connection(self) -> Iterator[Any]:
        conn = self._get_conn()
        try:
            yield conn
        finally:
            conn.close()

    def fetch_schema(self, force: bool = False) -> dict:
        import time
        with self._schema_lock:
            now = time.time()
            if (not force and self._schema_cache is not None and self._schema_last_load is not None
                    and (now - self._schema_last_load) < self._SCHEMA_TTL_SECONDS):
                return self._schema_cache
            schema: dict[str, list[dict[str, str]]] = {}
            with self.connection() as conn:
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
            self._schema_cache = schema
            self._schema_last_load = now
            return schema

    def schema_cache_age_seconds(self) -> float | None:
        import time
        if self._schema_last_load is None:
            return None
        return time.time() - self._schema_last_load

    def fetch_primary_keys(self, force: bool = False) -> dict[str, str]:
        if self._pk_cache is not None and not force:
            return self._pk_cache
        pks: dict[str, str] = {}
        with self.connection() as conn:
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
        self._pk_cache = pks
        return pks

    def get_primary_key(self, table: str) -> str:
        return self.fetch_primary_keys().get(table, "id")

    def safe_execute_select(self, query: str, params: tuple | None = None, limit_default: int = 200):
        q = query.strip().rstrip(";")
        q_upper = q.upper()
        if not q_upper.startswith("SELECT "):
            return {"error": "Only SELECT statements allowed"}
        if " LIMIT " not in q_upper:
            q = f"{q} LIMIT {limit_default}"
        try:
            with self.connection() as conn:
                cur = conn.cursor()
                cur.execute(q, params or ())
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
            return {"columns": cols, "rows": rows}
        except Error as e:  # type: ignore
            return {"error": str(e), "sqlstate": getattr(e, 'sqlstate', None), "errno": getattr(e, 'errno', None)}
        except Exception as e:
            return {"error": str(e), "type": e.__class__.__name__}


############################################################
# Dummy / In-Memory Implementation (for non-SQL or custom ingestion)
############################################################

_ds: MySQLDataSource | None = None


def _ensure() -> MySQLDataSource:
    global _ds
    if _ds is None:
        _ds = MySQLDataSource()
        _ds.init()
    return _ds


def init_pool():
    return _ensure()


def connection() -> Iterator[Any]:
    return _ensure().connection()  # type: ignore


def fetch_schema(force: bool = False) -> dict:
    return _ensure().fetch_schema(force)


def schema_cache_age_seconds() -> float | None:
    return _ensure().schema_cache_age_seconds()


def get_primary_key(table: str) -> str:
    return _ensure().get_primary_key(table)


def safe_execute_select(query: str, params: tuple | None = None, limit_default: int = 200):
    return _ensure().safe_execute_select(query, params, limit_default)
