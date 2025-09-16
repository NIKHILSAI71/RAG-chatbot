from __future__ import annotations
from typing import Iterable, Any, Sequence, Tuple, Dict
import hashlib
import re
from ragchat.core.config import settings
from ragchat.core.db import connection, fetch_schema, get_primary_key
from ragchat.core.embeddings import embed_texts
import time
from ragchat.infra.vector_store import VectorStore

CHUNK_CHARS = settings.chunk_chars if getattr(settings, 'chunk_chars', None) else 1200

__all__ = ["build_index", "resolve_index_columns", "incremental_update"]


def chunk_text(text: str, size: int = CHUNK_CHARS):
    for i in range(0, len(text), size):
        yield text[i:i+size]


def fetch_rows(table: str, column: str, limit: int) -> Iterable[dict]:
    """Fetch a single column (legacy path)."""
    pk = get_primary_key(table)
    with connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            f"SELECT {pk} AS pk_val, {column} FROM {table} WHERE {column} IS NOT NULL LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()
    for row in rows:
        yield {"pk": row["pk_val"], "text": row[column], "table": table, "column": column}


def fetch_rows_all_text_columns(table: str, limit: int) -> Iterable[dict]:
    """Fetch and concatenate relevant columns dynamically.

    Strategy:
    - Start with columns whose data type is in text_types.
    - Force include columns whose names match always_include_patterns (case-insensitive substring or regex fragment).
    - Optionally strip column name labels for cleaner semantic signal if include_column_names is false.
    """
    schema = fetch_schema()
    cols_meta = schema.get(table, [])
    patterns = [re.compile(p, re.IGNORECASE) for p in settings.always_include_patterns]
    text_cols = []
    for col in cols_meta:
        name = col["name"]
        col_type = col["type"].lower()
        include = col_type in settings.text_types
        if not include:
            for pat in patterns:
                if pat.search(name):
                    include = True
                    break
        if include:
            text_cols.append(name)
    if not text_cols:
        return []
    pk = get_primary_key(table)
    # Extend with temporal columns (even if not text) for status inference
    schema_cols = schema.get(table, [])
    temporal_types = {"date", "datetime", "timestamp", "time", "year"}
    temporal_only = [c["name"] for c in schema_cols if c["type"].lower() in temporal_types and c["name"] not in text_cols]
    select_cols = ", ".join([pk] + text_cols + temporal_only) if temporal_only else ", ".join([pk] + text_cols)
    with connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(f"SELECT {select_cols} FROM {table} LIMIT %s", (limit,))
        rows = cur.fetchall()
    label_cols = settings.include_column_names
    for row in rows:
        pk_val = row.get(pk)
        parts = []
        for c in text_cols:
            val = row.get(c)
            if val is None:
                continue
            sval = str(val).strip()
            if not sval:
                continue
            if label_cols:
                parts.append(f"{c}: {sval}")
            else:
                parts.append(sval)
        if not parts:
            continue
        # include raw column values for temporal/status enrichment (per-row)
        raw_cols = {c: row.get(c) for c in (text_cols + temporal_only)}
        yield {"pk": pk_val, "text": " \n ".join(parts), "table": table, "column": "*", "raw_cols": raw_cols}


def resolve_index_columns() -> list[str]:
    raw = [r.strip() for r in settings.index_columns if r.strip()]
    schema = fetch_schema()
    # Optional table filter from env
    allowed_tables = {t for t in getattr(settings, 'index_tables', ['*']) if t != '*'}
    # Validation of explicit list
    if raw and raw != ["*"] and not (len(raw) == 1 and raw[0] == ""):
        valid: list[str] = []
        missing: list[str] = []
        schema_tables = {t: {c['name'] for c in cols} for t, cols in schema.items()}
        for spec in raw:
            if '.' not in spec:
                missing.append(spec)
                continue
            t, c = spec.split('.', 1)
            if allowed_tables and t not in allowed_tables:
                # skip columns not in allowed tables list
                missing.append(spec)
                continue
            if t in schema_tables and c in schema_tables[t]:
                valid.append(spec)
            else:
                missing.append(spec)
        if valid:
            if missing:
                print(f"[indexer] Ignored missing columns: {', '.join(missing)}")
            return valid
        # Nothing valid -> fallback if allowed
        print("[indexer] No valid INDEX_COLUMNS found; falling back to auto discovery.")
    if not settings.index_auto_discover:
        return []
    cols: list[str] = []
    for table, column_list in schema.items():
        if allowed_tables and table not in allowed_tables:
            continue
        for c in column_list:
            if c["type"].lower() in settings.text_types:
                cols.append(f"{table}.{c['name']}")
    return cols


def build_index(vs: VectorStore | None = None) -> VectorStore:
    # If existing vs passed we append; else create new after first embed
    texts: list[str] = []
    metas: list[dict[str, Any]] = []
    columns_specs = resolve_index_columns()
    tables_seen: set[str] = set()
    now_ts = int(time.time())
    schema = fetch_schema()
    datetime_types = {"date", "datetime", "timestamp", "time", "year"}
    for spec in columns_specs:
        if not spec or "." not in spec:
            continue
        table, col = spec.split(".", 1)
        if settings.combine_row_columns:
            if table in tables_seen:
                continue
            for row in fetch_rows_all_text_columns(table, settings.index_row_limit):
                for ch in chunk_text(row["text"]):
                    if len(ch) < settings.min_chunk_chars:
                        continue
                    texts.append(ch)
                    # capture temporal columns for table (first row basis)
                    temporal_meta = {}
                    cols_meta = schema.get(row["table"], [])
                    for c in cols_meta:
                        if c["type"].lower() in datetime_types and c["name"] in row.get("raw_cols", {}):
                            temporal_meta[c["name"]] = row["raw_cols"][c["name"]]
                    metas.append({
                        "table": row["table"],
                        "pk": row["pk"],
                        "column": row["column"],
                        "chunk": ch,
                        "indexed_at": now_ts,
                        **({"temporal": temporal_meta} if temporal_meta else {})
                    })
            tables_seen.add(table)
        else:
            for row in fetch_rows(table, col, settings.index_row_limit):
                for ch in chunk_text(row["text"]):
                    if len(ch) < settings.min_chunk_chars:
                        continue
                    texts.append(ch)
                    temporal_meta = {}
                    cols_meta = schema.get(row["table"], [])
                    for c in cols_meta:
                        if c["type"].lower() in datetime_types and c["name"] in row.get("raw_cols", {}):
                            temporal_meta[c["name"]] = row["raw_cols"][c["name"]]
                    metas.append({
                        "table": row["table"],
                        "pk": row["pk"],
                        "column": row["column"],
                        "chunk": ch,
                        "indexed_at": now_ts,
                        **({"temporal": temporal_meta} if temporal_meta else {})
                    })
    if not texts:
        raise RuntimeError("No texts collected for indexing.")
    vectors = embed_texts(texts)
    dim = len(vectors[0])
    if vs is None:
        vs = VectorStore(dim)
    vs.add(vectors, metas)
    vs.save()
    return vs


def _row_fingerprint(parts: Sequence[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


def _collect_table_rows(table: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in fetch_rows_all_text_columns(table, settings.index_row_limit):
        rows.append(row)
    return rows


def incremental_update(vs: VectorStore, tables: list[str] | None = None) -> dict[str, Any]:
    """Perform incremental update of specified tables (or all auto discovered tables).

    Strategy:
    - Build a hash (fingerprint) per row (concatenated chunk text before chunking) so we know if the row changed.
    - For changed rows: delete existing vectors (metadata delete by table+pk) then re-chunk and re-embed.
    - For new rows: simply add.
    - Unchanged rows: untouched to preserve existing vector IDs & scores.
    ID scheme: table::pk::chunkIndex (deterministic) for new incremental path.
    """
    columns_specs = resolve_index_columns()
    # Derive table list from columns specs if not provided
    discovered_tables = sorted({spec.split(".", 1)[0] for spec in columns_specs if "." in spec})
    target_tables = tables or discovered_tables

    # Build existing map of (table, pk) -> list of meta entries (legacy + new)
    existing_rows: Dict[Tuple[str, Any], list[dict[str, Any]]] = {}
    for m in vs.meta:
        t = m.get("table")
        pk = m.get("pk")
        if t is None or pk is None:
            continue
        existing_rows.setdefault((t, pk), []).append(m)

    added_vectors = 0
    reembedded_rows = 0
    skipped_rows = 0
    deleted_vectors = 0
    deleted_rows = 0
    new_meta: list[dict[str, Any]] = []  # we rebuild vs.meta for target tables we touch; others preserved

    # We keep untouched metas first
    untouched: list[dict[str, Any]] = []
    touched_keys: set[Tuple[str, Any]] = set()

    for table in target_tables:
        rows = _collect_table_rows(table)
        current_pks: set[Any] = set()
        # Map pk-> (original_text, chunks)
        for row in rows:
            pk = row["pk"]
            current_pks.add(pk)
            text_full = row["text"]
            touched_keys.add((table, pk))
            existing = existing_rows.get((table, pk))
            # Build previous fingerprint if any
            prev_fp = None
            if existing:
                # Reconstruct previous combined row text by concatenating chunks (may over-approximate)
                prev_chunks = [m.get("chunk", "") for m in existing]
                prev_fp = _row_fingerprint(prev_chunks)
            new_fp = _row_fingerprint([text_full])
            if prev_fp == new_fp:
                # Keep existing metas unchanged
                new_meta.extend(existing)
                skipped_rows += 1
                continue
            # Delete old vectors for this row
            if existing:
                # Use metadata delete for the whole row
                try:
                    vs.delete_row(table, pk)
                except Exception:
                    pass
                deleted_vectors += len(existing)
            # Re-chunk and embed
            row_texts: list[str] = []
            row_metas: list[dict[str, Any]] = []
            chunk_idx = 0
            for ch in chunk_text(text_full):
                if len(ch) < settings.min_chunk_chars:
                    continue
                row_texts.append(ch)
                # temporal capture
                temporal_meta = {}
                cols_meta = fetch_schema().get(table, [])
                for c in cols_meta:
                    if c["type"].lower() in {"date","datetime","timestamp","time","year"} and c["name"] in row.get("raw_cols", {}):
                        temporal_meta[c["name"]] = row["raw_cols"][c["name"]]
                row_metas.append({
                    "table": table,
                    "pk": pk,
                    "column": row["column"],
                    "chunk": ch,
                    "indexed_at": int(time.time()),
                    # deterministic id
                    "id": f"{table}::{pk}::{chunk_idx}",
                    **({"temporal": temporal_meta} if temporal_meta else {})
                })
                chunk_idx += 1
            if not row_texts:
                continue
            vecs = embed_texts(row_texts)
            vs.add(vecs, row_metas)
            new_meta.extend(row_metas)
            added_vectors += len(row_texts)
            reembedded_rows += 1

        # Handle deletions: any (table, pk) that existed but is not in current snapshot
        existing_pks = {k[1] for k in existing_rows.keys() if k[0] == table}
        to_delete = existing_pks - current_pks
        for pk in to_delete:
            try:
                vs.delete_row(table, pk)
            except Exception:
                pass
            touched_keys.add((table, pk))
            deleted_vectors += len(existing_rows.get((table, pk), []) or [])
            deleted_rows += 1

    # Preserve metas for tables not touched
    for m in vs.meta:
        key = (m.get("table"), m.get("pk"))
        if key in touched_keys:
            continue
        untouched.append(m)

    vs.meta = untouched + new_meta
    changed = (added_vectors + reembedded_rows + deleted_vectors) > 0
    if changed:
        vs.save()
    return {
        "tables": target_tables,
        "added_vectors": added_vectors,
        "reembedded_rows": reembedded_rows,
        "skipped_rows": skipped_rows,
        "deleted_vectors": deleted_vectors,
        "deleted_rows": deleted_rows,
        "total_meta": len(vs.meta),
        "saved": changed,
    }


