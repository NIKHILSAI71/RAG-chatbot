from __future__ import annotations
from typing import Any
from ragchat.core.db import connection, fetch_schema, get_primary_key
from ragchat.core.config import settings
from ragchat.core.embeddings import embed_query
from ragchat.infra.vector_store import VectorStore

# VectorStore instance injected at runtime
_vs: VectorStore | None = None

def set_vector_store(vs: VectorStore):
    global _vs
    _vs = vs


_text_columns_cache: dict[str, list[str]] | None = None


def _get_text_columns() -> dict[str, list[str]]:
    global _text_columns_cache
    if _text_columns_cache is not None:
        return _text_columns_cache
    schema = fetch_schema()
    out: dict[str, list[str]] = {}
    for table, cols in schema.items():
        txt = [c["name"] for c in cols if c["type"].lower() in settings.text_types][:4]
        if txt:
            out[table] = txt
    _text_columns_cache = out
    return out


def fulltext_search(query: str, limit: int = 50) -> list[dict[str, Any]]:
    columns_map = _get_text_columns()
    results: list[dict[str, Any]] = []
    q_like = f"%{query[:64]}%"
    with connection() as conn:
        cur = conn.cursor(dictionary=True)
        for table, text_cols in columns_map.items():
            if not text_cols:
                continue
            pk = get_primary_key(table)
            if not settings.fulltext_required:
                select_cols = ", ".join(text_cols)
                first_col = text_cols[0]
                try:
                    cur.execute(
                        f"SELECT {pk} AS pk_val, {select_cols} FROM {table} WHERE {first_col} LIKE %s LIMIT %s",
                        (q_like, limit),
                    )
                    for row in cur:
                        parts = []
                        for col in text_cols:
                            val = row.get(col)
                            if val:
                                parts.append(str(val))
                        snippet = " | ".join(parts)[:500]
                        results.append({
                            "table": table,
                            "pk": row["pk_val"],
                            "title": "",
                            "snippet": snippet,
                            "score_ft": 0.0,
                        })
                except Exception:
                    continue
            else:
                cols_expr = ", ".join(text_cols[:3])
                try:
                    cur.execute(
                        f"SELECT {pk} AS pk_val, {text_cols[0]} AS c0{', ' + text_cols[1] + ' AS c1' if len(text_cols) > 1 else ''}{', ' + text_cols[2] + ' AS c2' if len(text_cols) > 2 else ''}, "
                        f"MATCH({cols_expr}) AGAINST (%s IN BOOLEAN MODE) AS ft_score FROM {table} "
                        f"WHERE MATCH({cols_expr}) AGAINST (%s IN BOOLEAN MODE) LIMIT %s",
                        (query, query, limit),
                    )
                    for row in cur:
                        parts = [row.get('c0') or '', row.get('c1') or '', row.get('c2') or '']
                        snippet = " | ".join([p for p in parts if p])[:500]
                        results.append({
                            "table": table,
                            "pk": row["pk_val"],
                            "title": "",
                            "snippet": snippet,
                            "score_ft": float(row.get("ft_score") or 0.0),
                        })
                except Exception:
                    continue
    return results


def hybrid(query: str, k_vec: int = 20, k_ft: int = 50) -> list[dict[str, Any]]:
    if _vs is None:
        return []
    ft = fulltext_search(query, k_ft)
    qv = embed_query(query)
    sem = _vs.search(qv, k=k_vec)

    merged: dict[tuple, dict[str, Any]] = {}

    for r in ft:
        key = (r["table"], r["pk"])
        merged[key] = {
            "table": r["table"],
            "pk": r["pk"],
            "title": r.get("title", ""),
            "snippet": r.get("snippet", "")[:500],
            "score_ft": float(r.get("score_ft", 0.0)),
            "score_sem": 0.0,
        }

    for r in sem:
        key = (r["table"], r["pk"])
        new_snip = (r.get("chunk") or r.get("snippet") or "")[:250]
        if key not in merged:
            merged[key] = {
                "table": r["table"],
                "pk": r["pk"],
                "title": "",
                "snippet": new_snip,
                "score_ft": 0.0,
                "score_sem": r.get("score_sem", 0.0),
                "combined_text": r.get("chunk") or r.get("snippet") or "",
            }
        else:
            merged[key]["score_sem"] = max(merged[key]["score_sem"], r.get("score_sem", 0.0))
            existing = merged[key]["snippet"]
            if new_snip and new_snip not in existing:
                combined = existing + " | " + new_snip
                merged[key]["snippet"] = combined[:500]
            if not merged[key].get("combined_text"):
                merged[key]["combined_text"] = r.get("chunk") or r.get("snippet") or ""

    out = list(merged.values())
    out.sort(key=lambda x: 0.6 * x["score_sem"] + 0.4 * x["score_ft"], reverse=True)
    if getattr(settings, 'row_enrich', False):
        top_enrich = out[: min(5, len(out))]
        schema = fetch_schema()
        with connection() as conn:
            cur = conn.cursor(dictionary=True)
            for item in top_enrich:
                table = item["table"]
                pk = get_primary_key(table)
                if table not in schema:
                    continue
                cols = [c['name'] for c in schema[table]][:50]
                col_list = ", ".join(cols)
                try:
                    cur.execute(f"SELECT {col_list} FROM {table} WHERE {pk} = %s LIMIT 1", (item["pk"],))
                    row = cur.fetchone()
                    if row:
                        parts = []
                        for c, v in row.items():
                            if v is None:
                                continue
                            sval = str(v).strip()
                            if not sval:
                                continue
                            parts.append(f"{c}: {sval}")
                        enriched_text = " | ".join(parts)
                        if enriched_text:
                            item["enriched"] = enriched_text[: settings.row_enrich_chars]
                            lower_snip = item.get("snippet", "").lower()
                            for term in settings.essential_terms:
                                if term in enriched_text.lower() and term not in lower_snip:
                                    item["snippet"] = enriched_text[:500]
                                    break
                except Exception:
                    continue
    return out[: max(k_vec, 10)]
