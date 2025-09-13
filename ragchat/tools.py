from __future__ import annotations
from typing import Any
from ragchat.core.db import fetch_schema, safe_execute_select, connection, get_primary_key
from ragchat.core.retrieval import hybrid

# Functions exposed for Gemini function calling

def get_schema() -> dict:
    return fetch_schema()


def execute_sql(query: str) -> dict:
    return safe_execute_select(query)


def search_knowledge(question: str) -> list[dict[str, Any]]:
    return hybrid(question)

def list_random_rows(table: str, limit: int = 5) -> dict:
    table = table.strip()
    if not table:
        return {"error": "table required"}
    try:
        pk = get_primary_key(table)
        with connection() as conn:
            cur = conn.cursor()
            # ORDER BY RAND() for MySQL randomness; safe because limit small.
            cur.execute(f"SELECT * FROM {table} ORDER BY RAND() LIMIT %s", (limit,))
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return {"columns": cols, "rows": rows, "table": table}
    except Exception as e:
        return {"error": str(e)}
