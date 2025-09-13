from __future__ import annotations
from typing import Any
from google import genai
from google.genai import types
from ragchat.core.config import settings
from ragchat.core.retrieval import hybrid as hybrid_search

_client: genai.Client | None = None

def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def chat_once(user_query: str) -> str:
    retrieved = hybrid_search(user_query)[: settings.context_snippets]
    context_lines = []
    limit = settings.context_chars
    essentials = settings.essential_terms
    for r in retrieved:
        raw = (r.get("enriched") or r.get("combined_text") or r.get("snippet") or r.get("chunk") or "").replace("\n", " ").strip()
        if not raw:
            continue
        truncated = raw[:limit]
        lower_raw = raw.lower()
        max_needed = len(truncated)
        for term in essentials:
            idx = lower_raw.find(term)
            if idx != -1 and idx + len(term) > max_needed:
                max_needed = min(max(idx + len(term) + 40, max_needed), min(len(raw), limit * 2))
        snippet = raw[:max_needed]
        meta_bits = []
        if r.get("table"): meta_bits.append(f"table={r['table']}")
        if r.get("pk") is not None: meta_bits.append(f"pk={r['pk']}")
        if r.get("status"): meta_bits.append(f"status={r['status']}")
        if r.get("indexed_at"): meta_bits.append(f"indexed_at={r['indexed_at']}")
        meta_str = (" [" + ", ".join(meta_bits) + "]") if meta_bits else ""
        context_lines.append(f"- {snippet}{meta_str}")
    context_block = ("\n\nContext (most relevant records with metadata):\n" + "\n".join(context_lines)) if context_lines else ""
    client = get_client()
    config = types.GenerateContentConfig(
        system_instruction=(
            "You are an assistive analytical AI. Use ONLY provided context for factual claims. "
            "If temporal metadata (status/indexed_at) suggests upcoming/ongoing/completed events, mention that explicitly. "
            "Synthesize details, group related facts, highlight timelines. If context is empty, say you have no information."
        ),
        temperature=0.25,
    )
    conv = client.chats.create(model="gemini-2.0-flash", config=config)
    resp = conv.send_message(user_query + context_block)
    return resp.text or "No information available."
