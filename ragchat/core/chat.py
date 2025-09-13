from __future__ import annotations
from typing import Any
import logging
from google import genai
from google.genai import types
from ragchat.core.config import settings
from ragchat.core.retrieval import hybrid as hybrid_search

_client: genai.Client | None = None
logger = logging.getLogger(__name__)

def get_client():
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        try:
            _client = genai.Client(api_key=settings.gemini_api_key)
        except Exception as e:
            logger.error("[chat] Failed to create Gemini client: %s", e)
            raise RuntimeError(f"Failed to initialize Gemini client: {e}")
    return _client


SYSTEM_PROMPT = (
    "You are an assistive analytical AI. Use ONLY provided context for factual claims. "
    "If temporal metadata (status/indexed_at) suggests upcoming/ongoing/completed events, mention that explicitly. "
    "Synthesize details, group related facts, highlight timelines. If context is empty, say you have no information."
)

def chat_once(user_query: str) -> str:
    try:
        retrieved = hybrid_search(user_query)[: settings.context_snippets]
    except Exception as e:
        logger.error("[chat] Retrieval failure: %s", e)
        retrieved = []
        # If retrieval fails completely, we can still try to answer without context
        if getattr(settings, 'debug_errors', False):
            logger.warning("[chat] Proceeding without retrieval context due to error: %s", e)
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
    
    try:
        client = get_client()
    except Exception as e:
        logger.error("[chat] Client initialization failed: %s", e)
        return f"Configuration error: Unable to initialize AI client. Please check your GEMINI_API_KEY."
    
    # Single request generation (avoids separate chat session creation latency)
    cfg = types.GenerateContentConfig(temperature=0.25, system_instruction=SYSTEM_PROMPT)
    prompt = user_query + ("\n\n" + context_block if context_block else "")
    
    try:
        # Try the newer API first, fall back to older if needed
        try:
            contents = [types.Content(parts=[types.Part.from_text(prompt)])]
        except AttributeError:
            # Fallback for older API version
            contents = [types.Content(parts=[types.Part(text=prompt)])]
        
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=cfg,
        )
        
        if not resp or not hasattr(resp, 'text') or not resp.text:
            logger.warning("[chat] Empty response from Gemini API")
            return "I couldn't generate a response. Please try rephrasing your question."
            
        return resp.text
        
    except Exception as e:
        error_msg = str(e)
        logger.error("[chat] Generation failure: %s", e)
        
        # Provide more specific error messages based on common issues
        if "api key" in error_msg.lower() or "authentication" in error_msg.lower():
            return "Authentication error: Please check your GEMINI_API_KEY configuration."
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            return "API quota exceeded. Please try again later or check your Gemini API usage limits."
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            return "Model access error: The gemini-2.0-flash model may not be available. Please check your API access."
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            return "Network error: Unable to connect to Gemini API. Please check your internet connection."
        else:
            # For debugging purposes, include the actual error in development
            if getattr(settings, 'debug_errors', False):
                return f"Generation error: {error_msg[:500]}"
            return "I'm experiencing technical difficulties. Please try again in a moment." 
