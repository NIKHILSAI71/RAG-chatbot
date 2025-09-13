from __future__ import annotations
from google.genai import types  # type: ignore

# Centralized helper for building Gemini content objects in a version-agnostic way.
# Falls back to raw string if object construction fails (older/newer SDK changes).

def build_content(text: str):
    try:
        Part = types.Part  # type: ignore
        part = None
        if hasattr(Part, "from_text"):
            try:
                part = Part.from_text(text)  # type: ignore
            except TypeError:
                part = Part(text=text)  # type: ignore
        else:
            part = Part(text=text)  # type: ignore
        return types.Content(parts=[part])  # type: ignore
    except Exception:
        return text


def build_contents(texts: list[str]):
    return [build_content(t) for t in texts]
