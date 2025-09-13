"""Top-level package shims to preserve original import paths.

This file re-exports key modules from the reorganized package layout.
"""

from importlib import import_module

__all__ = [
	"app", "chat", "config", "db", "embeddings", "indexer", "retrieval", "tools", "vector_store", "metrics", "tracing"
]

# Lazy import modules mapping
_MAP = {
	"chat": "ragchat.core.chat",
	"config": "ragchat.core.config",
	"db": "ragchat.core.db",
	"embeddings": "ragchat.core.embeddings",
	"indexer": "ragchat.core.indexer",
	"retrieval": "ragchat.core.retrieval",
	# `tools` lives at top-level `ragchat.tools`
	"tools": "ragchat.tools",
	"vector_store": "ragchat.infra.vector_store",
	"metrics": "ragchat.observability.metrics",
	"tracing": "ragchat.observability.tracing",
}


def __getattr__(name: str):
	if name in _MAP:
		mod = import_module(_MAP[name])
		globals()[name] = mod
		return mod
	raise AttributeError(name)
