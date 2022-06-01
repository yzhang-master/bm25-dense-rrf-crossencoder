"""Deterministic, from-scratch sparse, dense and neural retrieval."""

__version__ = "0.1.0"


def scaffold_info() -> dict[str, object]:
    return {
        "package": "hybrid_retrieval",
        "version": __version__,
        "pipeline": ("bm25", "dense", "rrf", "crossencoder"),
    }
