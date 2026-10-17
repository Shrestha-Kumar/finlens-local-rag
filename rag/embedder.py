"""Embeddings: turn text into normalised vectors (local model)."""
import numpy as np

from . import config

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # imported lazily (slow import)
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def embed_documents(texts: list[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
    vecs = get_model().encode(texts, batch_size=batch_size, show_progress_bar=show_progress,
                              normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    if "bge" in config.EMBED_MODEL.lower():
        query = config.QUERY_PREFIX + query
    vec = get_model().encode([query], normalize_embeddings=True)
    return np.asarray(vec, dtype="float32")
