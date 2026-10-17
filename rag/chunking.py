"""Chunking: split each page into overlapping word windows.

Chunking is done per page so every chunk keeps an exact (source, page) citation.
"""
from . import config


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def make_chunks(pages: list[dict], size: int = config.CHUNK_WORDS,
                overlap: int = config.CHUNK_OVERLAP,
                min_words: int = config.MIN_CHUNK_WORDS) -> list[dict]:
    chunks: list[dict] = []
    for p in pages:
        for piece in chunk_text(p["text"], size, overlap):
            if len(piece.split()) >= min_words:
                chunks.append({"id": len(chunks), "source": p["source"],
                               "page": p["page"], "text": piece})
    return chunks
