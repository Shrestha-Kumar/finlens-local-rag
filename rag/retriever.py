"""Retrieval: embed the question, return the top-k most similar chunks."""
from . import config, embedder


def retrieve(index, query: str, k: int = config.TOP_K) -> list[dict]:
    scores, ids = index.search(embedder.embed_query(query), k)
    results = []
    for score, i in zip(scores[0], ids[0]):
        if i < 0:
            continue
        chunk = dict(index.chunks[i])
        chunk["score"] = float(score)
        results.append(chunk)
    return results
