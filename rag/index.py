"""Vector index: FAISS inner-product search over normalised vectors
(inner product on unit vectors == cosine similarity)."""
import json
from pathlib import Path

import faiss
import numpy as np

from . import config, embedder


class VectorIndex:
    def __init__(self, index: "faiss.Index", chunks: list[dict]):
        self.index = index
        self.chunks = chunks

    @classmethod
    def build(cls, chunks: list[dict], show_progress: bool = True) -> "VectorIndex":
        vecs = embedder.embed_documents([c["text"] for c in chunks], show_progress=show_progress)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return cls(index, chunks)

    def search(self, query_vec: np.ndarray, k: int):
        return self.index.search(query_vec, min(k, len(self.chunks)))

    def save(self, directory: Path = config.INDEX_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "faiss.index"))
        (directory / "chunks.json").write_text(json.dumps(self.chunks), encoding="utf-8")
        meta = {"embed_model": config.EMBED_MODEL, "num_chunks": len(self.chunks)}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path = config.INDEX_DIR) -> "VectorIndex":
        if not (directory / "faiss.index").exists():
            raise FileNotFoundError("No index found. Run:  python build_index.py")
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if meta["embed_model"] != config.EMBED_MODEL:
            print(f"  ! Index was built with {meta['embed_model']} but config uses "
                  f"{config.EMBED_MODEL}. Rebuild with: python build_index.py")
        index = faiss.read_index(str(directory / "faiss.index"))
        chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        return cls(index, chunks)
