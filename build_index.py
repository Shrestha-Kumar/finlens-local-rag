"""Build the index:  python build_index.py [--chunk-words 350] [--overlap 60]"""
import argparse
import time

from rag import config
from rag.chunking import make_chunks
from rag.index import VectorIndex
from rag.ingest import load_documents


def main():
    ap = argparse.ArgumentParser(description="Ingest documents and build the vector index.")
    ap.add_argument("--chunk-words", type=int, default=config.CHUNK_WORDS)
    ap.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1/3] Reading documents from {config.DATA_DIR}")
    pages = load_documents(config.DATA_DIR)

    print(f"[2/3] Chunking {len(pages)} pages (size={args.chunk_words}, overlap={args.overlap})")
    chunks = make_chunks(pages, args.chunk_words, args.overlap)
    print(f"      -> {len(chunks)} chunks")

    print(f"[3/3] Embedding with {config.EMBED_MODEL} and saving index")
    VectorIndex.build(chunks).save()
    print(f"Done in {time.time() - t0:.1f}s. Index saved to {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
