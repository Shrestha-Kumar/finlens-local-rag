"""Retrieval evaluation.

  python evaluate.py                      evaluate the saved index
  python evaluate.py --sweep 150,300,500  compare chunk sizes (re-embeds for each size)

eval/questions.json is a list of:
  {"question": "...", "source": "file.pdf", "pages": [12, 13], "keywords": ["total income"]}
A retrieved chunk counts as relevant if (source matches AND page is in `pages`)
OR it contains any of the `keywords` (case-insensitive). Give either or both.
"""
import argparse
import json

from rag import config
from rag.chunking import make_chunks
from rag.index import VectorIndex
from rag.ingest import load_documents
from rag.retriever import retrieve

K_VALUES = (1, 3, 5)


def load_questions(path=config.EVAL_FILE) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy eval/questions.example.json to eval/questions.json "
            "and replace the examples with your own questions.")
    questions = json.loads(path.read_text(encoding="utf-8"))
    for q in questions:
        if not (q.get("keywords") or (q.get("source") and q.get("pages"))):
            raise ValueError(f"Question needs 'keywords' or 'source'+'pages': {q['question']}")
    return questions


def is_relevant(chunk: dict, q: dict) -> bool:
    if q.get("source") and q.get("pages"):
        if chunk["source"].lower() == q["source"].lower() and chunk["page"] in q["pages"]:
            return True
    text = chunk["text"].lower()
    return any(kw.lower() in text for kw in q.get("keywords", []))


def evaluate(index, questions: list[dict]) -> dict:
    kmax = max(K_VALUES)
    ranks, misses = [], []
    for q in questions:
        results = retrieve(index, q["question"], kmax)
        rank = next((i for i, c in enumerate(results, start=1) if is_relevant(c, q)), None)
        ranks.append(rank)
        if rank is None:
            misses.append((q["question"], results))
    n = len(questions)
    metrics = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in K_VALUES}
    metrics["mrr"] = sum(1 / r for r in ranks if r) / n
    metrics["n"] = n
    return {"metrics": metrics, "misses": misses}


def print_metrics(label: str, m: dict, extra: str = "") -> None:
    print(f"{label:<14}" + "  ".join(f"hit@{k}={m[f'hit@{k}']:.2f}" for k in K_VALUES)
          + f"  MRR={m['mrr']:.2f}  (n={m['n']}) {extra}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    ap.add_argument("--sweep", help="comma-separated chunk sizes, e.g. 150,300,500")
    args = ap.parse_args()
    questions = load_questions()

    if args.sweep:
        pages = load_documents(config.DATA_DIR)
        print()
        for size in (int(s) for s in args.sweep.split(",")):
            overlap = min(config.CHUNK_OVERLAP, size // 5)
            chunks = make_chunks(pages, size, overlap)
            idx = VectorIndex.build(chunks, show_progress=False)
            result = evaluate(idx, questions)
            print_metrics(f"chunk={size}", result["metrics"], f"[{len(chunks)} chunks]")
        return

    result = evaluate(VectorIndex.load(), questions)
    print_metrics("saved index", result["metrics"])
    if result["misses"]:
        print(f"\n{len(result['misses'])} question(s) with no relevant chunk in the top {max(K_VALUES)}:")
        for question, results in result["misses"]:
            print(f"  - {question}")
            for c in results[:3]:
                print(f"      got: {c['source']} p.{c['page']} (score {c['score']:.3f})")


if __name__ == "__main__":
    main()
