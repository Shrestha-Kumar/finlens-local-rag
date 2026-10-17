"""Ask questions from the terminal.

  python ask.py                         interactive mode
  python ask.py -q "What was total income?"
  python ask.py -q "..." --retrieve-only   search only, no LLM (works without Ollama)
"""
import argparse

from rag import config
from rag.generator import format_source
from rag.index import VectorIndex
from rag.pipeline import answer_question


def show(result: dict) -> None:
    if result["answer"] is not None:
        print("\nANSWER:\n" + result["answer"])
    print("\nSOURCES:")
    for i, c in enumerate(result["sources"], start=1):
        preview = c["text"][:160].replace("\n", " ")
        print(f"  [{i}] {format_source(c)}  (score {c['score']:.3f})\n      {preview}...")
    print()


def main():
    ap = argparse.ArgumentParser(description="Ask questions about your documents (fully local).")
    ap.add_argument("-q", "--question")
    ap.add_argument("-k", type=int, default=config.TOP_K, help="number of chunks to retrieve")
    ap.add_argument("--retrieve-only", action="store_true", help="skip the LLM, show retrieved chunks only")
    args = ap.parse_args()

    index = VectorIndex.load()
    if args.question:
        show(answer_question(index, args.question, args.k, args.retrieve_only))
        return
    print("Ask a question (empty line or Ctrl+C to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            break
        try:
            show(answer_question(index, q, args.k, args.retrieve_only))
        except RuntimeError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
