"""Pipeline: one function shared by the CLI and the web UI."""
from . import config
from .generator import generate
from .retriever import retrieve


def answer_question(index, question: str, k: int = config.TOP_K, retrieve_only: bool = False) -> dict:
    chunks = retrieve(index, question, k)
    answer = None if retrieve_only else generate(question, chunks)
    return {"answer": answer, "sources": chunks}
