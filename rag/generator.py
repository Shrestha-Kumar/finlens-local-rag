"""Generation: answer from the retrieved chunks with a local LLM (Ollama)."""
import requests

from . import config

SYSTEM_PROMPT = (
    "You are a careful assistant answering questions about financial documents. "
    "Use ONLY the numbered context passages provided. Cite the passages you used like [1] or [2][3]. "
    "Quote figures exactly as written, including units and currency. "
    "If the passages do not contain the answer, reply exactly: "
    "\"I could not find this in the provided documents.\" Never use outside knowledge."
)


def format_source(chunk: dict) -> str:
    page = f", p.{chunk['page']}" if chunk.get("page") else ""
    return f"{chunk['source']}{page}"


def build_prompt(question: str, chunks: list[dict]) -> str:
    blocks = [f"[{i}] ({format_source(c)})\n{c['text']}" for i, c in enumerate(chunks, start=1)]
    return ("Context passages:\n\n" + "\n\n".join(blocks) +
            f"\n\nQuestion: {question}\n\nAnswer (cite passages like [1]):")


def generate(question: str, chunks: list[dict]) -> str:
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": config.OLLAMA_NUM_CTX},
    }
    try:
        r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=config.OLLAMA_TIMEOUT)
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot reach Ollama at " + config.OLLAMA_URL +
                           ". Start it with `ollama serve` (or open the Ollama app).")
    if r.status_code == 404:
        raise RuntimeError(f"Model '{config.OLLAMA_MODEL}' not found. Run: ollama pull {config.OLLAMA_MODEL}")
    r.raise_for_status()
    return r.json()["message"]["content"].strip()
