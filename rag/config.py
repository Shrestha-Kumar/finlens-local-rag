"""Configuration. Every tunable setting lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"          # put your PDFs / .txt files here
INDEX_DIR = ROOT / "index"                # built index is saved here
EVAL_FILE = ROOT / "eval" / "questions.json"

# --- Embeddings (runs locally via sentence-transformers) ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"    # ~130 MB; alternative: "sentence-transformers/all-MiniLM-L6-v2"
# bge models work better when queries (not documents) get this prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- Chunking (measured in words) ---
CHUNK_WORDS = 350
CHUNK_OVERLAP = 60
MIN_CHUNK_WORDS = 30                      # drop tiny fragments (page numbers, headers)

# --- Retrieval ---
TOP_K = 5

# --- Generation (local LLM served by Ollama) ---
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"              # try "qwen2.5:3b" or "llama3.1:8b" if you have the RAM/GPU
OLLAMA_NUM_CTX = 4096                     # context window; must fit TOP_K chunks + question
OLLAMA_TIMEOUT = 300                      # seconds
