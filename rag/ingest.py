"""Ingestion: read PDFs / text files into page-level records."""
import re
from pathlib import Path

from pypdf import PdfReader

MIN_PAGE_WORDS = 20  # skip blank / image-only pages


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)     # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> list[dict]:
    pages = []
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if len(text.split()) >= MIN_PAGE_WORDS:
            pages.append({"source": path.name, "page": i, "text": text})
    if not pages:
        print(f"  ! No extractable text in {path.name} (scanned PDF? it would need OCR).")
    return pages


def load_text(path: Path) -> list[dict]:
    text = _clean(path.read_text(encoding="utf-8", errors="ignore"))
    if len(text.split()) < MIN_PAGE_WORDS:
        return []
    return [{"source": path.name, "page": None, "text": text}]


def load_documents(data_dir: Path) -> list[dict]:
    """Return a list of {source, page, text} for every supported file in data_dir."""
    pages: list[dict] = []
    files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"})
    if not files:
        raise FileNotFoundError(f"No .pdf/.txt/.md files found in {data_dir}. Add some documents first.")
    for path in files:
        loaded = load_pdf(path) if path.suffix.lower() == ".pdf" else load_text(path)
        print(f"  {path.name}: {len(loaded)} page(s) with text")
        pages.extend(loaded)
    return pages
