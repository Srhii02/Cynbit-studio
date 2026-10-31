"""Document parsing: PDF (PyMuPDF), DOCX (python-docx) and plain text -> clean text."""
from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class ParsingError(ValueError):
    """Raised when a document cannot be parsed into usable text."""


@dataclass
class ParsedDocument:
    filename: str
    text: str
    pages: int
    file_type: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


_BULLETS = "•●▪■◦‣∙·►▶✓✔➢➤-–—*"
_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}


def clean_text(text: str) -> str:
    """Normalise unicode, ligatures, bullets and whitespace while keeping line structure."""
    if not text:
        return ""
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    lines = []
    for raw in text.split("\n"):
        line = re.sub(r"[  ]+", " ", raw).strip()
        # Normalise any leading bullet glyph to "- "
        line = re.sub(rf"^[{re.escape(_BULLETS)}]+\s*", "- ", line) if line[:1] in _BULLETS and len(line) > 2 else line
        lines.append(line)
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _parse_pdf(data: bytes) -> tuple[str, int]:
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24
    except ImportError:  # pragma: no cover
        import fitz

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # corrupted / not a PDF
        raise ParsingError(f"Could not open PDF: {exc}") from exc
    with doc:
        if doc.needs_pass:
            raise ParsingError("PDF is password protected.")
        # sort=True gives natural reading order for multi-column layouts
        text = "\n".join(page.get_text("text", sort=True) for page in doc)
        return text, doc.page_count


def _parse_docx(data: bytes) -> tuple[str, int]:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise ParsingError(f"Could not open DOCX: {exc}") from exc
    parts: list[str] = [p.text for p in document.paragraphs]
    for table in document.tables:  # many resume templates put content in tables
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(dict.fromkeys(cells)))
    return "\n".join(parts), 1


def parse_bytes(data: bytes, filename: str) -> ParsedDocument:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ParsingError(f"Unsupported file type '{ext}'. Use PDF, DOCX or TXT.")
    if not data:
        raise ParsingError("File is empty.")
    if ext == ".pdf":
        raw, pages = _parse_pdf(data)
    elif ext == ".docx":
        raw, pages = _parse_docx(data)
    else:
        raw, pages = data.decode("utf-8", errors="ignore"), 1
    text = clean_text(raw)
    if len(text.split()) < 20:
        raise ParsingError(
            "Very little text was extracted. If this is a scanned/image PDF, export it as a text PDF or DOCX."
        )
    return ParsedDocument(filename=filename, text=text, pages=pages, file_type=ext.lstrip("."))


def parse_file(path: str | Path) -> ParsedDocument:
    p = Path(path)
    return parse_bytes(p.read_bytes(), p.name)
