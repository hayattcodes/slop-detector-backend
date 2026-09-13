"""
Adapted from Repo 5 (IviweBooi)'s FileParserFactory pattern: one function
per supported format, all normalized down to plain text before engines
ever see it.
"""
import io

from pypdf import PdfReader
from docx import Document

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


class UnsupportedFileType(Exception):
    pass


def parse_file(filename: str, content: bytes) -> str:
    lower = filename.lower()

    if lower.endswith(".txt"):
        return content.decode("utf-8", errors="ignore")

    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if lower.endswith(".docx"):
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)

    raise UnsupportedFileType(
        f"'{filename}' isn't supported yet. Upload a .txt, .pdf, or .docx file."
    )
