"""
PDF parsing for math-heavy documents.
Uses PyMuPDF for text + pdfplumber as fallback for complex layouts.
"""
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ParsedPage:
    page_num: int
    text: str
    source: str


def parse_pdf(path: str | Path) -> list[ParsedPage]:
    """
    Extract text from each page of a PDF.
    Tries PyMuPDF first; falls back to pdfplumber for scanned/complex pages.
    """
    path = Path(path)
    pages: list[ParsedPage] = []

    doc = fitz.open(str(path))
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()

        if not text:
            # Fallback: pdfplumber (better for table-heavy pages)
            with pdfplumber.open(str(path)) as pdf:
                if i < len(pdf.pages):
                    text = pdf.pages[i].extract_text() or ""

        if text:
            pages.append(ParsedPage(page_num=i + 1, text=text, source=path.name))

    doc.close()
    return pages
