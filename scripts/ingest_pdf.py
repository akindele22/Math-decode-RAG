#!/usr/bin/env python3
"""
Ingest a PDF into the Math & Quant AI knowledge base.

Usage:
    python scripts/ingest_pdf.py path/to/document.pdf
"""
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.pdf_parser import parse_pdf
from ingestion.chunker import chunk_pages
from ingestion.embedder import embed_and_store


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_pdf.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}")
        sys.exit(1)

    print(f"\nIngesting: {pdf_path.name}")
    print("─" * 50)

    pages = parse_pdf(pdf_path)
    print(f"  Pages extracted : {len(pages)}")

    chunks = chunk_pages(pages)
    print(f"  Chunks created  : {len(chunks)}")

    count = embed_and_store(chunks)
    print(f"  Chunks stored   : {count}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
