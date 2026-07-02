"""
Basic smoke tests for the ingestion pipeline.
Run with: pytest tests/
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ingestion.chunker import chunk_pages
from ingestion.pdf_parser import ParsedPage


def make_pages(n: int) -> list[ParsedPage]:
    return [
        ParsedPage(page_num=i + 1, text=f"This is test content for page {i + 1}. " * 20, source="test.pdf")
        for i in range(n)
    ]


def test_chunker_produces_chunks():
    pages = make_pages(3)
    chunks = chunk_pages(pages)
    assert len(chunks) > 0


def test_chunker_chunk_has_required_keys():
    pages = make_pages(1)
    chunks = chunk_pages(pages)
    for chunk in chunks:
        assert "text" in chunk
        assert "source" in chunk
        assert "page_num" in chunk
        assert "chunk_id" in chunk


def test_chunker_chunk_ids_are_unique():
    pages = make_pages(5)
    chunks = chunk_pages(pages)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


def test_chunker_preserves_source():
    pages = make_pages(2)
    chunks = chunk_pages(pages)
    for chunk in chunks:
        assert chunk["source"] == "test.pdf"
