"""
Smart text chunker for Math & Quant AI.

Strategy:
  - Short PDF pages (content already fits in one chunk) → kept as-is.
    Splitting mid-derivation or mid-theorem breaks retrieval quality.
  - Long pages and text documents → split with overlap via LangChain.

This matches the approach from the workshop RAG pipeline and is especially
important for math textbooks where a single page often contains one proof.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.pdf_parser import ParsedPage
from config.settings import get_settings

settings = get_settings()


def _is_short_page(text: str) -> bool:
    """
    Return True if the page text is short enough to keep as a single chunk.
    Threshold: chunk_size × short_page_factor (default: 800 × 1.5 = 1200 chars).
    Avoids splitting proofs, derivations, and theorem statements mid-way.
    """
    max_chars = int(settings.chunk_size * settings.short_page_factor)
    return len(text) <= max_chars


def chunk_pages(pages: list[ParsedPage]) -> list[dict]:
    """
    Split parsed PDF pages into retrieval-ready chunks.

    Short pages  → 1 chunk each (no splitting).
    Long pages   → split with RecursiveCharacterTextSplitter.

    Each chunk dict contains:
        text       : str   — the chunk content
        source     : str   — filename
        page_num   : int   — original page number
        chunk_id   : str   — unique identifier for ChromaDB upsert
        is_full_page: bool — True if the chunk is an unsplit page
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []

    for page in pages:
        text = page.text.strip()
        if not text:
            continue

        if _is_short_page(text):
            # Keep the whole page as one chunk — preserves mathematical context
            chunks.append({
                "text": text,
                "source": page.source,
                "page_num": page.page_num,
                "chunk_id": f"{page.source}_p{page.page_num}_full",
                "is_full_page": True,
            })
        else:
            # Split long pages normally
            splits = splitter.split_text(text)
            for idx, split in enumerate(splits):
                split = split.strip()
                if split:
                    chunks.append({
                        "text": split,
                        "source": page.source,
                        "page_num": page.page_num,
                        "chunk_id": f"{page.source}_p{page.page_num}_c{idx}",
                        "is_full_page": False,
                    })

    return chunks

