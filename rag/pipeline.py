"""
Main RAG pipeline: retrieve → generate.
"""
from rag.retriever import retrieve
from rag.llm_client import generate_answer
from config.settings import get_settings

settings = get_settings()


def query(user_query: str, top_k: int | None = None) -> dict:
    """
    Run the full RAG pipeline for a user query.
    Returns: { answer, sources, query }
    """
    chunks = retrieve(user_query, top_k=top_k or settings.top_k_results)

    if not chunks:
        return {
            "answer": "No relevant content found in the knowledge base.",
            "sources": [],
            "query": user_query,
        }

    answer = generate_answer(user_query, chunks)

    sources = [
        {"source": c["source"], "page_num": c["page_num"], "score": c["score"]}
        for c in chunks
    ]

    return {"answer": answer, "sources": sources, "query": user_query}
