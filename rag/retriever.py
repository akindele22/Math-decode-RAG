"""
Retrieve top-k relevant chunks for a user query.
"""
from typing import Any, Optional

from sentence_transformers import SentenceTransformer
import chromadb
from config.settings import get_settings

settings = get_settings()

_model: Optional[SentenceTransformer] = None
_client: Optional[Any] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        candidate_models = [settings.embedding_model, "all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2"]
        last_error: Exception | None = None

        for model_name in candidate_models:
            try:
                _model = SentenceTransformer(model_name, device="cpu")
                break
            except Exception as exc:  # pragma: no cover - exercised in runtime environments
                last_error = exc

        if _model is None:
            raise RuntimeError(
                f"Unable to load embedding model. Tried: {', '.join(candidate_models)}"
            ) from last_error
    return _model


def _get_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client.get_or_create_collection(settings.chroma_collection)


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """
    Returns top-k chunks most relevant to the query.
    Each result: { text, source, page_num, score }
    """
    k = top_k or settings.top_k_results
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": text,
            "source": meta.get("source", ""),
            "page_num": meta.get("page_num", 0),
            "score": round(1 - dist, 4),  # cosine similarity
        })

    return chunks
