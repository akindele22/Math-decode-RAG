"""
Embed chunks and store them in ChromaDB.
"""
from typing import Any, Optional

from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
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


def embed_and_store(chunks: list[dict], batch_size: int = 64) -> int:
    """
    Embed each chunk and upsert into ChromaDB.
    Returns the number of chunks stored.
    """
    model = _get_model()
    collection = _get_collection()

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {"source": c["source"], "page_num": c["page_num"]}
        for c in chunks
    ]

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
        collection.upsert(
            ids=ids[i : i + batch_size],
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=metadatas[i : i + batch_size],
        )

    return len(texts)
