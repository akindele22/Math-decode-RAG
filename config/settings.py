from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── LLM provider ──────────────────────────────────────────────────────────
    # Supported values: "ollama" | "gemini" | "anthropic" | "openai"
    # Default: ollama (free, local, no API key required)
    llm_provider: str = "ollama"

    # Ollama (local — primary for development)
    ollama_model: str = "deepseek-r1:14b"
    ollama_base_url: str = "http://localhost:11434"

    # Gemini (cloud — primary for production)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Anthropic (cloud — fallback)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # OpenAI (cloud — fallback)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "paraphrase-MiniLM-L3-v2"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "math_quant"

    # ── MySQL (metadata store) ────────────────────────────────────────────────
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_db: str = "math_quant_ai"

    # ── API ───────────────────────────────────────────────────────────────────
    api_secret_key: str = "change_me"
    api_host: str = "0.0.0.0"
    api_port: int = 7000

    # ── RAG ───────────────────────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 100
    # Pages shorter than chunk_size * this factor skip splitting (smart chunking)
    short_page_factor: float = 1.5
    top_k_results: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
