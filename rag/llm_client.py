"""
LLM client for Math & Quant AI.

Provider priority:
  Development  →  Ollama  (local, free, no API key)
  Production   →  Gemini  (fast, cheap cloud)
  Fallback     →  Anthropic | OpenAI

Numbered chunk context gives the LLM precise citation anchors.
"""
from __future__ import annotations

from config.settings import get_settings

settings = get_settings()

SYSTEM_PROMPT = (
    "You are a world-class quantitative finance professor and mathematician. "
    "Answer the user's question using ONLY the numbered context chunks provided. "
    "Cite every claim with its chunk number, e.g. [Chunk 2 | source.pdf, p.14]. "
    "If the answer is not present in the context, say so explicitly — do not guess. "
    "Format all mathematical expressions using LaTeX notation."
)


# ── Context formatting ─────────────────────────────────────────────────────────

def format_context(chunks: list[dict]) -> str:
    """
    Build a numbered context block from retrieved chunks.
    Each entry is labelled so the LLM can cite precisely.

    Output example:
        [Chunk 1 | finance.pdf, p.42]
        The Sharpe Ratio measures risk-adjusted return...

        [Chunk 2 | stochastic.pdf, p.7]
        Under the risk-neutral measure Q...
    """
    parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        header = f"[Chunk {idx} | {chunk['source']}, p.{chunk['page_num']}]"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


# ── LLM provider factory ───────────────────────────────────────────────────────

def _get_ollama_response(user_message: str) -> str:
    """Run inference via local Ollama (llama3.2 by default)."""
    from langchain_ollama import ChatOllama
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    return response.content


def _get_gemini_response(user_message: str) -> str:
    """Run inference via Google Gemini API."""
    if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_key_here":
        raise ValueError(
            "Missing GEMINI_API_KEY in .env. "
            "Get one free at https://aistudio.google.com/apikey "
            "or switch to LLM_PROVIDER=ollama for local use."
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    return response.content


def _get_anthropic_response(user_message: str) -> str:
    """Run inference via Anthropic Claude API (cloud fallback)."""
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    message = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text


def _get_openai_response(user_message: str) -> str:
    """Run inference via OpenAI API (cloud fallback)."""
    import openai

    client = openai.OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=2048,
    )
    return response.choices[0].message.content


# ── Public interface ───────────────────────────────────────────────────────────

_PROVIDERS = {
    "ollama":    _get_ollama_response,
    "gemini":    _get_gemini_response,
    "anthropic": _get_anthropic_response,
    "openai":    _get_openai_response,
}


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """
    Generate an answer using the configured LLM provider.

    Args:
        query:          The user's question.
        context_chunks: Retrieved chunks from the vector store.

    Returns:
        LLM-generated answer string with inline chunk citations.
    """
    provider = settings.llm_provider.strip().lower()

    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Choose one of: {', '.join(_PROVIDERS)}"
        )

    context = format_context(context_chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    return _PROVIDERS[provider](user_message)

