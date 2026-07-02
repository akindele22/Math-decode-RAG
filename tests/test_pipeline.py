
"""
Smoke tests for the RAG pipeline (mocked external calls).
Run with: pytest tests/
"""
import pytest
from unittest.mock import patch, MagicMock

from rag.pipeline import query


MOCK_CHUNKS = [
    {
        "text": "The Sharpe Ratio measures risk-adjusted return.",
        "source": "finance_textbook.pdf",
        "page_num": 42,
        "score": 0.91,
    }
]


@patch("rag.pipeline.retrieve", return_value=MOCK_CHUNKS)
@patch("rag.pipeline.generate_answer", return_value="The Sharpe Ratio is S = (Rp - Rf) / σp.")
def test_pipeline_returns_expected_keys(mock_gen, mock_ret):
    result = query("What is the Sharpe Ratio?")
    assert "answer" in result
    assert "sources" in result
    assert "query" in result


@patch("rag.pipeline.retrieve", return_value=[])
def test_pipeline_empty_knowledge_base(mock_ret):
    result = query("What is the Sharpe Ratio?")
    assert "No relevant content" in result["answer"]
    assert result["sources"] == []


@patch("rag.pipeline.retrieve", return_value=MOCK_CHUNKS)
@patch("rag.pipeline.generate_answer", return_value="Mock answer.")
def test_pipeline_sources_have_required_keys(mock_gen, mock_ret):
    result = query("Explain Black-Scholes")
    for source in result["sources"]:
        assert "source" in source
        assert "page_num" in source
        assert "score" in source


