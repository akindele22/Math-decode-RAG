from unittest.mock import patch

from rag import retriever


def test_get_model_falls_back_to_smaller_model():
    retriever._model = None
    calls = []

    def fake_sentence_transformer(model_name, device="cpu"):
        calls.append((model_name, device))
        if model_name == "all-MiniLM-L6-v2":
            raise RuntimeError("simulated failure")
        return object()

    with patch.object(retriever.settings, "embedding_model", "all-MiniLM-L6-v2"):
        with patch("rag.retriever.SentenceTransformer", side_effect=fake_sentence_transformer):
            model = retriever._get_model()

    assert model is not None
    assert calls[0][0] == "all-MiniLM-L6-v2"
    assert calls[1][0] == "paraphrase-MiniLM-L3-v2"
