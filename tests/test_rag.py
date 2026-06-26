from unittest.mock import MagicMock

import pytest

from app.rag import embeddings as embeddings_module
from app.rag import pipeline as pipeline_module
from app.rag.chunking import chunk_text
from app.shared.exceptions import ValidationError


def test_chunk_text_empty_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_shorter_than_chunk_size_returns_one_chunk():
    assert chunk_text("hello world", chunk_size=1000, overlap=100) == ["hello world"]


def test_chunk_text_splits_with_overlap():
    text = "".join(chr(ord("a") + i % 26) for i in range(250))
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) == 3
    assert chunks[0] == text[0:100]
    assert chunks[1] == text[80:180]
    assert chunks[2] == text[160:250]
    # the overlapping tail of one chunk is the leading slice of the next
    assert chunks[0][-20:] == chunks[1][:20]
    assert chunks[1][-20:] == chunks[2][:20]


def test_embed_texts_empty_list_short_circuits(monkeypatch):
    calls = []
    monkeypatch.setattr(embeddings_module, "_model", lambda: calls.append("called") or MagicMock())

    assert embeddings_module.embed_texts([]) == []
    assert calls == []


def test_embed_texts_returns_vectors_from_model(monkeypatch):
    fake_vector = MagicMock()
    fake_vector.tolist.return_value = [0.1, 0.2, 0.3]
    fake_model = MagicMock()
    fake_model.embed.return_value = iter([fake_vector, fake_vector])
    monkeypatch.setattr(embeddings_module, "_model", lambda: fake_model)

    result = embeddings_module.embed_texts(["a", "b"])

    assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_index_document_raises_on_empty_text(monkeypatch):
    with pytest.raises(ValidationError):
        pipeline_module.index_document("company-1", "doc-1", "   ")


def test_index_document_upserts_one_point_per_chunk(monkeypatch):
    monkeypatch.setattr(pipeline_module, "embed_texts", lambda texts: [[0.0] * 3 for _ in texts])
    monkeypatch.setattr(pipeline_module, "embedding_model_name", lambda: "fake-model")
    fake_client = MagicMock()
    monkeypatch.setattr(pipeline_module, "get_client", lambda: fake_client)
    monkeypatch.setattr(pipeline_module, "ensure_collection", lambda client: "document_chunks")

    result = pipeline_module.index_document("company-1", "doc-1", "short text")

    assert result.chunk_count == 1
    assert result.model == "fake-model"
    fake_client.upsert.assert_called_once()
    _, kwargs = fake_client.upsert.call_args
    assert kwargs["collection_name"] == "document_chunks"
    assert len(kwargs["points"]) == 1


def test_answer_question_builds_context_from_sources(monkeypatch):
    from app.llm.base import LLMResult
    from app.rag.retriever import RetrievedChunk

    monkeypatch.setattr(pipeline_module, "embed_texts", lambda texts: [[0.0] * 3])
    monkeypatch.setattr(pipeline_module, "get_client", lambda: MagicMock())
    fake_source = RetrievedChunk(score=0.9, document_id="doc-1", chunk_index=0, content="relevant context")
    monkeypatch.setattr(pipeline_module, "search", lambda *args, **kwargs: [fake_source])

    captured_messages = {}

    class FakeProvider:
        def complete(self, messages, model, temperature, max_tokens):
            captured_messages["messages"] = messages
            return LLMResult(content="the answer", model_used=model, prompt_tokens=1, completion_tokens=1, total_tokens=2)

    monkeypatch.setattr(pipeline_module, "get_llm_provider", lambda model: FakeProvider())

    answer = pipeline_module.answer_question(
        company_id="company-1",
        question="what is it?",
        model="gpt-4o-mini",
        system_prompt=None,
        temperature=0.7,
        max_tokens=500,
        top_k=5,
    )

    assert answer.llm_result.content == "the answer"
    assert answer.sources == [fake_source]
    system_message = captured_messages["messages"][0]
    assert "relevant context" in system_message.content
