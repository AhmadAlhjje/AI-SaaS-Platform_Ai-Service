import uuid
from dataclasses import dataclass

from app.llm.base import ChatMessage, LLMResult
from app.llm.factory import get_llm_provider
from app.rag.chunking import chunk_text
from app.rag.embeddings import embed_texts, embedding_model_name
from app.rag.retriever import RetrievedChunk, search
from app.shared.exceptions import ValidationError
from app.vector_store.collections import ensure_collection
from app.vector_store.qdrant_client import get_client

RAG_INSTRUCTION = "Answer using the context below when it is relevant. If it is not relevant, answer normally."


@dataclass(frozen=True)
class IndexResult:
    chunk_count: int
    model: str


@dataclass(frozen=True)
class RagAnswer:
    llm_result: LLMResult
    sources: list[RetrievedChunk]


def index_document(company_id: str, document_id: str, text: str) -> IndexResult:
    chunks = chunk_text(text)
    if not chunks:
        raise ValidationError("document text produced no chunks")

    vectors = embed_texts(chunks)
    client = get_client()
    collection = ensure_collection(client)

    from qdrant_client.models import PointStruct

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "companyId": company_id,
                "documentId": document_id,
                "chunkIndex": index,
                "content": content,
            },
        )
        for index, (content, vector) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=collection, points=points)

    return IndexResult(chunk_count=len(chunks), model=embedding_model_name())


def answer_question(
    company_id: str,
    question: str,
    model: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
    top_k: int,
) -> RagAnswer:
    [question_vector] = embed_texts([question])
    client = get_client()
    sources = search(client, company_id, question_vector, top_k)

    augmented_prompt = _build_system_prompt(system_prompt, sources)
    messages = [ChatMessage(role="system", content=augmented_prompt)] if augmented_prompt else []
    messages.append(ChatMessage(role="user", content=question))

    result = get_llm_provider(model).complete(messages, model, temperature, max_tokens)
    return RagAnswer(llm_result=result, sources=sources)


def _build_system_prompt(system_prompt: str | None, sources: list[RetrievedChunk]) -> str | None:
    if not sources:
        return system_prompt

    context = "\n\n".join(f"[{i + 1}] {chunk.content}" for i, chunk in enumerate(sources))
    parts = [p for p in (system_prompt, RAG_INSTRUCTION, f"Context:\n{context}") if p]
    return "\n\n".join(parts)
