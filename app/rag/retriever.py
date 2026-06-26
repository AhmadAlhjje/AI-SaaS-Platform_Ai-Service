from dataclasses import dataclass

from app.core.config import settings
from app.shared.exceptions import VectorStoreError


@dataclass(frozen=True)
class RetrievedChunk:
    score: float
    document_id: str
    chunk_index: int
    content: str


def search(client, company_id: str, vector: list[float], top_k: int, collection_name: str | None = None) -> list[RetrievedChunk]:
    """Company-scoped similarity search. Returns [] when the collection
    doesn't exist yet (nothing indexed) — that's "no context", not an error,
    matching backend's QdrantRetrieverProvider behavior."""
    name = collection_name or settings.qdrant_collection

    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if not client.collection_exists(name):
            return []

        results = client.query_points(
            collection_name=name,
            query=vector,
            limit=top_k,
            query_filter=Filter(must=[FieldCondition(key="companyId", match=MatchValue(value=company_id))]),
            with_payload=True,
        ).points
    except Exception as error:  # noqa: BLE001
        raise VectorStoreError(f"search failed: {error}") from error

    return [
        RetrievedChunk(
            score=point.score,
            document_id=point.payload.get("documentId", ""),
            chunk_index=point.payload.get("chunkIndex", 0),
            content=point.payload.get("content", ""),
        )
        for point in results
    ]
