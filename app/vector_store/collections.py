from app.core.config import settings
from app.shared.exceptions import VectorStoreError

_ensured: set[str] = set()


def ensure_collection(client, collection_name: str | None = None, vector_size: int | None = None) -> str:
    """Idempotent: creates the collection (Cosine distance, matching backend's
    QdrantIndexerProvider) only if it doesn't already exist, and only checks
    once per process. Returns the collection name used."""
    name = collection_name or settings.qdrant_collection
    size = vector_size or settings.embedding_vector_size

    if name in _ensured:
        return name

    try:
        from qdrant_client.models import Distance, VectorParams

        if not client.collection_exists(name):
            client.create_collection(name, vectors_config=VectorParams(size=size, distance=Distance.COSINE))
    except Exception as error:  # noqa: BLE001
        raise VectorStoreError(f"failed to ensure collection \"{name}\": {error}") from error

    _ensured.add(name)
    return name
