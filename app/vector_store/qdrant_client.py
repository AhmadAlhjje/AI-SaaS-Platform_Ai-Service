from functools import lru_cache

from app.core.config import settings
from app.shared.exceptions import VectorStoreError


@lru_cache(maxsize=1)
def get_client():
    try:
        from qdrant_client import QdrantClient
    except ImportError as error:
        raise VectorStoreError(f"qdrant-client is not installed: {error}") from error

    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
