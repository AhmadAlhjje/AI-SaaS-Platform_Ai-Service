from functools import lru_cache

from app.core.config import settings
from app.shared.exceptions import EmbeddingError


@lru_cache(maxsize=1)
def _model():
    # Imported lazily so a missing/slow-to-import onnxruntime doesn't block
    # app startup or unrelated routes (ask/health) from working.
    try:
        from fastembed import TextEmbedding
    except ImportError as error:
        raise EmbeddingError(f"fastembed is not installed: {error}") from error

    try:
        return TextEmbedding(model_name=settings.embedding_model)
    except Exception as error:  # noqa: BLE001 - model load can fail in many SDK-specific ways
        raise EmbeddingError(f"failed to load embedding model \"{settings.embedding_model}\": {error}") from error


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Real, deterministic, local embeddings — no network call, no API key.
    Vector dimension matches settings.embedding_vector_size (384 by default),
    which must stay in sync with backend's QDRANT_VECTOR_SIZE."""
    if not texts:
        return []

    try:
        vectors = list(_model().embed(texts))
    except EmbeddingError:
        raise
    except Exception as error:  # noqa: BLE001
        raise EmbeddingError(f"failed to encode {len(texts)} text(s): {error}") from error

    return [vector.tolist() for vector in vectors]


def embedding_model_name() -> str:
    return settings.embedding_model
