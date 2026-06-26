class AiServiceError(Exception):
    """Base class for errors this service raises deliberately (as opposed to
    unexpected bugs) — always caught by the handlers registered in main.py
    and turned into a JSON error response with an appropriate status code."""

    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class LLMProviderError(AiServiceError):
    """The configured LLM provider (OpenAI/Anthropic) could not complete the
    request — missing API key, network failure, rate limit, bad model id."""

    status_code = 502


class EmbeddingError(AiServiceError):
    """The embedding model failed to load or encode the given input."""

    status_code = 502


class VectorStoreError(AiServiceError):
    """Qdrant could not be reached or the operation failed."""

    status_code = 502


class ValidationError(AiServiceError):
    """Caller-supplied input failed a domain check that pydantic's schema
    validation can't express (e.g. an empty column list)."""

    status_code = 400
