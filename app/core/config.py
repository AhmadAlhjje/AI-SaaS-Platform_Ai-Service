from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    port: int = 8000
    log_level: str = "info"

    # LLM providers — model id prefix decides which one handles a request
    # (see app/llm/factory.py). Keys are optional at startup so the service
    # still boots without them; a request that needs a missing key fails
    # with a clear 502, not a crash.
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    # Local inference via Ollama (https://ollama.com) — no API key, runs on
    # this machine. Requires `ollama serve` running and the model pulled
    # (e.g. `ollama pull llama3.1:8b`).
    ollama_base_url: str = "http://localhost:11434/v1"
    llm_request_timeout_s: float = 60.0
    llm_max_retries: int = 2

    # Embedding model — local, free, no API key. Dimension must match
    # backend's QDRANT_VECTOR_SIZE (defaults to 384 on both sides).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_vector_size: int = 384

    # ai-service's own Qdrant access for the standalone RAG pipeline
    # (app/rag, app/vector_store). Independent of the backend's own
    # indexing path but points at the same collection by default.
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "document_chunks"

    # SQL agent — generates a structured QueryPlan (filters + limit), never
    # raw SQL text. This caps how large a plan's limit can be regardless of
    # what the LLM returns.
    sql_agent_max_limit: int = 100

    class Config:
        env_file = ".env"


settings = Settings()
