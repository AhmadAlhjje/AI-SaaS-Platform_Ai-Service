# Support OTP — AI Service

FastAPI gateway that the NestJS backend calls over HTTP for everything LLM-related (ROLE.md §8 — "AI Service تعتبر External Gateway"). It never touches the product database; it only ever reasons about text/schema it's given.

## Setup

```bash
cd ai-service
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # then fill in OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

## Endpoints

| Method | Path | Used by | Purpose |
| --- | --- | --- | --- |
| GET | `/api/v1/health` | — | Liveness check |
| POST | `/api/v1/ask` | backend `HttpAiProvider` | Plain chat completion (model picked by id prefix) |
| POST | `/api/v1/embeddings` | backend `HttpEmbeddingsProvider` | Real local embeddings (fastembed, 384-dim) |
| POST | `/api/v1/route` | backend `HttpQuestionRouterProvider` | Decides RAG vs SQL_AGENT vs HUMAN for a question, given which sources the company actually has |
| POST | `/api/v1/sql-agent/query` | backend `HttpSqlQueryGeneratorProvider` | NL question + table column schema → structured `QueryPlan` (filters + limit), never raw SQL |
| POST | `/api/v1/rag/index` | standalone | Chunk + embed + upsert a document's text into Qdrant |
| POST | `/api/v1/rag/query` | standalone | Embed question → retrieve from Qdrant → answer with LLM + context |

Note: the backend does its own Qdrant retrieval for the RAG route (`QdrantRetrieverProvider`) rather than calling `/rag/query` — `/rag/*` stays available as a self-contained alternative pipeline. `/route` and `/sql-agent/query` are both live in the conversation flow today (`AskQuestionUseCase` → `RouteQuestionUseCase` / `RunSqlAgentQueryUseCase`).

## Model selection

Model id prefix decides the provider (`app/llm/factory.py`):

- `gpt-*`, `o1*`, `o3*`, `o4*`, `chatgpt-*` → OpenAI (`OPENAI_API_KEY`)
- `claude-*` → Anthropic (`ANTHROPIC_API_KEY`)
- `deepseek-*` → DeepSeek (`DEEPSEEK_API_KEY`) — OpenAI-compatible API, much cheaper than OpenAI/Anthropic; a good low-cost default (e.g. `deepseek-chat`)

A request for a model with no configured key, or an unrecognized prefix, returns `502` — not a crash.

## Embeddings

`fastembed` with `BAAI/bge-small-en-v1.5` (384 dimensions) — local, deterministic-given-input, no API key, no network call. Must stay in sync with backend's `QDRANT_VECTOR_SIZE`.

## SQL Agent — why no raw SQL

Backend's dynamic tables are EAV-style (`DataTableRow.rowData` JSON), not real SQL tables — `ExecuteSqlQueryUseCase` only ever applies a parameterized JSON-path filter, after re-validating every filter's column against the table's own whitelist. So this service generates a `QueryPlan` (`{filters, limit}`), validated against the caller-supplied column list and a fixed operator set — never a SQL string. This makes SQL injection structurally impossible regardless of what the LLM returns.

## Tests

```bash
pytest
```

Tests mock the LLM/embedding/Qdrant clients — no API key or network access needed to run them.
