from app.core.logging import get_logger
from app.llm.base import ChatMessage
from app.llm.factory import get_llm_provider

logger = get_logger(__name__)

RAG = "RAG"
SQL_AGENT = "SQL_AGENT"
HUMAN = "HUMAN"

SYSTEM_PROMPT_TEMPLATE = """You decide which backend should handle a customer's question. Respond with a \
single JSON object only, no prose, no markdown fences: {{"route": "<ROUTE>"}}.

Available routes for this request: {available_routes}.

Route meanings:
- RAG: answer by searching this company's uploaded documents (free text — FAQs, policies, product \
descriptions, prices, specs, anything written in them). Use RAG whenever the answer could plausibly be \
found in free-text content, regardless of the question's topic — a price or product question is still \
RAG if no structured data table is available, as long as documents are.
- SQL_AGENT: answer by querying this company's structured data table (rows with columns — counts, \
filters, lookups over tabular records such as inventory, orders, customers).
- HUMAN: only for complaints, account-specific issues, refund/order disputes, or something clearly \
impossible for any available source to answer — not merely a topic that "sounds like" an unavailable \
route.

Rules:
- Only ever pick a route from the available list above.
- Strongly prefer an available RAG or SQL_AGENT route over HUMAN. HUMAN is a last resort for when the \
available sources genuinely cannot help, never a way to pick the "ideal" route when it isn't available.
- When both RAG and SQL_AGENT are available, prefer SQL_AGENT for precise structured lookups (stock, \
counts, filtering by attributes) and RAG for everything else."""


def decide_route(question: str, model: str, has_documents: bool, has_data_tables: bool) -> str:
    available = [r for r, available in ((RAG, has_documents), (SQL_AGENT, has_data_tables)) if available]
    available.append(HUMAN)

    if not has_documents and not has_data_tables:
        # Nothing to draw an answer from — skip the LLM call entirely.
        return HUMAN

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(available_routes=", ".join(available))

    try:
        parsed, _ = get_llm_provider(model).complete_json(
            messages=[ChatMessage(role="system", content=system_prompt), ChatMessage(role="user", content=question)],
            model=model,
            temperature=0.0,
            max_tokens=50,
        )
    except Exception as error:  # noqa: BLE001 - routing must never crash the request, fail safe to HUMAN
        logger.warning("question routing failed, defaulting to HUMAN: %s", error)
        return HUMAN

    route = str(parsed.get("route", "")).strip().upper()
    if route not in available:
        logger.warning('router picked unavailable/unknown route "%s", defaulting to HUMAN', route)
        return HUMAN

    return route
