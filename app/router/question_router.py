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
- SQL_AGENT: answer by querying this company's structured data table — rows with named columns (price, \
quantity, plan name, max count, date, status, etc.). Pick SQL_AGENT whenever the question asks for a \
specific value, number, name, or comparison that a spreadsheet row/column could hold — e.g. "how much does \
X cost", "what is the limit for Y", "which plan has Z", "what's the difference between A and B". This \
includes prices and plan/product attributes when a data table is available — do NOT assume prices belong \
to RAG.
- RAG: answer by searching this company's uploaded free-text documents (policies, FAQs, narrative \
descriptions, instructions — content that is written in prose, not stored as table rows).
- HUMAN: only for complaints, account-specific issues, refund/order disputes, or something clearly \
impossible for any available source to answer — not merely a topic that "sounds like" an unavailable \
route.

Rules:
- Only ever pick a route from the available list above.
- Strongly prefer an available SQL_AGENT or RAG route over HUMAN. HUMAN is a last resort for when the \
available sources genuinely cannot help, never a way to pick the "ideal" route when it isn't available.
- When both SQL_AGENT and RAG are available, default to SQL_AGENT for any question about a specific \
value, attribute, count, or comparison (prices, limits, plan names, quantities). Only choose RAG when the \
question is clearly about free-text narrative content (a policy, a procedure, an explanation) that would \
not be stored as table rows.

Examples:
- "كم سعر الخطة المتقدمة شهرياً؟" with SQL_AGENT available -> SQL_AGENT (a specific price lookup).
- "ما هي سياسة الاسترجاع؟" with RAG available -> RAG (a free-text policy explanation).
- "ما الفرق بين الخطة المجانية والأساسية؟" with SQL_AGENT available -> SQL_AGENT (comparing row attributes).
- "ما هي ساعات العمل؟" with RAG available -> RAG (free-text fact, not tabular)."""


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
