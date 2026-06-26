import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.v1.routes import ask, embeddings, health, rag, route_question, sql_agent
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.shared.exceptions import AiServiceError

configure_logging()
logger = get_logger(__name__)

app = FastAPI(title="Support OTP AI Service")

app.include_router(health.router, prefix="/api/v1")
app.include_router(ask.router, prefix="/api/v1")
app.include_router(embeddings.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")
app.include_router(sql_agent.router, prefix="/api/v1")
app.include_router(route_question.router, prefix="/api/v1")


@app.exception_handler(AiServiceError)
def handle_ai_service_error(_request: Request, exc: AiServiceError) -> JSONResponse:
    logger.warning("%s: %s", type(exc).__name__, exc.message)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, log_level=settings.log_level)
