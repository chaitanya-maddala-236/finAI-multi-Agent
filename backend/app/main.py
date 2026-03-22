from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.routes_chat import router as chat_router
from app.api.routes_market import router as market_router
from app.api.routes_portfolio import router as portfolio_router
from app.api.routes_upload import router as upload_router
from app.api.routes_user import router as user_router
from app.config import settings
from app.database.db import create_tables

logger = logging.getLogger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting up FinAI backend…")
    try:
        await create_tables()
        logger.info("Database tables created/verified.")
    except Exception as exc:
        logger.warning("DB setup failed (may be expected in test env): %s", exc)
    yield
    logger.info("Shutting down FinAI backend.")


# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="FinAI – Multi-Agent Financial Intelligence System",
        description=(
            "AI-powered financial advisory platform using LangGraph multi-agent orchestration. "
            "Provides portfolio analysis, market insights, risk profiling, and personalised investment advice."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Rate limiter state
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    prefix = "/api/v1"
    app.include_router(user_router, prefix=prefix)
    app.include_router(chat_router, prefix=prefix)
    app.include_router(upload_router, prefix=prefix)
    app.include_router(portfolio_router, prefix=prefix)
    app.include_router(market_router, prefix=prefix)

    # Health check
    @app.get("/health", tags=["health"])
    async def health_check() -> dict[str, Any]:
        return {
            "status": "healthy",
            "version": "1.0.0",
            "service": "FinAI Backend",
        }

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception on %s %s: %s", request.method, request.url, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected error occurred. Please try again later."},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
    )
