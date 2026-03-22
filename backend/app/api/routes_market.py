from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query

from app.services.news_service import NewsService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/market-insights", tags=["market"])

news_service = NewsService()


@router.get("")
async def get_market_insights() -> dict[str, Any]:
    """Return market trends, recent news, and overall sentiment."""
    news, sentiment = await _fetch_news_and_sentiment()
    return {
        "sentiment": sentiment,
        "news": news[:10],
        "trending_topics": _extract_topics(news),
    }


@router.get("/symbols")
async def get_symbol_data(
    symbols: str = Query(
        default="AAPL,GOOGL,MSFT",
        description="Comma-separated list of ticker symbols",
    )
) -> dict[str, Any]:
    """Return price and volume data for requested ticker symbols."""
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        return {"symbols": {}}
    market_data = await news_service.fetch_market_data(symbol_list)
    return {"symbols": market_data}


@router.get("/sentiment")
async def get_market_sentiment() -> dict[str, Any]:
    """Return standalone market sentiment analysis."""
    return await news_service.get_market_sentiment()


@router.get("/news")
async def get_financial_news(
    query: str = Query(default="financial markets", description="Search query for news"),
) -> dict[str, Any]:
    """Return financial news articles for the given query."""
    articles = await news_service.fetch_financial_news(query)
    return {"query": query, "articles": articles, "count": len(articles)}


# ── helpers ───────────────────────────────────────────────────────────────────

async def _fetch_news_and_sentiment():
    import asyncio
    news, sentiment = await asyncio.gather(
        news_service.fetch_financial_news("financial markets economy"),
        news_service.get_market_sentiment(),
    )
    return news, sentiment


def _extract_topics(news: list[dict[str, Any]]) -> list[str]:
    keywords = [
        "inflation", "interest rates", "GDP", "earnings", "AI", "tech",
        "gold", "oil", "crypto", "Fed", "RBI", "NIFTY", "S&P 500",
    ]
    titles = " ".join(n.get("title", "") + " " + n.get("description", "") for n in news).lower()
    return [kw for kw in keywords if kw.lower() in titles]
