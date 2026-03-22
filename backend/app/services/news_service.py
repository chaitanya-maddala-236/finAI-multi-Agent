from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

MOCK_NEWS: list[dict[str, Any]] = [
    {
        "title": "Fed signals cautious approach to rate cuts amid sticky inflation",
        "description": "Federal Reserve officials indicated they are in no hurry to cut interest rates as inflation remains above the 2% target.",
        "source": "Reuters",
        "published_at": "2024-03-15T10:00:00Z",
        "sentiment": "neutral",
    },
    {
        "title": "Tech stocks rally as AI investment boom continues",
        "description": "Major technology companies saw strong gains as investors bet on AI-driven revenue growth.",
        "source": "Bloomberg",
        "published_at": "2024-03-15T08:30:00Z",
        "sentiment": "positive",
    },
    {
        "title": "Emerging market currencies face pressure from strong dollar",
        "description": "The US dollar's strength is putting pressure on emerging market currencies and assets.",
        "source": "Financial Times",
        "published_at": "2024-03-14T14:00:00Z",
        "sentiment": "negative",
    },
    {
        "title": "Gold prices hit new highs amid geopolitical uncertainty",
        "description": "Gold surged past $2,100 per ounce as investors sought safe-haven assets.",
        "source": "CNBC",
        "published_at": "2024-03-14T11:00:00Z",
        "sentiment": "positive",
    },
    {
        "title": "India's GDP growth remains robust at 8.4% in Q3",
        "description": "India's economy continues to outperform expectations, driven by domestic consumption and infrastructure spending.",
        "source": "Economic Times",
        "published_at": "2024-03-13T09:00:00Z",
        "sentiment": "positive",
    },
]

MOCK_MARKET_DATA: dict[str, dict[str, Any]] = {
    "AAPL": {"symbol": "AAPL", "price": 178.50, "change": 1.25, "change_pct": 0.70, "volume": 55_000_000},
    "GOOGL": {"symbol": "GOOGL", "price": 141.80, "change": -0.90, "change_pct": -0.63, "volume": 22_000_000},
    "MSFT": {"symbol": "MSFT", "price": 415.30, "change": 3.10, "change_pct": 0.75, "volume": 18_000_000},
    "AMZN": {"symbol": "AMZN", "price": 178.25, "change": 2.45, "change_pct": 1.39, "volume": 30_000_000},
    "TSLA": {"symbol": "TSLA", "price": 197.60, "change": -5.30, "change_pct": -2.61, "volume": 90_000_000},
    "NIFTY50": {"symbol": "NIFTY50", "price": 22_500.0, "change": 120.0, "change_pct": 0.54, "volume": 0},
}


class NewsService:
    """Service for fetching financial news and market data."""

    def __init__(self) -> None:
        self._newsapi_client = None
        if settings.NEWS_API_KEY:
            try:
                from newsapi import NewsApiClient  # type: ignore[import-untyped]
                self._newsapi_client = NewsApiClient(api_key=settings.NEWS_API_KEY)
            except Exception:
                logger.warning("newsapi-python not available; falling back to mock news.")

    async def fetch_financial_news(self, query: str = "financial markets") -> list[dict[str, Any]]:
        """Fetch financial news articles. Falls back to mock data when API key is absent."""
        if self._newsapi_client:
            try:
                response = self._newsapi_client.get_everything(
                    q=query,
                    language="en",
                    sort_by="publishedAt",
                    page_size=10,
                )
                articles = response.get("articles", [])
                return [
                    {
                        "title": a.get("title", ""),
                        "description": a.get("description", ""),
                        "source": a.get("source", {}).get("name", "Unknown"),
                        "published_at": a.get("publishedAt", ""),
                        "url": a.get("url", ""),
                        "sentiment": "neutral",
                    }
                    for a in articles
                ]
            except Exception as exc:
                logger.warning("NewsAPI request failed (%s); using mock data.", exc)

        return MOCK_NEWS

    async def fetch_market_data(self, symbols: list[str]) -> dict[str, Any]:
        """Fetch live price data for the given ticker symbols via yfinance."""
        result: dict[str, Any] = {}
        if not symbols:
            return result

        if settings.ALPHA_VANTAGE_API_KEY or True:
            try:
                import yfinance as yf  # type: ignore[import-untyped]

                for symbol in symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.fast_info
                        result[symbol] = {
                            "symbol": symbol,
                            "price": getattr(info, "last_price", None),
                            "change": None,
                            "change_pct": None,
                            "volume": getattr(info, "three_month_average_volume", None),
                        }
                    except Exception as sym_exc:
                        logger.warning("yfinance fetch failed for %s: %s", symbol, sym_exc)
                        result[symbol] = MOCK_MARKET_DATA.get(symbol, {"symbol": symbol, "error": "not_found"})
                return result
            except ImportError:
                logger.warning("yfinance not installed; falling back to mock market data.")

        for symbol in symbols:
            result[symbol] = MOCK_MARKET_DATA.get(symbol, {"symbol": symbol, "price": None})
        return result

    async def get_market_sentiment(self) -> dict[str, Any]:
        """Derive an overall market sentiment score from recent news."""
        news = await self.fetch_financial_news("stock market economy")
        positive = sum(1 for n in news if n.get("sentiment") == "positive")
        negative = sum(1 for n in news if n.get("sentiment") == "negative")
        total = len(news) or 1

        score = (positive - negative) / total
        if score > 0.2:
            label = "bullish"
        elif score < -0.2:
            label = "bearish"
        else:
            label = "neutral"

        return {
            "sentiment": label,
            "score": round(score, 3),
            "positive_articles": positive,
            "negative_articles": negative,
            "total_articles": total,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
