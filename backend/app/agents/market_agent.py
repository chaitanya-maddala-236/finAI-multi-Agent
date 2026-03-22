from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from app.config import settings
from app.services.news_service import NewsService

logger = logging.getLogger(__name__)


class MarketAnalysis(BaseModel):
    trend: str
    risk_factors: list[str]
    opportunities: list[str]
    recommended_sectors: list[str]
    sentiment_score: float
    summary: str


def _mock_market_analysis(news: list[dict[str, Any]], sentiment: dict[str, Any]) -> MarketAnalysis:
    trend = sentiment.get("sentiment", "neutral")
    score = float(sentiment.get("score", 0.0))

    risk_factors = [
        "Elevated global interest rates putting pressure on equity valuations.",
        "Geopolitical tensions increasing commodity price volatility.",
        "Currency depreciation risk for emerging markets.",
    ]
    opportunities = [
        "AI and technology sector showing strong earnings momentum.",
        "Gold offering safe-haven value amid uncertainty.",
        "Domestic consumption-driven sectors in India remain resilient.",
    ]
    recommended_sectors = ["technology", "healthcare", "consumer_staples", "gold"]

    if trend == "bearish":
        recommended_sectors = ["gold", "fixed_income", "consumer_staples", "utilities"]

    return MarketAnalysis(
        trend=trend,
        risk_factors=risk_factors,
        opportunities=opportunities,
        recommended_sectors=recommended_sectors,
        sentiment_score=score,
        summary=(
            f"Markets are currently {trend} with a sentiment score of {score:.2f}. "
            "Mixed signals from global macro environment suggest a cautious approach."
        ),
    )


async def run_market_agent() -> MarketAnalysis:
    """Fetch news & market data, then return a structured market analysis."""
    news_service = NewsService()
    news = await news_service.fetch_financial_news("financial markets stocks economy")
    sentiment = await news_service.get_market_sentiment()

    if not settings.OPENAI_API_KEY:
        logger.info("No OpenAI key – using rule-based market analysis.")
        return _mock_market_analysis(news, sentiment)

    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from langchain.output_parsers import PydanticOutputParser

        parser = PydanticOutputParser(pydantic_object=MarketAnalysis)
        news_text = "\n".join(
            f"- {n['title']}: {n.get('description', '')}" for n in news[:5]
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert market analyst. Analyse the news and sentiment data and return a "
                    "JSON object matching the schema.\n{format_instructions}",
                ),
                (
                    "human",
                    "Recent news:\n{news}\n\nSentiment data: {sentiment}\n\nProvide market analysis.",
                ),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        chain = prompt | llm | parser
        result: MarketAnalysis = await chain.ainvoke(
            {"news": news_text, "sentiment": str(sentiment)}
        )
        return result
    except Exception as exc:
        logger.warning("LLM market agent failed (%s); falling back to rules.", exc)
        return _mock_market_analysis(news, sentiment)
