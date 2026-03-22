from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from app.agents.market_agent import MarketAnalysis
from app.agents.user_profile_agent import UserProfileAnalysis
from app.config import settings

logger = logging.getLogger(__name__)


class RiskAnalysis(BaseModel):
    risk_level: str
    risk_score: float
    max_equity_exposure: float
    recommended_debt_ratio: float
    reasoning: str
    key_risk_factors: list[str]


def _compute_risk_score(
    profile: UserProfileAnalysis, market: MarketAnalysis
) -> float:
    score = 5.0

    # Profile adjustments
    if profile.financial_health_score >= 70:
        score += 1.5
    elif profile.financial_health_score < 40:
        score -= 1.5

    if profile.savings_rate >= 20:
        score += 1.0
    elif profile.savings_rate < 5:
        score -= 1.0

    if profile.emergency_fund_months >= 6:
        score += 0.5
    elif profile.emergency_fund_months < 3:
        score -= 0.5

    # Market adjustments
    if market.trend == "bearish":
        score -= 1.5
    elif market.trend == "bullish":
        score += 1.0

    return round(min(10.0, max(1.0, score)), 1)


def _mock_risk_analysis(
    profile: UserProfileAnalysis, market: MarketAnalysis
) -> RiskAnalysis:
    risk_score = _compute_risk_score(profile, market)

    if risk_score <= 3.5:
        risk_level = "conservative"
        max_equity = 30.0
        debt_ratio = 60.0
    elif risk_score <= 6.5:
        risk_level = "moderate"
        max_equity = 60.0
        debt_ratio = 30.0
    else:
        risk_level = "aggressive"
        max_equity = 80.0
        debt_ratio = 10.0

    return RiskAnalysis(
        risk_level=risk_level,
        risk_score=risk_score,
        max_equity_exposure=max_equity,
        recommended_debt_ratio=debt_ratio,
        reasoning=(
            f"Based on your financial health score of {profile.financial_health_score:.0f}/100 "
            f"and a {market.trend} market environment, your risk profile is {risk_level}."
        ),
        key_risk_factors=market.risk_factors[:3],
    )


async def run_risk_agent(
    profile: UserProfileAnalysis, market: MarketAnalysis, user_data: dict[str, Any]
) -> RiskAnalysis:
    """Compute risk profile from user data and market conditions."""
    if not settings.OPENAI_API_KEY:
        logger.info("No OpenAI key – using rule-based risk analysis.")
        return _mock_risk_analysis(profile, market)

    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from langchain.output_parsers import PydanticOutputParser

        parser = PydanticOutputParser(pydantic_object=RiskAnalysis)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert risk analyst. Evaluate the user's risk tolerance based on their "
                    "financial profile and current market conditions. Return a JSON object matching the "
                    "schema. Risk score is 1 (very conservative) to 10 (very aggressive).\n"
                    "{format_instructions}",
                ),
                (
                    "human",
                    "User profile analysis:\n{profile}\n\n"
                    "Market analysis:\n{market}\n\n"
                    "Additional user data:\n{user_data}\n\n"
                    "Provide a comprehensive risk assessment.",
                ),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.2,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        chain = prompt | llm | parser
        result: RiskAnalysis = await chain.ainvoke(
            {
                "profile": profile.model_dump_json(indent=2),
                "market": market.model_dump_json(indent=2),
                "user_data": str(user_data),
            }
        )
        return result
    except Exception as exc:
        logger.warning("LLM risk agent failed (%s); falling back to rules.", exc)
        return _mock_risk_analysis(profile, market)
