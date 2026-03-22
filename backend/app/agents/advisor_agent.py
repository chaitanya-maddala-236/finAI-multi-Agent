from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from app.agents.market_agent import MarketAnalysis
from app.agents.risk_agent import RiskAnalysis
from app.agents.strategy_agent import InvestmentStrategy
from app.agents.user_profile_agent import UserProfileAnalysis
from app.config import settings

logger = logging.getLogger(__name__)


class FinancialAdvice(BaseModel):
    summary: str
    detailed_advice: str
    action_items: list[str]
    warnings: list[str]
    confidence_score: float
    next_review_date: str


def _mock_advice(
    profile: UserProfileAnalysis,
    market: MarketAnalysis,
    risk: RiskAnalysis,
    strategy: InvestmentStrategy,
    message: str,
) -> FinancialAdvice:
    allocation_str = ", ".join(
        f"{k}: {v:.0f}%" for k, v in strategy.allocation.items()
    )

    warnings: list[str] = []
    if profile.emergency_fund_months < 3:
        warnings.append("⚠️ Emergency fund is critically low. Prioritise building it before investing.")
    if risk.risk_level == "aggressive" and market.trend == "bearish":
        warnings.append("⚠️ Aggressive strategy in a bearish market carries elevated downside risk.")
    if profile.debt_to_income_ratio > 0.4:
        warnings.append("⚠️ High debt-to-income ratio. Focus on debt reduction before aggressive investing.")

    action_items = [
        f"Allocate portfolio as: {allocation_str}.",
        *strategy.specific_recommendations[:3],
        f"Target expected returns of {strategy.expected_return_range}.",
    ]

    return FinancialAdvice(
        summary=(
            f"Based on your financial health score of {profile.financial_health_score:.0f}/100 and "
            f"a {market.trend} market, a {risk.risk_level} investment strategy is recommended."
        ),
        detailed_advice=(
            f"Your net worth stands at ₹{profile.net_worth:,.0f} with a savings rate of "
            f"{profile.savings_rate:.1f}%. {strategy.strategy_rationale} "
            f"The suggested portfolio allocation is: {allocation_str}. "
            f"With an investment horizon of {strategy.investment_horizon}, "
            f"expected returns are {strategy.expected_return_range}."
        ),
        action_items=action_items,
        warnings=warnings,
        confidence_score=min(
            1.0, round(profile.financial_health_score / 100 * 0.7 + 0.3, 2)
        ),
        next_review_date="6 months from today",
    )


async def run_advisor_agent(
    profile: UserProfileAnalysis,
    market: MarketAnalysis,
    risk: RiskAnalysis,
    strategy: InvestmentStrategy,
    message: str,
    user_data: dict[str, Any],
) -> FinancialAdvice:
    """Synthesise all agent outputs into comprehensive financial advice."""
    if not settings.OPENAI_API_KEY:
        logger.info("No OpenAI key – using rule-based advisor.")
        return _mock_advice(profile, market, risk, strategy, message)

    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from langchain.output_parsers import PydanticOutputParser

        parser = PydanticOutputParser(pydantic_object=FinancialAdvice)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior certified financial advisor. Synthesise all analysis results "
                    "into clear, actionable financial advice tailored to the user's question. "
                    "Be specific, empathetic, and practical. Return JSON matching the schema.\n"
                    "{format_instructions}",
                ),
                (
                    "human",
                    "User question: {message}\n\n"
                    "Profile analysis:\n{profile}\n\n"
                    "Market analysis:\n{market}\n\n"
                    "Risk analysis:\n{risk}\n\n"
                    "Investment strategy:\n{strategy}\n\n"
                    "Generate comprehensive financial advice.",
                ),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.4,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        chain = prompt | llm | parser
        result: FinancialAdvice = await chain.ainvoke(
            {
                "message": message,
                "profile": profile.model_dump_json(indent=2),
                "market": market.model_dump_json(indent=2),
                "risk": risk.model_dump_json(indent=2),
                "strategy": strategy.model_dump_json(indent=2),
            }
        )
        return result
    except Exception as exc:
        logger.warning("LLM advisor agent failed (%s); falling back to rules.", exc)
        return _mock_advice(profile, market, risk, strategy, message)
