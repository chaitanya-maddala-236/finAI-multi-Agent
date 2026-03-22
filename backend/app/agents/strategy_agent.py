from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from app.agents.market_agent import MarketAnalysis
from app.agents.risk_agent import RiskAnalysis
from app.config import settings

logger = logging.getLogger(__name__)


class InvestmentStrategy(BaseModel):
    allocation: dict[str, float]
    specific_recommendations: list[str]
    rebalancing_needed: bool
    expected_return_range: str
    investment_horizon: str
    strategy_rationale: str


_ALLOCATIONS: dict[str, dict[str, float]] = {
    "conservative": {
        "fixed_income": 50.0,
        "stock": 20.0,
        "etf": 10.0,
        "gold": 10.0,
        "mutual_fund": 5.0,
        "cash": 5.0,
    },
    "moderate": {
        "stock": 35.0,
        "etf": 20.0,
        "mutual_fund": 15.0,
        "fixed_income": 15.0,
        "gold": 10.0,
        "cash": 5.0,
    },
    "aggressive": {
        "stock": 50.0,
        "etf": 20.0,
        "mutual_fund": 15.0,
        "gold": 5.0,
        "fixed_income": 5.0,
        "crypto": 5.0,
    },
}

_RETURNS: dict[str, str] = {
    "conservative": "6-9% per annum",
    "moderate": "10-14% per annum",
    "aggressive": "15-20% per annum",
}


def _mock_strategy(
    risk: RiskAnalysis, market: MarketAnalysis, user_goals: str
) -> InvestmentStrategy:
    level = risk.risk_level
    allocation = _ALLOCATIONS.get(level, _ALLOCATIONS["moderate"])

    if market.trend == "bearish":
        # Shift towards defensive assets
        allocation = {k: v for k, v in allocation.items()}
        allocation["gold"] = allocation.get("gold", 5.0) + 5.0
        allocation["fixed_income"] = allocation.get("fixed_income", 10.0) + 5.0
        stock_reduction = 10.0
        if "stock" in allocation:
            allocation["stock"] = max(0.0, allocation["stock"] - stock_reduction)

    recs = [
        f"Invest in diversified {market.recommended_sectors[0]} ETFs for sector exposure.",
        "Add Sovereign Gold Bonds for inflation hedge and tax efficiency.",
        "Maintain a liquid emergency fund equal to 6 months of expenses.",
        "Consider SIP in large-cap equity mutual funds for rupee-cost averaging.",
        "Review and rebalance portfolio every 6 months.",
    ]

    return InvestmentStrategy(
        allocation=allocation,
        specific_recommendations=recs,
        rebalancing_needed=market.trend == "bearish",
        expected_return_range=_RETURNS.get(level, "10-14% per annum"),
        investment_horizon="5-7 years" if level != "aggressive" else "7-10 years",
        strategy_rationale=(
            f"Given your {level} risk profile and a {market.trend} market, "
            "we recommend a balanced approach with emphasis on quality assets."
        ),
    )


async def run_strategy_agent(
    risk: RiskAnalysis,
    market: MarketAnalysis,
    user_data: dict[str, Any],
) -> InvestmentStrategy:
    """Generate an investment strategy based on risk profile and market analysis."""
    user_goals = str(user_data.get("financial_goals", "long-term wealth creation"))

    if not settings.OPENAI_API_KEY:
        logger.info("No OpenAI key – using rule-based strategy generation.")
        return _mock_strategy(risk, market, user_goals)

    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from langchain.output_parsers import PydanticOutputParser

        parser = PydanticOutputParser(pydantic_object=InvestmentStrategy)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a certified financial planner. Generate a tailored investment strategy. "
                    "Allocation values must sum to 100. Return JSON matching the schema.\n"
                    "{format_instructions}",
                ),
                (
                    "human",
                    "Risk analysis:\n{risk}\n\n"
                    "Market analysis:\n{market}\n\n"
                    "User goals: {goals}\n\n"
                    "Generate a comprehensive investment strategy.",
                ),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        chain = prompt | llm | parser
        result: InvestmentStrategy = await chain.ainvoke(
            {
                "risk": risk.model_dump_json(indent=2),
                "market": market.model_dump_json(indent=2),
                "goals": user_goals,
            }
        )
        return result
    except Exception as exc:
        logger.warning("LLM strategy agent failed (%s); falling back to rules.", exc)
        return _mock_strategy(risk, market, user_goals)
