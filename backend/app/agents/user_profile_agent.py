from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)


class UserProfileAnalysis(BaseModel):
    net_worth: float
    savings_rate: float
    financial_health_score: float
    monthly_savings: float
    debt_to_income_ratio: float
    emergency_fund_months: float
    insights: list[str]
    risk_capacity: str
    recommended_actions: list[str]


def _mock_profile_analysis(user_data: dict[str, Any]) -> UserProfileAnalysis:
    salary = float(user_data.get("salary") or 0)
    expenses = float(user_data.get("monthly_expenses") or 0)
    assets = float(user_data.get("total_assets") or 0)
    liabilities = float(user_data.get("total_liabilities") or 0)
    age = int(user_data.get("age") or 30)

    net_worth = assets - liabilities
    monthly_savings = salary / 12 - expenses if salary > 0 else 0
    savings_rate = (monthly_savings / (salary / 12) * 100) if salary > 0 else 0
    emergency_fund = assets / expenses if expenses > 0 else 0

    score = 50.0
    if savings_rate > 20:
        score += 20
    elif savings_rate > 10:
        score += 10
    if net_worth > 0:
        score += 15
    if emergency_fund >= 6:
        score += 15
    score = min(100.0, max(0.0, score))

    capacity = "conservative"
    if score >= 70 and age < 45:
        capacity = "aggressive"
    elif score >= 50:
        capacity = "moderate"

    return UserProfileAnalysis(
        net_worth=net_worth,
        savings_rate=round(savings_rate, 2),
        financial_health_score=round(score, 2),
        monthly_savings=round(monthly_savings, 2),
        debt_to_income_ratio=round(liabilities / (salary or 1), 2),
        emergency_fund_months=round(emergency_fund, 1),
        insights=[
            f"Your net worth is ₹{net_worth:,.0f}.",
            f"You save approximately {savings_rate:.1f}% of your income.",
            f"Emergency fund covers ~{emergency_fund:.1f} months of expenses.",
        ],
        risk_capacity=capacity,
        recommended_actions=[
            "Build an emergency fund of at least 6 months of expenses.",
            "Increase SIP contributions to boost long-term wealth.",
            "Review and reduce discretionary spending.",
        ],
    )


async def run_user_profile_agent(user_data: dict[str, Any]) -> UserProfileAnalysis:
    """Analyse user financial profile and return structured insights."""
    if not settings.OPENAI_API_KEY:
        logger.info("No OpenAI key – using rule-based profile analysis.")
        return _mock_profile_analysis(user_data)

    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from langchain.output_parsers import PydanticOutputParser

        parser = PydanticOutputParser(pydantic_object=UserProfileAnalysis)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert financial advisor. Analyse the user profile and return a JSON "
                    "object matching the schema. All monetary values are in INR.\n"
                    "{format_instructions}",
                ),
                (
                    "human",
                    "User profile:\n{profile}\n\nProvide a thorough financial analysis.",
                ),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.2,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        chain = prompt | llm | parser
        result: UserProfileAnalysis = await chain.ainvoke({"profile": str(user_data)})
        return result
    except Exception as exc:
        logger.warning("LLM profile agent failed (%s); falling back to rules.", exc)
        return _mock_profile_analysis(user_data)
