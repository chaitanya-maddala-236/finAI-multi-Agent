from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from typing_extensions import TypedDict

from app.agents.advisor_agent import FinancialAdvice, run_advisor_agent
from app.agents.market_agent import MarketAnalysis, run_market_agent
from app.agents.risk_agent import RiskAnalysis, run_risk_agent
from app.agents.strategy_agent import InvestmentStrategy, run_strategy_agent
from app.agents.user_profile_agent import UserProfileAnalysis, run_user_profile_agent
from app.config import settings

logger = logging.getLogger(__name__)


# ── Graph state ───────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    user_data: dict[str, Any]
    message: str
    conversation_id: str
    user_profile: Optional[UserProfileAnalysis]
    market_analysis: Optional[MarketAnalysis]
    risk_analysis: Optional[RiskAnalysis]
    investment_strategy: Optional[InvestmentStrategy]
    financial_advice: Optional[FinancialAdvice]
    error: Optional[str]


# ── Node functions ────────────────────────────────────────────────────────────

async def node_user_profile(state: GraphState) -> GraphState:
    try:
        analysis = await run_user_profile_agent(state["user_data"])
        return {**state, "user_profile": analysis}
    except Exception as exc:
        logger.error("user_profile node error: %s", exc)
        return {**state, "error": str(exc)}


async def node_risk_analysis(state: GraphState) -> GraphState:
    try:
        if state.get("user_profile") is None or state.get("market_analysis") is None:
            raise ValueError("Prerequisite nodes have not completed successfully.")
        analysis = await run_risk_agent(
            state["user_profile"],  # type: ignore[arg-type]
            state["market_analysis"],  # type: ignore[arg-type]
            state["user_data"],
        )
        return {**state, "risk_analysis": analysis}
    except Exception as exc:
        logger.error("risk_analysis node error: %s", exc)
        return {**state, "error": str(exc)}


async def node_market_research(state: GraphState) -> GraphState:
    try:
        analysis = await run_market_agent()
        return {**state, "market_analysis": analysis}
    except Exception as exc:
        logger.error("market_research node error: %s", exc)
        return {**state, "error": str(exc)}


async def node_investment_strategy(state: GraphState) -> GraphState:
    try:
        if state.get("risk_analysis") is None or state.get("market_analysis") is None:
            raise ValueError("Prerequisite nodes have not completed successfully.")
        strategy = await run_strategy_agent(
            state["risk_analysis"],  # type: ignore[arg-type]
            state["market_analysis"],  # type: ignore[arg-type]
            state["user_data"],
        )
        return {**state, "investment_strategy": strategy}
    except Exception as exc:
        logger.error("investment_strategy node error: %s", exc)
        return {**state, "error": str(exc)}


async def node_advisor(state: GraphState) -> GraphState:
    try:
        advice = await run_advisor_agent(
            state["user_profile"],  # type: ignore[arg-type]
            state["market_analysis"],  # type: ignore[arg-type]
            state["risk_analysis"],  # type: ignore[arg-type]
            state["investment_strategy"],  # type: ignore[arg-type]
            state["message"],
            state["user_data"],
        )
        return {**state, "financial_advice": advice}
    except Exception as exc:
        logger.error("advisor node error: %s", exc)
        return {**state, "error": str(exc)}


# ── Graph builder ─────────────────────────────────────────────────────────────

def _build_graph():
    """Build and compile the LangGraph StateGraph."""
    try:
        from langgraph.graph import StateGraph, END  # type: ignore[import-untyped]

        builder: StateGraph = StateGraph(GraphState)

        builder.add_node("user_profile", node_user_profile)
        builder.add_node("market_research", node_market_research)
        builder.add_node("risk_analysis", node_risk_analysis)
        builder.add_node("investment_strategy", node_investment_strategy)
        builder.add_node("advisor", node_advisor)

        builder.set_entry_point("user_profile")
        builder.add_edge("user_profile", "market_research")
        builder.add_edge("market_research", "risk_analysis")
        builder.add_edge("risk_analysis", "investment_strategy")
        builder.add_edge("investment_strategy", "advisor")
        builder.add_edge("advisor", END)

        return builder.compile()
    except Exception as exc:
        logger.warning("LangGraph not available (%s); will run pipeline manually.", exc)
        return None


_graph = _build_graph()


# ── Redis helpers ─────────────────────────────────────────────────────────────

async def _get_redis():
    """Return a Redis connection or None if unavailable."""
    try:
        import redis.asyncio as aioredis  # type: ignore[import-untyped]
        client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        await client.ping()
        return client
    except Exception:
        return None


async def save_conversation(
    conversation_id: str, user_id: int, state: dict[str, Any]
) -> None:
    client = await _get_redis()
    if client is None:
        return
    try:
        payload = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message": state.get("message", ""),
            "advice_summary": state.get("financial_advice", {}).get("summary", "") if isinstance(state.get("financial_advice"), dict) else "",
        }
        key = f"conversation:{user_id}:{conversation_id}"
        await client.lpush(key, json.dumps(payload))
        await client.expire(key, 86400 * 7)
    except Exception as exc:
        logger.warning("Failed to save conversation to Redis: %s", exc)
    finally:
        await client.aclose()


async def get_conversation_history(user_id: int) -> list[dict[str, Any]]:
    client = await _get_redis()
    if client is None:
        return []
    try:
        pattern = f"conversation:{user_id}:*"
        keys = await client.keys(pattern)
        history: list[dict[str, Any]] = []
        for key in keys:
            items = await client.lrange(key, 0, -1)
            for item in items:
                try:
                    history.append(json.loads(item))
                except json.JSONDecodeError:
                    pass
        return history
    except Exception as exc:
        logger.warning("Failed to retrieve conversation history: %s", exc)
        return []
    finally:
        await client.aclose()


# ── Public entry point ────────────────────────────────────────────────────────

async def run_financial_graph(user_data: dict[str, Any], message: str) -> dict[str, Any]:
    """Execute the full financial advisory pipeline and return structured outputs."""
    conversation_id = str(uuid.uuid4())
    initial_state: GraphState = {
        "user_data": user_data,
        "message": message,
        "conversation_id": conversation_id,
        "user_profile": None,
        "market_analysis": None,
        "risk_analysis": None,
        "investment_strategy": None,
        "financial_advice": None,
        "error": None,
    }

    if _graph is not None:
        try:
            final_state: GraphState = await _graph.ainvoke(initial_state)
        except Exception as exc:
            logger.error("LangGraph execution failed: %s", exc)
            final_state = await _run_pipeline_manually(initial_state)
    else:
        final_state = await _run_pipeline_manually(initial_state)

    result = _state_to_response(final_state)
    user_id = int(user_data.get("user_id", 0))
    await save_conversation(conversation_id, user_id, result)
    return result


async def _run_pipeline_manually(state: GraphState) -> GraphState:
    """Fallback sequential pipeline when LangGraph is unavailable."""
    state = await node_user_profile(state)
    state = await node_market_research(state)
    state = await node_risk_analysis(state)
    state = await node_investment_strategy(state)
    state = await node_advisor(state)
    return state


def _state_to_response(state: GraphState) -> dict[str, Any]:
    def _safe_dict(obj: Any) -> Any:
        if obj is None:
            return None
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return obj

    advice = state.get("financial_advice")
    return {
        "conversation_id": state.get("conversation_id", ""),
        "response": _safe_dict(advice).get("summary", "Unable to generate advice at this time.") if advice else "Unable to generate advice at this time.",
        "financial_advice": _safe_dict(advice),
        "agent_outputs": {
            "user_profile": _safe_dict(state.get("user_profile")),
            "market_analysis": _safe_dict(state.get("market_analysis")),
            "risk_analysis": _safe_dict(state.get("risk_analysis")),
            "investment_strategy": _safe_dict(state.get("investment_strategy")),
        },
        "error": state.get("error"),
    }
