from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.graph.financial_graph import get_conversation_history, run_financial_graph

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    user_id: int
    user_data: Optional[dict[str, Any]] = None
    conversation_id: Optional[str] = None
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    agent_outputs: dict[str, Any]
    financial_advice: Optional[dict[str, Any]] = None


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Run the full LangGraph financial advisory pipeline."""
    user_data = request.user_data or {}
    user_data["user_id"] = request.user_id

    try:
        result = await run_financial_graph(user_data=user_data, message=request.message)
    except Exception as exc:
        logger.error("Graph execution error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return ChatResponse(
        response=result.get("response", ""),
        conversation_id=result.get("conversation_id", ""),
        agent_outputs=result.get("agent_outputs", {}),
        financial_advice=result.get("financial_advice"),
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest, req: Request) -> StreamingResponse:
    """Stream the financial advisory pipeline results via Server-Sent Events."""
    user_data = request.user_data or {}
    user_data["user_id"] = request.user_id

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            yield _sse_event({"status": "started", "message": "Analysing your financial profile…"})
            await asyncio.sleep(0)

            result = await run_financial_graph(user_data=user_data, message=request.message)

            # Stream intermediate agent outputs
            for agent_name, agent_output in result.get("agent_outputs", {}).items():
                if agent_output:
                    yield _sse_event({"status": "agent_complete", "agent": agent_name, "data": agent_output})
                    await asyncio.sleep(0)

            yield _sse_event({"status": "complete", "data": result})
            yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            logger.info("Client disconnected from SSE stream.")
        except Exception as exc:
            logger.error("SSE stream error: %s", exc)
            yield _sse_event({"status": "error", "message": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history/{user_id}")
async def get_history(user_id: int) -> dict[str, Any]:
    """Return conversation history for a user from Redis."""
    history = await get_conversation_history(user_id)
    return {"user_id": user_id, "history": history, "count": len(history)}


def _sse_event(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"
