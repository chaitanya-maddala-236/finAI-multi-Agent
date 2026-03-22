from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db import get_db
from app.models.portfolio import PortfolioItemCreate, PortfolioItemResponse, PortfolioSummaryResponse
from app.services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/portfolio", tags=["portfolio"])

portfolio_service = PortfolioService()


@router.get("/{user_id}", response_model=PortfolioSummaryResponse)
async def get_portfolio(user_id: int, db: AsyncSession = Depends(get_db)) -> PortfolioSummaryResponse:
    """Return portfolio summary with allocation percentages for a user."""
    summary = await portfolio_service.get_portfolio_summary(user_id, db)
    return PortfolioSummaryResponse(**summary)


@router.get("/{user_id}/items", response_model=list[PortfolioItemResponse])
async def get_portfolio_items(
    user_id: int, db: AsyncSession = Depends(get_db)
) -> list[PortfolioItemResponse]:
    """Return all raw portfolio items for a user."""
    items = await portfolio_service.get_user_portfolio(user_id, db)
    return [PortfolioItemResponse.model_validate(item) for item in items]


@router.post("/{user_id}/item", response_model=PortfolioItemResponse, status_code=201)
async def add_portfolio_item(
    user_id: int,
    item: PortfolioItemCreate,
    db: AsyncSession = Depends(get_db),
) -> PortfolioItemResponse:
    """Add a single portfolio item for a user."""
    db_item = await portfolio_service.add_portfolio_item(user_id, item, db)
    return PortfolioItemResponse.model_validate(db_item)


@router.delete("/{user_id}/item/{item_id}", status_code=204)
async def delete_portfolio_item(
    user_id: int,
    item_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Remove a portfolio item."""
    deleted = await portfolio_service.delete_portfolio_item(user_id, item_id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Portfolio item not found.")


@router.get("/{user_id}/rebalancing")
async def get_rebalancing_suggestions(
    user_id: int, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Return rebalancing suggestions comparing current vs ideal allocation."""
    summary = await portfolio_service.get_portfolio_summary(user_id, db)
    current_allocation = summary.get("allocation", {})
    suggestions = portfolio_service.calculate_rebalancing_suggestions(current_allocation)
    return {
        "user_id": user_id,
        "current_allocation": current_allocation,
        "suggestions": suggestions,
        "rebalancing_needed": len(suggestions) > 0,
    }
