from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel
from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.db import Base


class AssetType(str, enum.Enum):
    stock = "stock"
    etf = "etf"
    mutual_fund = "mutual_fund"
    fixed_income = "fixed_income"
    gold = "gold"
    crypto = "crypto"
    cash = "cash"


class PortfolioItem(Base):
    __tablename__ = "portfolio_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    asset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    asset_type: Mapped[str] = mapped_column(Enum(AssetType), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    quantity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    purchase_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PortfolioItemCreate(BaseModel):
    asset_name: str
    asset_type: AssetType
    value: float
    quantity: Optional[float] = None
    purchase_price: Optional[float] = None
    current_price: Optional[float] = None


class PortfolioItemResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    user_id: int
    asset_name: str
    asset_type: AssetType
    value: float
    quantity: Optional[float] = None
    purchase_price: Optional[float] = None
    current_price: Optional[float] = None
    created_at: datetime
    updated_at: datetime


class PortfolioSummary(BaseModel):
    total_value: float
    allocation: Dict[str, float]
    performance: Dict[str, Any]


class PortfolioSummaryResponse(BaseModel):
    total_value: float
    allocation: Dict[str, float]
    performance: Dict[str, Any]
    diversification_score: float
    items_count: int
