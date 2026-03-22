from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy import Boolean, DateTime, Enum, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.db import Base


class RiskTolerance(str, enum.Enum):
    conservative = "conservative"
    moderate = "moderate"
    aggressive = "aggressive"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    salary: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    monthly_expenses: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_tolerance: Mapped[Optional[str]] = mapped_column(
        Enum(RiskTolerance), nullable=True, default=RiskTolerance.moderate
    )
    financial_goals: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    age: Optional[int] = None
    salary: Optional[float] = None
    monthly_expenses: Optional[float] = None
    risk_tolerance: Optional[RiskTolerance] = RiskTolerance.moderate
    financial_goals: Optional[str] = None

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    age: Optional[int] = None
    salary: Optional[float] = None
    monthly_expenses: Optional[float] = None
    risk_tolerance: Optional[RiskTolerance] = None
    financial_goals: Optional[str] = None


class UserResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    email: str
    full_name: str
    age: Optional[int] = None
    salary: Optional[float] = None
    monthly_expenses: Optional[float] = None
    risk_tolerance: Optional[RiskTolerance] = None
    financial_goals: Optional[str] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[int] = None
    email: Optional[str] = None
