from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database.db import get_db
from app.models.user import Token, TokenData, User, UserCreate, UserResponse, UserUpdate

logger = logging.getLogger(__name__)
router = APIRouter(tags=["auth & users"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    return pwd_context.hash(password)


def _verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def _create_access_token(data: dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload["exp"] = expire
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


async def _get_user_by_email(email: str, db: AsyncSession) -> Optional[User]:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def _get_user_by_id(user_id: int, db: AsyncSession) -> Optional[User]:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


# ── Auth dependency ───────────────────────────────────────────────────────────

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: AsyncSession = Depends(get_db),
) -> User:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: Optional[int] = payload.get("sub")
        if user_id is None:
            raise credentials_exc
    except JWTError:
        raise credentials_exc

    user = await _get_user_by_id(int(user_id), db)
    if user is None or not user.is_active:
        raise credentials_exc
    return user


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/auth/register", response_model=UserResponse, status_code=201)
async def register(user_create: UserCreate, db: AsyncSession = Depends(get_db)) -> UserResponse:
    """Register a new user."""
    existing = await _get_user_by_email(user_create.email, db)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered.")

    db_user = User(
        email=user_create.email,
        hashed_password=_hash_password(user_create.password),
        full_name=user_create.full_name,
        age=user_create.age,
        salary=user_create.salary,
        monthly_expenses=user_create.monthly_expenses,
        risk_tolerance=user_create.risk_tolerance,
        financial_goals=user_create.financial_goals,
    )
    db.add(db_user)
    await db.flush()
    await db.refresh(db_user)
    return UserResponse.model_validate(db_user)


@router.post("/auth/login", response_model=Token)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: AsyncSession = Depends(get_db),
) -> Token:
    """Authenticate and return a JWT access token."""
    user = await _get_user_by_email(form_data.username, db)
    if not user or not _verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = _create_access_token({"sub": str(user.id), "email": user.email})
    return Token(access_token=token)


@router.get("/user-profile/{user_id}", response_model=UserResponse)
async def get_user_profile(
    user_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Return a user's profile. Users can only fetch their own profile."""
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Access denied.")
    return UserResponse.model_validate(current_user)


@router.put("/user-profile/{user_id}", response_model=UserResponse)
async def update_user_profile(
    user_id: int,
    user_update: UserUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Update a user's profile fields."""
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Access denied.")

    update_data = user_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(current_user, field, value)

    db.add(current_user)
    await db.flush()
    await db.refresh(current_user)
    return UserResponse.model_validate(current_user)
