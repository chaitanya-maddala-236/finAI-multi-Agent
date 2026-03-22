from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db import get_db
from app.services.portfolio_service import PortfolioService
from app.utils.csv_parser import (
    parse_expenses_csv,
    parse_portfolio_csv,
    validate_portfolio_data,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/upload", tags=["upload"])

portfolio_service = PortfolioService()

ALLOWED_CONTENT_TYPES = {
    "text/csv",
    "application/csv",
    "application/json",
    "text/plain",
}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


@router.post("/financial-data")
async def upload_financial_data(
    file: Annotated[UploadFile, File(description="CSV or JSON file")],
    data_type: Annotated[
        Literal["portfolio", "expenses", "income"],
        Form(description="Type of data being uploaded"),
    ],
    user_id: Annotated[int, Form(description="User ID")],
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Upload a CSV/JSON file and process it into the database."""
    if file.content_type not in ALLOWED_CONTENT_TYPES and not (
        file.filename or ""
    ).endswith((".csv", ".json")):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Upload CSV or JSON.",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File size exceeds 5 MB limit.")

    try:
        if data_type == "portfolio":
            return await _process_portfolio_upload(user_id, content, file.filename or "", db)
        elif data_type == "expenses":
            return await _process_expenses_upload(content, file.filename or "")
        elif data_type == "income":
            return await _process_income_upload(content, file.filename or "")
        else:
            raise HTTPException(status_code=400, detail=f"Unknown data_type: {data_type}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Upload processing error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to process uploaded file.") from exc


async def _process_portfolio_upload(
    user_id: int, content: bytes, filename: str, db: AsyncSession
) -> dict[str, Any]:
    if filename.endswith(".json"):
        raw = json.loads(content)
        items = raw if isinstance(raw, list) else [raw]
    else:
        items = parse_portfolio_csv(content)

    validated = validate_portfolio_data(items)
    created = await portfolio_service.update_portfolio_from_csv(user_id, validated, db)

    total_value = sum(float(i.get("value", 0)) for i in validated)
    return {
        "success": True,
        "data_type": "portfolio",
        "items_processed": len(created),
        "items_skipped": len(items) - len(validated),
        "summary": {
            "total_value": round(total_value, 2),
            "asset_types": list({i.get("asset_type", "unknown") for i in validated}),
        },
    }


async def _process_expenses_upload(content: bytes, filename: str) -> dict[str, Any]:
    if filename.endswith(".json"):
        raw = json.loads(content)
        items = raw if isinstance(raw, list) else [raw]
    else:
        items = parse_expenses_csv(content)

    total_amount = sum(float(i.get("amount", 0)) for i in items)
    categories = list({i.get("category", "unknown") for i in items})

    return {
        "success": True,
        "data_type": "expenses",
        "items_processed": len(items),
        "summary": {
            "total_amount": round(total_amount, 2),
            "categories": categories,
            "average_transaction": round(total_amount / len(items), 2) if items else 0,
        },
    }


async def _process_income_upload(content: bytes, filename: str) -> dict[str, Any]:
    if filename.endswith(".json"):
        raw = json.loads(content)
        items = raw if isinstance(raw, list) else [raw]
    else:
        # Generic CSV – expect date, source, amount columns
        import io
        import pandas as pd

        df = pd.read_csv(io.BytesIO(content))
        df.columns = [c.strip().lower() for c in df.columns]
        items = df.to_dict(orient="records")

    total_income = sum(float(i.get("amount", 0)) for i in items)
    return {
        "success": True,
        "data_type": "income",
        "items_processed": len(items),
        "summary": {
            "total_income": round(total_income, 2),
            "average_monthly": round(total_income / max(len(items), 1), 2),
        },
    }
