from __future__ import annotations

import io
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PORTFOLIO_REQUIRED_COLS = {"asset", "type", "value"}
EXPENSES_REQUIRED_COLS = {"date", "category", "amount"}

VALID_ASSET_TYPES = {
    "stock", "etf", "mutual_fund", "fixed_income", "gold", "crypto", "cash",
}


def parse_portfolio_csv(file_content: bytes) -> list[dict[str, Any]]:
    """Parse portfolio CSV with columns: asset, type, value (and optional quantity, purchase_price)."""
    try:
        df = pd.read_csv(io.BytesIO(file_content))
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = PORTFOLIO_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        record: dict[str, Any] = {
            "asset_name": str(row["asset"]).strip(),
            "asset_type": str(row["type"]).strip().lower(),
            "value": float(row["value"]),
        }
        if "quantity" in df.columns and pd.notna(row.get("quantity")):
            record["quantity"] = float(row["quantity"])
        if "purchase_price" in df.columns and pd.notna(row.get("purchase_price")):
            record["purchase_price"] = float(row["purchase_price"])
        if "current_price" in df.columns and pd.notna(row.get("current_price")):
            record["current_price"] = float(row["current_price"])
        records.append(record)

    return records


def parse_expenses_csv(file_content: bytes) -> list[dict[str, Any]]:
    """Parse expenses CSV with columns: date, category, amount, description."""
    try:
        df = pd.read_csv(io.BytesIO(file_content))
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = EXPENSES_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        record: dict[str, Any] = {
            "date": str(row["date"]).strip(),
            "category": str(row["category"]).strip(),
            "amount": float(row["amount"]),
            "description": str(row.get("description", "")).strip() if pd.notna(row.get("description")) else "",
        }
        records.append(record)

    return records


def validate_portfolio_data(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and clean portfolio data, removing invalid entries."""
    cleaned: list[dict[str, Any]] = []
    for item in data:
        asset_type = item.get("asset_type", "").lower().replace(" ", "_")
        if asset_type not in VALID_ASSET_TYPES:
            logger.warning("Skipping item with unknown asset_type=%r", asset_type)
            continue
        value = item.get("value", 0)
        if not isinstance(value, (int, float)) or value < 0:
            logger.warning("Skipping item with invalid value=%r", value)
            continue
        item["asset_type"] = asset_type
        item["value"] = float(value)
        cleaned.append(item)
    return cleaned
