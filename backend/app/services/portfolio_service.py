from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.portfolio import AssetType, PortfolioItem, PortfolioItemCreate

logger = logging.getLogger(__name__)

# Ideal balanced allocation used for rebalancing suggestions
DEFAULT_TARGET_ALLOCATION: dict[str, float] = {
    "stock": 40.0,
    "etf": 20.0,
    "mutual_fund": 10.0,
    "fixed_income": 15.0,
    "gold": 5.0,
    "crypto": 5.0,
    "cash": 5.0,
}


class PortfolioService:
    """Business logic for portfolio management."""

    async def get_user_portfolio(self, user_id: int, db: AsyncSession) -> list[PortfolioItem]:
        result = await db.execute(select(PortfolioItem).where(PortfolioItem.user_id == user_id))
        return list(result.scalars().all())

    async def get_portfolio_summary(self, user_id: int, db: AsyncSession) -> dict[str, Any]:
        items = await self.get_user_portfolio(user_id, db)
        if not items:
            return {
                "total_value": 0.0,
                "allocation": {},
                "performance": {},
                "diversification_score": 0.0,
                "items_count": 0,
            }

        total_value = sum(item.value for item in items)

        # Allocation by asset type (percentage)
        type_totals: dict[str, float] = {}
        for item in items:
            type_totals[item.asset_type] = type_totals.get(item.asset_type, 0.0) + item.value
        allocation = {
            k: round((v / total_value) * 100, 2) for k, v in type_totals.items()
        }

        # Simple gain/loss performance
        performance: dict[str, Any] = {}
        for item in items:
            if item.purchase_price and item.current_price and item.quantity:
                gain = (item.current_price - item.purchase_price) * item.quantity
                gain_pct = ((item.current_price - item.purchase_price) / item.purchase_price) * 100
                performance[item.asset_name] = {
                    "gain_loss": round(gain, 2),
                    "gain_loss_pct": round(gain_pct, 2),
                }

        diversification_score = self._calculate_diversification_score(allocation)

        return {
            "total_value": round(total_value, 2),
            "allocation": allocation,
            "performance": performance,
            "diversification_score": diversification_score,
            "items_count": len(items),
        }

    async def add_portfolio_item(
        self, user_id: int, item: PortfolioItemCreate, db: AsyncSession
    ) -> PortfolioItem:
        db_item = PortfolioItem(
            user_id=user_id,
            asset_name=item.asset_name,
            asset_type=item.asset_type,
            value=item.value,
            quantity=item.quantity,
            purchase_price=item.purchase_price,
            current_price=item.current_price,
        )
        db.add(db_item)
        await db.flush()
        await db.refresh(db_item)
        return db_item

    async def delete_portfolio_item(self, user_id: int, item_id: int, db: AsyncSession) -> bool:
        result = await db.execute(
            delete(PortfolioItem).where(
                PortfolioItem.id == item_id, PortfolioItem.user_id == user_id
            )
        )
        return result.rowcount > 0

    async def update_portfolio_from_csv(
        self, user_id: int, items: list[dict[str, Any]], db: AsyncSession
    ) -> list[PortfolioItem]:
        created: list[PortfolioItem] = []
        for item_data in items:
            db_item = PortfolioItem(
                user_id=user_id,
                asset_name=item_data.get("asset_name", "Unknown"),
                asset_type=item_data.get("asset_type", AssetType.stock),
                value=float(item_data.get("value", 0)),
                quantity=item_data.get("quantity"),
                purchase_price=item_data.get("purchase_price"),
                current_price=item_data.get("current_price"),
            )
            db.add(db_item)
            created.append(db_item)
        await db.flush()
        return created

    def calculate_rebalancing_suggestions(
        self,
        current_allocation: dict[str, float],
        target_allocation: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        target = target_allocation or DEFAULT_TARGET_ALLOCATION
        suggestions: list[dict[str, Any]] = []

        all_types = set(current_allocation) | set(target)
        for asset_type in all_types:
            current_pct = current_allocation.get(asset_type, 0.0)
            target_pct = target.get(asset_type, 0.0)
            diff = target_pct - current_pct
            if abs(diff) < 1.0:
                continue
            action = "increase" if diff > 0 else "decrease"
            suggestions.append(
                {
                    "asset_type": asset_type,
                    "current_allocation": round(current_pct, 2),
                    "target_allocation": round(target_pct, 2),
                    "difference": round(diff, 2),
                    "action": action,
                    "message": (
                        f"{action.capitalize()} {asset_type} allocation by "
                        f"{abs(diff):.1f}% (current: {current_pct:.1f}%, target: {target_pct:.1f}%)"
                    ),
                }
            )

        suggestions.sort(key=lambda x: abs(x["difference"]), reverse=True)
        return suggestions

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _calculate_diversification_score(allocation: dict[str, float]) -> float:
        """Return a 0-100 score based on how evenly assets are spread."""
        if not allocation:
            return 0.0
        num_types = len(allocation)
        ideal = 100.0 / num_types if num_types else 100.0
        variance = sum((pct - ideal) ** 2 for pct in allocation.values()) / num_types
        # Normalise: max variance ≈ ideal² * (num_types-1)
        max_variance = (ideal ** 2) * max(num_types - 1, 1)
        score = max(0.0, 100.0 - (variance / max_variance) * 100) if max_variance else 100.0
        return round(score, 2)
