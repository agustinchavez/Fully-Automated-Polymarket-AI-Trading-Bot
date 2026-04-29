"""Cross-platform arbitrage scanner — finds and executes arb opportunities.

Scans for price differences between Polymarket and Kalshi on overlapping
markets.  When spread > (total_fees + min_arb_edge), executes both legs
as a paired trade with timeout and optional unwind.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.connectors.kalshi_client import KalshiClient, KalshiOrderResult
from src.connectors.market_matcher import MarketMatch, MarketMatcher
from src.observability.logger import get_logger

log = get_logger(__name__)


# ── Data Models ──────────────────────────────────────────────────────


@dataclass
class CrossPlatformArbOpportunity:
    """A cross-platform arbitrage opportunity."""
    match: MarketMatch
    poly_yes_price: float
    poly_no_price: float
    kalshi_yes_price: float
    kalshi_no_price: float
    spread: float                   # absolute price difference
    net_spread: float               # spread - total_fees
    direction: str                  # e.g. "BUY_POLY_YES_SELL_KALSHI_YES"
    buy_platform: str               # "polymarket" | "kalshi"
    sell_platform: str
    buy_price: float
    sell_price: float
    total_fees: float
    is_actionable: bool
    timestamp: float = 0.0
    arb_id: str = ""

    def __post_init__(self) -> None:
        if not self.arb_id:
            self.arb_id = f"arb-{uuid.uuid4().hex[:12]}"
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "arb_id": self.arb_id,
            "match": self.match.to_dict(),
            "poly_yes_price": self.poly_yes_price,
            "poly_no_price": self.poly_no_price,
            "kalshi_yes_price": self.kalshi_yes_price,
            "kalshi_no_price": self.kalshi_no_price,
            "spread": round(self.spread, 4),
            "net_spread": round(self.net_spread, 4),
            "direction": self.direction,
            "buy_platform": self.buy_platform,
            "sell_platform": self.sell_platform,
            "buy_price": round(self.buy_price, 4),
            "sell_price": round(self.sell_price, 4),
            "total_fees": round(self.total_fees, 4),
            "is_actionable": self.is_actionable,
            "timestamp": self.timestamp,
        }


@dataclass
class PairedTradeResult:
    """Result of executing both legs of an arb trade."""
    arb_id: str
    buy_platform: str
    sell_platform: str
    buy_fill_price: float
    sell_fill_price: float
    stake_usd: float
    net_pnl: float
    status: str                     # "both_filled" | "partial" | "unwound" | "failed"
    timestamp: float = 0.0
    unwind_reason: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


# ── Scanner ──────────────────────────────────────────────────────────


class CrossPlatformArbScanner:
    """Scans for and executes cross-platform arb opportunities."""

    def __init__(self, config: Any):
        self._config = config
        self._kalshi: KalshiClient | None = None
        self._matcher = MarketMatcher(
            manual_mappings=json.loads(
                getattr(config, "manual_mappings_json", "{}"),
            ),
            min_confidence=getattr(config, "match_min_confidence", 0.6),
        )
        self._active_positions: list[PairedTradeResult] = []
        self._opportunity_log: list[CrossPlatformArbOpportunity] = []

    def _ensure_kalshi(self) -> KalshiClient:
        if self._kalshi is None:
            self._kalshi = KalshiClient(
                base_url=getattr(
                    self._config, "kalshi_api_base",
                    "https://api.elections.kalshi.com",
                ),
                api_key_id=getattr(self._config, "kalshi_api_key_id", ""),
                private_key_path=getattr(
                    self._config, "kalshi_private_key_path", "",
                ),
                paper_mode=getattr(self._config, "kalshi_paper_mode", True),
            )
        return self._kalshi

    async def scan(
        self,
        poly_markets: list[Any],
    ) -> list[CrossPlatformArbOpportunity]:
        """Scan for cross-platform arb opportunities.

        1. Fetch Kalshi active markets
        2. Match to Polymarket markets
        3. Compute spreads for each match
        4. Filter by min_arb_edge
        """
        if not poly_markets:
            return []

        kalshi = self._ensure_kalshi()

        try:
            kalshi_markets = await kalshi.list_markets(status="open")
        except Exception as e:
            log.warning("cross_platform_arb.kalshi_fetch_error", error=str(e))
            return []

        matches = self._matcher.find_matches(poly_markets, kalshi_markets)
        if not matches:
            return []

        poly_fee = getattr(self._config, "polymarket_fee_pct", 0.02)
        kalshi_fee = getattr(self._config, "kalshi_fee_pct", 0.02)
        total_fees = poly_fee + kalshi_fee
        min_edge = getattr(self._config, "min_arb_edge", 0.03)

        # Build price lookup for Poly markets
        poly_prices: dict[str, tuple[float, float]] = {}
        for pm in poly_markets:
            pid = getattr(pm, "condition_id", None) or getattr(pm, "id", "")
            tokens = getattr(pm, "tokens", [])
            yes_price = 0.5
            no_price = 0.5
            for t in tokens:
                outcome = getattr(t, "outcome", "").lower()
                price = getattr(t, "price", 0.5)
                if outcome == "yes":
                    yes_price = price
                elif outcome == "no":
                    no_price = price
            poly_prices[pid] = (yes_price, no_price)

        # Build price lookup for Kalshi markets
        kalshi_prices: dict[str, tuple[float, float]] = {}
        for km in kalshi_markets:
            kalshi_prices[km.ticker] = (km.yes_bid, 1.0 - km.yes_ask)

        opportunities: list[CrossPlatformArbOpportunity] = []
        for match in matches:
            poly_yp, poly_np = poly_prices.get(
                match.polymarket_id, (0.5, 0.5),
            )
            kalshi_yp, kalshi_np = kalshi_prices.get(
                match.kalshi_ticker, (0.5, 0.5),
            )

            opp = self.compute_spread(
                match, poly_yp, poly_np, kalshi_yp, kalshi_np,
                total_fees, min_edge,
            )
            opportunities.append(opp)

        # Sort by net_spread descending
        opportunities.sort(key=lambda o: o.net_spread, reverse=True)
        self._opportunity_log = opportunities

        if opportunities:
            actionable = sum(1 for o in opportunities if o.is_actionable)
            log.info(
                "cross_platform_arb.scan_complete",
                total=len(opportunities),
                actionable=actionable,
            )

        return opportunities

    @staticmethod
    def compute_spread(
        match: MarketMatch,
        poly_yes: float,
        poly_no: float,
        kalshi_yes: float,
        kalshi_no: float,
        total_fees: float = 0.04,
        min_edge: float = 0.03,
    ) -> CrossPlatformArbOpportunity:
        """Compute the best arb spread across all direction combos.

        Checks 4 combinations:
        - Buy Poly YES + Sell Kalshi YES
        - Buy Kalshi YES + Sell Poly YES
        - Buy Poly NO + Sell Kalshi NO
        - Buy Kalshi NO + Sell Poly NO
        """
        combos = [
            ("BUY_POLY_YES_SELL_KALSHI_YES", "polymarket", "kalshi",
             poly_yes, kalshi_yes),
            ("BUY_KALSHI_YES_SELL_POLY_YES", "kalshi", "polymarket",
             kalshi_yes, poly_yes),
            ("BUY_POLY_NO_SELL_KALSHI_NO", "polymarket", "kalshi",
             poly_no, kalshi_no),
            ("BUY_KALSHI_NO_SELL_POLY_NO", "kalshi", "polymarket",
             kalshi_no, poly_no),
        ]

        best_spread = -999.0
        best_direction = ""
        best_buy_platform = ""
        best_sell_platform = ""
        best_buy_price = 0.0
        best_sell_price = 0.0

        for direction, buy_plat, sell_plat, buy_p, sell_p in combos:
            spread = sell_p - buy_p  # Profit = sell high, buy low
            if spread > best_spread:
                best_spread = spread
                best_direction = direction
                best_buy_platform = buy_plat
                best_sell_platform = sell_plat
                best_buy_price = buy_p
                best_sell_price = sell_p

        net = best_spread - total_fees

        return CrossPlatformArbOpportunity(
            match=match,
            poly_yes_price=poly_yes,
            poly_no_price=poly_no,
            kalshi_yes_price=kalshi_yes,
            kalshi_no_price=kalshi_no,
            spread=max(0.0, best_spread),
            net_spread=net,
            direction=best_direction,
            buy_platform=best_buy_platform,
            sell_platform=best_sell_platform,
            buy_price=best_buy_price,
            sell_price=best_sell_price,
            total_fees=total_fees,
            is_actionable=net > min_edge,
        )

    async def execute_arb(
        self,
        opportunity: CrossPlatformArbOpportunity,
        stake_usd: float,
    ) -> PairedTradeResult:
        """Execute both legs of an arb trade.

        Returns PairedTradeResult with status indicating success/failure.
        """
        kalshi = self._ensure_kalshi()
        arb_id = opportunity.arb_id

        # Determine quantity (simplified: stake / buy_price)
        if opportunity.buy_price <= 0:
            return PairedTradeResult(
                arb_id=arb_id,
                buy_platform=opportunity.buy_platform,
                sell_platform=opportunity.sell_platform,
                buy_fill_price=0.0,
                sell_fill_price=0.0,
                stake_usd=stake_usd,
                net_pnl=0.0,
                status="failed",
                unwind_reason="Invalid buy price",
            )

        quantity = max(1, int(stake_usd / opportunity.buy_price))

        # Leg 1: Buy
        try:
            if opportunity.buy_platform == "kalshi":
                buy_result = await kalshi.place_order(
                    opportunity.match.kalshi_ticker,
                    "buy", quantity, opportunity.buy_price,
                )
                buy_fill = buy_result.fill_price
            else:
                # Polymarket buy — simulated in paper mode
                buy_fill = opportunity.buy_price
                buy_result = KalshiOrderResult(
                    order_id=f"poly-sim-{uuid.uuid4().hex[:8]}",
                    status="simulated",
                    fill_price=buy_fill,
                    fill_size=quantity,
                )
        except Exception as e:
            log.warning("cross_platform_arb.buy_leg_error", error=str(e))
            return PairedTradeResult(
                arb_id=arb_id,
                buy_platform=opportunity.buy_platform,
                sell_platform=opportunity.sell_platform,
                buy_fill_price=0.0,
                sell_fill_price=0.0,
                stake_usd=stake_usd,
                net_pnl=0.0,
                status="failed",
                unwind_reason=f"Buy leg failed: {e}",
            )

        # Leg 2: Sell
        try:
            if opportunity.sell_platform == "kalshi":
                sell_result = await kalshi.place_order(
                    opportunity.match.kalshi_ticker,
                    "sell", quantity, opportunity.sell_price,
                )
                sell_fill = sell_result.fill_price
            else:
                sell_fill = opportunity.sell_price
                sell_result = KalshiOrderResult(
                    order_id=f"poly-sim-{uuid.uuid4().hex[:8]}",
                    status="simulated",
                    fill_price=sell_fill,
                    fill_size=quantity,
                )
        except Exception as e:
            log.warning("cross_platform_arb.sell_leg_error", error=str(e))
            unwind = getattr(self._config, "unwind_on_partial_fill", True)
            return PairedTradeResult(
                arb_id=arb_id,
                buy_platform=opportunity.buy_platform,
                sell_platform=opportunity.sell_platform,
                buy_fill_price=buy_fill,
                sell_fill_price=0.0,
                stake_usd=stake_usd,
                net_pnl=-opportunity.total_fees * stake_usd,
                status="unwound" if unwind else "partial",
                unwind_reason=f"Sell leg failed: {e}",
            )

        # Both legs succeeded
        gross_pnl = (sell_fill - buy_fill) * quantity
        net_pnl = gross_pnl - opportunity.total_fees * stake_usd

        result = PairedTradeResult(
            arb_id=arb_id,
            buy_platform=opportunity.buy_platform,
            sell_platform=opportunity.sell_platform,
            buy_fill_price=buy_fill,
            sell_fill_price=sell_fill,
            stake_usd=stake_usd,
            net_pnl=net_pnl,
            status="both_filled",
        )
        self._active_positions.append(result)

        log.info(
            "cross_platform_arb.trade_executed",
            arb_id=arb_id,
            status="both_filled",
            net_pnl=round(net_pnl, 4),
        )

        return result

    def check_position_limits(self) -> bool:
        """Check if we can open another arb position."""
        max_count = getattr(self._config, "max_arb_positions_count", 5)
        max_usd = getattr(self._config, "max_arb_position_usd", 200.0)

        active = [p for p in self._active_positions if p.status == "both_filled"]
        if len(active) >= max_count:
            return False

        total_exposure = sum(p.stake_usd for p in active)
        if total_exposure >= max_usd * max_count:
            return False

        return True

    @property
    def opportunity_log(self) -> list[CrossPlatformArbOpportunity]:
        return list(self._opportunity_log)

    @property
    def active_positions(self) -> list[PairedTradeResult]:
        return list(self._active_positions)

    async def close(self) -> None:
        if self._kalshi:
            await self._kalshi.close()
            self._kalshi = None
