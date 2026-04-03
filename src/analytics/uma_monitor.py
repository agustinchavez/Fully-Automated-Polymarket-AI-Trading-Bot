"""UMA Oracle dispute resolution monitor.

Checks for active UMA disputes that could affect Polymarket market
resolution. Markets with open disputes have high tail risk (20-40% price
swings) and should be avoided.

API: https://oracle.uma.xyz — free, no auth, 60 req/min.
"""

from __future__ import annotations

import time
from typing import Any

from src.observability.logger import get_logger

log = get_logger(__name__)


class UMAMonitor:
    """Monitor UMA Oracle for active market disputes."""

    BASE = "https://oracle.uma.xyz/api/v1"

    def __init__(self, refresh_interval_mins: int = 15) -> None:
        self._disputed: set[str] = set()
        self._last_refresh: float = 0.0
        self._interval = refresh_interval_mins * 60

    async def refresh_disputes(self) -> None:
        """Fetch open disputes from UMA Oracle API."""
        now = time.time()
        if now - self._last_refresh < self._interval:
            return

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE}/disputes",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    data = await resp.json()

            self._disputed = {
                d["ancillaryData"]
                for d in data
                if d.get("status") == "open" and d.get("ancillaryData")
            }
            self._last_refresh = now
            log.info(
                "uma.refresh_complete",
                disputed_count=len(self._disputed),
            )
        except Exception as exc:
            log.warning("uma.refresh_failed", error=str(exc))

    def is_disputed(self, condition_id: str) -> bool:
        """Check if a market condition_id has an active UMA dispute."""
        return condition_id in self._disputed

    def get_dispute_risk(self, condition_id: str) -> float:
        """Return dispute risk score: 1.0 if disputed, 0.0 otherwise."""
        return 1.0 if self.is_disputed(condition_id) else 0.0
