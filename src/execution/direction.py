"""Direction normalization — canonical action_side + outcome_side.

The legacy codebase uses multiple direction formats:
  - "BUY_YES", "BUY_NO" (edge calculator output / thesis direction)
  - "SELL" (exit orders)
  - "BUY" (generic buy)

This module provides a canonical two-field representation:
  - action_side: "BUY" or "SELL" (what you do at the exchange)
  - outcome_side: "YES" or "NO" (which token you operate on)
"""

from __future__ import annotations


def parse_direction(raw: str) -> tuple[str, str]:
    """Parse a raw direction string into (action_side, outcome_side).

    Returns:
        Tuple of (action_side, outcome_side). Either may be "" if
        the raw value does not contain enough information.

    Examples:
        >>> parse_direction("BUY_YES")
        ('BUY', 'YES')
        >>> parse_direction("BUY_NO")
        ('BUY', 'NO')
        >>> parse_direction("SELL")
        ('SELL', '')
        >>> parse_direction("BUY")
        ('BUY', '')
    """
    if raw == "BUY_YES":
        return ("BUY", "YES")
    if raw == "BUY_NO":
        return ("BUY", "NO")
    if raw == "SELL":
        return ("SELL", "")
    if raw == "BUY":
        return ("BUY", "")
    return ("", "")


def canonical_direction(action_side: str, outcome_side: str) -> str:
    """Reconstruct a legacy direction string from canonical fields.

    Useful for backward compatibility with code that reads the old format.
    """
    if action_side == "BUY" and outcome_side == "YES":
        return "BUY_YES"
    if action_side == "BUY" and outcome_side == "NO":
        return "BUY_NO"
    if action_side == "SELL":
        return "SELL"
    if action_side == "BUY":
        return "BUY"
    return ""


def is_long(action_side: str, outcome_side: str) -> bool:
    """Return True if this direction represents a long YES position."""
    return action_side == "BUY" and outcome_side == "YES"


def is_short(action_side: str, outcome_side: str) -> bool:
    """Return True if this direction is buying the NO token (bearish)."""
    return action_side == "BUY" and outcome_side == "NO"
