"""Unit tests for src/execution/direction.py."""

from __future__ import annotations

import pytest

from src.execution.direction import (
    parse_direction,
    canonical_direction,
    is_long,
    is_short,
)


class TestParseDirection:
    def test_buy_yes(self):
        assert parse_direction("BUY_YES") == ("BUY", "YES")

    def test_buy_no(self):
        assert parse_direction("BUY_NO") == ("BUY", "NO")

    def test_sell(self):
        assert parse_direction("SELL") == ("SELL", "")

    def test_buy(self):
        assert parse_direction("BUY") == ("BUY", "")

    def test_unknown(self):
        assert parse_direction("HOLD") == ("", "")

    def test_empty_string(self):
        assert parse_direction("") == ("", "")

    def test_lowercase_not_matched(self):
        assert parse_direction("buy_yes") == ("", "")


class TestCanonicalDirection:
    def test_buy_yes_roundtrip(self):
        a, o = parse_direction("BUY_YES")
        assert canonical_direction(a, o) == "BUY_YES"

    def test_buy_no_roundtrip(self):
        a, o = parse_direction("BUY_NO")
        assert canonical_direction(a, o) == "BUY_NO"

    def test_sell_roundtrip(self):
        a, o = parse_direction("SELL")
        assert canonical_direction(a, o) == "SELL"

    def test_buy_roundtrip(self):
        a, o = parse_direction("BUY")
        assert canonical_direction(a, o) == "BUY"

    def test_empty_returns_empty(self):
        assert canonical_direction("", "") == ""

    def test_sell_with_outcome_still_sell(self):
        assert canonical_direction("SELL", "YES") == "SELL"


class TestIsLong:
    def test_long_yes(self):
        assert is_long("BUY", "YES") is True

    def test_long_no(self):
        assert is_long("BUY", "NO") is False

    def test_sell(self):
        assert is_long("SELL", "YES") is False

    def test_empty(self):
        assert is_long("", "") is False


class TestIsShort:
    def test_short_no(self):
        assert is_short("BUY", "NO") is True

    def test_short_yes(self):
        assert is_short("BUY", "YES") is False

    def test_sell(self):
        assert is_short("SELL", "NO") is False

    def test_empty(self):
        assert is_short("", "") is False
