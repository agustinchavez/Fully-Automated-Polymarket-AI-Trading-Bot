"""Tests for base rate dashboard endpoint (Phase 2 — Batch D)."""

from __future__ import annotations

import pytest


@pytest.fixture
def app():
    from src.dashboard.app import app as flask_app
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


class TestBaseRateEndpoint:

    def test_base_rates_returns_patterns(self, client) -> None:
        """GET /api/forecast/base-rates returns patterns."""
        resp = client.get("/api/forecast/base-rates")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["pattern_count"] >= 50
        assert len(data["patterns"]) >= 50

    def test_base_rates_pattern_structure(self, client) -> None:
        """Each pattern has required fields."""
        resp = client.get("/api/forecast/base-rates")
        data = resp.get_json()
        for p in data["patterns"][:5]:
            assert "pattern" in p
            assert "category" in p
            assert "description" in p
            assert "base_rate" in p
            assert "source" in p

    def test_base_rates_includes_empirical(self, client) -> None:
        """Response includes empirical_rates dict."""
        resp = client.get("/api/forecast/base-rates")
        data = resp.get_json()
        assert "empirical_rates" in data
        assert isinstance(data["empirical_rates"], dict)

    def test_base_rates_categories_present(self, client) -> None:
        """Patterns span multiple categories."""
        resp = client.get("/api/forecast/base-rates")
        data = resp.get_json()
        categories = {p["category"] for p in data["patterns"]}
        assert len(categories) >= 5

    def test_base_rates_valid_rates(self, client) -> None:
        """All base rates are between 0 and 1."""
        resp = client.get("/api/forecast/base-rates")
        data = resp.get_json()
        for p in data["patterns"]:
            assert 0.0 < p["base_rate"] < 1.0, f"Invalid rate in: {p['description']}"
