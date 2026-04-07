"""Shared test fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set fake API keys so SDK constructors don't raise during tests.
# The actual API calls are always mocked — these just satisfy init-time checks.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key-for-testing")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-fake-key-for-testing")
