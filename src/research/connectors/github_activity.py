"""GitHub activity connector — development activity signal for TECHNOLOGY markets.

Fetches commit frequency, release cadence, and star trends from the GitHub
REST API for repositories mentioned in prediction market questions.

Uses GITHUB_TOKEN env var for authenticated requests (optional but
recommended — unauthenticated rate limit is 60 req/h).

Rate-limited via a dedicated ``github`` bucket.
"""

from __future__ import annotations

import os
import re
from typing import Any

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_GITHUB_API = "https://api.github.com"

# Project name → GitHub owner/repo mapping
_PROJECT_REPOS: dict[str, str] = {
    "bitcoin": "bitcoin/bitcoin",
    "btc": "bitcoin/bitcoin",
    "ethereum": "ethereum/go-ethereum",
    "eth": "ethereum/go-ethereum",
    "solana": "solana-labs/solana",
    "sol": "solana-labs/solana",
    "cardano": "IntersectMBO/cardano-node",
    "ada": "IntersectMBO/cardano-node",
    "polkadot": "paritytech/polkadot-sdk",
    "dot": "paritytech/polkadot-sdk",
    "cosmos": "cosmos/cosmos-sdk",
    "atom": "cosmos/cosmos-sdk",
    "avalanche": "ava-labs/avalanchego",
    "avax": "ava-labs/avalanchego",
    "polygon": "maticnetwork/bor",
    "matic": "maticnetwork/bor",
    "rust": "rust-lang/rust",
    "python": "python/cpython",
    "typescript": "microsoft/TypeScript",
    "react": "facebook/react",
    "pytorch": "pytorch/pytorch",
    "tensorflow": "tensorflow/tensorflow",
    "openai": "openai/openai-python",
    "llama": "meta-llama/llama",
    "linux": "torvalds/linux",
    "kubernetes": "kubernetes/kubernetes",
    "docker": "moby/moby",
}


class GitHubActivityConnector(BaseResearchConnector):
    """GitHub development activity signal for TECHNOLOGY and CRYPTO markets."""

    @property
    def name(self) -> str:
        return "github_activity"

    def relevant_categories(self) -> set[str]:
        return {"TECHNOLOGY", "CRYPTO"}

    def is_relevant(self, question: str, market_type: str) -> bool:
        if market_type not in self.relevant_categories():
            return False
        return self._extract_repo(question) is not None

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        repo = self._extract_repo(question)
        if not repo:
            return []

        token = os.environ.get("GITHUB_TOKEN", "")
        headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        await rate_limiter.get("github").acquire()

        client = self._get_client(timeout=10.0)

        # Fetch repo metadata
        resp = await client.get(
            f"{_GITHUB_API}/repos/{repo}",
            headers=headers,
        )
        resp.raise_for_status()
        repo_data = resp.json()

        stars = repo_data.get("stargazers_count", 0)
        forks = repo_data.get("forks_count", 0)
        open_issues = repo_data.get("open_issues_count", 0)
        pushed_at = repo_data.get("pushed_at", "")

        # Fetch recent commit activity (last 4 weeks)
        await rate_limiter.get("github").acquire()
        resp2 = await client.get(
            f"{_GITHUB_API}/repos/{repo}/stats/commit_activity",
            headers=headers,
        )
        if resp2.status_code == 200:
            weekly_data = resp2.json()
            if isinstance(weekly_data, list) and len(weekly_data) >= 4:
                recent_4w = weekly_data[-4:]
                commits_4w = sum(w.get("total", 0) for w in recent_4w)
                prev_4w = weekly_data[-8:-4] if len(weekly_data) >= 8 else []
                commits_prev = sum(w.get("total", 0) for w in prev_4w)
            else:
                commits_4w = 0
                commits_prev = 0
        else:
            commits_4w = 0
            commits_prev = 0

        # Commit trend
        if commits_prev > 0:
            commit_change = (commits_4w - commits_prev) / commits_prev
        else:
            commit_change = 0.0

        if commit_change > 0.2:
            activity_trend = "increasing"
        elif commit_change < -0.2:
            activity_trend = "decreasing"
        else:
            activity_trend = "stable"

        # Fetch latest release
        await rate_limiter.get("github").acquire()
        resp3 = await client.get(
            f"{_GITHUB_API}/repos/{repo}/releases/latest",
            headers=headers,
        )
        latest_release = ""
        if resp3.status_code == 200:
            release_data = resp3.json()
            latest_release = release_data.get("tag_name", "")

        repo_name = repo_data.get("full_name", repo)
        content = (
            f"GitHub Activity: {repo_name}\n"
            f"  Stars: {stars:,}\n"
            f"  Forks: {forks:,}\n"
            f"  Open issues: {open_issues:,}\n"
            f"  Commits (4 weeks): {commits_4w}\n"
            f"  Commit trend: {activity_trend} ({commit_change:+.0%} vs prior 4w)\n"
            f"  Latest release: {latest_release or 'N/A'}\n"
            f"  Last push: {pushed_at[:10] if pushed_at else 'N/A'}\n"
            f"  Source: GitHub API"
        )

        return [
            self._make_source(
                title=f"GitHub: {repo_name}",
                url=f"https://github.com/{repo}",
                snippet=(
                    f"{repo_name}: {stars:,} stars, "
                    f"{commits_4w} commits/4w ({activity_trend})"
                ),
                publisher="GitHub",
                content=content,
                authority_score=0.60,
                raw={
                    "behavioral_signal": {
                        "source": "github_activity",
                        "signal_type": "dev_activity",
                        "value": commits_4w,
                        "stars": stars,
                        "forks": forks,
                        "commit_change_pct": round(commit_change, 4),
                        "activity_trend": activity_trend,
                        "latest_release": latest_release,
                        "repo": repo,
                    }
                },
            )
        ]

    @staticmethod
    def _extract_repo(question: str) -> str | None:
        """Extract GitHub repo from question text.

        First tries explicit 'owner/repo' patterns, then keyword mapping.
        """
        # Try explicit owner/repo pattern
        m = re.search(r"github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)", question)
        if m:
            return m.group(1)

        q = question.lower()
        for keyword, repo in _PROJECT_REPOS.items():
            if keyword in q:
                return repo
        return None
