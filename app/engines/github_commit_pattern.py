"""
GitHub Repo Commit-Pattern engine — an original implementation of the
"commit pattern" signal that competitor tools (isvibecoded.com, vibedetector,
etc.) describe using: AI-assisted / "vibe-coded" repos often show bursts of
large, multi-file commits within the same minute, with generic messages
like "fix bug" or "update ui" — patterns a human typing normally wouldn't
produce. This engine calls the public GitHub REST API (no auth needed for
public repos, rate-limited) to fetch recent commit history and check for
exactly those patterns.
"""
import re
from collections import Counter
from datetime import datetime

import requests

from app.engines.base import Engine, EngineOutput

GENERIC_MESSAGE_PATTERNS = [
    r"^fix(ed)?\s*(bug|bugs|issue)?s?$",
    r"^update(d)?\s*(ui|code|files?|readme)?$",
    r"^wip$",
    r"^initial commit$",
    r"^changes?$",
    r"^minor (fix|update|changes?)$",
    r"^cleanup$",
    r"^refactor(ed)?$",
    r"^\.$",
    r"^style$",
    r"^formatting$",
]

REQUEST_TIMEOUT = 10


class GithubRepoError(Exception):
    pass


def parse_github_url(url: str) -> tuple[str, str]:
    match = re.search(r"github\.com/([^/]+)/([^/?#]+)", url)
    if not match:
        raise GithubRepoError("Couldn't parse a GitHub owner/repo from this URL.")
    owner, repo = match.group(1), match.group(2).removesuffix(".git")
    return owner, repo


class GithubCommitPatternEngine(Engine):
    engine_id = "github_commit_pattern"
    engine_name = "GitHub Commit Pattern"
    category = "website"
    weight = 1.0

    def applies_to(self, input_type: str) -> bool:
        return input_type == "github_repo"

    def run(self, *, text: str = "", html: str = "", url: str = "", raw_html_before_js: str = "") -> EngineOutput:
        try:
            owner, repo = parse_github_url(url)
            resp = requests.get(
                f"https://api.github.com/repos/{owner}/{repo}/commits",
                params={"per_page": 50},
                headers={"Accept": "application/vnd.github+json"},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 404:
                raise GithubRepoError("Repository not found — check the URL and that it's public.")
            if resp.status_code == 403:
                raise GithubRepoError("GitHub API rate limit hit — try again in a bit.")
            resp.raise_for_status()
            commits = resp.json()
        except requests.RequestException as e:
            raise GithubRepoError(f"Could not reach GitHub: {e}") from e

        if not commits:
            return EngineOutput(
                engine_id=self.engine_id,
                engine_name=self.engine_name,
                category=self.category,
                score=0.0,
                confidence="low",
                summary="No commit history found.",
                details={"reason": "no_commits"},
            )

        messages = [c["commit"]["message"].splitlines()[0].strip().lower() for c in commits]
        timestamps = [
            datetime.fromisoformat(c["commit"]["author"]["date"].replace("Z", "+00:00"))
            for c in commits
        ]

        generic_count = sum(
            1 for m in messages if any(re.match(p, m) for p in GENERIC_MESSAGE_PATTERNS)
        )
        generic_ratio = generic_count / len(messages)

        # Burst detection: commits sharing the exact same minute
        minute_buckets = Counter(t.strftime("%Y-%m-%d %H:%M") for t in timestamps)
        burst_minutes = sum(1 for count in minute_buckets.values() if count >= 3)
        burst_ratio = burst_minutes / len(minute_buckets) if minute_buckets else 0

        points = 0
        max_points = 100
        if generic_ratio > 0.6:
            points += 50
        elif generic_ratio > 0.35:
            points += 25
        if burst_ratio > 0.15:
            points += 50
        elif burst_ratio > 0.05:
            points += 25

        overall_score = min(100, points)
        confidence = "high" if len(commits) >= 20 else "medium"

        return EngineOutput(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            category=self.category,
            score=overall_score,
            confidence=confidence,
            summary=(
                f"{generic_ratio:.0%} of recent commit messages are generic; "
                f"{burst_minutes} minute(s) had 3+ commits at once."
            ),
            details={
                "commits_analyzed": len(commits),
                "generic_message_ratio": round(generic_ratio, 3),
                "burst_minutes_detected": burst_minutes,
                "score_breakdown": {
                    "generic_commit_messages": min(50, points if generic_ratio > 0.35 else 0),
                    "burst_commit_pattern": min(50, points if burst_ratio > 0.05 else 0),
                },
                "max_points_possible": max_points,
            },
        )
