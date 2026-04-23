"""Configuration helpers for the Technical Debt Hotspot Dashboard."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DEFAULT_BRANCH = "main"
MAINTENANCE_KEYWORDS = [
    "refactor",
    "cleanup",
    "technical debt",
    "maintainability",
    "code smell",
]


@dataclass(slots=True)
class AppConfig:
    """Application inputs selected from the Streamlit sidebar."""

    github_owner: str
    github_repo: str
    default_branch: str
    start_date: date
    end_date: date
    github_token: str
    sonar_base_url: str
    sonar_token: str
    sonar_project_key: str
    weight_churn: float
    weight_sonar: float
    weight_github_issues: float
    use_mock_data: bool = False


def ensure_data_directories() -> None:
    """Create cache directories used for raw and processed API payloads."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def get_default_sidebar_values() -> dict[str, object]:
    """Load default sidebar values from the environment."""
    load_dotenv(BASE_DIR / ".env", override=False)
    today = date.today()
    return {
        "github_owner": os.getenv("GITHUB_OWNER", ""),
        "github_repo": os.getenv("GITHUB_REPO", ""),
        "default_branch": os.getenv("GITHUB_DEFAULT_BRANCH", DEFAULT_BRANCH),
        "start_date": today - timedelta(days=90),
        "end_date": today,
        "github_token": os.getenv("GITHUB_TOKEN", ""),
        "sonar_base_url": os.getenv("SONARQUBE_BASE_URL", ""),
        "sonar_token": os.getenv("SONARQUBE_TOKEN", ""),
        "sonar_project_key": os.getenv("SONARQUBE_PROJECT_KEY", ""),
        "use_mock_data": os.getenv("USE_MOCK_DATA", "false").lower() == "true",
    }


def build_cache_key(*parts: object) -> str:
    """Create a filename-safe cache key from user inputs."""
    raw_key = "__".join(str(part).strip() for part in parts if str(part).strip())
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_key)
    return normalized.strip("_") or "cache"
