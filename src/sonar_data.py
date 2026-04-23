"""SonarQube data collection and mock fallback helpers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests
import streamlit as st

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, build_cache_key


class SonarAPIError(RuntimeError):
    """Raised when a SonarQube API call fails."""


ProgressCallback = Callable[[float, str], None]


SONAR_ISSUE_COLUMNS = ["issue_key", "file_path", "severity", "type", "status", "message"]
SONAR_SUMMARY_COLUMNS = ["file_path", "sonar_issue_count", "highest_severity", "severity_breakdown", "type_breakdown"]


def _emit_progress(callback: ProgressCallback | None, progress: float, message: str) -> None:
    """Send a bounded progress update if a callback is available."""
    if callback is None:
        return
    callback(max(0.0, min(1.0, progress)), message)


def _request_json(base_url: str, token: str, params: dict[str, Any]) -> dict[str, Any]:
    """Execute a JSON request against SonarQube."""
    normalized_base_url = base_url.rstrip("/")
    response = requests.get(
        f"{normalized_base_url}/api/issues/search",
        params=params,
        auth=(token, ""),
        timeout=30,
    )
    if response.status_code >= 400:
        detail = ""
        try:
            detail = response.json().get("errors", [{}])[0].get("msg", "")
        except ValueError:
            detail = response.text[:250]
        raise SonarAPIError(f"{response.status_code} response from SonarQube. {detail}".strip())
    return response.json()


def _extract_file_path(component: str) -> str:
    """Extract a repository-relative file path from a SonarQube component key."""
    if ":" not in component:
        return component
    return component.split(":", maxsplit=1)[1]


def _write_json(path: Path, payload: Any) -> None:
    """Persist a JSON payload to local cache storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_dataframe(path: Path, dataframe: pd.DataFrame) -> None:
    """Persist a dataframe to CSV cache storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def _read_dataframe(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a cached CSV dataframe and guarantee its expected columns."""
    if not path.exists():
        return pd.DataFrame(columns=columns)
    dataframe = pd.read_csv(path)
    for column in columns:
        if column not in dataframe.columns:
            dataframe[column] = pd.NA
    return dataframe[columns]


def _sonar_cache_paths(project_key: str) -> tuple[Path, Path]:
    """Return the cache paths used for SonarCloud collection."""
    cache_key = build_cache_key("sonarqube_issues", project_key)
    return DATA_RAW_DIR / f"{cache_key}.json", DATA_PROCESSED_DIR / f"{cache_key}.csv"


def _load_sonar_cache(raw_path: Path, summary_path: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Load SonarCloud data from local cache when available."""
    if not (raw_path.exists() and summary_path.exists()):
        return None

    try:
        raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
        issue_rows = []
        for issue in raw_payload:
            issue_rows.append(
                {
                    "issue_key": issue.get("key"),
                    "file_path": _extract_file_path(issue.get("component", "")),
                    "severity": issue.get("severity", "UNKNOWN"),
                    "type": issue.get("type", "UNKNOWN"),
                    "status": issue.get("status", "UNKNOWN"),
                    "message": issue.get("message", ""),
                }
            )
        issue_df = pd.DataFrame(issue_rows)
        if issue_df.empty:
            issue_df = pd.DataFrame(columns=SONAR_ISSUE_COLUMNS)
        else:
            issue_df = issue_df[SONAR_ISSUE_COLUMNS]

        summary_df = _read_dataframe(summary_path, SONAR_SUMMARY_COLUMNS)
        return issue_df, summary_df
    except Exception:
        return None


def _fetch_sonar_issues_impl(
    base_url: str,
    token: str,
    project_key: str,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch SonarQube code smell issues and summarize them by file."""
    if not base_url or not token or not project_key:
        raise SonarAPIError("SonarQube base URL, token, and project key are required.")

    raw_cache_path, summary_cache_path = _sonar_cache_paths(project_key)
    cached_sonar_data = _load_sonar_cache(raw_cache_path, summary_cache_path)
    if cached_sonar_data is not None:
        cached_issue_df, cached_summary_df = cached_sonar_data
        _emit_progress(
            progress_callback,
            1.0,
            f"SonarCloud: loaded issue data from local cache. Found {len(cached_issue_df)} issues.",
        )
        return cached_issue_df, cached_summary_df

    issues: list[dict[str, Any]] = []
    page = 1
    page_size = 500

    _emit_progress(progress_callback, 0.05, "SonarCloud: requesting open code smell issues...")

    while True:
        payload = _request_json(
            base_url,
            token,
            {
                "componentKeys": project_key,
                "types": "CODE_SMELL",
                "resolved": "false",
                "p": page,
                "ps": page_size,
            },
        )
        batch = payload.get("issues", [])
        total = int(payload.get("total", 0))
        if not batch:
            break

        issues.extend(batch)
        progress_value = 0.15 if total == 0 else 0.15 + (0.75 * (len(issues) / total))
        _emit_progress(
            progress_callback,
            progress_value,
            f"SonarCloud: fetched {len(issues)}/{total} open code smell issues.",
        )

        if len(issues) >= total:
            break
        page += 1

    if not issues:
        empty_issue_df = pd.DataFrame(columns=SONAR_ISSUE_COLUMNS)
        empty_summary_df = pd.DataFrame(columns=SONAR_SUMMARY_COLUMNS)
        _emit_progress(progress_callback, 1.0, "SonarCloud: no open code smell issues were returned.")
        return empty_issue_df, empty_summary_df

    issue_rows: list[dict[str, Any]] = []
    for issue in issues:
        issue_rows.append(
            {
                "issue_key": issue.get("key"),
                "file_path": _extract_file_path(issue.get("component", "")),
                "severity": issue.get("severity", "UNKNOWN"),
                "type": issue.get("type", "UNKNOWN"),
                "status": issue.get("status", "UNKNOWN"),
                "message": issue.get("message", ""),
            }
        )

    issue_df = pd.DataFrame(issue_rows)[SONAR_ISSUE_COLUMNS]

    summary_rows: list[dict[str, Any]] = []
    severity_order = {"BLOCKER": 5, "CRITICAL": 4, "MAJOR": 3, "MINOR": 2, "INFO": 1, "UNKNOWN": 0}
    for file_path, group in issue_df.groupby("file_path", dropna=True):
        severity_counts = Counter(group["severity"])
        type_counts = Counter(group["type"])
        highest_severity = max(severity_counts, key=lambda severity: severity_order.get(severity, 0))
        summary_rows.append(
            {
                "file_path": file_path,
                "sonar_issue_count": int(len(group)),
                "highest_severity": highest_severity,
                "severity_breakdown": ", ".join(
                    f"{severity}:{count}" for severity, count in sorted(severity_counts.items(), reverse=True)
                ),
                "type_breakdown": ", ".join(
                    f"{issue_type}:{count}" for issue_type, count in sorted(type_counts.items(), reverse=True)
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("sonar_issue_count", ascending=False)
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=SONAR_SUMMARY_COLUMNS)

    _write_json(raw_cache_path, issues)
    _write_dataframe(summary_cache_path, summary_df)
    _emit_progress(progress_callback, 1.0, "SonarCloud: issue aggregation complete.")

    return issue_df, summary_df


@st.cache_data(show_spinner=False, ttl=900)
def fetch_sonar_issues(base_url: str, token: str, project_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached wrapper around SonarQube issue collection."""
    return _fetch_sonar_issues_impl(base_url, token, project_key)


def fetch_sonar_issues_with_progress(
    base_url: str,
    token: str,
    project_key: str,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uncached SonarQube issue collection with live progress reporting."""
    return _fetch_sonar_issues_impl(base_url, token, project_key, progress_callback)


@st.cache_data(show_spinner=False)
def load_mock_sonar_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load small mock SonarQube data for local demos and offline work."""
    issue_df = pd.DataFrame(
        [
            {"issue_key": "SQ-1", "file_path": "src/api.py", "severity": "MAJOR", "type": "CODE_SMELL", "status": "OPEN", "message": "Large method should be split."},
            {"issue_key": "SQ-2", "file_path": "src/api.py", "severity": "CRITICAL", "type": "CODE_SMELL", "status": "OPEN", "message": "Nested conditionals reduce readability."},
            {"issue_key": "SQ-3", "file_path": "src/auth.py", "severity": "MAJOR", "type": "CODE_SMELL", "status": "OPEN", "message": "Duplicate validation logic."},
            {"issue_key": "SQ-4", "file_path": "src/reporting.py", "severity": "MAJOR", "type": "CODE_SMELL", "status": "OPEN", "message": "Function exceeds cognitive complexity threshold."},
            {"issue_key": "SQ-5", "file_path": "src/reporting.py", "severity": "MINOR", "type": "CODE_SMELL", "status": "OPEN", "message": "Rename ambiguous variable."},
            {"issue_key": "SQ-6", "file_path": "src/dashboard.py", "severity": "MINOR", "type": "CODE_SMELL", "status": "OPEN", "message": "Repeated layout construction."},
        ]
    )

    summary_df = pd.DataFrame(
        [
            {
                "file_path": "src/api.py",
                "sonar_issue_count": 2,
                "highest_severity": "CRITICAL",
                "severity_breakdown": "CRITICAL:1, MAJOR:1",
                "type_breakdown": "CODE_SMELL:2",
            },
            {
                "file_path": "src/reporting.py",
                "sonar_issue_count": 2,
                "highest_severity": "MAJOR",
                "severity_breakdown": "MAJOR:1, MINOR:1",
                "type_breakdown": "CODE_SMELL:2",
            },
            {
                "file_path": "src/auth.py",
                "sonar_issue_count": 1,
                "highest_severity": "MAJOR",
                "severity_breakdown": "MAJOR:1",
                "type_breakdown": "CODE_SMELL:1",
            },
            {
                "file_path": "src/dashboard.py",
                "sonar_issue_count": 1,
                "highest_severity": "MINOR",
                "severity_breakdown": "MINOR:1",
                "type_breakdown": "CODE_SMELL:1",
            },
        ]
    )
    return issue_df, summary_df
