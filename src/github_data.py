"""GitHub data collection and lightweight issue-to-file heuristics."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import requests
import streamlit as st

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, MAINTENANCE_KEYWORDS, build_cache_key


class GitHubAPIError(RuntimeError):
    """Raised when a GitHub API call fails."""


ProgressCallback = Callable[[float, str], None]


COMMIT_DF_COLUMNS = ["sha", "commit_date", "contributor", "additions", "deletions", "files_changed", "message"]
COMMIT_SUMMARY_COLUMNS = ["file_path", "commit_count", "additions", "deletions", "total_churn", "unique_contributors"]
DAILY_CHURN_COLUMNS = ["commit_date", "total_churn", "commit_count"]
ISSUE_COLUMNS = [
    "issue_number",
    "title",
    "body",
    "state",
    "created_at",
    "updated_at",
    "html_url",
    "matched_keywords",
]


def _emit_progress(callback: ProgressCallback | None, progress: float, message: str) -> None:
    """Send a bounded progress update if a callback is available."""
    if callback is None:
        return
    callback(max(0.0, min(1.0, progress)), message)


def _utc_iso(date_value: date, end_of_day: bool = False) -> str:
    """Convert a date into a UTC ISO 8601 string for GitHub queries."""
    target_time = time.max if end_of_day else time.min
    iso_value = datetime.combine(date_value, target_time, tzinfo=timezone.utc).isoformat()
    return iso_value.replace("+00:00", "Z")


def _parse_timestamp(value: str) -> pd.Timestamp:
    """Parse an API timestamp into a pandas timestamp."""
    return pd.to_datetime(value, utc=True)


def _github_headers(token: str) -> dict[str, str]:
    """Build GitHub request headers."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request_response(url: str, token: str, params: dict[str, Any] | None = None) -> requests.Response:
    """Execute a request against the GitHub API and return the raw response."""
    response = requests.get(url, headers=_github_headers(token), params=params, timeout=30)
    if response.status_code >= 400:
        detail = ""
        try:
            detail = response.json().get("message", "")
        except ValueError:
            detail = response.text[:250]
        raise GitHubAPIError(f"{response.status_code} response from GitHub at {url}. {detail}".strip())
    return response


def _request_json(url: str, token: str, params: dict[str, Any] | None = None) -> Any:
    """Execute a JSON request against the GitHub API."""
    return _request_response(url, token, params).json()


def _paginate_list(url: str, token: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect all records from a paginated list endpoint."""
    records: list[dict[str, Any]] = []
    next_url = url
    next_params: dict[str, Any] | None = {**params, "per_page": 100}

    while next_url:
        response = _request_response(next_url, token, next_params)
        batch = response.json()
        if not batch:
            break
        if not isinstance(batch, list):
            raise GitHubAPIError(f"Unexpected list payload from GitHub at {next_url}.")

        records.extend(batch)
        next_url = response.links.get("next", {}).get("url")
        next_params = None

    return records


def _paginate_search(url: str, token: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect all records from a paginated GitHub search endpoint."""
    records: list[dict[str, Any]] = []
    next_url = url
    next_params: dict[str, Any] | None = {**params, "per_page": 100}

    while next_url:
        response = _request_response(next_url, token, next_params)
        payload = response.json()
        batch = payload.get("items", [])
        if not batch:
            break

        records.extend(batch)
        next_url = response.links.get("next", {}).get("url")
        next_params = None

        if not next_url and len(records) >= payload.get("total_count", 0):
            break

    return records


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


def _commit_cache_paths(
    owner: str,
    repo: str,
    branch: str,
    start_date: date,
    end_date: date,
) -> tuple[Path, Path, Path]:
    """Return the cache paths used for commit collection."""
    cache_key = build_cache_key("github_commits", owner, repo, branch, start_date, end_date)
    return (
        DATA_RAW_DIR / f"{cache_key}.json",
        DATA_PROCESSED_DIR / f"{cache_key}_summary.csv",
        DATA_PROCESSED_DIR / f"{cache_key}_daily.csv",
    )


def _issue_cache_paths(owner: str, repo: str, start_date: date, end_date: date) -> tuple[Path, Path]:
    """Return the cache paths used for issue collection."""
    cache_key = build_cache_key("github_issues", owner, repo, start_date, end_date)
    return DATA_RAW_DIR / f"{cache_key}.json", DATA_PROCESSED_DIR / f"{cache_key}.csv"


def _load_commit_cache(raw_path: Path, summary_path: Path, daily_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Load commit data from local cache when available."""
    if not (raw_path.exists() and summary_path.exists() and daily_path.exists()):
        return None

    try:
        raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
        commit_rows = []
        for item in raw_payload:
            stats = item.get("stats", {}) or {}
            files = item.get("files", []) or []
            commit_rows.append(
                {
                    "sha": item.get("sha"),
                    "commit_date": item.get("commit_date"),
                    "contributor": item.get("contributor", "unknown"),
                    "additions": stats.get("additions", 0),
                    "deletions": stats.get("deletions", 0),
                    "files_changed": len(files),
                    "message": item.get("message", ""),
                }
            )

        commit_df = pd.DataFrame(commit_rows)
        if commit_df.empty:
            commit_df = pd.DataFrame(columns=COMMIT_DF_COLUMNS)
        else:
            commit_df = commit_df.reindex(columns=COMMIT_DF_COLUMNS)
            commit_df["commit_date"] = pd.to_datetime(commit_df["commit_date"], errors="coerce").dt.date

        summary_df = _read_dataframe(summary_path, COMMIT_SUMMARY_COLUMNS)
        daily_df = _read_dataframe(daily_path, DAILY_CHURN_COLUMNS)
        if not daily_df.empty:
            daily_df["commit_date"] = pd.to_datetime(daily_df["commit_date"], errors="coerce").dt.date

        return commit_df, summary_df, daily_df
    except Exception:
        return None


def _load_issue_cache(raw_path: Path, csv_path: Path) -> pd.DataFrame | None:
    """Load issue data from local cache when available."""
    if csv_path.exists():
        try:
            issue_df = _read_dataframe(csv_path, ISSUE_COLUMNS)
            for column in ["created_at", "updated_at"]:
                issue_df[column] = pd.to_datetime(issue_df[column], errors="coerce").dt.date
            return issue_df
        except Exception:
            pass

    if raw_path.exists():
        try:
            raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
            issue_df = pd.DataFrame(raw_payload)
            for column in ISSUE_COLUMNS:
                if column not in issue_df.columns:
                    issue_df[column] = pd.NA
            for column in ["created_at", "updated_at"]:
                issue_df[column] = pd.to_datetime(issue_df[column], errors="coerce").dt.date
            return issue_df[ISSUE_COLUMNS]
        except Exception:
            return None

    return None


def _empty_commit_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return stable empty GitHub commit dataframes."""
    commit_df = pd.DataFrame(columns=COMMIT_DF_COLUMNS)
    file_summary_df = pd.DataFrame(columns=COMMIT_SUMMARY_COLUMNS)
    daily_churn_df = pd.DataFrame(columns=DAILY_CHURN_COLUMNS)
    return commit_df, file_summary_df, daily_churn_df


def _fetch_commit_history_impl(
    owner: str,
    repo: str,
    branch: str,
    start_date: date,
    end_date: date,
    token: str = "",
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch commit history and aggregate churn metrics by file and by day."""
    raw_cache_path, summary_cache_path, daily_cache_path = _commit_cache_paths(owner, repo, branch, start_date, end_date)
    cached_commit_data = _load_commit_cache(raw_cache_path, summary_cache_path, daily_cache_path)
    if cached_commit_data is not None:
        _emit_progress(progress_callback, 1.0, "GitHub: loaded commit history from local cache.")
        return cached_commit_data

    commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    _emit_progress(progress_callback, 0.02, f"GitHub: requesting commit list for {owner}/{repo}...")
    commit_refs = _paginate_list(
        commits_url,
        token,
        {
            "sha": branch,
            "since": _utc_iso(start_date),
            "until": _utc_iso(end_date, end_of_day=True),
        },
    )

    total_commits = len(commit_refs)
    if total_commits == 0:
        _emit_progress(progress_callback, 1.0, "GitHub: no commits found in the selected date range.")
        return _empty_commit_frames()

    _emit_progress(progress_callback, 0.1, f"GitHub: found {total_commits} commits. Fetching file-level details...")

    commit_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    raw_commit_payload: list[dict[str, Any]] = []
    checkpoint = max(1, total_commits // 10)

    for index, commit_ref in enumerate(commit_refs, start=1):
        sha = commit_ref["sha"]
        detail = _request_json(f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}", token)

        commit_date = _parse_timestamp(detail["commit"]["author"]["date"])
        github_author = detail.get("author") or {}
        commit_author = detail["commit"].get("author") or {}
        contributor = (
            github_author.get("login")
            or commit_author.get("name")
            or commit_author.get("email")
            or "unknown"
        )
        stats = detail.get("stats", {}) or {}
        files = detail.get("files", []) or []

        commit_rows.append(
            {
                "sha": sha,
                "commit_date": commit_date.date(),
                "contributor": contributor,
                "additions": stats.get("additions", 0),
                "deletions": stats.get("deletions", 0),
                "files_changed": len(files),
                "message": detail["commit"]["message"].splitlines()[0],
            }
        )

        raw_file_stats: list[dict[str, Any]] = []
        for file_stat in files:
            filename = file_stat.get("filename")
            if not filename:
                continue

            additions = int(file_stat.get("additions", 0))
            deletions = int(file_stat.get("deletions", 0))
            total_churn = additions + deletions

            file_rows.append(
                {
                    "sha": sha,
                    "commit_date": commit_date.date(),
                    "contributor": contributor,
                    "file_path": filename,
                    "status": file_stat.get("status", "modified"),
                    "additions": additions,
                    "deletions": deletions,
                    "total_churn": total_churn,
                }
            )
            raw_file_stats.append(
                {
                    "file_path": filename,
                    "status": file_stat.get("status", "modified"),
                    "additions": additions,
                    "deletions": deletions,
                    "total_churn": total_churn,
                }
            )

        raw_commit_payload.append(
            {
                "sha": sha,
                "commit_date": detail["commit"]["author"]["date"],
                "contributor": contributor,
                "message": detail["commit"]["message"].splitlines()[0],
                "stats": {
                    "additions": stats.get("additions", 0),
                    "deletions": stats.get("deletions", 0),
                    "total": stats.get("total", 0),
                },
                "files": raw_file_stats,
            }
        )

        if index == 1 or index == total_commits or index % checkpoint == 0:
            progress_value = 0.1 + (0.8 * (index / total_commits))
            _emit_progress(progress_callback, progress_value, f"GitHub: processed commit details {index}/{total_commits}.")

    commit_df = pd.DataFrame(commit_rows, columns=COMMIT_DF_COLUMNS)
    file_df = pd.DataFrame(file_rows)

    if file_df.empty:
        _, file_summary_df, daily_churn_df = _empty_commit_frames()
    else:
        file_summary_df = (
            file_df.groupby("file_path", dropna=True)
            .agg(
                commit_count=("sha", "nunique"),
                additions=("additions", "sum"),
                deletions=("deletions", "sum"),
                total_churn=("total_churn", "sum"),
                unique_contributors=("contributor", "nunique"),
            )
            .reset_index()
            .sort_values(["total_churn", "commit_count"], ascending=[False, False])
        )

        daily_churn_df = (
            file_df.groupby("commit_date", dropna=True)
            .agg(total_churn=("total_churn", "sum"), commit_count=("sha", "nunique"))
            .reset_index()
            .sort_values("commit_date")
        )

    _write_json(raw_cache_path, raw_commit_payload)
    _write_dataframe(summary_cache_path, file_summary_df)
    _write_dataframe(daily_cache_path, daily_churn_df)
    _emit_progress(progress_callback, 1.0, "GitHub: churn aggregation complete.")

    return commit_df, file_summary_df, daily_churn_df


@st.cache_data(show_spinner=False, ttl=900)
def fetch_commit_history(
    owner: str,
    repo: str,
    branch: str,
    start_date: date,
    end_date: date,
    token: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cached wrapper around GitHub commit history collection."""
    return _fetch_commit_history_impl(owner, repo, branch, start_date, end_date, token)


def fetch_commit_history_with_progress(
    owner: str,
    repo: str,
    branch: str,
    start_date: date,
    end_date: date,
    token: str = "",
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Uncached commit history collection with live progress reporting."""
    return _fetch_commit_history_impl(owner, repo, branch, start_date, end_date, token, progress_callback)


def _fetch_maintenance_issues_impl(
    owner: str,
    repo: str,
    start_date: date,
    end_date: date,
    token: str = "",
    keywords: Iterable[str] = MAINTENANCE_KEYWORDS,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Fetch maintenance-oriented GitHub issues using keyword search."""
    raw_cache_path, csv_cache_path = _issue_cache_paths(owner, repo, start_date, end_date)
    cached_issue_data = _load_issue_cache(raw_cache_path, csv_cache_path)
    if cached_issue_data is not None:
        _emit_progress(
            progress_callback,
            1.0,
            f"GitHub: loaded maintenance issue results from local cache. Found {len(cached_issue_data)} issues.",
        )
        return cached_issue_data

    _emit_progress(progress_callback, 0.05, "GitHub: searching maintenance-related issues...")
    keyword_terms = " OR ".join(f'"{keyword}"' for keyword in keywords)
    query = (
        f"repo:{owner}/{repo} is:issue "
        f"updated:{start_date.isoformat()}..{end_date.isoformat()} ({keyword_terms})"
    )
    items = _paginate_search("https://api.github.com/search/issues", token, {"q": query})

    issue_rows: list[dict[str, Any]] = []
    for item in items:
        title = item.get("title", "")
        body = item.get("body", "") or ""
        issue_text = f"{title}\n{body}".lower()
        matched_keywords = [keyword for keyword in keywords if keyword.lower() in issue_text]
        issue_rows.append(
            {
                "issue_number": item.get("number"),
                "title": title,
                "body": body,
                "state": item.get("state"),
                "created_at": _parse_timestamp(item["created_at"]).date() if item.get("created_at") else None,
                "updated_at": _parse_timestamp(item["updated_at"]).date() if item.get("updated_at") else None,
                "html_url": item.get("html_url"),
                "matched_keywords": ", ".join(matched_keywords),
            }
        )

    issue_df = pd.DataFrame(issue_rows).sort_values("updated_at", ascending=False) if issue_rows else pd.DataFrame(columns=ISSUE_COLUMNS)
    if not issue_df.empty:
        issue_df = issue_df.reindex(columns=ISSUE_COLUMNS)

    _write_json(raw_cache_path, issue_rows)
    _write_dataframe(csv_cache_path, issue_df)
    _emit_progress(
        progress_callback,
        1.0,
        f"GitHub: maintenance-related issue search complete. Found {len(issue_df)} issues.",
    )
    return issue_df


@st.cache_data(show_spinner=False, ttl=900)
def fetch_maintenance_issues(
    owner: str,
    repo: str,
    start_date: date,
    end_date: date,
    token: str = "",
    keywords: Iterable[str] = MAINTENANCE_KEYWORDS,
) -> pd.DataFrame:
    """Cached wrapper around GitHub maintenance issue collection."""
    return _fetch_maintenance_issues_impl(owner, repo, start_date, end_date, token, keywords)


def fetch_maintenance_issues_with_progress(
    owner: str,
    repo: str,
    start_date: date,
    end_date: date,
    token: str = "",
    keywords: Iterable[str] = MAINTENANCE_KEYWORDS,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Uncached maintenance issue collection with live progress reporting."""
    return _fetch_maintenance_issues_impl(owner, repo, start_date, end_date, token, keywords, progress_callback)


def build_issue_signal(issue_df: pd.DataFrame, file_paths: list[str]) -> pd.DataFrame:
    """Build a simple per-file GitHub issue signal using path and filename mentions."""
    if not file_paths:
        return pd.DataFrame(columns=["file_path", "github_issue_matches", "github_issue_signal", "repo_issue_signal"])

    unique_file_paths = sorted(set(file_paths))
    repo_issue_signal = int(len(issue_df))
    baseline = repo_issue_signal / max(len(unique_file_paths), 1)

    basename_index: dict[str, list[str]] = defaultdict(list)
    lower_path_index: dict[str, str] = {}
    for file_path in unique_file_paths:
        basename_index[Path(file_path).name.lower()].append(file_path)
        lower_path_index[file_path.lower()] = file_path

    direct_matches = {file_path: 0 for file_path in unique_file_paths}
    signal_values = {file_path: baseline for file_path in unique_file_paths}

    for issue in issue_df.to_dict("records"):
        issue_text = f"{issue.get('title', '')}\n{issue.get('body', '')}".lower()
        matched_files: set[str] = set()

        for lower_path, original_path in lower_path_index.items():
            if lower_path and lower_path in issue_text:
                matched_files.add(original_path)

        if not matched_files:
            for basename, related_paths in basename_index.items():
                if not basename:
                    continue
                if re.search(rf"(?<![\w/.-]){re.escape(basename)}(?![\w/.-])", issue_text):
                    matched_files.update(related_paths)

        for file_path in matched_files:
            direct_matches[file_path] += 1
            signal_values[file_path] += 1

    return pd.DataFrame(
        [
            {
                "file_path": file_path,
                "github_issue_matches": direct_matches[file_path],
                "github_issue_signal": signal_values[file_path],
                "repo_issue_signal": repo_issue_signal,
            }
            for file_path in unique_file_paths
        ]
    )


@st.cache_data(show_spinner=False)
def load_mock_github_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load small mock GitHub data so the UI can run without live API credentials."""
    commit_df = pd.DataFrame(
        [
            {"sha": "c1", "commit_date": date(2026, 1, 7), "contributor": "alice", "additions": 120, "deletions": 30, "files_changed": 4, "message": "Initial API refactor"},
            {"sha": "c2", "commit_date": date(2026, 1, 12), "contributor": "bob", "additions": 40, "deletions": 80, "files_changed": 3, "message": "Auth cleanup"},
            {"sha": "c3", "commit_date": date(2026, 1, 14), "contributor": "alice", "additions": 15, "deletions": 10, "files_changed": 2, "message": "Reporting fix"},
            {"sha": "c4", "commit_date": date(2026, 1, 20), "contributor": "carol", "additions": 90, "deletions": 45, "files_changed": 5, "message": "Dashboard improvements"},
            {"sha": "c5", "commit_date": date(2026, 1, 27), "contributor": "bob", "additions": 30, "deletions": 70, "files_changed": 2, "message": "Serializer cleanup"},
            {"sha": "c6", "commit_date": date(2026, 2, 3), "contributor": "dana", "additions": 110, "deletions": 60, "files_changed": 4, "message": "API adjustments"},
            {"sha": "c7", "commit_date": date(2026, 2, 11), "contributor": "alice", "additions": 25, "deletions": 20, "files_changed": 2, "message": "Minor auth fix"},
            {"sha": "c8", "commit_date": date(2026, 2, 19), "contributor": "eric", "additions": 75, "deletions": 95, "files_changed": 3, "message": "Reporting pipeline cleanup"},
        ],
        columns=COMMIT_DF_COLUMNS,
    )

    churn_summary_df = pd.DataFrame(
        [
            {"file_path": "src/api.py", "commit_count": 6, "additions": 210, "deletions": 150, "total_churn": 360, "unique_contributors": 4},
            {"file_path": "src/auth.py", "commit_count": 5, "additions": 165, "deletions": 140, "total_churn": 305, "unique_contributors": 3},
            {"file_path": "src/reporting.py", "commit_count": 4, "additions": 125, "deletions": 95, "total_churn": 220, "unique_contributors": 3},
            {"file_path": "src/dashboard.py", "commit_count": 4, "additions": 115, "deletions": 55, "total_churn": 170, "unique_contributors": 2},
            {"file_path": "src/utils/serialization.py", "commit_count": 3, "additions": 55, "deletions": 65, "total_churn": 120, "unique_contributors": 2},
            {"file_path": "tests/test_api.py", "commit_count": 2, "additions": 35, "deletions": 12, "total_churn": 47, "unique_contributors": 2},
        ]
    )

    daily_churn_df = pd.DataFrame(
        [
            {"commit_date": date(2026, 1, 7), "total_churn": 150, "commit_count": 1},
            {"commit_date": date(2026, 1, 12), "total_churn": 120, "commit_count": 1},
            {"commit_date": date(2026, 1, 14), "total_churn": 25, "commit_count": 1},
            {"commit_date": date(2026, 1, 20), "total_churn": 135, "commit_count": 1},
            {"commit_date": date(2026, 1, 27), "total_churn": 100, "commit_count": 1},
            {"commit_date": date(2026, 2, 3), "total_churn": 170, "commit_count": 1},
            {"commit_date": date(2026, 2, 11), "total_churn": 45, "commit_count": 1},
            {"commit_date": date(2026, 2, 19), "total_churn": 170, "commit_count": 1},
        ],
        columns=DAILY_CHURN_COLUMNS,
    )

    github_issue_df = pd.DataFrame(
        [
            {
                "issue_number": 101,
                "title": "Refactor src/api.py request handling",
                "body": "The current code in src/api.py has grown brittle and needs cleanup.",
                "state": "open",
                "created_at": date(2026, 1, 10),
                "updated_at": date(2026, 1, 22),
                "html_url": "https://example.com/issues/101",
                "matched_keywords": "refactor, cleanup",
            },
            {
                "issue_number": 102,
                "title": "Technical debt in auth flow",
                "body": "src/auth.py and src/api.py both need maintainability improvements.",
                "state": "open",
                "created_at": date(2026, 1, 25),
                "updated_at": date(2026, 2, 2),
                "html_url": "https://example.com/issues/102",
                "matched_keywords": "technical debt, maintainability",
            },
            {
                "issue_number": 103,
                "title": "Cleanup reporting pipeline",
                "body": "src/reporting.py is accumulating code smell follow-ups.",
                "state": "closed",
                "created_at": date(2026, 2, 5),
                "updated_at": date(2026, 2, 12),
                "html_url": "https://example.com/issues/103",
                "matched_keywords": "cleanup, code smell",
            },
        ],
        columns=ISSUE_COLUMNS,
    )

    github_issue_signal_df = build_issue_signal(github_issue_df, churn_summary_df["file_path"].tolist())
    return commit_df, churn_summary_df, daily_churn_df, github_issue_df, github_issue_signal_df
