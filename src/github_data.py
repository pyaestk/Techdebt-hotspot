"""GitHub data collection and lightweight issue-to-file heuristics."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date, datetime, time, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

import pandas as pd
import requests
import streamlit as st

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, MAINTENANCE_KEYWORDS, build_cache_key


class GitHubAPIError(RuntimeError):
    """Raised when a GitHub API call fails."""


ProgressCallback = Callable[[float, str], None]


COMMIT_DF_COLUMNS = ["sha", "commit_date", "contributor", "additions", "deletions", "files_changed", "message"]
COMMIT_SUMMARY_COLUMNS = [
    "file_path",
    "commit_count",
    "additions",
    "deletions",
    "total_churn",
    "unique_contributors",
    "active_days",
    "first_commit_date",
    "last_commit_date",
    "days_since_last_touch",
    "avg_churn_per_commit",
    "ownership_concentration",
    "bugfix_commit_count",
    "bugfix_commit_ratio",
    "churn_burstiness",
]
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
ISSUE_SIGNAL_COLUMNS = [
    "file_path",
    "github_issue_matches",
    "github_issue_signal",
    "repo_issue_signal",
    "github_exact_path_matches",
    "github_suffix_path_matches",
    "github_basename_matches",
    "github_directory_matches",
    "github_weighted_mentions",
    "github_issue_attribution_confidence",
    "github_issue_examples",
]

BUGFIX_PATTERN = re.compile(r"\b(fix|bug|defect|hotfix|regression|failure|error|patch)\b", re.IGNORECASE)
PATH_PATTERN = re.compile(r"(?:[a-z0-9_.-]+/)+[a-z0-9_.-]+", re.IGNORECASE)
FILENAME_PATTERN = re.compile(r"\b[a-z0-9_.-]+\.[a-z0-9_.-]+\b", re.IGNORECASE)
DIRECTORY_PATTERN = re.compile(r"(?:[a-z0-9_.-]+/)+", re.IGNORECASE)
COMMON_DIRECTORY_TOKENS = {"src", "app", "main", "lib", "test", "tests", "java", "kotlin", "python"}


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


def _normalize_path(value: str) -> str:
    """Normalize GitHub paths into a lowercase forward-slash representation."""
    return value.replace("\\", "/").strip().lower().strip("`'\" ")


def _normalize_text(value: str) -> str:
    """Normalize free text for path and filename matching."""
    return _normalize_path(value)


def _is_bugfix_message(message: str) -> bool:
    """Flag commit messages that look like bug-fix work."""
    return bool(BUGFIX_PATTERN.search(message or ""))


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


def _empty_commit_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return stable empty GitHub commit dataframes."""
    commit_df = pd.DataFrame(columns=COMMIT_DF_COLUMNS)
    file_summary_df = pd.DataFrame(columns=COMMIT_SUMMARY_COLUMNS)
    daily_churn_df = pd.DataFrame(columns=DAILY_CHURN_COLUMNS)
    return commit_df, file_summary_df, daily_churn_df


def _build_commit_frames_from_raw(raw_payload: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild commit-level and file-level frames from cached raw payloads."""
    commit_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []

    for item in raw_payload:
        commit_date_value = item.get("commit_date")
        commit_date = pd.to_datetime(commit_date_value, errors="coerce")
        if pd.isna(commit_date):
            continue

        contributor = item.get("contributor") or "unknown"
        message = item.get("message", "")
        stats = item.get("stats", {}) or {}
        files = item.get("files", []) or []

        commit_rows.append(
            {
                "sha": item.get("sha"),
                "commit_date": commit_date.date(),
                "contributor": contributor,
                "additions": int(stats.get("additions", 0)),
                "deletions": int(stats.get("deletions", 0)),
                "files_changed": len(files),
                "message": message,
            }
        )

        bugfix_flag = _is_bugfix_message(message)
        for file_item in files:
            file_path = file_item.get("file_path") or file_item.get("filename")
            if not file_path:
                continue
            additions = int(file_item.get("additions", 0))
            deletions = int(file_item.get("deletions", 0))
            total_churn = int(file_item.get("total_churn", additions + deletions))
            file_rows.append(
                {
                    "sha": item.get("sha"),
                    "commit_date": commit_date.date(),
                    "contributor": contributor,
                    "file_path": file_path,
                    "status": file_item.get("status", "modified"),
                    "additions": additions,
                    "deletions": deletions,
                    "total_churn": total_churn,
                    "message": message,
                    "bugfix_flag": bugfix_flag,
                }
            )

    commit_df = pd.DataFrame(commit_rows, columns=COMMIT_DF_COLUMNS)
    file_df = pd.DataFrame(file_rows)
    return commit_df, file_df


def _aggregate_commit_metrics(file_df: pd.DataFrame, end_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate file-level engineering signals and daily churn metrics."""
    if file_df.empty:
        return pd.DataFrame(columns=COMMIT_SUMMARY_COLUMNS), pd.DataFrame(columns=DAILY_CHURN_COLUMNS)

    summary_df = (
        file_df.groupby("file_path", dropna=True)
        .agg(
            commit_count=("sha", "nunique"),
            additions=("additions", "sum"),
            deletions=("deletions", "sum"),
            total_churn=("total_churn", "sum"),
            unique_contributors=("contributor", "nunique"),
            active_days=("commit_date", "nunique"),
            first_commit_date=("commit_date", "min"),
            last_commit_date=("commit_date", "max"),
        )
        .reset_index()
    )

    contributor_touches = (
        file_df.groupby(["file_path", "contributor"], dropna=True)["sha"]
        .nunique()
        .reset_index(name="contributor_touch_count")
    )
    ownership_df = (
        contributor_touches.groupby("file_path", dropna=True)["contributor_touch_count"]
        .max()
        .reset_index(name="max_contributor_touch_count")
    )

    bugfix_df = (
        file_df[file_df["bugfix_flag"]]
        .groupby("file_path", dropna=True)["sha"]
        .nunique()
        .reset_index(name="bugfix_commit_count")
    )

    burstiness_df = (
        file_df.groupby(["file_path", "sha"], dropna=True)["total_churn"]
        .sum()
        .reset_index()
        .groupby("file_path", dropna=True)["total_churn"]
        .max()
        .reset_index(name="max_single_commit_churn")
    )

    summary_df = summary_df.merge(ownership_df, on="file_path", how="left")
    summary_df = summary_df.merge(bugfix_df, on="file_path", how="left")
    summary_df = summary_df.merge(burstiness_df, on="file_path", how="left")
    summary_df["bugfix_commit_count"] = pd.to_numeric(summary_df["bugfix_commit_count"], errors="coerce").fillna(0).astype(int)
    summary_df["max_contributor_touch_count"] = pd.to_numeric(summary_df["max_contributor_touch_count"], errors="coerce").fillna(0)
    summary_df["max_single_commit_churn"] = pd.to_numeric(summary_df["max_single_commit_churn"], errors="coerce").fillna(0)

    summary_df["days_since_last_touch"] = summary_df["last_commit_date"].apply(
        lambda value: max((end_date - value).days, 0) if pd.notna(value) else 0
    )
    summary_df["avg_churn_per_commit"] = summary_df["total_churn"] / summary_df["commit_count"].replace(0, 1)
    summary_df["ownership_concentration"] = summary_df["max_contributor_touch_count"] / summary_df["commit_count"].replace(0, 1)
    summary_df["bugfix_commit_ratio"] = summary_df["bugfix_commit_count"] / summary_df["commit_count"].replace(0, 1)
    summary_df["churn_burstiness"] = summary_df["max_single_commit_churn"] / summary_df["total_churn"].replace(0, 1)

    summary_df = summary_df[COMMIT_SUMMARY_COLUMNS].sort_values(["total_churn", "commit_count"], ascending=[False, False])

    daily_churn_df = (
        file_df.groupby("commit_date", dropna=True)
        .agg(total_churn=("total_churn", "sum"), commit_count=("sha", "nunique"))
        .reset_index()
        .sort_values("commit_date")
    )

    return summary_df, daily_churn_df


def _load_commit_cache(
    raw_path: Path,
    summary_path: Path,
    daily_path: Path,
    end_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Load commit data from local cache when available."""
    if not raw_path.exists():
        return None

    try:
        raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
        commit_df, file_df = _build_commit_frames_from_raw(raw_payload)
        summary_df, daily_df = _aggregate_commit_metrics(file_df, end_date)

        if not summary_df.empty:
            _write_dataframe(summary_path, summary_df)
        if not daily_df.empty:
            _write_dataframe(daily_path, daily_df)

        return commit_df, summary_df, daily_df
    except Exception:
        if summary_path.exists() and daily_path.exists():
            try:
                commit_df = pd.DataFrame(columns=COMMIT_DF_COLUMNS)
                summary_df = _read_dataframe(summary_path, COMMIT_SUMMARY_COLUMNS)
                daily_df = _read_dataframe(daily_path, DAILY_CHURN_COLUMNS)
                if not daily_df.empty:
                    daily_df["commit_date"] = pd.to_datetime(daily_df["commit_date"], errors="coerce").dt.date
                return commit_df, summary_df, daily_df
            except Exception:
                return None
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
    cached_commit_data = _load_commit_cache(raw_cache_path, summary_cache_path, daily_cache_path, end_date)
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

    raw_commit_payload: list[dict[str, Any]] = []
    checkpoint = max(1, total_commits // 10)

    for index, commit_ref in enumerate(commit_refs, start=1):
        sha = commit_ref["sha"]
        detail = _request_json(f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}", token)

        commit_date = _parse_timestamp(detail["commit"]["author"]["date"])
        github_author = detail.get("author") or {}
        commit_author = detail["commit"].get("author") or {}
        contributor = github_author.get("login") or commit_author.get("name") or commit_author.get("email") or "unknown"
        stats = detail.get("stats", {}) or {}
        files = detail.get("files", []) or []
        message = detail["commit"]["message"].splitlines()[0]

        raw_file_stats: list[dict[str, Any]] = []
        for file_stat in files:
            filename = file_stat.get("filename")
            if not filename:
                continue

            additions = int(file_stat.get("additions", 0))
            deletions = int(file_stat.get("deletions", 0))
            total_churn = additions + deletions
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
                "message": message,
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

    commit_df, file_df = _build_commit_frames_from_raw(raw_commit_payload)
    file_summary_df, daily_churn_df = _aggregate_commit_metrics(file_df, end_date)

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


def _extract_path_mentions(issue_text: str) -> set[str]:
    """Extract path-like mentions from issue text."""
    return {
        _normalize_path(match.group(0).strip("`'\"()[]{}<>,.;:"))
        for match in PATH_PATTERN.finditer(issue_text)
        if match.group(0)
    }


def _extract_filename_mentions(issue_text: str) -> set[str]:
    """Extract filename-like mentions from issue text."""
    return {
        _normalize_path(match.group(0).strip("`'\"()[]{}<>,.;:"))
        for match in FILENAME_PATTERN.finditer(issue_text)
        if "/" not in match.group(0)
    }


def _extract_directory_mentions(issue_text: str) -> set[str]:
    """Extract directory-like mentions from issue text."""
    return {
        _normalize_path(match.group(0).rstrip("/"))
        for match in DIRECTORY_PATTERN.finditer(issue_text)
        if match.group(0)
    }


def _build_file_reference_index(file_paths: list[str]) -> tuple[dict[str, str], dict[str, list[str]], dict[str, list[str]], dict[str, dict[str, Any]]]:
    """Pre-compute exact path, suffix, basename, and directory indexes."""
    exact_path_index: dict[str, str] = {}
    suffix_index: dict[str, list[str]] = defaultdict(list)
    basename_index: dict[str, list[str]] = defaultdict(list)
    metadata_by_path: dict[str, dict[str, Any]] = {}

    for file_path in sorted(set(file_paths)):
        normalized_path = _normalize_path(file_path)
        parts = [part.lower() for part in PurePosixPath(normalized_path).parts if part]
        if not parts:
            continue

        basename = parts[-1]
        parent_parts = parts[:-1]
        parent_path = "/".join(parent_parts)
        last_directory = parent_parts[-1] if parent_parts else ""
        meaningful_parent_tokens = {
            token for token in parent_parts if len(token) > 3 and token not in COMMON_DIRECTORY_TOKENS
        }
        parent_suffixes = {
            "/".join(parent_parts[-size:])
            for size in (1, 2, 3)
            if len(parent_parts) >= size and "/".join(parent_parts[-size:])
        }

        exact_path_index[normalized_path] = file_path
        basename_index[basename].append(file_path)
        for size in (2, 3):
            if len(parts) >= size:
                suffix_index["/".join(parts[-size:])].append(file_path)

        metadata_by_path[file_path] = {
            "normalized_path": normalized_path,
            "basename": basename,
            "parent_path": parent_path,
            "last_directory": last_directory,
            "meaningful_parent_tokens": meaningful_parent_tokens,
            "parent_suffixes": parent_suffixes,
        }

    return exact_path_index, suffix_index, basename_index, metadata_by_path


def _issue_matches_file_context(
    issue_text: str,
    issue_tokens: set[str],
    directory_mentions: set[str],
    metadata: dict[str, Any],
) -> bool:
    """Check whether issue text contains directory context for a candidate file."""
    parent_path = metadata.get("parent_path", "")
    if parent_path and parent_path in issue_text:
        return True

    last_directory = metadata.get("last_directory", "")
    if last_directory and last_directory in issue_tokens and last_directory not in COMMON_DIRECTORY_TOKENS:
        return True

    if metadata.get("meaningful_parent_tokens", set()) & issue_tokens:
        return True

    if metadata.get("parent_suffixes", set()) & directory_mentions:
        return True

    return False


def build_issue_signal(issue_df: pd.DataFrame, file_paths: list[str]) -> pd.DataFrame:
    """Build a file-level GitHub issue signal using weighted mention heuristics."""
    if not file_paths:
        return pd.DataFrame(columns=ISSUE_SIGNAL_COLUMNS)

    unique_file_paths = sorted(set(file_paths))
    repo_issue_signal = int(len(issue_df))
    baseline = (repo_issue_signal / max(len(unique_file_paths), 1)) * 0.05

    exact_path_index, suffix_index, basename_index, metadata_by_path = _build_file_reference_index(unique_file_paths)
    signal_state = {
        file_path: {
            "github_issue_matches": 0,
            "github_issue_signal": baseline,
            "repo_issue_signal": repo_issue_signal,
            "github_exact_path_matches": 0,
            "github_suffix_path_matches": 0,
            "github_basename_matches": 0,
            "github_directory_matches": 0,
            "github_weighted_mentions": 0.0,
            "github_issue_attribution_confidence": 0.0,
            "github_issue_examples": [],
        }
        for file_path in unique_file_paths
    }

    for issue in issue_df.to_dict("records"):
        issue_number = issue.get("issue_number")
        issue_text = _normalize_text(f"{issue.get('title', '')}\n{issue.get('body', '')}")
        issue_tokens = set(re.findall(r"[a-z0-9_.-]+", issue_text))
        path_mentions = _extract_path_mentions(issue_text)
        filename_mentions = _extract_filename_mentions(issue_text)
        directory_mentions = _extract_directory_mentions(issue_text)
        evidence_by_file: dict[str, dict[str, Any]] = {}

        def ensure_evidence(file_path: str) -> dict[str, Any]:
            return evidence_by_file.setdefault(
                file_path,
                {"score": 0.0, "exact": False, "suffix": False, "basename": False, "directory": False},
            )

        for mention in path_mentions:
            exact_match = exact_path_index.get(mention)
            if exact_match:
                evidence = ensure_evidence(exact_match)
                evidence["score"] = max(evidence["score"], 1.0)
                evidence["exact"] = True
                continue

            for candidate_path in suffix_index.get(mention, []):
                evidence = ensure_evidence(candidate_path)
                evidence["score"] = max(evidence["score"], 0.85)
                evidence["suffix"] = True

        for basename in filename_mentions:
            candidate_paths = basename_index.get(basename, [])
            if not candidate_paths:
                continue

            if len(candidate_paths) == 1:
                evidence = ensure_evidence(candidate_paths[0])
                evidence["score"] = max(evidence["score"], 0.65)
                evidence["basename"] = True
                continue

            narrowed_paths = [
                candidate_path
                for candidate_path in candidate_paths
                if _issue_matches_file_context(issue_text, issue_tokens, directory_mentions, metadata_by_path[candidate_path])
            ]
            for candidate_path in narrowed_paths:
                evidence = ensure_evidence(candidate_path)
                evidence["score"] = max(evidence["score"], 0.65)
                evidence["basename"] = True

        for file_path, evidence in evidence_by_file.items():
            metadata = metadata_by_path[file_path]
            if _issue_matches_file_context(issue_text, issue_tokens, directory_mentions, metadata):
                evidence["directory"] = True
                evidence["score"] = min(1.0, evidence["score"] + 0.15)

        for file_path, evidence in evidence_by_file.items():
            if evidence["score"] <= 0:
                continue
            file_state = signal_state[file_path]
            file_state["github_issue_matches"] += 1
            file_state["github_weighted_mentions"] += evidence["score"]
            file_state["github_issue_signal"] += evidence["score"]
            file_state["github_exact_path_matches"] += int(evidence["exact"])
            file_state["github_suffix_path_matches"] += int(evidence["suffix"])
            file_state["github_basename_matches"] += int(evidence["basename"])
            file_state["github_directory_matches"] += int(evidence["directory"])
            if issue_number is not None and len(file_state["github_issue_examples"]) < 5:
                file_state["github_issue_examples"].append(str(issue_number))

    rows = []
    for file_path in unique_file_paths:
        file_state = signal_state[file_path]
        matches = file_state["github_issue_matches"]
        weighted_mentions = file_state["github_weighted_mentions"]
        rows.append(
            {
                "file_path": file_path,
                "github_issue_matches": matches,
                "github_issue_signal": file_state["github_issue_signal"],
                "repo_issue_signal": repo_issue_signal,
                "github_exact_path_matches": file_state["github_exact_path_matches"],
                "github_suffix_path_matches": file_state["github_suffix_path_matches"],
                "github_basename_matches": file_state["github_basename_matches"],
                "github_directory_matches": file_state["github_directory_matches"],
                "github_weighted_mentions": weighted_mentions,
                "github_issue_attribution_confidence": 0.0 if matches == 0 else weighted_mentions / matches,
                "github_issue_examples": ", ".join(file_state["github_issue_examples"]),
            }
        )

    return pd.DataFrame(rows, columns=ISSUE_SIGNAL_COLUMNS)


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
            {"file_path": "src/api.py", "commit_count": 6, "additions": 210, "deletions": 150, "total_churn": 360, "unique_contributors": 4, "active_days": 5, "first_commit_date": date(2026, 1, 7), "last_commit_date": date(2026, 2, 3), "days_since_last_touch": 16, "avg_churn_per_commit": 60.0, "ownership_concentration": 0.50, "bugfix_commit_count": 1, "bugfix_commit_ratio": 0.17, "churn_burstiness": 0.33},
            {"file_path": "src/auth.py", "commit_count": 5, "additions": 165, "deletions": 140, "total_churn": 305, "unique_contributors": 3, "active_days": 4, "first_commit_date": date(2026, 1, 12), "last_commit_date": date(2026, 2, 11), "days_since_last_touch": 8, "avg_churn_per_commit": 61.0, "ownership_concentration": 0.40, "bugfix_commit_count": 2, "bugfix_commit_ratio": 0.40, "churn_burstiness": 0.39},
            {"file_path": "src/reporting.py", "commit_count": 4, "additions": 125, "deletions": 95, "total_churn": 220, "unique_contributors": 3, "active_days": 4, "first_commit_date": date(2026, 1, 14), "last_commit_date": date(2026, 2, 19), "days_since_last_touch": 0, "avg_churn_per_commit": 55.0, "ownership_concentration": 0.50, "bugfix_commit_count": 1, "bugfix_commit_ratio": 0.25, "churn_burstiness": 0.45},
            {"file_path": "src/dashboard.py", "commit_count": 4, "additions": 115, "deletions": 55, "total_churn": 170, "unique_contributors": 2, "active_days": 3, "first_commit_date": date(2026, 1, 20), "last_commit_date": date(2026, 1, 20), "days_since_last_touch": 30, "avg_churn_per_commit": 42.5, "ownership_concentration": 0.75, "bugfix_commit_count": 0, "bugfix_commit_ratio": 0.0, "churn_burstiness": 0.79},
            {"file_path": "src/utils/serialization.py", "commit_count": 3, "additions": 55, "deletions": 65, "total_churn": 120, "unique_contributors": 2, "active_days": 2, "first_commit_date": date(2026, 1, 27), "last_commit_date": date(2026, 1, 27), "days_since_last_touch": 23, "avg_churn_per_commit": 40.0, "ownership_concentration": 0.67, "bugfix_commit_count": 0, "bugfix_commit_ratio": 0.0, "churn_burstiness": 0.83},
            {"file_path": "tests/test_api.py", "commit_count": 2, "additions": 35, "deletions": 12, "total_churn": 47, "unique_contributors": 2, "active_days": 2, "first_commit_date": date(2026, 1, 14), "last_commit_date": date(2026, 2, 3), "days_since_last_touch": 16, "avg_churn_per_commit": 23.5, "ownership_concentration": 0.50, "bugfix_commit_count": 1, "bugfix_commit_ratio": 0.50, "churn_burstiness": 0.53},
        ],
        columns=COMMIT_SUMMARY_COLUMNS,
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
