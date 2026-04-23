"""Scoring helpers for hotspot ranking."""

from __future__ import annotations

import pandas as pd


def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a numeric series to the 0-1 range with a safe constant fallback."""
    numeric_series = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if numeric_series.empty:
        return numeric_series

    min_value = float(numeric_series.min())
    max_value = float(numeric_series.max())
    if max_value == min_value:
        return pd.Series(0.0, index=numeric_series.index)
    return (numeric_series - min_value) / (max_value - min_value)


def compute_hotspot_scores(
    churn_summary_df: pd.DataFrame,
    sonar_summary_df: pd.DataFrame,
    github_issue_signal_df: pd.DataFrame,
    weight_churn: float = 0.5,
    weight_sonar: float = 0.3,
    weight_github_issues: float = 0.2,
) -> pd.DataFrame:
    """Merge signals, normalize them, and calculate the weighted hotspot score."""
    merged_df = churn_summary_df.merge(sonar_summary_df, on="file_path", how="outer").merge(
        github_issue_signal_df,
        on="file_path",
        how="outer",
    )

    if merged_df.empty:
        return pd.DataFrame()

    fill_defaults = {
        "commit_count": 0,
        "additions": 0,
        "deletions": 0,
        "total_churn": 0,
        "unique_contributors": 0,
        "sonar_issue_count": 0,
        "highest_severity": "NONE",
        "severity_breakdown": "",
        "type_breakdown": "",
        "github_issue_matches": 0,
        "github_issue_signal": 0.0,
        "repo_issue_signal": 0,
    }

    for column, default_value in fill_defaults.items():
        if column not in merged_df.columns:
            merged_df[column] = default_value
        else:
            merged_df[column] = merged_df[column].fillna(default_value)

    merged_df["churn_normalized"] = normalize_series(merged_df["total_churn"])
    merged_df["sonar_normalized"] = normalize_series(merged_df["sonar_issue_count"])
    merged_df["github_issues_normalized"] = normalize_series(merged_df["github_issue_signal"])

    total_weight = weight_churn + weight_sonar + weight_github_issues
    if total_weight <= 0:
        total_weight = 1.0

    merged_df["hotspot_score"] = (
        (merged_df["churn_normalized"] * weight_churn)
        + (merged_df["sonar_normalized"] * weight_sonar)
        + (merged_df["github_issues_normalized"] * weight_github_issues)
    ) / total_weight

    merged_df = merged_df.sort_values(["hotspot_score", "total_churn"], ascending=[False, False]).reset_index(drop=True)
    merged_df["rank"] = merged_df.index + 1
    return merged_df


def summarize_hotspots(
    hotspot_df: pd.DataFrame,
    github_issue_df: pd.DataFrame,
    sonar_issue_df: pd.DataFrame,
) -> dict[str, float]:
    """Create top-level summary metrics for the dashboard."""
    if hotspot_df.empty:
        return {
            "files_analyzed": 0,
            "total_churn": 0,
            "maintenance_issues": 0,
            "sonar_issues": 0,
            "average_hotspot_score": 0.0,
            "median_hotspot_score": 0.0,
        }

    hotspot_scores = pd.to_numeric(hotspot_df["hotspot_score"], errors="coerce").fillna(0)
    return {
        "files_analyzed": int(hotspot_df["file_path"].nunique()),
        "total_churn": int(pd.to_numeric(hotspot_df["total_churn"], errors="coerce").fillna(0).sum()),
        "maintenance_issues": int(len(github_issue_df)),
        "sonar_issues": int(len(sonar_issue_df)),
        "average_hotspot_score": float(hotspot_scores.mean()),
        "median_hotspot_score": float(hotspot_scores.median()),
    }
