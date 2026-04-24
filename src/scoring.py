"""Scoring helpers for hotspot ranking."""

from __future__ import annotations

import pandas as pd


SEVERITY_SCORES = {
    "NONE": 0,
    "UNKNOWN": 0,
    "INFO": 1,
    "MINOR": 2,
    "MAJOR": 3,
    "CRITICAL": 4,
    "BLOCKER": 5,
}


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


def invert_normalized_series(series: pd.Series) -> pd.Series:
    """Normalize a series and invert it so lower raw values become higher scores."""
    return 1.0 - normalize_series(series)


def _severity_score_series(series: pd.Series) -> pd.Series:
    """Convert categorical Sonar severities into numeric severity scores."""
    return pd.to_numeric(series.map(SEVERITY_SCORES), errors="coerce").fillna(0)


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
        "active_days": 0,
        "first_commit_date": pd.NA,
        "last_commit_date": pd.NA,
        "days_since_last_touch": 0,
        "avg_churn_per_commit": 0.0,
        "ownership_concentration": 0.0,
        "bugfix_commit_count": 0,
        "bugfix_commit_ratio": 0.0,
        "churn_burstiness": 0.0,
        "sonar_issue_count": 0,
        "highest_severity": "NONE",
        "severity_breakdown": "",
        "type_breakdown": "",
        "github_issue_matches": 0,
        "github_issue_signal": 0.0,
        "repo_issue_signal": 0,
        "github_exact_path_matches": 0,
        "github_suffix_path_matches": 0,
        "github_basename_matches": 0,
        "github_directory_matches": 0,
        "github_weighted_mentions": 0.0,
        "github_issue_attribution_confidence": 0.0,
        "github_issue_examples": "",
    }

    for column, default_value in fill_defaults.items():
        if column not in merged_df.columns:
            merged_df[column] = default_value
        else:
            merged_df[column] = merged_df[column].fillna(default_value)

    merged_df["churn_normalized"] = normalize_series(merged_df["total_churn"])
    merged_df["commit_count_normalized"] = normalize_series(merged_df["commit_count"])
    merged_df["active_days_normalized"] = normalize_series(merged_df["active_days"])
    merged_df["contributors_normalized"] = normalize_series(merged_df["unique_contributors"])
    merged_df["avg_churn_per_commit_normalized"] = normalize_series(merged_df["avg_churn_per_commit"])
    merged_df["recency_normalized"] = invert_normalized_series(merged_df["days_since_last_touch"])
    merged_df["ownership_normalized"] = normalize_series(merged_df["ownership_concentration"])
    merged_df["bugfix_ratio_normalized"] = normalize_series(merged_df["bugfix_commit_ratio"])
    merged_df["burstiness_normalized"] = normalize_series(merged_df["churn_burstiness"])

    merged_df["code_activity_normalized"] = (
        (merged_df["churn_normalized"] * 0.30)
        + (merged_df["commit_count_normalized"] * 0.10)
        + (merged_df["active_days_normalized"] * 0.10)
        + (merged_df["contributors_normalized"] * 0.10)
        + (merged_df["avg_churn_per_commit_normalized"] * 0.10)
        + (merged_df["recency_normalized"] * 0.10)
        + (merged_df["ownership_normalized"] * 0.10)
        + (merged_df["bugfix_ratio_normalized"] * 0.05)
        + (merged_df["burstiness_normalized"] * 0.05)
    )

    merged_df["sonar_count_normalized"] = normalize_series(merged_df["sonar_issue_count"])
    merged_df["sonar_severity_normalized"] = normalize_series(_severity_score_series(merged_df["highest_severity"]))
    merged_df["sonar_signal_normalized"] = (
        (merged_df["sonar_count_normalized"] * 0.80)
        + (merged_df["sonar_severity_normalized"] * 0.20)
    )

    merged_df["github_issues_normalized"] = normalize_series(merged_df["github_issue_signal"])
    merged_df["github_confidence_normalized"] = normalize_series(merged_df["github_issue_attribution_confidence"])
    merged_df["github_signal_normalized"] = (
        (merged_df["github_issues_normalized"] * 0.80)
        + (merged_df["github_confidence_normalized"] * 0.20)
    )

    total_weight = weight_churn + weight_sonar + weight_github_issues
    if total_weight <= 0:
        total_weight = 1.0

    merged_df["hotspot_score"] = (
        (merged_df["code_activity_normalized"] * weight_churn)
        + (merged_df["sonar_signal_normalized"] * weight_sonar)
        + (merged_df["github_signal_normalized"] * weight_github_issues)
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


def compute_validation_insights(
    hotspot_df: pd.DataFrame,
    validation_hotspot_df: pd.DataFrame,
    top_n: int = 20,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Compare current hotspot ranks against a future validation window."""
    if hotspot_df.empty or validation_hotspot_df.empty:
        return {
            "top_n": top_n,
            "top_n_overlap": 0,
            "score_correlation": 0.0,
            "avg_future_score_top_n": 0.0,
            "overall_future_avg": 0.0,
            "top_n_future_lift": 0.0,
            "validation_files": 0,
        }, hotspot_df.copy()

    validation_view = validation_hotspot_df[
        [
            "file_path",
            "rank",
            "hotspot_score",
            "total_churn",
            "sonar_issue_count",
            "github_issue_signal",
        ]
    ].rename(
        columns={
            "rank": "future_rank",
            "hotspot_score": "future_hotspot_score",
            "total_churn": "future_total_churn",
            "sonar_issue_count": "future_sonar_issue_count",
            "github_issue_signal": "future_github_issue_signal",
        }
    )

    comparison_df = hotspot_df.merge(validation_view, on="file_path", how="left")
    for column in [
        "future_rank",
        "future_hotspot_score",
        "future_total_churn",
        "future_sonar_issue_count",
        "future_github_issue_signal",
    ]:
        comparison_df[column] = pd.to_numeric(comparison_df[column], errors="coerce").fillna(0)

    current_top_files = set(hotspot_df.head(top_n)["file_path"])
    future_top_files = set(validation_hotspot_df.head(top_n)["file_path"])
    top_n_overlap = len(current_top_files & future_top_files)

    score_correlation = 0.0
    correlation_input = comparison_df[["hotspot_score", "future_hotspot_score"]].copy()
    if len(correlation_input) >= 2 and correlation_input["future_hotspot_score"].nunique() > 1:
        score_correlation = float(correlation_input.corr(method="spearman").iloc[0, 1])
        if pd.isna(score_correlation):
            score_correlation = 0.0

    avg_future_score_top_n = float(comparison_df.head(top_n)["future_hotspot_score"].mean()) if not comparison_df.empty else 0.0
    overall_future_avg = float(comparison_df["future_hotspot_score"].mean()) if not comparison_df.empty else 0.0
    top_n_future_lift = 0.0 if overall_future_avg <= 0 else avg_future_score_top_n / overall_future_avg

    return {
        "top_n": top_n,
        "top_n_overlap": top_n_overlap,
        "score_correlation": score_correlation,
        "avg_future_score_top_n": avg_future_score_top_n,
        "overall_future_avg": overall_future_avg,
        "top_n_future_lift": top_n_future_lift,
        "validation_files": int(validation_hotspot_df["file_path"].nunique()),
    }, comparison_df
