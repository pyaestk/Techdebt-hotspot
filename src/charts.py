"""Plotly chart builders for the hotspot dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _empty_figure(title: str) -> go.Figure:
    """Return an empty placeholder figure with a readable message."""
    figure = go.Figure()
    figure.add_annotation(
        text="No data available for the current selection.",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14},
    )
    figure.update_layout(title=title, template="plotly_white", height=420)
    return figure


def create_hotspot_bar_chart(hotspot_df: pd.DataFrame) -> go.Figure:
    """Create the top hotspot files bar chart."""
    if hotspot_df.empty:
        return _empty_figure("Top 20 Potential Technical Debt Hotspots by File")

    top_df = hotspot_df.head(20).sort_values("hotspot_score", ascending=True)
    figure = px.bar(
        top_df,
        x="hotspot_score",
        y="file_path",
        orientation="h",
        color="total_churn",
        color_continuous_scale="Tealgrn",
        title="Top 20 Potential Technical Debt Hotspots by File",
        labels={
            "hotspot_score": "Weighted hotspot score (0-1)",
            "file_path": "Repository file path",
            "total_churn": "Total code churn (additions + deletions)",
        },
        hover_data=["commit_count", "sonar_issue_count", "github_issue_signal"],
    )
    figure.update_layout(
        template="plotly_white",
        height=520,
        coloraxis_colorbar_title="Code churn",
        xaxis_title="Weighted hotspot score (0-1)",
        yaxis_title="Repository file path",
    )
    return figure


def create_scatter_chart(hotspot_df: pd.DataFrame) -> go.Figure:
    """Create the churn versus SonarQube scatter plot."""
    if hotspot_df.empty:
        return _empty_figure("File-Level Churn vs SonarCloud Maintainability Findings")

    figure = px.scatter(
        hotspot_df,
        x="total_churn",
        y="sonar_issue_count",
        color="hotspot_score",
        size="commit_count",
        hover_name="file_path",
        color_continuous_scale="Sunset",
        title="File-Level Churn vs SonarCloud Maintainability Findings",
        labels={
            "total_churn": "Total code churn (additions + deletions)",
            "sonar_issue_count": "Open SonarCloud code smell findings",
            "hotspot_score": "Weighted hotspot score (0-1)",
            "commit_count": "Commits touching file",
        },
    )
    figure.update_layout(
        template="plotly_white",
        height=520,
        xaxis_title="Total code churn (additions + deletions)",
        yaxis_title="Open SonarCloud code smell findings",
        coloraxis_colorbar_title="Hotspot score",
    )
    return figure


def create_churn_over_time_chart(daily_churn_df: pd.DataFrame) -> go.Figure:
    """Create the line chart showing churn over time."""
    if daily_churn_df.empty:
        return _empty_figure("Repository Code Churn Trend Over Time")

    sorted_df = daily_churn_df.sort_values("commit_date")
    figure = px.line(
        sorted_df,
        x="commit_date",
        y="total_churn",
        markers=True,
        title="Repository Code Churn Trend Over Time",
        labels={
            "commit_date": "Commit date",
            "total_churn": "Daily code churn (additions + deletions)",
        },
    )
    figure.update_traces(line={"color": "#0f766e", "width": 3})
    figure.update_layout(
        template="plotly_white",
        height=420,
        xaxis_title="Commit date",
        yaxis_title="Daily code churn (additions + deletions)",
    )
    return figure
