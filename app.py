"""Streamlit entrypoint for the Technical Debt Hotspot Dashboard."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st

from src.charts import (
    create_churn_over_time_chart,
    create_hotspot_bar_chart,
    create_scatter_chart,
)
from src.config import AppConfig, ensure_data_directories, get_default_sidebar_values
from src.github_data import (
    GitHubAPIError,
    build_issue_signal,
    fetch_commit_history_with_progress,
    fetch_maintenance_issues_with_progress,
    load_mock_github_data,
)
from src.scoring import compute_hotspot_scores, summarize_hotspots
from src.sonar_data import SonarAPIError, fetch_sonar_issues_with_progress, load_mock_sonar_data


st.set_page_config(
    page_title="Technical Debt Hotspot Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)


ProgressUpdater = Callable[[float, str], None]


def _empty_sonar_summary() -> pd.DataFrame:
    """Return an empty Sonar summary with a stable schema."""
    return pd.DataFrame(
        columns=[
            "file_path",
            "sonar_issue_count",
            "highest_severity",
            "severity_breakdown",
            "type_breakdown",
        ]
    )


def _format_hotspot_score(value: float) -> str:
    """Format hotspot scores so very small values stay visible in KPIs."""
    if value >= 0.1:
        return f"{value:.2f}"
    if value >= 0.01:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _render_sidebar(defaults: dict[str, Any]) -> tuple[AppConfig, bool]:
    """Render the sidebar inputs and return the selected configuration."""
    with st.sidebar.form("analysis_inputs"):
        st.subheader("Repository Settings")
        github_owner = st.text_input("GitHub owner", value=defaults["github_owner"])
        github_repo = st.text_input("GitHub repo", value=defaults["github_repo"])
        default_branch = st.text_input("Default branch", value=defaults["default_branch"])

        date_range = st.date_input(
            "Analysis date range",
            value=(defaults["start_date"], defaults["end_date"]),
            help="Commit churn and GitHub issue signals are calculated across this window.",
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = defaults["start_date"]
            end_date = defaults["end_date"]

        st.subheader("GitHub Access")
        github_token = st.text_input(
            "GitHub token",
            value=defaults["github_token"],
            type="password",
            help="Optional for public repositories, but recommended to avoid rate limits.",
        )

        st.subheader("SonarQube Access")
        sonar_base_url = st.text_input(
            "SonarQube base URL",
            value=defaults["sonar_base_url"],
            placeholder="https://sonarqube.example.com",
        )
        sonar_token = st.text_input(
            "SonarQube token",
            value=defaults["sonar_token"],
            type="password",
        )
        sonar_project_key = st.text_input(
            "SonarQube project key",
            value=defaults["sonar_project_key"],
        )

        st.subheader("Scoring Weights")
        weight_churn = st.slider("Churn weight", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        weight_sonar = st.slider("Sonar weight", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        weight_github_issues = st.slider(
            "GitHub issue weight",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
        )

        use_mock_data = st.checkbox(
            "Use mock data",
            value=defaults["use_mock_data"],
            help="Useful for demos or when API access is not configured yet.",
        )

        run_analysis = st.form_submit_button("Run analysis", use_container_width=True)

    config = AppConfig(
        github_owner=github_owner.strip(),
        github_repo=github_repo.strip(),
        default_branch=default_branch.strip() or "main",
        start_date=start_date,
        end_date=end_date,
        github_token=github_token.strip(),
        sonar_base_url=sonar_base_url.strip(),
        sonar_token=sonar_token.strip(),
        sonar_project_key=sonar_project_key.strip(),
        weight_churn=weight_churn,
        weight_sonar=weight_sonar,
        weight_github_issues=weight_github_issues,
        use_mock_data=use_mock_data,
    )
    return config, run_analysis


def _render_kpis(summary: dict[str, Any]) -> None:
    """Display the dashboard KPI strip."""
    columns = st.columns(6)
    columns[0].metric("Files analyzed", f"{summary['files_analyzed']:,}")
    columns[1].metric("Total churn", f"{summary['total_churn']:,}")
    columns[2].metric("GitHub maintenance issues", f"{summary['maintenance_issues']:,}")
    columns[3].metric("Sonar code smells", f"{summary['sonar_issues']:,}")
    columns[4].metric("Avg hotspot score", _format_hotspot_score(summary["average_hotspot_score"]))
    columns[5].metric("Median hotspot score", _format_hotspot_score(summary["median_hotspot_score"]))


def _build_signal_summary(config: AppConfig, sonar_issue_df: pd.DataFrame) -> str:
    """Build a concise summary of which signals are active in this run."""
    signal_labels = ["GitHub churn", "GitHub maintenance issue search"]
    if config.use_mock_data:
        signal_labels.append("Mock SonarQube data")
    elif not sonar_issue_df.empty:
        signal_labels.append("SonarCloud code smell issues")
    elif config.sonar_base_url and config.sonar_token and config.sonar_project_key:
        signal_labels.append("SonarCloud configured")
    else:
        signal_labels.append("SonarCloud omitted")
    return ", ".join(signal_labels)


def _render_repository_summary(
    config: AppConfig,
    commit_df: pd.DataFrame,
    hotspot_df: pd.DataFrame,
    sonar_issue_df: pd.DataFrame,
) -> None:
    """Render a compact repository summary card section."""
    commit_count = int(commit_df["sha"].nunique()) if not commit_df.empty else 0
    contributor_count = int(commit_df["contributor"].nunique()) if not commit_df.empty else 0
    analysis_days = (config.end_date - config.start_date).days + 1
    hotspot_candidates = int(hotspot_df["file_path"].nunique()) if not hotspot_df.empty else 0
    signal_summary = _build_signal_summary(config, sonar_issue_df)

    st.subheader("Repository summary")
    columns = st.columns(5)
    card_values = [
        ("Repository", f"{config.github_owner}/{config.github_repo}"),
        ("Branch", config.default_branch),
        ("Analysis window", f"{analysis_days} days"),
        ("Commits analyzed", f"{commit_count:,}"),
        ("Contributors observed", f"{contributor_count:,}"),
    ]

    for column, (label, value) in zip(columns, card_values):
        with column:
            with st.container(border=True):
                st.caption(label)
                st.markdown(f"**{value}**")

    st.caption(
        f"Date range: {config.start_date.isoformat()} to {config.end_date.isoformat()} | "
        f"Hotspot candidates: {hotspot_candidates:,} | Signals used: {signal_summary}"
    )


def _render_methodology(config: AppConfig) -> None:
    """Render the methodology section for seminar use."""
    st.subheader("Methodology")
    left_col, right_col = st.columns([1.3, 0.7])

    with left_col:
        st.markdown(
            """
            This dashboard combines multiple repository signals to identify **potential technical debt hotspots**.

            **Data sources**
            - GitHub commit history: file-level commit count, additions, deletions, total churn, and contributor spread.
            - GitHub issue search: maintenance-related issues matching keywords such as `refactor`, `cleanup`, `technical debt`, `maintainability`, and `code smell`.
            - SonarCloud maintainability findings: unresolved `CODE_SMELL` issues aggregated by file when SonarCloud access is configured.

            **Metrics**
            - `Total churn` = additions + deletions per file across the selected period.
            - `GitHub issue signal` combines repository-level maintenance issue frequency with simple file-name or path mentions in issue text.
            - `Sonar issue count` is the number of SonarCloud `CODE_SMELL` findings associated with each file.
            """
        )

    with right_col:
        st.markdown("**Scoring formula**")
        st.latex(
            r"""
            hotspot\_score =
            \frac{(w_c \cdot churn_{norm}) + (w_s \cdot sonar_{norm}) + (w_g \cdot github_{norm})}
            {w_c + w_s + w_g}
            """
        )
        st.caption(
            f"Current weights: churn={config.weight_churn:.2f}, "
            f"sonar={config.weight_sonar:.2f}, github issues={config.weight_github_issues:.2f}"
        )
        st.markdown(
            """
            All three signals are min-max normalized to a 0-1 scale before weighting so that no single raw metric dominates only because of its numeric range.
            """
        )


def _render_limitations() -> None:
    """Render the limitations section for seminar use."""
    st.subheader("Limitations")
    st.markdown(
        """
        - Code churn and maintenance issue frequency are **proxy indicators**. They can suggest maintenance pressure, but they are not direct proof of technical debt.
        - High churn can reflect healthy feature work, active refactoring, or release preparation rather than problematic code.
        - GitHub issue signals depend on issue-writing practices. If teams do not document refactoring or code smell work in issues, the issue signal will be understated.
        - SonarCloud issue counts depend on analysis coverage and rule configuration. They should be interpreted as one input among several, not as a definitive quality score.
        - File-level attribution from GitHub issues is heuristic because it relies on simple path or filename mentions in issue text.
        """
    )


def _create_progress_updater() -> tuple[ProgressUpdater, Callable[[str], None]]:
    """Create UI elements used to report live analysis progress."""
    progress_bar = st.progress(0.0, text="Waiting to start analysis...")
    status_placeholder = st.empty()
    log_placeholder = st.empty()
    log_messages: list[str] = []

    def update(progress: float, message: str) -> None:
        bounded_progress = max(0.0, min(1.0, progress))
        progress_bar.progress(bounded_progress, text=message)
        status_placeholder.caption(message)
        if not log_messages or log_messages[-1] != message:
            log_messages.append(message)
            recent_messages = log_messages[-8:]
            log_placeholder.markdown(
                "**Collection progress**\n" + "\n".join(f"- {entry}" for entry in recent_messages)
            )

    def finish(message: str) -> None:
        update(1.0, message)

    return update, finish


def _stage_progress(
    updater: ProgressUpdater,
    start: float,
    end: float,
) -> ProgressUpdater:
    """Map a stage-local 0-1 progress value onto the overall progress bar."""
    span = max(end - start, 0.0)

    def inner(local_progress: float, message: str) -> None:
        bounded_local = max(0.0, min(1.0, local_progress))
        updater(start + (bounded_local * span), message)

    return inner


def _render_results(
    config: AppConfig,
    commit_df: pd.DataFrame,
    hotspot_df: pd.DataFrame,
    daily_churn_df: pd.DataFrame,
    github_issue_df: pd.DataFrame,
    sonar_issue_df: pd.DataFrame,
    data_source_messages: list[str],
) -> None:
    """Render the main dashboard content once data has been computed."""
    st.caption(
        "Heuristic dashboard for surfacing potential technical debt hotspots using repository churn, "
        "maintenance-related GitHub issues, and SonarQube code smell findings."
    )

    st.info(
        "Hotspot scores are heuristic indicators of potential technical debt hotspots. "
        "They are not ground-truth measurements and should be used alongside engineering judgment."
    )

    if config.use_mock_data:
        st.warning("Mock mode is active. The charts below use bundled demo data instead of live APIs.")

    for message in data_source_messages:
        st.warning(message)

    summary = summarize_hotspots(hotspot_df, github_issue_df, sonar_issue_df)
    _render_repository_summary(config, commit_df, hotspot_df, sonar_issue_df)
    _render_kpis(summary)

    left_col, right_col = st.columns([1.1, 0.9])
    with left_col:
        st.plotly_chart(create_hotspot_bar_chart(hotspot_df), use_container_width=True)
    with right_col:
        st.plotly_chart(create_scatter_chart(hotspot_df), use_container_width=True)

    st.plotly_chart(create_churn_over_time_chart(daily_churn_df), use_container_width=True)

    # _render_methodology(config)
    # _render_limitations()

    display_columns = [
        "rank",
        "file_path",
        "hotspot_score",
        "total_churn",
        "commit_count",
        "unique_contributors",
        "sonar_issue_count",
        "github_issue_signal",
        "github_issue_matches",
        "highest_severity",
        "severity_breakdown",
    ]
    available_columns = [column for column in display_columns if column in hotspot_df.columns]
    export_df = hotspot_df[available_columns].copy()
    export_filename = (
        f"{config.github_owner}_{config.github_repo}_hotspot_ranking_"
        f"{config.start_date.isoformat()}_{config.end_date.isoformat()}.csv"
    )

    st.subheader("Hotspot ranking")
    st.download_button(
        label="Download hotspot ranking as CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=export_filename,
        mime="text/csv",
    )
    st.dataframe(
        export_df,
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    """Run the Streamlit dashboard."""
    ensure_data_directories()
    defaults = get_default_sidebar_values()
    config, run_analysis = _render_sidebar(defaults)

    st.title("Technical Debt Hotspot Dashboard")
    st.caption(
        "Configure the repository and click `Run analysis` to surface potential technical debt hotspots."
    )
    st.info(
        "This dashboard combines heuristic indicators. It does not detect technical debt with certainty."
    )

    if not run_analysis:
        st.stop()

    if config.start_date > config.end_date:
        st.error("The start date must be on or before the end date.")
        st.stop()

    data_source_messages: list[str] = []
    progress_updater, finish_progress = _create_progress_updater()

    if config.use_mock_data:
        progress_updater(0.1, "Mock mode: loading bundled GitHub data...")
        commit_df, churn_summary_df, daily_churn_df, github_issue_df, github_issue_signal_df = load_mock_github_data()
        progress_updater(0.6, "Mock mode: loading bundled SonarQube data...")
        sonar_issue_df, sonar_summary_df = load_mock_sonar_data()
        progress_updater(0.9, "Calculating hotspot scores from mock data...")
    else:
        if not config.github_owner or not config.github_repo:
            st.error("GitHub owner and repo are required for live analysis.")
            st.stop()

        try:
            commit_df, churn_summary_df, daily_churn_df = fetch_commit_history_with_progress(
                owner=config.github_owner,
                repo=config.github_repo,
                branch=config.default_branch,
                start_date=config.start_date,
                end_date=config.end_date,
                token=config.github_token,
                progress_callback=_stage_progress(progress_updater, 0.0, 0.72),
            )
            github_issue_df = fetch_maintenance_issues_with_progress(
                owner=config.github_owner,
                repo=config.github_repo,
                start_date=config.start_date,
                end_date=config.end_date,
                token=config.github_token,
                progress_callback=_stage_progress(progress_updater, 0.72, 0.84),
            )
        except GitHubAPIError as exc:
            st.error(f"GitHub data collection failed: {exc}")
            st.stop()

        sonar_issue_df = pd.DataFrame()
        sonar_summary_df = _empty_sonar_summary()

        if config.sonar_base_url and config.sonar_token and config.sonar_project_key:
            try:
                sonar_issue_df, sonar_summary_df = fetch_sonar_issues_with_progress(
                    base_url=config.sonar_base_url,
                    token=config.sonar_token,
                    project_key=config.sonar_project_key,
                    progress_callback=_stage_progress(progress_updater, 0.84, 0.96),
                )
            except SonarAPIError as exc:
                data_source_messages.append(
                    f"SonarQube data collection failed: {exc}. Scores will use GitHub-only signals."
                )
                progress_updater(0.96, "SonarCloud access failed. Continuing with GitHub-only signals...")
        else:
            data_source_messages.append(
                "SonarQube settings are incomplete. SonarQube signals were omitted from this run."
            )
            progress_updater(0.96, "Skipping SonarCloud because required settings are incomplete.")

        all_file_paths = sorted(
            set(churn_summary_df.get("file_path", pd.Series(dtype=str)).dropna().tolist())
            | set(sonar_summary_df.get("file_path", pd.Series(dtype=str)).dropna().tolist())
        )
        github_issue_signal_df = build_issue_signal(github_issue_df, all_file_paths)
        progress_updater(0.98, "Combining GitHub and SonarCloud signals...")

    hotspot_df = compute_hotspot_scores(
        churn_summary_df=churn_summary_df,
        sonar_summary_df=sonar_summary_df,
        github_issue_signal_df=github_issue_signal_df,
        weight_churn=config.weight_churn,
        weight_sonar=config.weight_sonar,
        weight_github_issues=config.weight_github_issues,
    )

    finish_progress("Analysis complete.")

    if hotspot_df.empty:
        st.warning("No hotspot candidates were produced for the selected configuration.")
        st.stop()

    _render_results(
        config=config,
        commit_df=commit_df,
        hotspot_df=hotspot_df,
        daily_churn_df=daily_churn_df,
        github_issue_df=github_issue_df,
        sonar_issue_df=sonar_issue_df,
        data_source_messages=data_source_messages,
    )


if __name__ == "__main__":
    main()
