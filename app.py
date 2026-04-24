"""Streamlit entrypoint for the Technical Debt Hotspot Dashboard."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Callable

import pandas as pd
import streamlit as st

from src.charts import (
    create_churn_over_time_chart,
    create_hotspot_bar_chart,
    create_scatter_chart,
    create_validation_comparison_chart,
)
from src.config import AppConfig, ensure_data_directories, get_default_sidebar_values
from src.github_data import (
    GitHubAPIError,
    build_issue_signal,
    fetch_commit_history_with_progress,
    fetch_maintenance_issues_with_progress,
    load_mock_github_data,
)
from src.scoring import compute_hotspot_scores, compute_validation_insights, summarize_hotspots
from src.sonar_data import SonarAPIError, fetch_sonar_issues_with_progress, load_mock_sonar_data


st.set_page_config(
    page_title="Potential Technical Debt Hotspot Dashboard",
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


def _derive_time_windows(config: AppConfig) -> dict[str, Any]:
    """Split the selected study range into analysis and validation windows when enabled."""
    study_start = config.start_date
    study_end = config.end_date
    windows = {
        "study_start": study_start,
        "study_end": study_end,
        "analysis_start": study_start,
        "analysis_end": study_end,
        "validation_enabled": False,
        "validation_start": None,
        "validation_end": None,
    }

    if not config.enable_time_sliced_validation:
        return windows

    total_days = (study_end - study_start).days + 1
    if total_days < 28:
        raise ValueError("Time-sliced validation needs at least 28 days in the selected study range.")

    validation_days = max(14, int(round(total_days * config.validation_window_ratio)))
    analysis_days = total_days - validation_days
    if analysis_days < 14:
        raise ValueError("The validation split is too large for the selected date range. Reduce the validation share or widen the range.")

    analysis_end = study_start + timedelta(days=analysis_days - 1)
    validation_start = analysis_end + timedelta(days=1)

    windows.update(
        {
            "analysis_end": analysis_end,
            "validation_enabled": True,
            "validation_start": validation_start,
            "validation_end": study_end,
        }
    )
    return windows


def _render_sidebar(defaults: dict[str, Any]) -> tuple[AppConfig, bool]:
    """Render the sidebar inputs and return the selected configuration."""
    with st.sidebar.form("analysis_inputs"):
        st.subheader("Repository Settings")
        github_owner = st.text_input("GitHub owner", value=defaults["github_owner"])
        github_repo = st.text_input("GitHub repo", value=defaults["github_repo"])
        default_branch = st.text_input("Default branch", value=defaults["default_branch"])

        date_range = st.date_input(
            "Study date range",
            value=(defaults["start_date"], defaults["end_date"]),
            help="This is the full historical window used for analysis, and optionally split into analysis and validation slices.",
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
        weight_churn = st.slider(
            "Code activity weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Controls the combined code-activity signal built from churn, recency, ownership concentration, contributor spread, and burstiness.",
        )
        weight_sonar = st.slider("Sonar weight", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        weight_github_issues = st.slider(
            "GitHub issue weight",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
        )

        st.subheader("Time-Sliced Validation")
        enable_time_sliced_validation = st.checkbox(
            "Enable time-sliced validation",
            value=bool(defaults.get("enable_time_sliced_validation", False)),
            help="Splits the selected study range into an analysis slice and a future validation slice, so the system can compare current hotspot scores with later maintenance pressure.",
        )
        validation_window_ratio = defaults.get("validation_window_ratio", 0.25)
        if enable_time_sliced_validation:
            validation_window_ratio = st.slider(
                "Validation window share",
                min_value=0.15,
                max_value=0.50,
                value=float(validation_window_ratio),
                step=0.05,
                help="Fraction of the study range reserved for the future validation slice.",
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
        enable_time_sliced_validation=enable_time_sliced_validation,
        validation_window_ratio=float(validation_window_ratio),
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


def _build_signal_summary(config: AppConfig, sonar_issue_df: pd.DataFrame, validation_enabled: bool) -> str:
    """Build a concise summary of which signals are active in this run."""
    signal_labels = ["GitHub code-activity signals", "GitHub maintenance issue search"]
    if config.use_mock_data:
        signal_labels.append("Mock SonarQube data")
    elif not sonar_issue_df.empty:
        signal_labels.append("SonarCloud code smell issues")
    elif config.sonar_base_url and config.sonar_token and config.sonar_project_key:
        signal_labels.append("SonarCloud configured")
    else:
        signal_labels.append("SonarCloud omitted")
    if validation_enabled:
        signal_labels.append("time-sliced validation")
    return ", ".join(signal_labels)


def _render_repository_summary(
    config: AppConfig,
    commit_df: pd.DataFrame,
    hotspot_df: pd.DataFrame,
    sonar_issue_df: pd.DataFrame,
    windows: dict[str, Any],
) -> None:
    """Render a compact repository summary card section."""
    commit_count = int(commit_df["sha"].nunique()) if not commit_df.empty else 0
    contributor_count = int(commit_df["contributor"].nunique()) if not commit_df.empty else 0
    analysis_days = (windows["analysis_end"] - windows["analysis_start"]).days + 1
    hotspot_candidates = int(hotspot_df["file_path"].nunique()) if not hotspot_df.empty else 0
    signal_summary = _build_signal_summary(config, sonar_issue_df, windows["validation_enabled"])

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

    study_range = f"{windows['study_start'].isoformat()} to {windows['study_end'].isoformat()}"
    analysis_range = f"{windows['analysis_start'].isoformat()} to {windows['analysis_end'].isoformat()}"
    caption_parts = [
        f"Study range: {study_range}",
        f"Analysis slice: {analysis_range}",
        f"Hotspot candidates: {hotspot_candidates:,}",
        f"Signals used: {signal_summary}",
    ]
    if windows["validation_enabled"]:
        caption_parts.insert(
            2,
            f"Validation slice: {windows['validation_start'].isoformat()} to {windows['validation_end'].isoformat()}",
        )
    st.caption(" | ".join(caption_parts))


def _render_methodology(config: AppConfig, windows: dict[str, Any]) -> None:
    """Render the methodology section for seminar use."""
    st.subheader("Methodology")
    left_col, right_col = st.columns([1.3, 0.7])

    with left_col:
        validation_note = ""
        if windows["validation_enabled"]:
            validation_note = (
                f"\n\n**Time-sliced validation**\n"
                f"- Analysis slice: `{windows['analysis_start'].isoformat()}` to `{windows['analysis_end'].isoformat()}`\n"
                f"- Validation slice: `{windows['validation_start'].isoformat()}` to `{windows['validation_end'].isoformat()}`\n"
                f"- Current hotspot scores are computed on the analysis slice and then compared with later signals from the validation slice."
            )

        st.markdown(
            f"""
            This dashboard combines multiple repository signals to identify **potential technical debt hotspots**.

            **Data sources**
            - GitHub commit history: file-level commit count, additions, deletions, total churn, contributor spread, recency, and burstiness.
            - GitHub issue search: maintenance-related issues matching keywords such as `refactor`, `cleanup`, `technical debt`, `maintainability`, and `code smell`.
            - SonarCloud maintainability findings: unresolved `CODE_SMELL` issues aggregated by file when SonarCloud access is configured.

            **Metrics**
            - `Code activity signal` combines total churn with commit frequency, active days, contributor spread, average churn per commit, recency of last touch, ownership concentration, bug-fix ratio, and change burstiness.
            - `GitHub issue signal` combines repository-level maintenance issue frequency with weighted file attributions using exact path mentions, path suffix mentions, basename mentions, and directory context.
            - `Sonar signal` combines SonarCloud code-smell count with severity information at file level.
            {validation_note}
            """
        )

    with right_col:
        st.markdown("**Scoring formula**")
        st.latex(
            r"""
            hotspot\_score =
            \frac{(w_a \cdot activity_{norm}) + (w_s \cdot sonar_{norm}) + (w_g \cdot github_{norm})}
            {w_a + w_s + w_g}
            """
        )
        st.caption(
            f"Current weights: activity={config.weight_churn:.2f}, "
            f"sonar={config.weight_sonar:.2f}, github issues={config.weight_github_issues:.2f}"
        )
        st.markdown(
            """
            Each signal is min-max normalized before weighting. The GitHub and Sonar components also include confidence-oriented secondary features so that stronger evidence is favored over weak textual mentions or severity-free counts.
            """
        )


def _render_limitations() -> None:
    """Render the limitations section for seminar use."""
    st.subheader("Limitations")
    st.markdown(
        """
        - Code churn, recency, ownership concentration, and issue frequency are **proxy indicators**. They can suggest maintenance pressure, but they are not direct proof of technical debt.
        - High code activity can still reflect healthy feature work, active refactoring, or release preparation rather than problematic code.
        - GitHub issue attribution is heuristic. Exact path mentions are stronger evidence than basename-only mentions, but both can still miss context or over-attribute ambiguous files.
        - SonarCloud issue counts depend on analysis coverage and rule configuration. They should be interpreted as one input among several, not as a definitive quality score.
        - Time-sliced validation compares one historical slice with a later slice, but it still does not establish causality or a ground-truth debt label.
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
            recent_messages = log_messages[-10:]
            log_placeholder.markdown(
                "**Collection progress**\n" + "\n".join(f"- {entry}" for entry in recent_messages)
            )

    def finish(message: str) -> None:
        update(1.0, message)

    return update, finish


def _stage_progress(updater: ProgressUpdater, start: float, end: float) -> ProgressUpdater:
    """Map a stage-local 0-1 progress value onto the overall progress bar."""
    span = max(end - start, 0.0)

    def inner(local_progress: float, message: str) -> None:
        bounded_local = max(0.0, min(1.0, local_progress))
        updater(start + (bounded_local * span), message)

    return inner


def _prefixed_stage_progress(updater: ProgressUpdater, label: str, start: float, end: float) -> ProgressUpdater:
    """Prefix progress messages with the active analysis slice."""
    staged_updater = _stage_progress(updater, start, end)

    def inner(local_progress: float, message: str) -> None:
        staged_updater(local_progress, f"{label}: {message}")

    return inner


def _build_issue_signal_frame(
    churn_summary_df: pd.DataFrame,
    sonar_summary_df: pd.DataFrame,
    github_issue_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build issue signal frame for the union of files present in churn and Sonar data."""
    all_file_paths = sorted(
        set(churn_summary_df.get("file_path", pd.Series(dtype=str)).dropna().tolist())
        | set(sonar_summary_df.get("file_path", pd.Series(dtype=str)).dropna().tolist())
    )
    return build_issue_signal(github_issue_df, all_file_paths)


def _collect_live_dataset(
    config: AppConfig,
    start_date: Any,
    end_date: Any,
    progress_specs: dict[str, tuple[float, float]],
    progress_updater: ProgressUpdater,
    dataset_label: str,
    data_source_messages: list[str],
) -> dict[str, pd.DataFrame]:
    """Collect GitHub and optional Sonar data for one analysis slice."""
    commit_df, churn_summary_df, daily_churn_df = fetch_commit_history_with_progress(
        owner=config.github_owner,
        repo=config.github_repo,
        branch=config.default_branch,
        start_date=start_date,
        end_date=end_date,
        token=config.github_token,
        progress_callback=_prefixed_stage_progress(progress_updater, dataset_label, *progress_specs["commits"]),
    )
    github_issue_df = fetch_maintenance_issues_with_progress(
        owner=config.github_owner,
        repo=config.github_repo,
        start_date=start_date,
        end_date=end_date,
        token=config.github_token,
        progress_callback=_prefixed_stage_progress(progress_updater, dataset_label, *progress_specs["issues"]),
    )

    sonar_issue_df = pd.DataFrame()
    sonar_summary_df = _empty_sonar_summary()
    if config.sonar_base_url and config.sonar_token and config.sonar_project_key:
        try:
            sonar_issue_df, sonar_summary_df = fetch_sonar_issues_with_progress(
                base_url=config.sonar_base_url,
                token=config.sonar_token,
                project_key=config.sonar_project_key,
                progress_callback=_prefixed_stage_progress(progress_updater, dataset_label, *progress_specs["sonar"]),
            )
        except SonarAPIError as exc:
            data_source_messages.append(
                f"{dataset_label}: SonarQube data collection failed: {exc}. Scores for this slice use GitHub-only signals."
            )
    else:
        data_source_messages.append(f"{dataset_label}: SonarQube settings are incomplete. SonarQube signals were omitted.")
        progress_updater(progress_specs["sonar"][1], f"{dataset_label}: Skipping SonarCloud because required settings are incomplete.")

    return {
        "commit_df": commit_df,
        "churn_summary_df": churn_summary_df,
        "daily_churn_df": daily_churn_df,
        "github_issue_df": github_issue_df,
        "sonar_issue_df": sonar_issue_df,
        "sonar_summary_df": sonar_summary_df,
    }


def _render_validation_section(windows: dict[str, Any], validation_insights: dict[str, float], comparison_df: pd.DataFrame) -> None:
    """Render time-sliced validation outputs when available."""
    if comparison_df.empty or not windows["validation_enabled"]:
        return

    st.subheader("Time-sliced validation")
    st.caption(
        f"Analysis slice: {windows['analysis_start'].isoformat()} to {windows['analysis_end'].isoformat()} | "
        f"Validation slice: {windows['validation_start'].isoformat()} to {windows['validation_end'].isoformat()}"
    )

    columns = st.columns(4)
    columns[0].metric(
        "Top-20 overlap",
        f"{validation_insights['top_n_overlap']}/{validation_insights['top_n']}",
    )
    columns[1].metric("Rank correlation", f"{validation_insights['score_correlation']:.2f}")
    columns[2].metric(
        "Avg future score of current top 20",
        _format_hotspot_score(validation_insights["avg_future_score_top_n"]),
    )
    columns[3].metric("Top-20 future lift", f"{validation_insights['top_n_future_lift']:.2f}x")

    st.plotly_chart(create_validation_comparison_chart(comparison_df), use_container_width=True)

    validation_columns = [
        "rank",
        "file_path",
        "hotspot_score",
        "future_rank",
        "future_hotspot_score",
        "future_total_churn",
        "future_sonar_issue_count",
        "future_github_issue_signal",
    ]
    st.dataframe(comparison_df[[column for column in validation_columns if column in comparison_df.columns]].head(15), use_container_width=True, hide_index=True)


def _render_results(
    config: AppConfig,
    commit_df: pd.DataFrame,
    hotspot_df: pd.DataFrame,
    daily_churn_df: pd.DataFrame,
    github_issue_df: pd.DataFrame,
    sonar_issue_df: pd.DataFrame,
    data_source_messages: list[str],
    windows: dict[str, Any],
    validation_insights: dict[str, float] | None = None,
    comparison_df: pd.DataFrame | None = None,
) -> None:
    """Render the main dashboard content once data has been computed."""
    st.caption(
        "Heuristic dashboard for surfacing potential technical debt hotspots using repository code activity, "
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
    _render_repository_summary(config, commit_df, hotspot_df, sonar_issue_df, windows)
    _render_kpis(summary)

    left_col, right_col = st.columns([1.1, 0.9])
    with left_col:
        st.plotly_chart(create_hotspot_bar_chart(hotspot_df), use_container_width=True)
    with right_col:
        st.plotly_chart(create_scatter_chart(hotspot_df), use_container_width=True)

    st.plotly_chart(create_churn_over_time_chart(daily_churn_df), use_container_width=True)

    if validation_insights is not None and comparison_df is not None:
        _render_validation_section(windows, validation_insights, comparison_df)

    # _render_methodology(config, windows)
    # _render_limitations()

    export_df = hotspot_df.copy()
    if comparison_df is not None and not comparison_df.empty:
        export_df = comparison_df.copy()

    display_columns = [
        "rank",
        "file_path",
        "hotspot_score",
        "code_activity_normalized",
        "total_churn",
        "commit_count",
        "active_days",
        "days_since_last_touch",
        "ownership_concentration",
        "bugfix_commit_ratio",
        "churn_burstiness",
        "sonar_issue_count",
        "github_issue_signal",
        "github_issue_matches",
        "github_issue_attribution_confidence",
        "github_exact_path_matches",
        "github_suffix_path_matches",
        "github_basename_matches",
        "github_directory_matches",
        "github_issue_examples",
        "future_rank",
        "future_hotspot_score",
        "future_total_churn",
        "future_sonar_issue_count",
        "future_github_issue_signal",
        "highest_severity",
        "severity_breakdown",
    ]
    available_columns = [column for column in display_columns if column in export_df.columns]
    export_view_df = export_df[available_columns].copy()
    export_filename = (
        f"{config.github_owner}_{config.github_repo}_hotspot_ranking_"
        f"{config.start_date.isoformat()}_{config.end_date.isoformat()}.csv"
    )

    st.subheader("Hotspot ranking")
    st.download_button(
        label="Download hotspot ranking as CSV",
        data=export_view_df.to_csv(index=False).encode("utf-8"),
        file_name=export_filename,
        mime="text/csv",
    )
    st.dataframe(export_view_df, use_container_width=True, hide_index=True)


def main() -> None:
    """Run the Streamlit dashboard."""
    ensure_data_directories()
    defaults = get_default_sidebar_values()
    config, run_analysis = _render_sidebar(defaults)

    st.title("Maintainability-Related Technical Debt Hotspots")
    st.caption(
        "Configure the repository and click `Run analysis` to surface potential maintainability-related technical debt hotspots."
    )
    st.info("This dashboard combines heuristic indicators. It does not detect technical debt with certainty.")

    if not run_analysis:
        st.stop()

    if config.start_date > config.end_date:
        st.error("The start date must be on or before the end date.")
        st.stop()

    try:
        windows = _derive_time_windows(config)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    data_source_messages: list[str] = []
    progress_updater, finish_progress = _create_progress_updater()

    validation_insights: dict[str, float] | None = None
    comparison_df: pd.DataFrame | None = None

    if config.use_mock_data:
        progress_updater(0.15, "Mock mode: loading bundled GitHub data...")
        commit_df, churn_summary_df, daily_churn_df, github_issue_df, github_issue_signal_df = load_mock_github_data()
        progress_updater(0.60, "Mock mode: loading bundled SonarQube data...")
        sonar_issue_df, sonar_summary_df = load_mock_sonar_data()
        progress_updater(0.90, "Calculating hotspot scores from mock data...")
        if windows["validation_enabled"]:
            data_source_messages.append("Time-sliced validation is disabled in mock mode. Switch to live data to compare current and future slices.")
            windows["validation_enabled"] = False
    else:
        if not config.github_owner or not config.github_repo:
            st.error("GitHub owner and repo are required for live analysis.")
            st.stop()

        try:
            if windows["validation_enabled"]:
                analysis_dataset = _collect_live_dataset(
                    config,
                    windows["analysis_start"],
                    windows["analysis_end"],
                    {
                        "commits": (0.00, 0.28),
                        "issues": (0.28, 0.38),
                        "sonar": (0.38, 0.48),
                    },
                    progress_updater,
                    "Analysis window",
                    data_source_messages,
                )
                validation_dataset = _collect_live_dataset(
                    config,
                    windows["validation_start"],
                    windows["validation_end"],
                    {
                        "commits": (0.48, 0.74),
                        "issues": (0.74, 0.84),
                        "sonar": (0.84, 0.94),
                    },
                    progress_updater,
                    "Validation window",
                    data_source_messages,
                )
            else:
                analysis_dataset = _collect_live_dataset(
                    config,
                    windows["analysis_start"],
                    windows["analysis_end"],
                    {
                        "commits": (0.00, 0.62),
                        "issues": (0.62, 0.78),
                        "sonar": (0.78, 0.94),
                    },
                    progress_updater,
                    "Analysis window",
                    data_source_messages,
                )
                validation_dataset = None
        except GitHubAPIError as exc:
            st.error(f"GitHub data collection failed: {exc}")
            st.stop()

        commit_df = analysis_dataset["commit_df"]
        churn_summary_df = analysis_dataset["churn_summary_df"]
        daily_churn_df = analysis_dataset["daily_churn_df"]
        github_issue_df = analysis_dataset["github_issue_df"]
        sonar_issue_df = analysis_dataset["sonar_issue_df"]
        sonar_summary_df = analysis_dataset["sonar_summary_df"]
        github_issue_signal_df = _build_issue_signal_frame(churn_summary_df, sonar_summary_df, github_issue_df)

    hotspot_df = compute_hotspot_scores(
        churn_summary_df=churn_summary_df,
        sonar_summary_df=sonar_summary_df,
        github_issue_signal_df=github_issue_signal_df,
        weight_churn=config.weight_churn,
        weight_sonar=config.weight_sonar,
        weight_github_issues=config.weight_github_issues,
    )

    if not config.use_mock_data and windows["validation_enabled"] and validation_dataset is not None:
        validation_issue_signal_df = _build_issue_signal_frame(
            validation_dataset["churn_summary_df"],
            validation_dataset["sonar_summary_df"],
            validation_dataset["github_issue_df"],
        )
        validation_hotspot_df = compute_hotspot_scores(
            churn_summary_df=validation_dataset["churn_summary_df"],
            sonar_summary_df=validation_dataset["sonar_summary_df"],
            github_issue_signal_df=validation_issue_signal_df,
            weight_churn=config.weight_churn,
            weight_sonar=config.weight_sonar,
            weight_github_issues=config.weight_github_issues,
        )
        validation_insights, comparison_df = compute_validation_insights(hotspot_df, validation_hotspot_df)

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
        windows=windows,
        validation_insights=validation_insights,
        comparison_df=comparison_df,
    )


if __name__ == "__main__":
    main()
