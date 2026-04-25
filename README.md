# Potential Maintainability related Technical Debt Hotspot Dashboard

Streamlit dashboard for surfacing **potential technical debt hotspots** in a GitHub repository by combining heuristic indicators from:

- GitHub code churn
- maintenance-related GitHub issues
- SonarQube maintainability-oriented code smell findings

The dashboard does **not** claim to detect technical debt with certainty. It highlights files that may deserve closer review.

## Features

- Sidebar configuration for GitHub and SonarQube inputs
- GitHub commit history ingestion with per-file churn metrics
- Keyword-based GitHub maintenance issue signal
- SonarQube `CODE_SMELL` issue aggregation by file
- Weighted hotspot scoring with adjustable weights
- KPI cards, bar chart, scatter plot, churn-over-time line chart, and sortable ranking table
- Local caching to `data/raw` and `data/processed`
- Mock-data mode so the app can run without live credentials

## Project Structure

```text
.
|- app.py
|- requirements.txt
|- README.md
|- .env.example
|- data/
|  |- processed/
|  `- raw/
`- src/
   |- charts.py
   |- config.py
   |- github_data.py
   |- scoring.py
   `- sonar_data.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in any defaults you want available in the sidebar.
4. Run the app:

```bash
python -m streamlit run app.py
```

## Environment Variables

Supported variables:

- `GITHUB_OWNER`
- `GITHUB_REPO`
- `GITHUB_DEFAULT_BRANCH`
- `GITHUB_TOKEN`
- `SONARQUBE_BASE_URL`
- `SONARQUBE_TOKEN`
- `SONARQUBE_PROJECT_KEY`
- `USE_MOCK_DATA`

## Data Collection Notes

### GitHub

- Commit history is fetched for the selected branch and date range.
- Per-file metrics include:
  - commit count
  - additions
  - deletions
  - total churn
  - unique contributors
- Maintenance-related issues are discovered using these keywords:
  - `refactor`
  - `cleanup`
  - `technical debt`
  - `maintainability`
  - `code smell`
- GitHub issue signal is heuristic. It uses keyword matches and simple file-path or filename mentions in issue text when available.

### SonarQube

- SonarQube data uses `/api/issues/search`
- Version 1 focuses on unresolved `CODE_SMELL` issues as a maintainability signal
- Results are aggregated by file path, with severity and type breakdowns when available

## Scoring

Signals are min-max normalized to a `0-1` range, then combined into a weighted score:

```text
hotspot_score = weighted average of normalized churn, sonar issue count, and github issue signal
```

Default weights:

- churn: `0.5`
- sonar: `0.3`
- GitHub issues: `0.2`

You can change the weights in the sidebar.

## Mock Mode

Enable `Use mock data` in the sidebar to run the UI without GitHub or SonarQube credentials. This is useful for demos, local UI work, and environments where live API access is not available yet.

## Local Caching

The app stores intermediate data in:

- `data/raw` for raw JSON API responses
- `data/processed` for aggregated CSV outputs

This keeps version 1 simple without requiring a database.

## Known Limitations

- Hotspot scores are heuristic indicators, not ground truth technical debt measurements.
- GitHub commit detail collection can be slow on large repositories because file-level churn requires per-commit API requests.
- File attribution from GitHub issues is approximate and depends on issue text mentioning relevant paths or filenames.
- SonarQube component keys are assumed to map cleanly to repository-relative file paths.
