# Inflation Intelligence Engine

## Purpose

The Inflation Intelligence Engine is a standalone subproject focused on building a production-quality inflation analytics pipeline.

It is also the first module of a broader Macro Regime Engine.

The project should be developed in isolation as its own Git/GitHub repository so it can stand alone as a portfolio project.

Its purpose is to:

- Ingest CPI/HICP data across multiple countries
- Standardize inflation data into a clean schema
- Calculate inflation momentum metrics
- Build inflation regime features
- Generate charts and reports
- Add a dashboard
- Later add AI-generated macro commentary
- Later feed inflation signals into a broader macro regime model

---

## Project Repository

Recommended GitHub repository name:

```text
inflation-intelligence-engine
```

This should be separate from the broader macro regime research repo.

Reason:

- cleaner Git history
- easier to explain to employers
- easier to deploy independently
- easier to test and package
- easier to later import as a module into a larger macro platform

Future relationship:

```text
macro-regime-platform/
    ├── inflation-intelligence-engine/
    ├── growth-engine/
    ├── liquidity-engine/
    └── regime-engine/
```

For now, build this project independently.

---

## Core Vision

```text
Inflation data sources
    ↓
Data ingestion
    ↓
Standardized CPI/HICP panel
    ↓
Inflation metrics
    ↓
Momentum and regime features
    ↓
Charts and reports
    ↓
Dashboard
    ↓
AI macro commentary
    ↓
Telegram distribution
    ↓
Macro regime engine input
```

---

## Initial Country Coverage

### United States

Source:

- FRED

Initial series:

- Headline CPI
- Core CPI

Future series:

- Shelter
- Services
- Supercore
- PPI
- Breakevens

---

### Euro Area Countries

Source:

- Eurostat HICP

Initial countries:

- Germany
- France
- Italy
- Spain

Initial series:

- Headline HICP

Future series:

- Core HICP
- Energy
- Food
- Services
- Goods

---

### Nordics

Sources:

- Statistics Denmark
- SSB Norway
- Statistics Sweden

Initial countries:

- Denmark
- Norway
- Sweden

Initial series:

- Headline CPI

Future series:

- Core CPI
- Services
- Housing-related components
- Energy

---

## Future Component Decomposition

After the headline and core pipeline is stable, expand the source fetchers to
ingest detailed CPI/HICP subcomponents and their corresponding expenditure
weights.

The weights are needed because component inflation rates alone show where
prices are moving, but not how much each component contributed to headline
inflation. Combining component observations with the applicable weights will
allow the engine to:

- attribute headline inflation to categories such as food, energy, housing,
  goods, and services
- explain whether inflation is broad-based or concentrated in a few components
- compare contribution changes across countries and through time
- build country decomposition charts and deterministic commentary

Weights should be stored separately from the monthly observation panel because
they are generally annual and can change when statistical agencies update
consumption baskets. Each weight must match the component's country, index
family, classification, and effective year. Calculated contributions must also
account for source-specific chain-linking and aggregation methodology; use
official contribution series where available.

Planned outputs:

```text
data/raw/cpi_components.parquet
data/raw/cpi_weights.parquet
```

This expansion is intentionally deferred until the normalized headline and core
series, metrics, and reporting pipeline are working reliably.

---

## Proposed Folder Structure

This is the recommended standalone structure for the GitHub repository.

```text
inflation-intelligence-engine/
│
├── README.md
├── inflation-intelligence-engine.md
├── .gitignore
├── .env.example
├── pyproject.toml
├── run_pipeline.py
│
├── src/
│   └── inflation_engine/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── fetch_fred.py
│       │   ├── fetch_eurostat.py
│       │   ├── fetch_statbank_dk.py
│       │   ├── fetch_ssb.py
│       │   ├── fetch_sweden.py
│       │   └── combine_sources.py
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── inflation_metrics.py
│       │   ├── decomposition.py
│       │   ├── momentum.py
│       │   └── regime_features.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── regime_classifier.py
│       │   ├── nowcast.py
│       │   └── evaluation.py
│       │
│       ├── reports/
│       │   ├── __init__.py
│       │   ├── charts.py
│       │   ├── generate_report.py
│       │   └── templates/
│       │       └── inflation_report.md.jinja
│       │
│       ├── dashboard/
│       │   ├── __init__.py
│       │   └── streamlit_app.py
│       │
│       ├── ai/
│       │   ├── __init__.py
│       │   ├── summarize.py
│       │   ├── prompts.py
│       │   └── schemas.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── config.py
│           ├── paths.py
│           ├── schema.py
│           └── logging.py
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── outputs/
│   ├── charts/
│   ├── reports/
│   └── ai_summaries/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_inflation_momentum.ipynb
│   ├── 03_country_comparison.ipynb
│   └── 04_regime_features.ipynb
│
├── tests/
│   ├── test_schema.py
│   ├── test_inflation_metrics.py
│   ├── test_data_loaders.py
│   └── test_pipeline.py
│
├── docs/
│   ├── data_sources.md
│   ├── methodology.md
│   └── roadmap.md
│
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Why Use `src/inflation_engine/`?

This structure makes the project behave like a real Python package.

Instead of loose scripts, code is organized as an importable module:

```python
from inflation_engine.features.inflation_metrics import calculate_inflation_metrics
```

This is better for:

- testing
- packaging
- GitHub professionalism
- future deployment
- importing into the larger macro regime platform

---

## Standard Data Schema

Every CPI/HICP series should be normalized into the same structure.

```text
date
country
country_code
region
series_type
component
value
source
frequency
unit
```

Example:

```text
2025-04-01 | United States | US | North America | headline | all_items | 319.2 | FRED | monthly | index
```

Schema discipline is one of the most important parts of this project.

Bad schemas create fragile systems.

---

## Data Layer

### Goal

Create reliable, repeatable data ingestion from official/open sources.

### Initial files

```text
src/inflation_engine/data/fetch_fred.py
src/inflation_engine/data/fetch_eurostat.py
src/inflation_engine/data/combine_sources.py
```

### Raw data outputs

```text
data/raw/fred_us_cpi.parquet
data/raw/eurostat_hicp.parquet
data/raw/nordics_cpi.parquet
```

### Processed output

```text
data/processed/cpi_panel.parquet
```

---

## Feature Layer

### Initial Inflation Metrics

The core metrics should include:

- YoY inflation
- MoM inflation
- 3-month sequential annualized inflation
- True 3m/3m annualized inflation
- 6-month annualized inflation
- Momentum signal
- 3m/3m minus YoY spread

---

### YoY Inflation

```text
CPI_t / CPI_t-12 - 1
```

---

### MoM Inflation

```text
CPI_t / CPI_t-1 - 1
```

---

### 3-Month Sequential Annualized Inflation

This measures the change in the CPI index over the latest three months, annualized.

```text
(CPI_t / CPI_t-3)^4 - 1
```

This is useful for faster turning points.

---

### True 3m/3m Annualized Inflation

This compares the latest three-month average with the previous three-month average, annualized.

```text
(
  average(CPI_t, CPI_t-1, CPI_t-2)
  /
  average(CPI_t-3, CPI_t-4, CPI_t-5)
)^4 - 1
```

This is the preferred smoother macro momentum measure.

---

### 6-Month Annualized Inflation

```text
(CPI_t / CPI_t-6)^2 - 1
```

---

### Momentum Signal

Initial rule-based signal:

```text
if 3m3m_ann > yoy + 0.5:
    accelerating
elif 3m3m_ann < yoy - 0.5:
    slowing
else:
    stable
```

This can later become a probabilistic model.

---

## Report Layer

### Initial Report Output

```text
outputs/reports/latest_inflation_report.md
```

The report should include:

```text
# Inflation Intelligence Report

## Latest Observations

## Country Ranking by YoY Inflation

## Country Ranking by 3m/3m Annualized Inflation

## Momentum Signal

## Data Coverage

## Methodology Notes
```

---

## Chart Layer

Initial charts:

```text
outputs/charts/cpi_yoy_by_country.png
outputs/charts/cpi_3m3m_momentum.png
outputs/charts/inflation_heatmap.png
```

Later charts:

- Country decomposition charts
- Core vs headline charts
- Momentum spread charts
- Cross-country dispersion charts
- Regime probability charts

---

## Dashboard Layer

The first dashboard should be built after the pipeline works.

Recommended tool:

```text
Streamlit
```

Initial dashboard pages:

1. Country overview
2. Inflation momentum
3. Cross-country heatmap
4. Latest report
5. Data coverage

Future dashboard pages:

1. Inflation decomposition
2. Regime classification
3. Nowcast model
4. AI commentary
5. Macro regime integration

---

## AI Layer

The AI layer should be added only after the deterministic pipeline works.

Correct architecture:

```text
Python calculates the signals.
LLM explains the signals.
Automation distributes the output.
```

Initial AI task:

```text
Turn structured inflation metrics into a concise macro commentary.
```

Example input:

```json
{
  "country": "Sweden",
  "headline_yoy": 2.1,
  "core_yoy": 3.4,
  "ann_3m3m": 1.8,
  "momentum_signal": "slowing"
}
```

Example output:

```text
Swedish inflation momentum has slowed relative to its 12-month trend, but core inflation remains elevated. This suggests disinflation is underway, though underlying price pressure is not yet fully normalized.
```

Future AI tasks:

- Daily/weekly inflation summary
- CPI release reaction notes
- Telegram alerts
- Report rewriting
- Historical comparison
- RAG over previous reports

---

## Week 1 Deliverables

By the end of Week 1, this should run:

```bash
python run_pipeline.py
```

And produce:

```text
data/processed/cpi_panel.parquet

outputs/charts/
    cpi_yoy_by_country.png
    cpi_3m3m_momentum.png

outputs/reports/
    latest_inflation_report.md
```

---

## Week 1 Implementation Plan

### Day 1: Project Setup

Create repo:

```bash
mkdir inflation-intelligence-engine
cd inflation-intelligence-engine
git init
```

Create folders:

```bash
mkdir -p src/inflation_engine/{data,features,models,reports,dashboard,ai,utils}
mkdir -p data/{raw,interim,processed}
mkdir -p outputs/{charts,reports,ai_summaries}
mkdir -p notebooks tests docs .github/workflows
touch README.md inflation-intelligence-engine.md .gitignore .env.example pyproject.toml run_pipeline.py
```

Create `__init__.py` files:

```bash
touch src/inflation_engine/__init__.py
touch src/inflation_engine/data/__init__.py
touch src/inflation_engine/features/__init__.py
touch src/inflation_engine/models/__init__.py
touch src/inflation_engine/reports/__init__.py
touch src/inflation_engine/dashboard/__init__.py
touch src/inflation_engine/ai/__init__.py
touch src/inflation_engine/utils/__init__.py
```

Install first dependencies:

```bash
pip install pandas requests matplotlib pyarrow python-dotenv
```

Optional developer tools:

```bash
pip install ruff pytest
```

Commit:

```bash
git checkout -b feature/project-setup
git add .
git commit -m "feat: initialize inflation intelligence engine"
```

---

### Day 2: Schema and Config

Create:

```text
src/inflation_engine/utils/schema.py
src/inflation_engine/utils/config.py
src/inflation_engine/utils/paths.py
```

Add country metadata:

```python
COUNTRIES = {
    "US": {"name": "United States", "region": "North America"},
    "DE": {"name": "Germany", "region": "Euro Area"},
    "FR": {"name": "France", "region": "Euro Area"},
    "IT": {"name": "Italy", "region": "Euro Area"},
    "ES": {"name": "Spain", "region": "Euro Area"},
    "DK": {"name": "Denmark", "region": "Nordics"},
    "NO": {"name": "Norway", "region": "Nordics"},
    "SE": {"name": "Sweden", "region": "Nordics"},
}
```

Commit:

```bash
git add .
git commit -m "feat: define CPI schema and country metadata"
```

---

### Day 3: US CPI Ingestion

Create:

```text
src/inflation_engine/data/fetch_fred.py
```

Goal:

- Fetch US headline CPI
- Fetch US core CPI
- Normalize into standard schema
- Save raw parquet file

Output:

```text
data/raw/fred_us_cpi.parquet
```

Run:

```bash
python -m inflation_engine.data.fetch_fred
```

Commit:

```bash
git add .
git commit -m "feat: add FRED CPI ingestion"
```

---

### Day 4: Eurostat HICP Ingestion

Create:

```text
src/inflation_engine/data/fetch_eurostat.py
```

Goal:

- Fetch headline HICP for Germany, France, Italy, Spain
- Normalize into standard schema
- Save raw parquet file

Output:

```text
data/raw/eurostat_hicp.parquet
```

Commit:

```bash
git add .
git commit -m "feat: add Eurostat HICP ingestion"
```

---

### Day 5: Combine Sources

Create:

```text
src/inflation_engine/data/combine_sources.py
```

Goal:

- Load raw FRED and Eurostat files
- Combine into one clean panel
- Validate required columns
- Save intermediate panel

Output:

```text
data/interim/cpi_panel_raw_combined.parquet
```

Commit:

```bash
git add .
git commit -m "feat: combine CPI sources into standard panel"
```

---

### Day 6: Inflation Metrics

Create:

```text
src/inflation_engine/features/inflation_metrics.py
```

Goal:

- Calculate YoY
- Calculate MoM
- Calculate 3-month sequential annualized inflation
- Calculate true 3m/3m annualized inflation
- Calculate 6-month annualized inflation
- Add momentum signal

Output:

```text
data/processed/cpi_panel.parquet
```

Commit:

```bash
git add .
git commit -m "feat: calculate inflation momentum metrics"
```

---

### Day 7: Charts and Report

Create:

```text
src/inflation_engine/reports/charts.py
src/inflation_engine/reports/generate_report.py
```

Goal:

- Generate first charts
- Generate first Markdown report
- Wire everything into `run_pipeline.py`

Output:

```text
outputs/charts/cpi_yoy_by_country.png
outputs/charts/cpi_3m3m_momentum.png
outputs/reports/latest_inflation_report.md
```

Commit:

```bash
git add .
git commit -m "feat: generate first inflation intelligence report"
```

---

## GitHub Workflow

### 1. Work locally first

```bash
git init
git add .
git commit -m "feat: initialize inflation intelligence engine"
```

### 2. Create GitHub repo

Create a new GitHub repository named:

```text
inflation-intelligence-engine
```

### 3. Connect local repo to GitHub

```bash
git remote add origin git@github.com:<username>/inflation-intelligence-engine.git
git branch -M main
git push -u origin main
```

### 4. Use feature branches

```bash
git checkout -b feature/fred-ingestion
git checkout -b feature/eurostat-ingestion
git checkout -b feature/inflation-metrics
git checkout -b feature/report-generation
```

### 5. Push branches

```bash
git push origin feature/fred-ingestion
```

### 6. Merge through GitHub pull requests

This gives the project a professional Git history.

---

## What Not To Do In Week 1

Avoid:

- AI summaries
- Telegram bot
- nowcasting
- dashboard
- regime classifier
- LangChain
- agents
- deep learning
- complex deployment

Week 1 is only about:

```text
data ingestion
schema discipline
inflation metrics
charts
report generation
Git/GitHub workflow
```

---

## Phase 2: Dashboard

Add Streamlit after the Week 1 pipeline works.

Command:

```bash
streamlit run src/inflation_engine/dashboard/streamlit_app.py
```

Initial dashboard should show:

- Country selector
- CPI YoY chart
- 3m/3m annualized momentum chart
- Latest country rankings
- Inflation heatmap

---

## Phase 3: Inflation Regime Classification

Start with simple rule-based classifications:

```text
Disinflation
Reacceleration
Sticky inflation
Deflation pressure
Energy-driven inflation
Broad-based inflation
```

Then move to probabilistic models:

- Logistic scoring
- KMeans
- Gaussian Mixture Models
- Hidden Markov Models

---

## Phase 4: Nowcasting

Use leading indicators:

- Energy prices
- FX
- PPI
- wages
- import prices
- commodity baskets
- PMI price components

Start simple:

- Linear regression
- Ridge regression
- Rolling backtests

Then later:

- XGBoost
- Bayesian models

---

## Phase 5: AI Macro Commentary

Add LLM commentary only after the deterministic pipeline works.

Recommended flow:

```text
inflation metrics JSON
    ↓
structured AI summary
    ↓
Markdown report
    ↓
Telegram message
```

Use AI for explanation, not calculation.

---

## Phase 6: Telegram Distribution

Add automated distribution:

- CPI release summaries
- Weekly inflation monitor
- Regime change alerts
- Dashboard/report links

---

## Phase 7: Macro Regime Integration

The output of this engine should eventually feed into the broader macro regime model.

Inflation features to export:

```text
headline_yoy
core_yoy
headline_3m3m_ann
core_3m3m_ann
inflation_momentum_signal
inflation_reacceleration_flag
inflation_dispersion
inflation_regime_probability
```

These become inputs into:

- HMM
- Markov-Switching VAR
- Change-point detection
- Bayesian state-space models

---

## Guiding Principle

Use deterministic code to calculate signals.

Use AI to explain signals.

Use automation to distribute signals.

Correct architecture:

```text
Data
    ↓
Features
    ↓
Signals
    ↓
AI Explanation
    ↓
Reports / Dashboard / Telegram
```

This keeps the system:

- reliable
- explainable
- debuggable
- market-relevant
- useful for AI engineering roles
