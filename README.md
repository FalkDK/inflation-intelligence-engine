# Inflation Intelligence Engine

A Python pipeline for collecting, normalizing, and analyzing cross-country
inflation data.

It covers national CPI/HICP and US PCE, headline and core inflation, item-level
contributions, spending-weighted breadth, and release-aware local automation.
The terminal/browser Inflation Weather interface is currently a design concept.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Create a local environment file for the FRED API key:

```bash
cp .env.example .env
```

Then set `FRED_API_KEY` and `BLS_API_KEY` in `.env`. The detailed US CPI fetch
uses BLS for series not available on FRED. Denmark and the other public providers
do not need these keys. Real credentials belong only in your local `.env`;
see [Security](SECURITY.md). Generated data and runtime logs are ignored by Git.

## Fetch Inflation Data

When the package is installed with `pip install -e .`:

```bash
python3 -m inflation_engine.data.fetch_fred
python3 -m inflation_engine.data.fetch_ons
python3 -m inflation_engine.data.fetch_eurostat
python3 -m inflation_engine.data.fetch_statbank_dk
python3 -m inflation_engine.data.fetch_sweden
python3 -m inflation_engine.data.fetch_ssb
python3 -m inflation_engine.data.fetch_bea_pce
```

Without installing the package:

```bash
PYTHONPATH=src python3 -m inflation_engine.data.fetch_fred
PYTHONPATH=src python3 -m inflation_engine.data.fetch_ons
PYTHONPATH=src python3 -m inflation_engine.data.fetch_eurostat
PYTHONPATH=src python3 -m inflation_engine.data.fetch_statbank_dk
PYTHONPATH=src python3 -m inflation_engine.data.fetch_sweden
PYTHONPATH=src python3 -m inflation_engine.data.fetch_ssb
PYTHONPATH=src python3 -m inflation_engine.data.fetch_bea_pce
```

The commands write:

```text
data/raw/fred_us_cpi.parquet
data/raw/ons_uk_cpi.parquet
data/raw/eurostat_hicp.parquet
data/raw/statbank_dk_cpi.parquet
data/raw/scb_sweden_cpi.parquet
data/raw/ssb_norway_cpi.parquet
data/raw/bea_us_pce_cpi.parquet
data/raw/bea_us_pce.parquet
```

Both commands print the destination path. The HICP command also prints the
latest headline and core index values and their observation dates.

Every successful fetch also updates:

```text
data/raw/ingestion_manifest.json
```

The manifest records retrieval and source-update timestamps, output checksum,
row count, date coverage, source datasets, index bases, adjustment flags, and
observation-status counts. It is generated locally and ignored by Git.

## Combine Raw Sources

After the six raw fetches have completed:

```bash
PYTHONPATH=src python3 -m inflation_engine.data.combine_sources
```

This validates every raw file, combines them under the standard schema, checks
for duplicate observation keys, and writes:

```text
data/interim/cpi_panel_raw_combined.parquet
```

## Calculate Inflation Metrics

After creating the combined panel:

```bash
PYTHONPATH=src python3 -m inflation_engine.features.inflation_metrics
```

This writes:

```text
data/processed/cpi_panel.parquet
```

The processed panel preserves the normalized CPI columns and adds percentage
rates for MoM, YoY, 3-month annualized, true 3m/3m annualized, 6-month
annualized, and true 6m/6m annualized inflation. It also adds the 3m/3m minus
YoY spread, the seasonality-safe three-month change in YoY inflation, and a
rule-based momentum signal based on that YoY change.

By default, metrics use each source's published index. To seasonally adjust
non-adjusted series with the SEATS decomposition in X-13ARIMA-SEATS:

```bash
PYTHONPATH=src python3 -m inflation_engine.features.inflation_metrics \
    --seasonal-adjustment x13 \
    --x13-path /path/to/x13as
```

The executable can alternatively be provided through `X13PATH` or `PATH`.
The official Census Bureau distribution and documentation are available at
<https://www.census.gov/data/software/x13as.html>. Census does not distribute
a supported macOS binary, so macOS users must provide a compatible compiled
executable.

The source `value` and `seasonal_adjustment` columns are never overwritten.
`analysis_value` records the index used for metrics, and
`analysis_seasonal_adjustment` records whether it is source data or an
`x13_arima_seats` result. Source series that are already seasonally adjusted
pass through unchanged.

Sequential rates are calendar-aligned and require contiguous monthly
observations. For series that are not seasonally adjusted, MoM and other
short-horizon measures still contain seasonal effects.

## Build Regime Features

The main pipeline also writes the versioned model contract:

```text
data/processed/inflation_regime_features.parquet
```

It separates observation, publication, and information dates. Reviewed release
dates are used when available; older observations use a clearly labeled
conservative lag. Fast annualized rates are exposed to the model only when the
underlying series is seasonally adjusted.

The ten-geography contribution decomposition and annual-weight update process is
documented in [Focused Inflation Decomposition](docs/decomposition.md).

## Generate Charts and Report

To rebuild the combined panel and metrics from existing raw files, then
generate charts and the Markdown report:

```bash
python3 run_pipeline.py
```

To fetch raw source data first, then rebuild the panel, metrics, charts, and
report:

```bash
python3 run_pipeline.py --fetch
```

Denmark, Sweden, and Norway use incremental refreshes by default. To force full
history refreshes for those sources:

```bash
python3 run_pipeline.py --fetch --full-refresh-incremental-sources
```

The pipeline writes:

```text
data/interim/cpi_panel_raw_combined.parquet
data/processed/cpi_panel.parquet
outputs/charts/cpi_yoy_by_country.png
outputs/charts/cpi_inflation_direction.png
outputs/reports/latest_inflation_report.md
```

FRED still requires `FRED_API_KEY` in `.env` or the process environment when
using `--fetch`.

## Run the Dashboard

Release-aware local scheduling (initially Denmark only) is documented in
[Automation](docs/automation.md). It checks locally each day and fetches only
for due or missed releases, with a weekly calendar refresh.

After `data/processed/cpi_panel.parquet` exists:

```bash
PYTHONPATH=src streamlit run src/inflation_engine/dashboard/streamlit_app.py
```

The Streamlit dashboard includes a country selector, headline/core metric
cards, history charts for MoM, YoY, 3-month annualized, true 3m/3m annualized,
and other available momentum metrics, plus latest country rankings and an
inflation heatmap.

## Data Coverage

- US CPI: headline, core, and the major components shown in BLS CPI Table A.
- UK CPI: headline CPI and CPI excluding energy, food, alcohol, and tobacco.
- European HICP: headline and core for Germany, France, Italy, Spain, and the
  euro area.
- Danish CPI: headline and Statistics Denmark's official core CPI.
- Swedish inflation: headline CPIF and core CPIF excluding energy.
- Norwegian CPI: headline CPI and CPI-ATE.
- US PCE (BEA, no API key): headline and core PCE price indexes plus goods,
  services, food, energy, and market-based aggregates under the separate
  `US_PCE` geography code, and every line of BEA's underlying-detail PCE
  tables for the item-level decomposition and item breadth, plus a derived
  "supercore" (services excluding energy and housing). US CPI is untouched
  by it.

All observations use the standard schema:

```text
date
country
country_code
region
series_type
series_role
component
value
source
source_dataset
source_series_id
frequency
unit
index_base
seasonal_adjustment
observation_status
```

`series_role` distinguishes the canonical aggregate indexes used in reports
from lower-level CPI components.

See [Data Sources](docs/data_sources.md) for series definitions, API details,
and source-specific methodology.

## Tests

The self-contained test suite runs on a fresh checkout without keys or downloads:

```bash
python -m pytest -m 'not integration' -q
```

After fetching data and building the required weight/decomposition outputs, run
`python -m pytest -q` to include the data-dependent integration checks.

## Preparing a standalone public repository

If this folder is inside a larger private Git repository, do not publish the
parent repository. Create a clean, audited source snapshot instead:

```sh
python scripts/audit_secrets.py --history
python scripts/prepare_public_repo.py
```

The snapshot defaults to `tmp/public-github/`. It includes code, tests,
documentation and public reference tables, while excluding `.env`, fetched
data, generated reports, runtime logs, local configuration and old Git history.
It is not pushed or published automatically. Source data attribution and
calculation limitations are documented in [Data Sources](docs/data_sources.md)
and [Decomposition](docs/decomposition.md); third-party data retains its
respective provider terms.
