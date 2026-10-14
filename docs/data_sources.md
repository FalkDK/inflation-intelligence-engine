# Data Sources

## Standard Output Schema

All ingestion modules normalize source observations to:

| Column | Description |
| --- | --- |
| `date` | First day of the monthly observation period |
| `country` | Display name |
| `country_code` | Internal geography code |
| `region` | Broad geographic region |
| `series_type` | `headline` or `core` |
| `series_role` | `aggregate` for canonical report series or `component` |
| `component` | Normalized CPI/HICP component name |
| `value` | Published index level |
| `source` | Publishing API or institution |
| `source_dataset` | Reproducible source table or dataflow |
| `source_series_id` | Exact source-native series identifier |
| `frequency` | `monthly` |
| `unit` | `index` |
| `index_base` | Source index reference base |
| `seasonal_adjustment` | Standardized adjustment flag |
| `observation_status` | Source status such as `normal` or `estimated` |

The raw files contain index levels. Inflation rates and momentum measures are
calculated later by the feature layer.

Canonical aggregate definitions are maintained once in
`inflation_engine.utils.config.PRIMARY_SERIES` and reused by ingestion,
validation, charts, and reports.

The schema enforces month-start dates, positive finite values, controlled
categorical values, country metadata consistency, and uniqueness on:

```text
date + country_code + series_type + component
```

APIs that do not publish row-level observation status use `not_provided`.
Raw index levels must not be compared across series with different
`index_base` values; calculated rates of change are comparable.

## Ingestion Manifest

Every successful fetch updates:

```text
data/raw/ingestion_manifest.json
```

The file has one entry per raw dataset. Each entry includes:

- `retrieved_at`: UTC time when this pipeline completed the fetch
- `source_updated_at`: provider update time, or `null` if unavailable
- `output_file`: generated Parquet path
- `sha256`: checksum of the generated file
- Row and column counts
- First and latest observation dates
- Country and source coverage
- Source dataset and source-series counts
- Index bases and seasonal-adjustment values
- Observation-status counts

The manifest is updated only after the Parquet file is written successfully.
Writes are atomic and protected by a file lock, allowing independent fetch
commands to run concurrently without dropping other dataset entries. The
manifest and its lock file are generated artifacts and are ignored by Git.

## Combined Raw Panel

Run:

```bash
PYTHONPATH=src python3 -m inflation_engine.data.combine_sources
```

The combine step loads all six normalized raw Parquet files, validates each
one independently, preserves the exact standard column order, checks the
combined unique observation key, sorts deterministically, and writes:

```text
data/interim/cpi_panel_raw_combined.parquet
```

The interim write is atomic, so an interrupted write does not replace a
previous valid panel with a partial file.

## US CPI From FRED

Module:

```text
src/inflation_engine/data/fetch_fred.py
```

Command:

```bash
PYTHONPATH=src python3 -m inflation_engine.data.fetch_fred
```

Output:

```text
data/raw/fred_us_cpi.parquet
```

API:

```text
https://api.stlouisfed.org/fred/series/observations
```

Authentication is provided through `FRED_API_KEY` in `.env` or the process
environment. The credential is never written to the output.

FRED rows preserve the exact FRED series ID, use index base
`1982-1984=100`, and are marked `seasonally_adjusted`. FRED does not expose a
comparable row-level observation status in this request, so status is
`not_provided`.

The fetcher downloads seasonally adjusted CPI-U index series. The roots are:

| Type | FRED series | Component |
| --- | --- | --- |
| Headline | `CPIAUCSL` | All items |
| Core | `CPILFESL` | All items less food and energy |

It also includes the major BLS Table A branches:

- Food, food at home, and food away from home
- Energy, energy commodities, gasoline, fuel oil and other fuels
- Energy services, electricity, and utility gas
- Core commodities and services
- New vehicles, used vehicles, apparel, and medical commodities
- Shelter, transportation services, and medical services

The closest seasonally adjusted FRED series for the Table A fuel-oil row is
published as **Fuel Oil and Other Fuels**. The normalized component preserves
that broader definition.

References:

- [FRED API documentation](https://fred.stlouisfed.org/docs/api/fred/)
- [BLS CPI news release and Table A](https://www.bls.gov/news.release/cpi.nr0.htm)

## US PCE From BEA

Module:

```text
src/inflation_engine/data/fetch_bea_pce.py
```

Command:

```bash
PYTHONPATH=src python3 -m inflation_engine.data.fetch_bea_pce
```

Outputs:

```text
data/raw/bea_us_pce_cpi.parquet   # standard schema, US_PCE aggregates
data/raw/bea_us_pce.parquet       # every table line, long format
```

Sources (keyless NIPA bulk files):

```text
https://apps.bea.gov/national/Release/TXT/SeriesRegister.txt
https://apps.bea.gov/national/Release/TXT/NipaDataM.txt
```

The detail file pairs NIPA Table 2.4.4U (Fisher price indexes, `2017=100`)
with Table 2.4.5U (nominal spending, millions of dollars, seasonally adjusted
annual rates) by table line. Its additive product tree comes from the
register's parent codes on 2.4.5U. Nodes are keyed by table line, because some
series appear on more than one line. For example, nonprofit hospitals appear
under household health care and again under the NPISH "Less: Receipts" branch.
Each line's parent is the closest preceding line among its listed parents.
"Less:" lines carry `aggregation_sign = -1`. Lines after the tree (core,
market-based and other addenda) are kept with `additive_tree = False`. Detail
columns: `date`, `table_line`, `label`, `parent_line`, `hierarchy_level`,
`aggregation_sign`, `additive_tree`, `price_series_id`, `nominal_series_id`,
`price_index`, `nominal_spending_musd`.

The panel file uses the separate geography code `US_PCE` ("United States
(PCE)"), so it can never collide with `US` CPI rows on the panel's unique key.
`ALTERNATIVE_MEASURE_CODES` in `utils/config.py` keeps it out of cross-country
rankings in the report.

| Type | BEA series | Component |
| --- | --- | --- |
| Headline | `DPCERG` | `all_items` (aggregate) |
| Headline | `DGDSRG` / `DDURRG` / `DNDGRG` / `DSERRG` | goods, durable, nondurable, services |
| Headline | `DFXARG` / `DNRGRG` / `DPFERG` | food off-premises, energy, food and energy |
| Headline | `DPCMRG` | market-based PCE |
| Core | `DPCCRG` | `all_items_less_food_and_energy` (aggregate) |
| Core | `IA000062` / `IA000063` | goods ex food and energy, services ex energy |
| Core | `DPCXRG` | market-based PCE ex food and energy |
| Core | `DERIVED:SERVICES_EX_ENERGY_HOUSING` | `services_less_energy_and_housing` ("supercore") |

BEA does not publish supercore. It is rebuilt here as a monthly-chained Fisher
index (`2017=100`) over the services frontier, leaving out the housing and
electricity-and-gas subtrees (`utils/pce_tree.py`). The same construction
reproduces BEA's published indexes closely. Mean absolute YoY error since 2000
is about 0.013pp for services excluding energy, 0.018pp for core and 0.035pp
for core goods, with the largest gaps in 2020-21.

BEA revises PCE history with each release; like the other sources, only the
latest vintage is stored. Publication dates come from the release calendar
(BEA Personal Income and Outlays); months it does not cover use the
conservative two-month information lag. `source_updated_at` is the
data file's HTTP `Last-Modified` time.

## UK CPI From ONS

Module:

```text
src/inflation_engine/data/fetch_ons.py
```

Command:

```bash
PYTHONPATH=src python3 -m inflation_engine.data.fetch_ons
```

Output:

```text
data/raw/ons_uk_cpi.parquet
```

No API key is required. The former `api.ons.gov.uk` time-series API was
retired on November 25, 2024. The ONS developer catalogue currently exposes
the `cpih01` CPIH dataset, but this project uses CPI for comparability with its
other national headline measures. The fetcher therefore downloads the official
machine-readable current MM23 CSV once and extracts:

| Type | ONS series | Component |
| --- | --- | --- |
| Headline | `D7BT` | CPI all items |
| Core | `DKC6` | CPI excluding energy, food, alcohol, and tobacco |
| Component | `DK9O` | Food, alcohol, and tobacco |
| Component | `DK9T` | Energy |
| Component | `DK9J` | Non-energy industrial goods |
| Component | `D7F5` | Services |
| Component | `D7CE` | Actual rents |

All series are monthly indices with base `2015=100` and are marked
`not_seasonally_adjusted`.

Annual CPI special-aggregate weights are extracted from the same MM23 database
by `fetch_ons_weights.py`.

References:

- [ONS developer portal](https://developer.ons.gov.uk/)
- [ONS headline CPI series
  D7BT](https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/d7bt/mm23)
- [ONS core CPI series
  DKC6](https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/dkc6/mm23)

## European HICP From The ECB Data Portal

Module:

```text
src/inflation_engine/data/fetch_eurostat.py
```

Command:

```bash
PYTHONPATH=src python3 -m inflation_engine.data.fetch_eurostat
```

Output:

```text
data/raw/eurostat_hicp.parquet
```

No API key is required.

Although the module and output retain `eurostat` in their names, the current
data is fetched from the ECB Data Portal. Eurostat remains the underlying
statistical source. This naming is retained to avoid breaking the planned
pipeline paths.

The download covers headline, core, food including alcohol and tobacco,
energy, non-energy industrial goods, services, and actual rents. Annual weights
are fetched separately from Eurostat dataset `prc_hicp_iw`.

API flow:

```text
https://data-api.ecb.europa.eu/service/data/HICP
```

The ECB discontinued its former `ICP` dataset on February 4, 2026 and replaced
it with the `HICP` dataset. The replacement uses the `ECB_ICP3` structure,
ECOICOP v2 classification, and a `2025=100` index base. The fetcher therefore
uses the new provider code `4D0` and index suffix `INX`.

Requested geographies:

| ECB code | Output code | Geography |
| --- | --- | --- |
| `DE` | `DE` | Germany |
| `DK` | `DK` | Denmark |
| `FR` | `FR` | France |
| `IT` | `IT` | Italy |
| `ES` | `ES` | Spain |
| `SE` | `SE` | Sweden |
| `U2` | `EA20` | Euro area |

Requested series:

| Type | ECB item | Component |
| --- | --- | --- |
| Headline | `000000` | All items |
| Core | `XEF000` | All items excluding energy, food, alcohol, and tobacco |
| Component | `FOOD00` | Food, alcohol, and tobacco |
| Component | `NRGY00` | Energy |
| Component | `IGXE00` | Non-energy industrial goods |
| Component | `SERV00` | Services |
| Component | `041000` | Actual rents |

The query uses monthly, neither seasonally nor working-day-adjusted index
levels. Recent observations may carry an estimated status and can subsequently
be revised. ECB status codes are normalized, including `normal`, `estimated`,
`provisional`, and `revised`. The exact HICP SDMX series key is retained in
`source_series_id`. The command prints the latest available date and value for
every geography and series.

References:

- [ECB HICP dataset information](https://data.ecb.europa.eu/data/datasets/HICP/data-information)
- [ECB HICP data structure](https://data.ecb.europa.eu/data/datasets/HICP/structure)
- [ECB API overview](https://data.ecb.europa.eu/help/api/overview)
- [Eurostat API getting started](https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/api-getting-started)

## Danish CPI From Statistics Denmark

Module:

```text
src/inflation_engine/data/fetch_statbank_dk.py
```

Command:

```bash
PYTHONPATH=src python3 -m inflation_engine.data.fetch_statbank_dk
```

Output:

```text
data/raw/statbank_dk_cpi.parquet
```

No API key is required. The fetcher uses table `PRIS01` from StatBank:

```text
https://api.statbank.dk/v1/data
```

Requested series:

| Type | StatBank code | Component |
| --- | --- | --- |
| Headline | `000000` | Consumer price index, total |
| Core | `151` | CPI excluding energy and unprocessed food |

The query requests all `VAREGR` commodity groups, unit code `100` (index), and
all available monthly periods. The current series use a 2025 annual-average
index base of 100. StatBank's published core aggregate is used directly; the
pipeline does not reconstruct core CPI from lower-level components. Rows retain
table `PRIS01`, the commodity group code, and the `not_seasonally_adjusted`
flag. `000000` and `151` are marked as aggregate headline/core rows; all other
commodity groups are stored as headline components.

The response is requested as JSON-stat and parsed using dimension codes rather
than translated labels. The command prints the latest available date and value
for headline and core CPI.

References:

- [Statistics Denmark API guide](https://www.dst.dk/da/Statistik/hjaelp-til-statistikbanken/api)
- [StatBank API](https://api.statbank.dk/)
- [PRIS01 table](https://www.statbank.dk/PRIS01)

## Swedish CPI From Statistics Sweden

Module:

```text
src/inflation_engine/data/fetch_sweden.py
```

Command:

```bash
PYTHONPATH=src python3 -m inflation_engine.data.fetch_sweden
```

Output:

```text
data/raw/scb_sweden_cpi.parquet
```

The fetcher uses Statistics Sweden's PXWeb API v2 beta:

```text
https://api.scb.se/OV0104/v2beta/api/v2/tables
```

Requested series:

| Type | Table | Content code | Component |
| --- | --- | --- | --- |
| Headline | `TAB6590` | `000007ZN` | CPIF |
| Core | `TAB6593` | `000007ZW` | CPIF excluding energy |
| Components | `TAB6613` | `0000083S` | CPIF COICOP component levels |

Both series are monthly indices with `2020=100` and continuous histories from
1987. CPIF holds mortgage interest rates fixed and is Sweden's policy-relevant
headline inflation measure. CPIF-XE additionally excludes energy and is stored
as the core measure. These definitions are recorded explicitly because they
are not identical to the core measures used for other countries. Table and
content codes are retained as source identifiers, and both series are marked
`not_seasonally_adjusted`.

The legacy SCB v1 table path `PR0101G/KPIF2020` corresponds to the current v2
table `TAB6590`.

Component rows are sourced from the detailed CPIF COICOP table `TAB6613`. The
normal ingest stores all non-total COICOP component levels as headline
components from `2006M01` onward. The fetcher uses SCB's GET selection API in
bounded component/month batches because one full query URL exceeds the
server-side request-length limit.

References:

- [Statistics Sweden API](https://www.scb.se/en/services/open-data-api/)
- [Statistics Sweden statistical database](https://www.statistikdatabasen.scb.se/)

## Norwegian CPI From Statistics Norway

Module:

```text
src/inflation_engine/data/fetch_ssb.py
```

Command:

```bash
PYTHONPATH=src python3 -m inflation_engine.data.fetch_ssb
```

Output:

```text
data/raw/ssb_norway_cpi.parquet
```

The fetcher uses Statbank Norway's PXWeb API v2:

```text
https://data.ssb.no/api/pxwebapi/v2/tables
```

Requested series:

| Type | Table | Selection | Component |
| --- | --- | --- | --- |
| Headline | `14710` | `KpiIndMnd` | CPI, all items |
| Core | `14706` | `KPI-JAE`, `KPIJustIndMnd` | CPI-ATE |
| Components | `14700` | `VareTjenesteGrp`, `KpiIndMnd` | CPI goods and services |

Both series use `2025=100`. CPI-ATE is adjusted for tax changes and excludes
energy products. The source's published aggregate is used directly. Table and
selection codes are retained as source identifiers, and both series are marked
`not_seasonally_adjusted`. Component rows are sourced from table `14700`; the
table total is excluded to avoid duplicating the headline aggregate, and all
other goods-and-services groups are stored as headline components.

References:

- [Statbank Norway API](https://www.ssb.no/en/api/pxwebapi)
- [PXWeb API v2 user guide](https://www.ssb.no/en/api/pxwebapiv2)
- [Statbank Norway](https://www.ssb.no/en/statbank/)

## CPI Release Calendar From TradingView

Module:

```bash
PYTHONPATH=src python -m inflation_engine.data.fetch_release_calendar
```

Writes `data/reference/inflation_release_calendar.csv` (one release per
observation month and geography) and prints the releases due in the next
45 days, including their times. It also runs as part of `run_pipeline.py --fetch`.
`publication_date` retains the UTC calendar date for existing daily analysis;
`publication_datetime_utc` retains the feed timestamp as ISO 8601 UTC, for
example `2026-06-10T12:30:00Z`. Legacy date-only calendars still load, without
inventing a release time. Display Danish local times with
`--timezone Europe/Copenhagen`; timezone conversion handles daylight saving.
Times are the feed's scheduled timestamps, not confirmation that a source API
has published its data. A scheduler should check the expected observation too.

Source is TradingView's public economic-calendar feed
(`economic-calendar.tradingview.com/events`, no key). It is the only single
source that covers all geographies with both history (from 2016) and the
forward schedule; BLS blocks scripted access to its own calendar, and the other
agencies each publish theirs in a different format. RSS feeds were not used:
they announce a release once it is out but carry no schedule. Every row keeps
the publishing agency's URL in `source_url`.

The feed silently truncates large responses, so requests go one month at a
time. It schedules roughly one month ahead, so rerun it monthly.

Mapping rules:

- Each geography uses the release that publishes the value the panel stores.
  Euro-area members and EA20 use Eurostat's full HICP release, not the flash
  or national preliminary CPI. Sweden uses the final CPI release, and US_PCE
  uses the BEA Personal Income and Outlays release.
- When one month has two matching events (for example a reissued "Final"),
  the later timestamp is kept.
- Releases at or before the current UTC instant are `actual` and future ones
  are `scheduled`, including releases later today. A `scheduled`
  row that matches a panel observation is a value already published as a flash
  estimate (for example Eurostat member-state flash HICP). It keeps the later
  full-release date as its information date.
- Months with no event keep the conservative two-month lag. The one gap is US
  CPI for October 2025, which BLS never published because of the government
  shutdown.
