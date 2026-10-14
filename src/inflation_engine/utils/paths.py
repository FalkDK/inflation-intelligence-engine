from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DATA_DIR = DATA_DIR / "reference"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
REPORTS_DIR = OUTPUTS_DIR / "reports"
INGESTION_MANIFEST_PATH = RAW_DATA_DIR / "ingestion_manifest.json"
FRED_US_CPI_PATH = RAW_DATA_DIR / "fred_us_cpi.parquet"
BEA_US_PCE_PATH = RAW_DATA_DIR / "bea_us_pce.parquet"
BEA_US_PCE_CPI_PATH = RAW_DATA_DIR / "bea_us_pce_cpi.parquet"
BLS_US_CPI_WEIGHTS_PATH = RAW_DATA_DIR / "bls_us_cpi_weights.parquet"
SSB_NORWAY_CPI_WEIGHTS_PATH = RAW_DATA_DIR / "ssb_norway_cpi_weights.parquet"
SSB_NORWAY_COMPONENT_CONTRACT_PATH = (
    RAW_DATA_DIR / "ssb_norway_component_contract.parquet"
)
SSB_NORWAY_FULL_WEIGHTS_PATH = RAW_DATA_DIR / "ssb_norway_full_weights.parquet"
BLS_US_CPI_WEIGHT_SNAPSHOT_PATH = (
    REFERENCE_DATA_DIR / "us_cpi_weights_2026.csv"
)
INFLATION_RELEASE_CALENDAR_PATH = (
    REFERENCE_DATA_DIR / "inflation_release_calendar.csv"
)
COMPONENT_TAXONOMY_PATH = REFERENCE_DATA_DIR / "component_taxonomy.csv"
DANISH_CPI_WEIGHTS_2026_PATH = (
    REFERENCE_DATA_DIR / "danish_cpi_weights_2026.xlsx"
)
DANISH_CPI_WEIGHTS_BACKCAST_PATH = (
    REFERENCE_DATA_DIR / "danish_cpi_weights_backcast_2001_2025.xlsx"
)
ONS_ANNEX_A_WEIGHTS_DIR = REFERENCE_DATA_DIR / "ons_cpi_weights"
BLS_CPI_RELATIVE_IMPORTANCE_PATH = (
    REFERENCE_DATA_DIR / "bls_cpi_relative_importance_2024weights.xlsx"
)
US_CPI_ITEM_SERIES_MAP_PATH = REFERENCE_DATA_DIR / "us_cpi_item_series_map.csv"
ONS_UK_CPI_PATH = RAW_DATA_DIR / "ons_uk_cpi.parquet"
ONS_UK_CPI_WEIGHTS_PATH = RAW_DATA_DIR / "ons_uk_cpi_weights.parquet"
EUROSTAT_HICP_PATH = RAW_DATA_DIR / "eurostat_hicp.parquet"
EUROSTAT_HICP_WEIGHTS_PATH = RAW_DATA_DIR / "eurostat_hicp_weights.parquet"
EUROSTAT_HICP_ITEMS_PATH = RAW_DATA_DIR / "eurostat_hicp_items.parquet"
EUROSTAT_HICP_ITEM_WEIGHTS_PATH = (
    RAW_DATA_DIR / "eurostat_hicp_item_weights.parquet"
)
STATBANK_DK_CPI_PATH = RAW_DATA_DIR / "statbank_dk_cpi.parquet"
SCB_SWEDEN_CPI_PATH = RAW_DATA_DIR / "scb_sweden_cpi.parquet"
SCB_SWEDEN_CPI_WEIGHTS_PATH = RAW_DATA_DIR / "scb_sweden_cpi_weights.parquet"
SSB_NORWAY_CPI_PATH = RAW_DATA_DIR / "ssb_norway_cpi.parquet"
CPI_PANEL_RAW_COMBINED_PATH = (
    INTERIM_DATA_DIR / "cpi_panel_raw_combined.parquet"
)
CPI_PANEL_PATH = PROCESSED_DATA_DIR / "cpi_panel.parquet"
INFLATION_REGIME_FEATURES_PATH = (
    PROCESSED_DATA_DIR / "inflation_regime_features.parquet"
)
INFLATION_BUCKET_DECOMPOSITION_PATH = (
    PROCESSED_DATA_DIR / "inflation_bucket_decomposition.parquet"
)
INFLATION_DECOMPOSITION_SUMMARY_PATH = (
    PROCESSED_DATA_DIR / "inflation_decomposition_summary.parquet"
)
DANISH_ITEM_DECOMPOSITION_PATH = (
    PROCESSED_DATA_DIR / "danish_item_decomposition.parquet"
)
DANISH_ITEM_AGGREGATION_VALIDATION_PATH = (
    PROCESSED_DATA_DIR / "danish_item_aggregation_validation.parquet"
)
DANISH_ITEM_CONTRIBUTIONS_PATH = (
    PROCESSED_DATA_DIR / "danish_item_contributions.parquet"
)
NORDIC_ITEM_CONTRIBUTIONS_PATH = (
    PROCESSED_DATA_DIR / "nordic_item_contributions.parquet"
)
UK_ITEM_DECOMPOSITION_PATH = (
    PROCESSED_DATA_DIR / "uk_item_decomposition.parquet"
)
US_ITEM_DECOMPOSITION_PATH = (
    PROCESSED_DATA_DIR / "us_item_decomposition.parquet"
)
US_PCE_ITEM_DECOMPOSITION_PATH = (
    PROCESSED_DATA_DIR / "us_pce_item_decomposition.parquet"
)
NORWAY_ITEM_DECOMPOSITION_PATH = (
    PROCESSED_DATA_DIR / "norway_item_decomposition.parquet"
)
EUROZONE_ITEM_DECOMPOSITION_PATH = (
    PROCESSED_DATA_DIR / "eurozone_item_decomposition.parquet"
)
BREADTH_BUCKETS_PATH = PROCESSED_DATA_DIR / "inflation_breadth_buckets.parquet"
BREADTH_ITEMS_PATH = PROCESSED_DATA_DIR / "inflation_breadth_items.parquet"
CPI_YOY_CHART_PATH = CHARTS_DIR / "cpi_yoy_by_country.png"
CPI_DIRECTION_CHART_PATH = CHARTS_DIR / "cpi_inflation_direction.png"
CPI_3M3M_CHART_PATH = CPI_DIRECTION_CHART_PATH
LATEST_INFLATION_REPORT_PATH = REPORTS_DIR / "latest_inflation_report.md"
