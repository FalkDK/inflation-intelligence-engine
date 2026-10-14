import argparse
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from inflation_engine.utils.config import series_role
from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.paths import (
    BEA_US_PCE_CPI_PATH,
    CPI_PANEL_RAW_COMBINED_PATH,
    EUROSTAT_HICP_ITEMS_PATH,
    EUROSTAT_HICP_PATH,
    FRED_US_CPI_PATH,
    ONS_UK_CPI_PATH,
    SCB_SWEDEN_CPI_PATH,
    SSB_NORWAY_CPI_PATH,
    STATBANK_DK_CPI_PATH,
)
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS, validate_cpi_schema


RAW_CPI_SOURCES = {
    "fred_us_cpi": FRED_US_CPI_PATH,
    "ons_uk_cpi": ONS_UK_CPI_PATH,
    "eurostat_hicp": EUROSTAT_HICP_PATH,
    "eurostat_hicp_items": EUROSTAT_HICP_ITEMS_PATH,
    "statbank_dk_cpi": STATBANK_DK_CPI_PATH,
    "scb_sweden_cpi": SCB_SWEDEN_CPI_PATH,
    "ssb_norway_cpi": SSB_NORWAY_CPI_PATH,
    "bea_us_pce": BEA_US_PCE_CPI_PATH,
}


def load_cpi_source(path: str | Path) -> pd.DataFrame:
    """Load one normalized raw CPI file and validate its schema."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Raw CPI source does not exist: {source_path}")

    frame = pd.read_parquet(source_path)
    if "series_role" not in frame:
        frame["series_role"] = [
            series_role(country_code, series_type, component)
            for country_code, series_type, component in frame[
                ["country_code", "series_type", "component"]
            ].itertuples(index=False, name=None)
        ]
    validate_cpi_schema(frame)
    return frame.loc[:, CPI_SCHEMA_COLUMNS].copy()


def combine_cpi_sources(
    source_paths: Mapping[str, str | Path] = RAW_CPI_SOURCES,
) -> pd.DataFrame:
    """Load and combine normalized raw CPI sources into one clean panel."""
    if not source_paths:
        raise ValueError("At least one raw CPI source is required")

    frames = []
    for dataset_id, path in source_paths.items():
        try:
            frames.append(load_cpi_source(path))
        except (FileNotFoundError, ValueError) as error:
            raise type(error)(f"{dataset_id}: {error}") from error

    combined = (
        pd.concat(frames, ignore_index=True)
        .loc[:, CPI_SCHEMA_COLUMNS]
        .sort_values(
            ["country_code", "series_type", "component", "date"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    validate_cpi_schema(combined)
    return combined


def save_combined_panel(
    frame: pd.DataFrame,
    output_path: str | Path = CPI_PANEL_RAW_COMBINED_PATH,
) -> Path:
    """Atomically save the validated combined CPI panel as Parquet."""
    validate_cpi_schema(frame)
    return atomic_write_parquet(
        frame,
        output_path,
        columns=CPI_SCHEMA_COLUMNS,
    )


def run(
    output_path: str | Path = CPI_PANEL_RAW_COMBINED_PATH,
) -> Path:
    frame = combine_cpi_sources()
    return save_combined_panel(frame, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine normalized raw CPI sources into one interim panel."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CPI_PANEL_RAW_COMBINED_PATH,
        help=f"Parquet output path (default: {CPI_PANEL_RAW_COMBINED_PATH})",
    )
    args = parser.parse_args()

    frame = combine_cpi_sources()
    destination = save_combined_panel(frame, args.output)
    print(f"Saved combined CPI panel to {destination}")
    print(
        f"Rows: {len(frame):,} | Countries: {frame['country_code'].nunique()} "
        f"| Series: {frame['source_series_id'].nunique()} "
        f"| Latest date: {frame['date'].max().date()}"
    )


if __name__ == "__main__":
    main()
