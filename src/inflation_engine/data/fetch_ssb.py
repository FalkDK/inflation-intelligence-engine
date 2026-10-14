import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from inflation_engine.utils.config import COUNTRIES, series_role
from inflation_engine.utils.components import (
    category_codes,
    category_labels,
    unique_component_slugs,
)
from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.jsonstat import parse_jsonstat_observations
from inflation_engine.utils.manifest import update_ingestion_manifest
from inflation_engine.utils.paths import SSB_NORWAY_CPI_PATH
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS, validate_cpi_schema


SSB_API_BASE_URL = "https://data.ssb.no/api/pxwebapi/v2/tables"
SSB_NORWAY_DATASET_ID = "ssb_norway_cpi"
SSB_CPI_COMPONENT_TABLE = {
    "table": "14700",
    "content_code": "KpiIndMnd",
    "component_dimension": "VareTjenesteGrp",
    "total_code": "00",
}

NORWAY_CPI_TABLES = {
    "headline": {
        "table": "14710",
        "source_series_id": "KpiIndMnd",
        "selection": [
            {"variableCode": "ContentsCode", "valueCodes": ["KpiIndMnd"]},
            {"variableCode": "Tid", "valueCodes": ["*"]},
        ],
        "component": "all_items",
    },
    "core": {
        "table": "14706",
        "source_series_id": "KPI-JAE:KPIJustIndMnd",
        "selection": [
            {"variableCode": "KPIavledetSerie", "valueCodes": ["KPI-JAE"]},
            {"variableCode": "ContentsCode", "valueCodes": ["KPIJustIndMnd"]},
            {"variableCode": "Tid", "valueCodes": ["*"]},
        ],
        "component": "cpi_adjusted_taxes_excluding_energy",
    },
}


def fetch_ssb_dataset(
    table: str,
    selection: list[dict[str, Any]],
    *,
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> Mapping[str, Any]:
    """Fetch one Norwegian CPI index series from Statistics Norway."""
    client = session or requests.Session()
    url = f"{SSB_API_BASE_URL}/{table}/data"

    try:
        response = client.post(
            url,
            params={"lang": "en", "outputFormat": "json-stat2"},
            json={"selection": selection},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(f"Statistics Norway request failed for table {table}") from error

    dataset = response.json()
    if dataset.get("class") != "dataset":
        raise ValueError(f"Invalid Statistics Norway response for table {table}")
    return dataset


def fetch_ssb_metadata(
    table: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> Mapping[str, Any]:
    """Fetch Statistics Norway table metadata used for incremental periods."""
    client = session or requests.Session()
    url = f"{SSB_API_BASE_URL}/{table}/metadata"

    try:
        response = client.get(
            url,
            params={"lang": "en"},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            f"Statistics Norway metadata request failed for table {table}"
        ) from error

    dataset = response.json()
    if dataset.get("class") != "dataset":
        raise ValueError(f"Invalid Statistics Norway metadata for table {table}")
    return dataset


def load_existing_norwegian_cpi(
    input_path: str | Path = SSB_NORWAY_CPI_PATH,
) -> pd.DataFrame | None:
    source = Path(input_path)
    if not source.is_file():
        return None
    try:
        frame = pd.read_parquet(source)
        validate_cpi_schema(frame)
    except (OSError, ValueError, KeyError):
        return None
    return frame.loc[:, CPI_SCHEMA_COLUMNS].copy()


def _period_code(date: pd.Timestamp) -> str:
    return pd.Timestamp(date).strftime("%YM%m")


def _existing_source_frame(
    existing_frame: pd.DataFrame | None,
    *,
    source_dataset: str,
) -> pd.DataFrame:
    if existing_frame is None:
        return pd.DataFrame(columns=CPI_SCHEMA_COLUMNS)
    frame = existing_frame.loc[
        existing_frame["source_dataset"].eq(source_dataset)
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=CPI_SCHEMA_COLUMNS)
    validate_cpi_schema(frame)
    return frame.loc[:, CPI_SCHEMA_COLUMNS]


def _incremental_periods(
    existing_source_frame: pd.DataFrame,
    source_periods: list[str],
) -> list[str] | None:
    if existing_source_frame.empty:
        return None
    latest_by_series = existing_source_frame.groupby("source_series_id")["date"].max()
    if latest_by_series.empty:
        return None
    latest_existing_period = _period_code(latest_by_series.min())
    return [period for period in source_periods if period > latest_existing_period]


def _selection_with_periods(
    selection: list[dict[str, Any]],
    periods: list[str] | None,
) -> list[dict[str, Any]]:
    if periods is None:
        return selection
    return [
        {
            **item,
            "valueCodes": periods
            if item.get("variableCode") == "Tid"
            else item.get("valueCodes", []),
        }
        for item in selection
    ]


def _merge_existing_and_new(
    existing_frame: pd.DataFrame | None,
    frames: list[pd.DataFrame],
) -> pd.DataFrame:
    merge_frames = []
    if existing_frame is not None and not existing_frame.empty:
        merge_frames.append(existing_frame)
    merge_frames.extend(frames)
    if not merge_frames:
        raise ValueError("No Norwegian CPI data available to merge")
    combined = (
        pd.concat(merge_frames, ignore_index=True)
        .drop_duplicates(
            ["date", "country_code", "series_type", "component"],
            keep="last",
        )
        .sort_values(["series_type", "component", "date"])
        .reset_index(drop=True)
    )
    validate_cpi_schema(combined)
    source_updates = [
        frame.attrs["source_updated_at"]
        for frame in frames
        if frame.attrs.get("source_updated_at")
    ]
    if source_updates:
        combined.attrs["source_updated_at"] = max(source_updates)
    return combined


def normalize_ssb_series(
    dataset: Mapping[str, Any],
    *,
    series_type: str,
    component: str,
    source_dataset: str,
    source_series_id: str,
) -> pd.DataFrame:
    """Normalize one Statistics Norway index series."""
    observations = parse_jsonstat_observations(dataset)
    if not {"Tid", "value"}.issubset(observations.columns):
        raise ValueError("Statistics Norway observations must contain Tid and value")

    frame = observations.loc[:, ["Tid", "value"]].copy()
    frame["date"] = pd.to_datetime(
        frame["Tid"].str.replace("M", "-", regex=False) + "-01",
        errors="coerce",
    )
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"])

    country = COUNTRIES["NO"]
    frame["country"] = country["name"]
    frame["country_code"] = "NO"
    frame["region"] = country["region"]
    frame["series_type"] = series_type
    frame["series_role"] = series_role("NO", series_type, component)
    frame["component"] = component
    frame["source"] = "Statistics Norway"
    frame["source_dataset"] = source_dataset
    frame["source_series_id"] = source_series_id
    frame["frequency"] = "monthly"
    frame["unit"] = "index"
    frame["index_base"] = "2025=100"
    frame["seasonal_adjustment"] = "not_seasonally_adjusted"
    frame["observation_status"] = "not_provided"

    frame = frame.loc[:, CPI_SCHEMA_COLUMNS].sort_values("date").reset_index(drop=True)
    validate_cpi_schema(frame)
    return frame


def normalize_ssb_component_series(
    dataset: Mapping[str, Any],
    *,
    component_dimension: str,
    source_dataset: str,
    source_series_id: str,
    exclude_component_codes: set[str] | None = None,
) -> pd.DataFrame:
    """Normalize one Statistics Norway component index table."""
    observations = parse_jsonstat_observations(dataset)
    required_columns = {component_dimension, "ContentsCode", "Tid", "value"}
    if not required_columns.issubset(observations.columns):
        raise ValueError(
            "Statistics Norway component observations must contain "
            f"{sorted(required_columns)}"
        )

    frame = observations.loc[
        observations["ContentsCode"].eq(source_series_id)
    ].copy()
    if exclude_component_codes:
        frame = frame.loc[
            ~frame[component_dimension].isin(exclude_component_codes)
        ].copy()

    frame["date"] = pd.to_datetime(
        frame["Tid"].str.replace("M", "-", regex=False) + "-01",
        errors="coerce",
    )
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"])

    labels = category_labels(dataset, component_dimension)
    component_slugs = unique_component_slugs(
        sorted(set(frame[component_dimension])),
        labels,
    )
    frame["component"] = frame[component_dimension].map(component_slugs)

    country = COUNTRIES["NO"]
    frame["country"] = country["name"]
    frame["country_code"] = "NO"
    frame["region"] = country["region"]
    frame["series_type"] = "headline"
    frame["series_role"] = [
        series_role("NO", "headline", component) for component in frame["component"]
    ]
    frame["source"] = "Statistics Norway"
    frame["source_dataset"] = source_dataset
    frame["source_series_id"] = (
        source_dataset + ":" + frame[component_dimension].astype(str)
    )
    frame["frequency"] = "monthly"
    frame["unit"] = "index"
    frame["index_base"] = "2025=100"
    frame["seasonal_adjustment"] = "not_seasonally_adjusted"
    frame["observation_status"] = "not_provided"

    frame = (
        frame.loc[:, CPI_SCHEMA_COLUMNS]
        .sort_values(["component", "date"])
        .reset_index(drop=True)
    )
    validate_cpi_schema(frame)
    return frame


def fetch_norwegian_cpi(
    *,
    session: requests.Session | None = None,
    existing_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fetch Norwegian headline CPI and CPI-ATE."""
    frames: list[pd.DataFrame] = []
    source_updates = []
    for series_type, metadata in NORWAY_CPI_TABLES.items():
        table_metadata = fetch_ssb_metadata(metadata["table"], session=session)
        if table_metadata.get("updated"):
            source_updates.append(table_metadata["updated"])
        periods = _incremental_periods(
            _existing_source_frame(
                existing_frame,
                source_dataset=metadata["table"],
            ),
            category_codes(table_metadata, "Tid"),
        )
        if periods == []:
            continue

        dataset = fetch_ssb_dataset(
            metadata["table"],
            _selection_with_periods(metadata["selection"], periods),
            session=session,
        )
        frame = normalize_ssb_series(
            dataset,
            series_type=series_type,
            component=metadata["component"],
            source_dataset=metadata["table"],
            source_series_id=metadata["source_series_id"],
        )
        frame.attrs["source_updated_at"] = table_metadata.get("updated")
        frames.append(frame)

    component_metadata = fetch_ssb_metadata(
        SSB_CPI_COMPONENT_TABLE["table"],
        session=session,
    )
    if component_metadata.get("updated"):
        source_updates.append(component_metadata["updated"])
    component_periods = _incremental_periods(
        _existing_source_frame(
            existing_frame,
            source_dataset=SSB_CPI_COMPONENT_TABLE["table"],
        ),
        category_codes(component_metadata, "Tid"),
    )
    if component_periods != []:
        component_dataset = fetch_ssb_dataset(
            SSB_CPI_COMPONENT_TABLE["table"],
            [
                {
                    "variableCode": SSB_CPI_COMPONENT_TABLE["component_dimension"],
                    "valueCodes": ["*"],
                },
                {
                    "variableCode": "ContentsCode",
                    "valueCodes": [SSB_CPI_COMPONENT_TABLE["content_code"]],
                },
                {"variableCode": "Tid", "valueCodes": component_periods or ["*"]},
            ],
            session=session,
        )
        frame = normalize_ssb_component_series(
            component_dataset,
            component_dimension=SSB_CPI_COMPONENT_TABLE["component_dimension"],
            source_dataset=SSB_CPI_COMPONENT_TABLE["table"],
            source_series_id=SSB_CPI_COMPONENT_TABLE["content_code"],
            exclude_component_codes={SSB_CPI_COMPONENT_TABLE["total_code"]},
        )
        frame.attrs["source_updated_at"] = component_metadata.get("updated")
        frames.append(frame)

    if existing_frame is not None and not frames:
        existing_frame.attrs["source_updated_at"] = (
            max(source_updates) if source_updates else None
        )
        return existing_frame

    combined = _merge_existing_and_new(existing_frame, frames)
    combined.attrs["source_updated_at"] = (
        max(source_updates) if source_updates else None
    )
    return combined


def save_norwegian_cpi(
    frame: pd.DataFrame,
    output_path: str | Path = SSB_NORWAY_CPI_PATH,
) -> Path:
    validate_cpi_schema(frame)
    return atomic_write_parquet(frame, output_path, columns=CPI_SCHEMA_COLUMNS)


def persist_norwegian_cpi(
    frame: pd.DataFrame,
    output_path: str | Path = SSB_NORWAY_CPI_PATH,
) -> Path:
    destination = save_norwegian_cpi(frame, output_path)
    update_ingestion_manifest(
        SSB_NORWAY_DATASET_ID,
        frame,
        destination,
        source_updated_at=frame.attrs.get("source_updated_at"),
    )
    return destination


def latest_norwegian_cpi_observations(frame: pd.DataFrame) -> pd.DataFrame:
    validate_cpi_schema(frame)
    aggregate_frame = frame.loc[frame["series_role"].eq("aggregate")]
    latest = (
        aggregate_frame.sort_values("date")
        .groupby("series_type", as_index=False)
        .tail(1)
    )
    return (
        latest.loc[:, ["series_type", "component", "date", "value"]]
        .sort_values("series_type")
        .reset_index(drop=True)
    )


def run(
    output_path: str | Path = SSB_NORWAY_CPI_PATH,
    *,
    full_refresh: bool = False,
) -> Path:
    existing_frame = None if full_refresh else load_existing_norwegian_cpi(output_path)
    frame = fetch_norwegian_cpi(existing_frame=existing_frame)
    return persist_norwegian_cpi(frame, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Norwegian headline and core CPI data from SSB."
    )
    parser.add_argument("--output", type=Path, default=SSB_NORWAY_CPI_PATH)
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Fetch full Norwegian CPI history instead of appending new periods.",
    )
    args = parser.parse_args()

    existing_frame = (
        None if args.full_refresh else load_existing_norwegian_cpi(args.output)
    )
    frame = fetch_norwegian_cpi(existing_frame=existing_frame)
    destination = persist_norwegian_cpi(frame, args.output)
    print(f"Saved Norwegian CPI data to {destination}")
    print("\nLatest observations:")
    print(
        latest_norwegian_cpi_observations(frame).to_string(
            index=False,
            formatters={
                "date": lambda value: value.strftime("%Y-%m-%d"),
                "value": lambda value: f"{value:.2f}",
            },
        )
    )


if __name__ == "__main__":
    main()
