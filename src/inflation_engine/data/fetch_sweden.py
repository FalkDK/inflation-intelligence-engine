import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

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
from inflation_engine.utils.paths import SCB_SWEDEN_CPI_PATH
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS, validate_cpi_schema


SCB_API_BASE_URL = "https://api.scb.se/OV0104/v2beta/api/v2/tables"
SCB_DATA_QUERY_BASE_URL = "https://statistikdatabasen.scb.se/api/v2/tables"
SCB_SWEDEN_DATASET_ID = "scb_sweden_cpi"
SCB_COMPONENT_BATCH_SIZE = 50
SCB_COMPONENT_MONTH_BATCH_SIZE = 75
SCB_COMPONENT_START_PERIOD = "2006M01"
SCB_CPIF_COMPONENT_TABLE = {
    "table": "TAB6613",
    "content_code": "0000083S",
    "component_dimension": "VaruTjanstegrupp",
    "total_code": "00",
}

SWEDEN_CPI_TABLES = {
    "headline": {
        "table": "TAB6590",
        "content_code": "000007ZN",
        "component": "cpif_fixed_interest_rate",
    },
    "core": {
        "table": "TAB6593",
        "content_code": "000007ZW",
        "component": "cpif_fixed_interest_rate_excluding_energy",
    },
}


def fetch_scb_dataset(
    table: str,
    content_code: str,
    *,
    component_dimension: str | None = None,
    component_codes: list[str] | None = None,
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> Mapping[str, Any]:
    """Fetch one Swedish CPI index series from Statistics Sweden."""
    client = session or requests.Session()
    url = f"{SCB_API_BASE_URL}/{table}/data"
    selection = []
    if component_dimension:
        selection.append(
            {
                "variableCode": component_dimension,
                "valueCodes": component_codes or ["*"],
            }
        )
    selection.extend(
        [
            {"variableCode": "ContentsCode", "valueCodes": [content_code]},
            {"variableCode": "Tid", "valueCodes": ["*"]},
        ]
    )
    request_payload = {
        "selection": selection
    }

    try:
        response = client.post(
            url,
            params={"lang": "en", "outputFormat": "json-stat2"},
            json=request_payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(f"Statistics Sweden request failed for table {table}") from error

    dataset = response.json()
    if dataset.get("class") != "dataset":
        raise ValueError(f"Invalid Statistics Sweden response for table {table}")
    return dataset


def fetch_scb_metadata(
    table: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 20.0,
) -> Mapping[str, Any]:
    """Fetch Statistics Sweden table metadata."""
    client = session or requests.Session()
    url = f"{SCB_API_BASE_URL}/{table}/metadata"

    try:
        response = client.get(
            url,
            params={"lang": "en"},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            f"Statistics Sweden metadata request failed for table {table}"
        ) from error

    dataset = response.json()
    if dataset.get("class") != "dataset":
        raise ValueError(f"Invalid Statistics Sweden metadata for table {table}")
    return dataset


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def _period_code(date: pd.Timestamp) -> str:
    return pd.Timestamp(date).strftime("%YM%m")


def fetch_scb_component_dataset(
    table: str,
    content_code: str,
    *,
    component_dimension: str,
    component_codes: list[str],
    periods: list[str],
    session: requests.Session | None = None,
    timeout: float = 20.0,
) -> Mapping[str, Any]:
    """Fetch one Swedish CPI component batch using SCB's GET selection API."""
    client = session or requests.Session()
    params = {
        "lang": "en",
        "outputFormat": "json-stat2",
        "valueCodes[ContentsCode]": content_code,
        f"valueCodes[{component_dimension}]": ",".join(component_codes),
        "valueCodes[Tid]": ",".join(periods),
    }
    url = (
        f"{SCB_DATA_QUERY_BASE_URL}/{table}/data?"
        + urlencode(params, safe=",")
    )

    try:
        response = client.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            f"Statistics Sweden component request failed for table {table}"
        ) from error

    dataset = response.json()
    if dataset.get("class") != "dataset":
        raise ValueError(f"Invalid Statistics Sweden component response for {table}")
    return dataset


def load_existing_swedish_cpi(
    input_path: str | Path = SCB_SWEDEN_CPI_PATH,
) -> pd.DataFrame | None:
    """Load a previous Swedish CPI file for incremental component refreshes."""
    source = Path(input_path)
    if not source.is_file():
        return None
    try:
        frame = pd.read_parquet(source)
        validate_cpi_schema(frame)
    except (OSError, ValueError, KeyError):
        return None
    return frame.loc[:, CPI_SCHEMA_COLUMNS].copy()


def _existing_component_frame(existing_frame: pd.DataFrame | None) -> pd.DataFrame:
    if existing_frame is None:
        return pd.DataFrame(columns=CPI_SCHEMA_COLUMNS)
    frame = existing_frame.loc[
        existing_frame["source_dataset"].eq(SCB_CPIF_COMPONENT_TABLE["table"])
        & existing_frame["source_series_id"].str.startswith(
            f"{SCB_CPIF_COMPONENT_TABLE['table']}:",
            na=False,
        )
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=CPI_SCHEMA_COLUMNS)
    validate_cpi_schema(frame)
    return frame.loc[:, CPI_SCHEMA_COLUMNS]


def normalize_scb_series(
    dataset: Mapping[str, Any],
    *,
    series_type: str,
    component: str,
    source_dataset: str,
    source_series_id: str,
) -> pd.DataFrame:
    """Normalize one Statistics Sweden index series."""
    observations = parse_jsonstat_observations(dataset)
    if not {"Tid", "value"}.issubset(observations.columns):
        raise ValueError("Statistics Sweden observations must contain Tid and value")

    frame = observations.loc[:, ["Tid", "value"]].copy()
    frame["date"] = pd.to_datetime(
        frame["Tid"].str.replace("M", "-", regex=False) + "-01",
        errors="coerce",
    )
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"])

    country = COUNTRIES["SE"]
    frame["country"] = country["name"]
    frame["country_code"] = "SE"
    frame["region"] = country["region"]
    frame["series_type"] = series_type
    frame["series_role"] = series_role("SE", series_type, component)
    frame["component"] = component
    frame["source"] = "Statistics Sweden"
    frame["source_dataset"] = source_dataset
    frame["source_series_id"] = source_series_id
    frame["frequency"] = "monthly"
    frame["unit"] = "index"
    frame["index_base"] = "2020=100"
    frame["seasonal_adjustment"] = "not_seasonally_adjusted"
    frame["observation_status"] = "not_provided"

    frame = frame.loc[:, CPI_SCHEMA_COLUMNS].sort_values("date").reset_index(drop=True)
    validate_cpi_schema(frame)
    return frame


def normalize_scb_component_series(
    dataset: Mapping[str, Any],
    *,
    component_dimension: str,
    source_dataset: str,
    source_series_id: str,
    exclude_component_codes: set[str] | None = None,
    component_slugs: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize one Statistics Sweden component index table."""
    observations = parse_jsonstat_observations(dataset)
    required_columns = {component_dimension, "ContentsCode", "Tid", "value"}
    if not required_columns.issubset(observations.columns):
        raise ValueError(
            "Statistics Sweden component observations must contain "
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

    if component_slugs is None:
        labels = category_labels(dataset, component_dimension)
        component_slugs = unique_component_slugs(
            sorted(set(frame[component_dimension])),
            labels,
        )
    frame["component"] = frame[component_dimension].map(dict(component_slugs))

    country = COUNTRIES["SE"]
    frame["country"] = country["name"]
    frame["country_code"] = "SE"
    frame["region"] = country["region"]
    frame["series_type"] = "headline"
    frame["series_role"] = [
        series_role("SE", "headline", component) for component in frame["component"]
    ]
    frame["source"] = "Statistics Sweden"
    frame["source_dataset"] = source_dataset
    frame["source_series_id"] = (
        source_dataset + ":" + frame[component_dimension].astype(str)
    )
    frame["frequency"] = "monthly"
    frame["unit"] = "index"
    frame["index_base"] = "2020=100"
    frame["seasonal_adjustment"] = "not_seasonally_adjusted"
    frame["observation_status"] = "not_provided"

    frame = (
        frame.loc[:, CPI_SCHEMA_COLUMNS]
        .sort_values(["component", "date"])
        .reset_index(drop=True)
    )
    validate_cpi_schema(frame)
    return frame


def fetch_swedish_cpi(
    *,
    session: requests.Session | None = None,
    existing_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fetch Swedish headline CPIF and core CPIF excluding energy."""
    frames = []
    source_updates = []
    for series_type, metadata in SWEDEN_CPI_TABLES.items():
        dataset = fetch_scb_dataset(
            metadata["table"],
            metadata["content_code"],
            session=session,
        )
        if dataset.get("updated"):
            source_updates.append(dataset["updated"])
        frames.append(
            normalize_scb_series(
                dataset,
                series_type=series_type,
                component=metadata["component"],
                source_dataset=metadata["table"],
                source_series_id=metadata["content_code"],
            )
        )

    existing_components = _existing_component_frame(existing_frame)
    try:
        component_metadata = fetch_scb_metadata(
            SCB_CPIF_COMPONENT_TABLE["table"],
            session=session,
        )
    except RuntimeError:
        if existing_components.empty:
            raise
        # Headline/core releases should still refresh when SCB's much larger
        # component metadata endpoint is temporarily unavailable. Preserve the
        # last validated component panel and make no claim of a component update.
        frames.append(existing_components)
        combined = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["series_type", "date"])
            .reset_index(drop=True)
        )
        validate_cpi_schema(combined)
        combined.attrs["source_updated_at"] = (
            max(source_updates) if source_updates else None
        )
        return combined
    if component_metadata.get("updated"):
        source_updates.append(component_metadata["updated"])
    component_dimension = SCB_CPIF_COMPONENT_TABLE["component_dimension"]
    component_codes = [
        code
        for code in category_codes(component_metadata, component_dimension)
        if code != SCB_CPIF_COMPONENT_TABLE["total_code"]
    ]
    periods = [
        code
        for code in category_codes(component_metadata, "Tid")
        if code >= SCB_COMPONENT_START_PERIOD
    ]
    if not existing_components.empty:
        latest_component_period = _period_code(existing_components["date"].max())
        periods = [code for code in periods if code > latest_component_period]
        frames.append(existing_components)

    component_slugs = unique_component_slugs(
        component_codes,
        category_labels(component_metadata, component_dimension),
    )
    new_component_frames = []
    try:
        for code_batch in _chunks(component_codes, SCB_COMPONENT_BATCH_SIZE):
            for period_batch in _chunks(periods, SCB_COMPONENT_MONTH_BATCH_SIZE):
                component_dataset = fetch_scb_component_dataset(
                    SCB_CPIF_COMPONENT_TABLE["table"],
                    SCB_CPIF_COMPONENT_TABLE["content_code"],
                    component_dimension=SCB_CPIF_COMPONENT_TABLE["component_dimension"],
                    component_codes=code_batch,
                    periods=period_batch,
                    session=session,
                )
                new_component_frames.append(
                    normalize_scb_component_series(
                        component_dataset,
                        component_dimension=SCB_CPIF_COMPONENT_TABLE[
                            "component_dimension"
                        ],
                        source_dataset=SCB_CPIF_COMPONENT_TABLE["table"],
                        source_series_id=SCB_CPIF_COMPONENT_TABLE["content_code"],
                        exclude_component_codes={SCB_CPIF_COMPONENT_TABLE["total_code"]},
                        component_slugs=component_slugs,
                    )
                )
    except RuntimeError:
        if existing_components.empty:
            raise
        # Avoid mixing a partial component month into the canonical panel. The
        # aggregate release remains usable while the last complete detailed
        # component vintage is preserved.
        new_component_frames = []
    frames.extend(new_component_frames)

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["series_type", "date"])
        .reset_index(drop=True)
    )
    validate_cpi_schema(combined)
    combined.attrs["source_updated_at"] = (
        max(source_updates) if source_updates else None
    )
    return combined


def save_swedish_cpi(
    frame: pd.DataFrame,
    output_path: str | Path = SCB_SWEDEN_CPI_PATH,
) -> Path:
    validate_cpi_schema(frame)
    return atomic_write_parquet(frame, output_path, columns=CPI_SCHEMA_COLUMNS)


def persist_swedish_cpi(
    frame: pd.DataFrame,
    output_path: str | Path = SCB_SWEDEN_CPI_PATH,
) -> Path:
    destination = save_swedish_cpi(frame, output_path)
    update_ingestion_manifest(
        SCB_SWEDEN_DATASET_ID,
        frame,
        destination,
        source_updated_at=frame.attrs.get("source_updated_at"),
    )
    return destination


def latest_swedish_cpi_observations(frame: pd.DataFrame) -> pd.DataFrame:
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
    output_path: str | Path = SCB_SWEDEN_CPI_PATH,
    *,
    full_refresh: bool = False,
) -> Path:
    existing_frame = None if full_refresh else load_existing_swedish_cpi(output_path)
    frame = fetch_swedish_cpi(existing_frame=existing_frame)
    return persist_swedish_cpi(frame, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Swedish headline and core CPI data from SCB."
    )
    parser.add_argument("--output", type=Path, default=SCB_SWEDEN_CPI_PATH)
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Fetch full Swedish component history instead of appending new periods.",
    )
    args = parser.parse_args()

    existing_frame = (
        None if args.full_refresh else load_existing_swedish_cpi(args.output)
    )
    frame = fetch_swedish_cpi(existing_frame=existing_frame)
    destination = persist_swedish_cpi(frame, args.output)
    print(f"Saved Swedish CPI data to {destination}")
    print("\nLatest observations:")
    print(
        latest_swedish_cpi_observations(frame).to_string(
            index=False,
            formatters={
                "date": lambda value: value.strftime("%Y-%m-%d"),
                "value": lambda value: f"{value:.2f}",
            },
        )
    )


if __name__ == "__main__":
    main()
