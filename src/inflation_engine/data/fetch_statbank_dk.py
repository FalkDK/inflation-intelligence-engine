import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from inflation_engine.utils.config import COUNTRIES, series_role
from inflation_engine.utils.components import category_labels, unique_component_slugs
from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.jsonstat import parse_jsonstat_observations
from inflation_engine.utils.manifest import update_ingestion_manifest
from inflation_engine.utils.paths import STATBANK_DK_CPI_PATH
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS, validate_cpi_schema


STATBANK_DATA_URL = "https://api.statbank.dk/v1/data"
STATBANK_TABLEINFO_URL = "https://api.statbank.dk/v1/tableinfo"
STATBANK_TABLE = "PRIS01"
STATBANK_DK_DATASET_ID = "statbank_dk_cpi"

DK_PRIMARY_CPI_SERIES = {
    "000000": {
        "series_type": "headline",
        "component": "all_items",
    },
    "151": {
        "series_type": "core",
        "component": "all_items_less_energy_unprocessed_food",
    },
}
DK_CPI_SERIES = DK_PRIMARY_CPI_SERIES


def fetch_statbank_payload(
    *,
    periods: list[str] | None = None,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> Mapping[str, Any]:
    """Fetch Danish headline and core CPI index histories from StatBank."""
    client = session or requests.Session()
    request_payload = {
        "table": STATBANK_TABLE,
        "format": "JSONSTAT",
        "lang": "en",
        "variables": [
            {"code": "VAREGR", "values": ["*"]},
            {"code": "ENHED", "values": ["100"]},
            {"code": "Tid", "values": periods or ["*"]},
        ],
    }

    try:
        response = client.post(
            STATBANK_DATA_URL,
            json=request_payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError("Statistics Denmark CPI request failed") from error

    payload = response.json()
    dataset = payload.get("dataset")
    if not isinstance(dataset, Mapping):
        message = payload.get("message", "response contains no dataset")
        raise ValueError(f"Invalid Statistics Denmark response: {message}")
    return dataset


def fetch_statbank_tableinfo(
    *,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> Mapping[str, Any]:
    """Fetch StatBank table metadata used to identify available months."""
    client = session or requests.Session()
    url = f"{STATBANK_TABLEINFO_URL}/{STATBANK_TABLE}"
    try:
        response = client.get(
            url,
            params={"lang": "en", "format": "JSON"},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError("Statistics Denmark table metadata request failed") from error

    metadata = response.json()
    if not isinstance(metadata, Mapping):
        raise ValueError("Invalid Statistics Denmark table metadata")
    return metadata


def statbank_period_codes(metadata: Mapping[str, Any]) -> list[str]:
    """Return StatBank month codes in source order."""
    variables = metadata.get("variables", [])
    for variable in variables:
        if not isinstance(variable, Mapping):
            continue
        if variable.get("id") != "Tid" and variable.get("code") != "Tid":
            continue
        values = variable.get("values", [])
        codes = [
            value.get("id") or value.get("code")
            for value in values
            if isinstance(value, Mapping)
        ]
        return [code for code in codes if isinstance(code, str)]
    return []


def load_existing_danish_cpi(
    input_path: str | Path = STATBANK_DK_CPI_PATH,
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


def _incremental_periods(
    existing_frame: pd.DataFrame | None,
    source_periods: list[str],
) -> list[str] | None:
    if existing_frame is None or existing_frame.empty:
        return None
    latest_by_series = existing_frame.groupby("source_series_id")["date"].max()
    if latest_by_series.empty:
        return None
    latest_existing_period = _period_code(latest_by_series.min())
    return [period for period in source_periods if period > latest_existing_period]


def _merge_existing_and_new(
    existing_frame: pd.DataFrame | None,
    new_frame: pd.DataFrame,
) -> pd.DataFrame:
    if existing_frame is None or existing_frame.empty:
        return new_frame
    key = ["date", "country_code", "series_type", "component"]
    combined = (
        pd.concat([existing_frame, new_frame], ignore_index=True)
        .drop_duplicates(key, keep="last")
        .sort_values(["series_type", "component", "date"])
        .reset_index(drop=True)
    )
    validate_cpi_schema(combined)
    combined.attrs["source_updated_at"] = new_frame.attrs.get("source_updated_at")
    return combined


def normalize_statbank_cpi(dataset: Mapping[str, Any]) -> pd.DataFrame:
    """Normalize Danish CPI observations into the common CPI schema."""
    observations = parse_jsonstat_observations(dataset)
    required_dimensions = {"VAREGR", "ENHED", "Tid", "value"}
    if not required_dimensions.issubset(observations.columns):
        raise ValueError(
            "StatBank observations must contain VAREGR, ENHED, Tid, and value"
        )

    frame = observations.loc[
        observations["ENHED"].eq("100")
    ].copy()

    returned_series = set(frame["VAREGR"])
    missing_series = set(DK_PRIMARY_CPI_SERIES) - returned_series
    if missing_series:
        raise ValueError(
            f"StatBank response is missing CPI series: {sorted(missing_series)}"
        )

    frame["date"] = pd.to_datetime(
        frame["Tid"].str.replace("M", "-", regex=False) + "-01",
        errors="coerce",
    )
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"])

    country = COUNTRIES["DK"]
    frame["country"] = country["name"]
    frame["country_code"] = "DK"
    frame["region"] = country["region"]
    frame["series_type"] = frame["VAREGR"].map(
        {
            code: metadata["series_type"]
            for code, metadata in DK_PRIMARY_CPI_SERIES.items()
        }
    ).fillna("headline")
    component_labels = category_labels(dataset, "VAREGR")
    primary_components = {
        code: metadata["component"] for code, metadata in DK_PRIMARY_CPI_SERIES.items()
    }
    component_slugs = unique_component_slugs(
        sorted(set(frame["VAREGR"])),
        component_labels,
        overrides=primary_components,
    )
    frame["component"] = frame["VAREGR"].map(component_slugs)
    frame["series_role"] = [
        series_role("DK", series_type, component)
        for series_type, component in frame[
            ["series_type", "component"]
        ].itertuples(index=False, name=None)
    ]
    frame["source"] = "Statistics Denmark"
    frame["source_dataset"] = STATBANK_TABLE
    frame["source_series_id"] = frame["VAREGR"]
    frame["frequency"] = "monthly"
    frame["unit"] = "index"
    frame["index_base"] = "2025=100"
    frame["seasonal_adjustment"] = "not_seasonally_adjusted"
    frame["observation_status"] = "not_provided"

    frame = (
        frame.loc[:, CPI_SCHEMA_COLUMNS]
        .sort_values(["series_type", "date"])
        .reset_index(drop=True)
    )
    validate_cpi_schema(frame)
    frame.attrs["source_updated_at"] = dataset.get("updated")
    return frame


def fetch_danish_cpi(
    *,
    session: requests.Session | None = None,
    existing_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fetch and normalize Danish headline and core CPI."""
    periods = None
    if existing_frame is not None:
        metadata = fetch_statbank_tableinfo(session=session)
        periods = _incremental_periods(
            existing_frame,
            statbank_period_codes(metadata),
        )
        if periods == []:
            existing_frame.attrs["source_updated_at"] = metadata.get("updated")
            return existing_frame

    dataset = fetch_statbank_payload(periods=periods, session=session)
    frame = normalize_statbank_cpi(dataset)
    return _merge_existing_and_new(existing_frame, frame)


def save_danish_cpi(
    frame: pd.DataFrame,
    output_path: str | Path = STATBANK_DK_CPI_PATH,
) -> Path:
    """Validate and save normalized Danish CPI data as Parquet."""
    validate_cpi_schema(frame)
    return atomic_write_parquet(frame, output_path, columns=CPI_SCHEMA_COLUMNS)


def persist_danish_cpi(
    frame: pd.DataFrame,
    output_path: str | Path = STATBANK_DK_CPI_PATH,
) -> Path:
    destination = save_danish_cpi(frame, output_path)
    update_ingestion_manifest(
        STATBANK_DK_DATASET_ID,
        frame,
        destination,
        source_updated_at=frame.attrs.get("source_updated_at"),
    )
    return destination


def latest_danish_cpi_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the latest Danish headline and core CPI observations."""
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
    output_path: str | Path = STATBANK_DK_CPI_PATH,
    *,
    full_refresh: bool = False,
) -> Path:
    existing_frame = None if full_refresh else load_existing_danish_cpi(output_path)
    frame = fetch_danish_cpi(existing_frame=existing_frame)
    return persist_danish_cpi(frame, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Danish headline and core CPI data from StatBank."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=STATBANK_DK_CPI_PATH,
        help=f"Parquet output path (default: {STATBANK_DK_CPI_PATH})",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Fetch full Danish CPI history instead of appending new periods.",
    )
    args = parser.parse_args()

    existing_frame = None if args.full_refresh else load_existing_danish_cpi(args.output)
    frame = fetch_danish_cpi(existing_frame=existing_frame)
    destination = persist_danish_cpi(frame, args.output)
    print(f"Saved Danish CPI data to {destination}")
    print("\nLatest observations:")
    print(
        latest_danish_cpi_observations(frame).to_string(
            index=False,
            formatters={
                "date": lambda value: value.strftime("%Y-%m-%d"),
                "value": lambda value: f"{value:.2f}",
            },
        )
    )


if __name__ == "__main__":
    main()
