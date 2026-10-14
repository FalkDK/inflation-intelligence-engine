import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from inflation_engine.utils.config import COUNTRIES, series_role
from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.manifest import update_ingestion_manifest
from inflation_engine.utils.paths import EUROSTAT_HICP_PATH
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS, validate_cpi_schema


ECB_HICP_API_URL = "https://data-api.ecb.europa.eu/service/data/HICP"
ECB_HICP_DATASET_ID = "eurostat_hicp"

HICP_GEOGRAPHIES = {
    "AT": "AT",
    "BE": "BE",
    "CY": "CY",
    "DE": "DE",
    "DK": "DK",
    "EE": "EE",
    "ES": "ES",
    "FI": "FI",
    "FR": "FR",
    "GR": "GR",
    "HR": "HR",
    "IE": "IE",
    "IT": "IT",
    "LT": "LT",
    "LU": "LU",
    "LV": "LV",
    "MT": "MT",
    "NL": "NL",
    "PT": "PT",
    "SE": "SE",
    "SI": "SI",
    "SK": "SK",
    "U2": "EA20",
}

HICP_SERIES = {
    "000000": {
        "series_type": "headline",
        "component": "all_items",
    },
    "XEF000": {
        "series_type": "core",
        "component": "all_items_less_food_energy_alcohol_tobacco",
    },
    "FOOD00": {
        "series_type": "headline",
        "component": "food_including_alcohol_tobacco",
    },
    "NRGY00": {
        "series_type": "headline",
        "component": "energy",
    },
    "IGXE00": {
        "series_type": "headline",
        "component": "non_energy_industrial_goods",
    },
    "SERV00": {
        "series_type": "headline",
        "component": "services",
    },
    "041000": {
        "series_type": "headline",
        "component": "actual_rentals_for_housing",
    },
}

ECB_OBSERVATION_STATUSES = {
    "A": "normal",
    "B": "break",
    "E": "estimated",
    "F": "forecast",
    "P": "provisional",
    "R": "revised",
}

ECB_HICP_SERIES_KEY = (
    f"M.{'+'.join(HICP_GEOGRAPHIES)}.N."
    f"{'+'.join(HICP_SERIES)}.4D0.INX"
)


def fetch_ecb_hicp_csv(
    *,
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> str:
    """Fetch current monthly HICP index histories from the ECB Data Portal."""
    client = session or requests.Session()
    url = f"{ECB_HICP_API_URL}/{ECB_HICP_SERIES_KEY}"

    try:
        response = client.get(
            url,
            params={"format": "csvdata"},
            headers={"Accept": "text/csv"},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError("ECB HICP request failed") from error

    return response.text


def normalize_ecb_hicp(csv_data: str) -> pd.DataFrame:
    """Normalize ECB HICP CSV data into the common CPI schema."""
    observations = pd.read_csv(StringIO(csv_data), low_memory=False)
    required_columns = {
        "REF_AREA",
        "ADJUSTMENT",
        "ICP_ITEM",
        "DATA_PROVIDER",
        "ICP_SUFFIX",
        "TIME_PERIOD",
        "OBS_VALUE",
        "OBS_STATUS",
        "UNIT_INDEX_BASE",
    }
    missing_columns = required_columns.difference(observations.columns)
    if missing_columns:
        raise ValueError(
            f"ECB HICP response is missing required columns: {sorted(missing_columns)}"
        )

    frame = observations.loc[
        observations["REF_AREA"].isin(HICP_GEOGRAPHIES)
        & observations["ICP_ITEM"].isin(HICP_SERIES)
        & observations["ADJUSTMENT"].eq("N")
        & observations["DATA_PROVIDER"].eq("4D0")
        & observations["ICP_SUFFIX"].eq("INX")
    ].copy()

    invalid_bases = set(frame["UNIT_INDEX_BASE"].dropna()) - {"2025 = 100"}
    if invalid_bases:
        raise ValueError(f"Unexpected ECB HICP index bases: {sorted(invalid_bases)}")

    frame["date"] = pd.to_datetime(
        frame["TIME_PERIOD"].astype(str) + "-01",
        errors="coerce",
    )
    frame["value"] = pd.to_numeric(frame["OBS_VALUE"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"])

    frame["country_code"] = frame["REF_AREA"].map(HICP_GEOGRAPHIES)
    frame["country"] = frame["country_code"].map(
        {code: COUNTRIES[code]["name"] for code in HICP_GEOGRAPHIES.values()}
    )
    frame["region"] = frame["country_code"].map(
        {code: COUNTRIES[code]["region"] for code in HICP_GEOGRAPHIES.values()}
    )
    frame["series_type"] = frame["ICP_ITEM"].map(
        {code: metadata["series_type"] for code, metadata in HICP_SERIES.items()}
    )
    frame["component"] = frame["ICP_ITEM"].map(
        {code: metadata["component"] for code, metadata in HICP_SERIES.items()}
    )
    # Denmark and Sweden retain their national headline series as the model
    # aggregates. Keep HICP all-items as an explicitly named component so the
    # harmonised decomposition can coexist without a duplicate canonical key.
    nordic_hicp_headline = frame["country_code"].isin({"DK", "SE"}) & frame[
        "ICP_ITEM"
    ].eq("000000")
    frame.loc[nordic_hicp_headline, "component"] = "hicp_all_items"
    frame["series_role"] = [
        series_role(country_code, series_type, component)
        for country_code, series_type, component in frame[
            ["country_code", "series_type", "component"]
        ].itertuples(index=False, name=None)
    ]
    frame["source"] = "ECB Data Portal"
    frame["source_dataset"] = "HICP"
    frame["source_series_id"] = (
        "HICP.M."
        + frame["REF_AREA"].astype(str)
        + ".N."
        + frame["ICP_ITEM"].astype(str)
        + ".4D0.INX"
    )
    frame["frequency"] = "monthly"
    frame["unit"] = "index"
    frame["index_base"] = "2025=100"
    frame["seasonal_adjustment"] = "not_seasonally_adjusted"
    frame["observation_status"] = (
        frame["OBS_STATUS"]
        .map(ECB_OBSERVATION_STATUSES)
        .fillna("not_provided")
    )

    frame = (
        frame.loc[:, CPI_SCHEMA_COLUMNS]
        .sort_values(["country_code", "series_type", "date"])
        .reset_index(drop=True)
    )
    validate_cpi_schema(frame)
    return frame


def fetch_hicp(
    *,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch current headline and core HICP for the requested geographies."""
    csv_data = fetch_ecb_hicp_csv(session=session)
    return normalize_ecb_hicp(csv_data)


def save_hicp(
    frame: pd.DataFrame,
    output_path: str | Path = EUROSTAT_HICP_PATH,
) -> Path:
    """Validate and save normalized HICP data as Parquet."""
    validate_cpi_schema(frame)
    return atomic_write_parquet(frame, output_path, columns=CPI_SCHEMA_COLUMNS)


def persist_hicp(
    frame: pd.DataFrame,
    output_path: str | Path = EUROSTAT_HICP_PATH,
) -> Path:
    destination = save_hicp(frame, output_path)
    update_ingestion_manifest(
        ECB_HICP_DATASET_ID,
        frame,
        destination,
        source_updated_at=frame.attrs.get("source_updated_at"),
    )
    return destination


def latest_hicp_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the latest observation for every HICP series and geography."""
    validate_cpi_schema(frame)
    latest = (
        frame.sort_values("date")
        .groupby(["country_code", "series_type", "component"], as_index=False)
        .tail(1)
    )
    return (
        latest.loc[
            :,
            ["country", "country_code", "series_type", "component", "date", "value"],
        ]
        .sort_values(["country_code", "series_type"])
        .reset_index(drop=True)
    )


def run(output_path: str | Path = EUROSTAT_HICP_PATH) -> Path:
    frame = fetch_hicp()
    return persist_hicp(frame, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch current headline and core HICP data from the ECB."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EUROSTAT_HICP_PATH,
        help=f"Parquet output path (default: {EUROSTAT_HICP_PATH})",
    )
    args = parser.parse_args()

    frame = fetch_hicp()
    destination = persist_hicp(frame, args.output)
    print(f"Saved ECB HICP data to {destination}")
    print("\nLatest observations:")
    print(
        latest_hicp_observations(frame).to_string(
            index=False,
            formatters={
                "date": lambda value: value.strftime("%Y-%m-%d"),
                "value": lambda value: f"{value:.3f}",
            },
        )
    )


if __name__ == "__main__":
    main()
