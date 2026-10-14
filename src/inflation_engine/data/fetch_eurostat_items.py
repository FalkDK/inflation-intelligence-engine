"""Fetch detailed COICOP-level HICP item indices for the euro-area item
decomposition (see inflation_engine.features.eurozone_item_decomposition).
This is a separate, finer-grained data source from fetch_eurostat.py's seven
harmonised HICP special aggregates -- the two are not related beyond both
ultimately coming from Eurostat/ECB HICP data.

Unlike the US/UK/Denmark item decompositions, which each need a bespoke,
manually curated weight-to-series crosswalk, Eurostat publishes both this
detailed index (``prc_hicp_midx``) and the matching annual weights (fetched
separately by fetch_eurostat_item_weights.py) under the same COICOP code, so
no separate crosswalk is needed here. The COICOP root, CP00 ("all-items
HICP"), is fetched here too and doubles as this module's own headline series
-- eurozone_item_decomposition.py reconciles each country's item frontier
against it directly, rather than against fetch_eurostat.py's separately
sourced headline (the two track each other to within ~0.02pp, but CP00 is
already in the same dataset and vintage as every other component here).
"""

import argparse
from pathlib import Path

import pandas as pd
import requests

from inflation_engine.utils.config import COUNTRIES, series_role
from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.jsonstat import parse_jsonstat_observations
from inflation_engine.utils.manifest import update_ingestion_manifest
from inflation_engine.utils.paths import EUROSTAT_HICP_ITEMS_PATH
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS, validate_cpi_schema


EUROSTAT_HICP_ITEM_INDEX_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "prc_hicp_midx"
)
EUROSTAT_HICP_ITEM_DATASET_ID = "eurostat_hicp_items"

# All 20 euro-area members. Maps this project's country_code to the
# geography code Eurostat's dissemination API expects -- identical to
# country_code for every member except Greece, whose Eurostat geography code
# is "EL" (its ISO 3166 code "GR" is what the *separate* ECB Data Portal
# fetch_eurostat.py uses instead -- verified live, not guessed). Both this
# map and fetch_eurostat_item_weights.py's own copy of it share the same
# Eurostat COICOP classification and API, so no per-country parsing is
# required the way the US/UK/Denmark item decompositions need.
HICP_ITEM_GEOGRAPHIES = {
    "AT": "AT",
    "BE": "BE",
    "CY": "CY",
    "DE": "DE",
    "EE": "EE",
    "ES": "ES",
    "FI": "FI",
    "FR": "FR",
    "GR": "EL",
    "HR": "HR",
    "IE": "IE",
    "IT": "IT",
    "LT": "LT",
    "LU": "LU",
    "LV": "LV",
    "MT": "MT",
    "NL": "NL",
    "PT": "PT",
    "SI": "SI",
    "SK": "SK",
}

# Index base is irrelevant to a percent-change decomposition; I15 (2015=100)
# is simply the unit prc_hicp_midx always publishes for the "coicop" (COICOP
# item) dimension, unlike prc_hicp_midx's own headline series which also
# carries older bases.
_ITEM_UNIT_CODE = "I15"

# Eurostat's "coicop" dimension mixes the true additive COICOP hierarchy
# (CP00 the all-items root, and its CPnn... descendants) with non-hierarchical
# overlapping combinations (e.g. "FOOD", "SERV", "NRGY00"); only the former is
# usable as an additive frontier -- the latter is already fetched, under its
# own ECB naming, by fetch_eurostat.py.
_COICOP_HIERARCHY_PATTERN = r"CP\d+"


def fetch_eurostat_item_index_dataset(
    geography: str,
    *,
    since_year: int = 1996,
    session: requests.Session | None = None,
    timeout: float = 180.0,
) -> dict:
    """Fetch one geography's full detailed monthly HICP index history."""
    client = session or requests.Session()
    params = [
        ("lang", "en"),
        ("geo", geography),
        ("unit", _ITEM_UNIT_CODE),
        ("sinceTimePeriod", str(since_year)),
    ]
    try:
        response = client.get(
            EUROSTAT_HICP_ITEM_INDEX_URL,
            params=params,
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError(
            f"Eurostat HICP item index request failed for {geography}"
        ) from error


def normalize_eurostat_item_index(datasets: dict[str, dict]) -> pd.DataFrame:
    """Map each geography's JSON-stat response into the common CPI schema."""
    frames = []
    for country_code, dataset in datasets.items():
        observations = parse_jsonstat_observations(dataset)
        required_columns = {"coicop", "geo", "time", "value"}
        missing_columns = required_columns.difference(observations.columns)
        if missing_columns:
            raise ValueError(
                "Eurostat HICP item index response is missing required "
                f"columns: {sorted(missing_columns)}"
            )
        observations = observations.loc[
            observations["coicop"].str.fullmatch(_COICOP_HIERARCHY_PATTERN)
        ].copy()
        observations["value"] = pd.to_numeric(
            observations["value"], errors="coerce"
        )
        # A handful of country/item combinations are genuinely published as
        # zero rather than withheld -- e.g. Ireland's CP0441 (water supply)
        # reads 0 for its entire history because domestic water charges were
        # suspended/abolished there. An index can't legitimately be zero or
        # negative, so treat these exactly like a missing observation: the
        # frontier's cascading residual then falls back to that item's
        # nearest ancestor with a usable index instead of erroring.
        observations = observations.loc[observations["value"] > 0]
        observations["date"] = pd.to_datetime(
            observations["time"].astype(str) + "-01", errors="coerce"
        )
        observations = observations.dropna(subset=["date"])
        observations["country_code"] = country_code
        observations["source_series_id"] = observations["coicop"]
        observations["component"] = "hicp_item_" + observations[
            "coicop"
        ].str.lower()
        frames.append(observations)

    if not frames:
        raise ValueError("No Eurostat HICP item index geographies were requested")
    frame = pd.concat(frames, ignore_index=True)

    frame["country"] = frame["country_code"].map(
        {code: COUNTRIES[code]["name"] for code in HICP_ITEM_GEOGRAPHIES}
    )
    frame["region"] = frame["country_code"].map(
        {code: COUNTRIES[code]["region"] for code in HICP_ITEM_GEOGRAPHIES}
    )
    frame["series_type"] = "headline"
    frame["series_role"] = [
        series_role(country_code, series_type, component)
        for country_code, series_type, component in frame[
            ["country_code", "series_type", "component"]
        ].itertuples(index=False, name=None)
    ]
    frame["source"] = "Eurostat"
    frame["source_dataset"] = "prc_hicp_midx"
    frame["frequency"] = "monthly"
    frame["unit"] = "index"
    frame["index_base"] = "2015=100"
    frame["seasonal_adjustment"] = "not_seasonally_adjusted"
    # Eurostat's per-observation reliability/break flags aren't captured here,
    # matching the same simplification fetch_fred.py already makes for every
    # FRED-sourced series in this project.
    frame["observation_status"] = "not_provided"

    frame = (
        frame.loc[:, CPI_SCHEMA_COLUMNS]
        .sort_values(["country_code", "component", "date"])
        .reset_index(drop=True)
    )
    validate_cpi_schema(frame)
    return frame


def fetch_hicp_items(
    *,
    since_year: int = 1996,
    session: requests.Session | None = None,
    geographies: dict[str, str] = HICP_ITEM_GEOGRAPHIES,
) -> pd.DataFrame:
    """Fetch and normalize detailed monthly HICP item indices."""
    datasets = {
        country_code: fetch_eurostat_item_index_dataset(
            geography, since_year=since_year, session=session
        )
        for country_code, geography in geographies.items()
    }
    return normalize_eurostat_item_index(datasets)


def save_hicp_items(
    frame: pd.DataFrame,
    output_path: str | Path = EUROSTAT_HICP_ITEMS_PATH,
) -> Path:
    validate_cpi_schema(frame)
    return atomic_write_parquet(frame, output_path, columns=CPI_SCHEMA_COLUMNS)


def persist_hicp_items(
    frame: pd.DataFrame,
    output_path: str | Path = EUROSTAT_HICP_ITEMS_PATH,
) -> Path:
    destination = save_hicp_items(frame, output_path)
    update_ingestion_manifest(
        EUROSTAT_HICP_ITEM_DATASET_ID,
        frame,
        destination,
        source_updated_at=frame.attrs.get("source_updated_at"),
    )
    return destination


def run(output_path: str | Path = EUROSTAT_HICP_ITEMS_PATH) -> Path:
    frame = fetch_hicp_items()
    return persist_hicp_items(frame, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch detailed COICOP-level HICP item indices from Eurostat."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EUROSTAT_HICP_ITEMS_PATH,
        help=f"Parquet output path (default: {EUROSTAT_HICP_ITEMS_PATH})",
    )
    args = parser.parse_args()

    frame = fetch_hicp_items()
    destination = persist_hicp_items(frame, args.output)
    print(f"Saved Eurostat HICP item indices to {destination}")
    print(
        frame.groupby("country_code")["component"].nunique().rename("n_items").to_string()
    )


if __name__ == "__main__":
    main()
