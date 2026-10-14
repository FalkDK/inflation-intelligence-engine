import argparse
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from inflation_engine.utils.config import COUNTRIES, series_role
from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.manifest import update_ingestion_manifest
from inflation_engine.utils.paths import FRED_US_CPI_PATH, PROJECT_ROOT, US_CPI_ITEM_SERIES_MAP_PATH
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS, validate_cpi_schema


FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_DATASET_ID = "fred_us_cpi"

# Not every BLS-published CPI series is mirrored on FRED -- FRED curates a
# subset. Series FRED 400s on fall back to BLS's own public timeseries API,
# which is authoritative for anything in cu.series (see
# build_us_cpi_item_series_map.py). Batched: up to 25 series and 20 years
# per request are the documented v2 API limits.
BLS_TIMESERIES_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
BLS_SERIES_BATCH_SIZE = 25
BLS_EARLIEST_YEAR = 1967  # oldest base period among BLS's granular CPI items
BLS_MONTH_PERIODS = {f"M{month:02d}" for month in range(1, 13)}


def _bls_year_windows(end_year: int, *, span: int = 20) -> list[tuple[int, int]]:
    """Non-overlapping <= 20-year windows from BLS_EARLIEST_YEAR to
    end_year, inclusive -- the BLS v2 API's max span per request."""
    windows = []
    start = BLS_EARLIEST_YEAR
    while start <= end_year:
        windows.append((start, min(start + span - 1, end_year)))
        start += span
    return windows

# The fixed aggregate/bucket series decomposition.py's five-bucket engine
# depends on by exact component name (food, energy, shelter, and their
# sub-splits). Item-level series -- the much larger, finer-grained set used
# for item breadth -- are appended from us_cpi_item_series_map.csv by
# build_us_cpi_series_registry() below, rather than hand-duplicated here.
US_CPI_STRUCTURAL_SERIES = {
    "CPIAUCSL": {
        "series_type": "headline",
        "component": "all_items",
    },
    "CPIUFDSL": {
        "series_type": "headline",
        "component": "food",
    },
    "CUSR0000SAF11": {
        "series_type": "headline",
        "component": "food_at_home",
    },
    "CUSR0000SEFV": {
        "series_type": "headline",
        "component": "food_away_from_home",
    },
    "CPIENGSL": {
        "series_type": "headline",
        "component": "energy",
    },
    "CUSR0000SACE": {
        "series_type": "headline",
        "component": "energy_commodities",
    },
    "CUSR0000SETB01": {
        "series_type": "headline",
        "component": "gasoline_all_types",
    },
    "CUSR0000SEHE": {
        "series_type": "headline",
        "component": "fuel_oil_and_other_fuels",
    },
    "CUSR0000SEHF": {
        "series_type": "headline",
        "component": "energy_services",
    },
    "CUSR0000SEHF01": {
        "series_type": "headline",
        "component": "electricity",
    },
    "CUSR0000SEHF02": {
        "series_type": "headline",
        "component": "utility_piped_gas_service",
    },
    "CPILFESL": {
        "series_type": "core",
        "component": "all_items_less_food_and_energy",
    },
    "CUSR0000SACL1E": {
        "series_type": "core",
        "component": "commodities_less_food_and_energy_commodities",
    },
    "CUSR0000SETA01": {
        "series_type": "core",
        "component": "new_vehicles",
    },
    "CUSR0000SETA02": {
        "series_type": "core",
        "component": "used_cars_and_trucks",
    },
    "CPIAPPSL": {
        "series_type": "core",
        "component": "apparel",
    },
    "CUSR0000SAM1": {
        "series_type": "core",
        "component": "medical_care_commodities",
    },
    "CUSR0000SASLE": {
        "series_type": "core",
        "component": "services_less_energy_services",
    },
    "CUSR0000SAH1": {
        "series_type": "core",
        "component": "shelter",
    },
    "CUSR0000SAS4": {
        "series_type": "core",
        "component": "transportation_services",
    },
    "CUSR0000SAM2": {
        "series_type": "core",
        "component": "medical_care_services",
    },
}

# BLS's Table 1 workbook appends a "Special aggregate indexes" section after
# the real additive tree -- overlapping combos like "Services less rent of
# shelter" or "All items less shelter" that deliberately double-count
# against the tree (that's why load_us_weight_tree in
# us_item_decomposition.py stops before this section). They're still real,
# independently useful published series -- just reference-only: never fed
# into the item frontier's weighted sum, which is why they live in their own
# dict here rather than us_cpi_item_series_map.csv. Every component name is
# prefixed so a query can always tell them apart from the additive item
# series. Six of BLS's published special aggregates are omitted here because
# they're already fetched under US_CPI_STRUCTURAL_SERIES -- Energy and All
# items less food and energy under their more standard CPIENGSL/CPILFESL
# IDs, and Commodities less food and energy commodities, Energy commodities,
# Services less energy services, and Transportation services under the
# exact same CUSR0000 series ID already listed above.
US_CPI_SPECIAL_AGGREGATE_SERIES = {
    "CUSR0000SAC": {
        "series_type": "headline",
        "component": "special_aggregate_commodities",
    },
    "CUSR0000SACL11": {
        "series_type": "headline",
        "component": "special_aggregate_commodities_less_food_and_beverages",
    },
    "CUSR0000SANL11": {
        "series_type": "headline",
        "component": "special_aggregate_nondurables_less_food_and_beverages",
    },
    "CUSR0000SANL113": {
        "series_type": "headline",
        "component": "special_aggregate_nondurables_less_food_beverages_and_apparel",
    },
    "CUSR0000SAD": {
        "series_type": "headline",
        "component": "special_aggregate_durables",
    },
    "CUSR0000SAS": {
        "series_type": "headline",
        "component": "special_aggregate_services",
    },
    "CUSR0000SAS2RS": {
        "series_type": "headline",
        "component": "special_aggregate_rent_of_shelter",
    },
    "CUSR0000SAS367": {
        "series_type": "headline",
        "component": "special_aggregate_other_services",
    },
    "CUSR0000SA0L1": {
        "series_type": "headline",
        "component": "special_aggregate_all_items_less_food",
    },
    "CUSR0000SA0L2": {
        "series_type": "headline",
        "component": "special_aggregate_all_items_less_shelter",
    },
    "CUSR0000SA0L5": {
        "series_type": "headline",
        "component": "special_aggregate_all_items_less_medical_care",
    },
    "CUSR0000SACL1": {
        "series_type": "headline",
        "component": "special_aggregate_commodities_less_food",
    },
    "CUSR0000SANL1": {
        "series_type": "headline",
        "component": "special_aggregate_nondurables_less_food",
    },
    "CUSR0000SANL13": {
        "series_type": "headline",
        "component": "special_aggregate_nondurables_less_food_and_apparel",
    },
    "CUSR0000SAN": {
        "series_type": "headline",
        "component": "special_aggregate_nondurables",
    },
    "CUSR0000SA311": {
        "series_type": "headline",
        "component": "special_aggregate_apparel_less_footwear",
    },
    "CUSR0000SASL2RS": {
        "series_type": "headline",
        "component": "special_aggregate_services_less_rent_of_shelter",
    },
    "CUSR0000SASL5": {
        "series_type": "headline",
        "component": "special_aggregate_services_less_medical_care_services",
    },
    "CUSR0000SA0LE": {
        "series_type": "headline",
        "component": "special_aggregate_all_items_less_energy",
    },
    "CUSR0000SAN1D": {
        "series_type": "headline",
        "component": "special_aggregate_domestically_produced_farm_food",
    },
    "CUSR0000SAS24": {
        "series_type": "headline",
        "component": "special_aggregate_utilities_and_public_transportation",
    },
}


def _slugify_component_label(label: str) -> str:
    """A stable ``component`` identifier derived from an item label -- e.g.
    "Owners' equivalent rent of primary residence" ->
    "owners_equivalent_rent_of_primary_residence"."""
    slug = label.lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", slug).strip("_")


def build_us_cpi_series_registry(
    *,
    structural_series: Mapping[str, Mapping[str, str]] = US_CPI_STRUCTURAL_SERIES,
    special_aggregate_series: Mapping[str, Mapping[str, str]] = US_CPI_SPECIAL_AGGREGATE_SERIES,
    item_series_map_path: str | Path = US_CPI_ITEM_SERIES_MAP_PATH,
) -> dict[str, dict[str, str]]:
    """The fixed structural/aggregate series, BLS's special aggregate
    (overlapping-combo) series, and every item-level series from
    ``us_cpi_item_series_map.csv`` -- deduplicated by series ID (a series
    that's listed more than once, e.g. shelter, keeps whichever entry was
    merged in first rather than being fetched twice)."""
    registry: dict[str, dict[str, str]] = dict(structural_series)
    used_components = {metadata["component"] for metadata in registry.values()}
    for series_id, metadata in special_aggregate_series.items():
        if series_id in registry:
            continue
        registry[series_id] = metadata
        used_components.add(metadata["component"])

    item_map = pd.read_csv(item_series_map_path)
    for row in item_map.itertuples(index=False):
        if row.series_id in registry:
            continue
        component = _slugify_component_label(row.label)
        if component in used_components:
            component = f"{component}_{row.series_id.lower()}"
        used_components.add(component)
        registry[row.series_id] = {"series_type": "headline", "component": component}
    return registry


US_CPI_SERIES = build_us_cpi_series_registry()


def get_fred_api_key(api_key: str | None = None) -> str:
    """Return an explicitly supplied key or load FRED_API_KEY from the environment."""
    load_dotenv(PROJECT_ROOT / ".env")
    resolved_key = api_key or os.getenv("FRED_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Add it to .env or export it in the shell."
        )
    return resolved_key


class FredSeriesNotFoundError(RuntimeError):
    """Raised when FRED returns a definitive "series does not exist" 400.

    Not every BLS-published CPI series is mirrored on FRED (BLS's own item
    catalog is the authority on what's published at all, not FRED's), so
    this is a real, permanent outcome for a handful of item-level series --
    distinct from a transient failure, and never worth retrying."""


def fetch_fred_observations(
    series_id: str,
    api_key: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 30.0,
    max_attempts: int = 3,
) -> Sequence[Mapping[str, Any]]:
    """Fetch all available observations for one FRED series.

    Retries transient request failures a couple of times with a short
    backoff -- fetching the full item-level frontier is now ~280 sequential
    calls in one run, and an occasional dropped connection or momentary
    rate-limit response shouldn't fail the whole pipeline run. A definitive
    "series does not exist" response is not retried."""
    client = session or requests.Session()
    last_error: requests.RequestException | None = None
    response = None
    for attempt in range(max_attempts):
        try:
            response = client.get(
                FRED_OBSERVATIONS_URL,
                params={
                    "series_id": series_id,
                    "api_key": api_key,
                    "file_type": "json",
                    "frequency": "m",
                    "aggregation_method": "avg",
                },
                timeout=timeout,
            )
            if response.status_code == 400 and "does not exist" in response.text:
                raise FredSeriesNotFoundError(
                    f"FRED has no series {series_id}"
                )
            response.raise_for_status()
            last_error = None
            break
        except FredSeriesNotFoundError:
            raise
        except requests.RequestException as error:
            last_error = error
            if attempt < max_attempts - 1:
                time.sleep(2**attempt)
    if last_error is not None:
        raise RuntimeError(
            f"FRED request failed for series {series_id}; credentials were redacted."
        ) from None

    payload = response.json()
    observations = payload.get("observations")
    if not isinstance(observations, list):
        message = payload.get("error_message", "response contains no observations")
        raise ValueError(f"Invalid FRED response for {series_id}: {message}")
    return observations


def normalize_fred_series(
    observations: Sequence[Mapping[str, Any]],
    *,
    series_id: str,
    series_type: str,
    component: str,
) -> pd.DataFrame:
    """Normalize FRED observations into the common CPI panel schema."""
    frame = pd.DataFrame(observations)
    if not {"date", "value"}.issubset(frame.columns):
        raise ValueError("FRED observations must contain date and value fields")

    frame = frame.loc[:, ["date", "value"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"])

    country = COUNTRIES["US"]
    frame["country"] = country["name"]
    frame["country_code"] = "US"
    frame["region"] = country["region"]
    frame["series_type"] = series_type
    frame["series_role"] = series_role("US", series_type, component)
    frame["component"] = component
    frame["source"] = "FRED"
    frame["source_dataset"] = "FRED CPI-U"
    frame["source_series_id"] = series_id
    frame["frequency"] = "monthly"
    frame["unit"] = "index"
    frame["index_base"] = "1982-1984=100"
    frame["seasonal_adjustment"] = "seasonally_adjusted"
    frame["observation_status"] = "not_provided"

    frame = frame.loc[:, CPI_SCHEMA_COLUMNS].sort_values("date").reset_index(drop=True)
    validate_cpi_schema(frame)
    return frame


def get_bls_api_key(api_key: str | None = None) -> str:
    """Return an explicitly supplied key or load BLS_API_KEY from the environment."""
    load_dotenv(PROJECT_ROOT / ".env")
    resolved_key = api_key or os.getenv("BLS_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "BLS_API_KEY is not set. Add it to .env or export it in the shell."
        )
    return resolved_key


def fetch_bls_series_batch(
    series_ids: Sequence[str],
    api_key: str,
    *,
    start_year: int,
    end_year: int,
    session: requests.Session | None = None,
    timeout: float = 30.0,
    max_attempts: int = 3,
) -> dict[str, list[Mapping[str, Any]]]:
    """Fetch up to 25 BLS series over one <= 20-year window in a single
    call. A window before a series' own history returns an empty result,
    not an error -- see BLS_EARLIEST_YEAR. Retries transient request
    failures a couple of times with a short backoff, the same as
    ``fetch_fred_observations`` -- fetching the full frontier is many
    sequential batched calls, and an occasional dropped connection
    shouldn't fail the whole run."""
    if len(series_ids) > BLS_SERIES_BATCH_SIZE:
        raise ValueError(f"BLS API accepts at most {BLS_SERIES_BATCH_SIZE} series per request")
    client = session or requests.Session()
    last_error: requests.RequestException | None = None
    response = None
    for attempt in range(max_attempts):
        try:
            response = client.post(
                BLS_TIMESERIES_URL,
                json={
                    "seriesid": list(series_ids),
                    "startyear": str(start_year),
                    "endyear": str(end_year),
                    "registrationkey": api_key,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            last_error = None
            break
        except requests.RequestException as error:
            last_error = error
            if attempt < max_attempts - 1:
                time.sleep(2**attempt)
    if last_error is not None:
        raise RuntimeError(
            "BLS API request failed; credentials were redacted."
        ) from None

    payload = response.json()
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise ValueError(f"BLS API request did not succeed: {payload.get('message')}")
    return {
        series["seriesID"]: series["data"]
        for series in payload.get("Results", {}).get("series", [])
    }


def normalize_bls_series(
    observations: Sequence[Mapping[str, Any]],
    *,
    series_id: str,
    series_type: str,
    component: str,
) -> pd.DataFrame:
    """Normalize BLS public-API observations into the common CPI panel
    schema. Monthly periods only (``M01``-``M12``) -- BLS's annual-average
    ``M13`` rows are dropped, matching FRED's monthly-only series."""
    frame = pd.DataFrame(observations)
    if frame.empty:
        return frame.reindex(columns=CPI_SCHEMA_COLUMNS)
    if not {"year", "period", "value"}.issubset(frame.columns):
        raise ValueError("BLS observations must contain year, period, and value fields")

    frame = frame.loc[frame["period"].isin(BLS_MONTH_PERIODS), ["year", "period", "value"]].copy()
    frame["date"] = pd.to_datetime(
        frame["year"] + "-" + frame["period"].str.removeprefix("M"), format="%Y-%m", errors="coerce"
    )
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"])

    country = COUNTRIES["US"]
    frame["country"] = country["name"]
    frame["country_code"] = "US"
    frame["region"] = country["region"]
    frame["series_type"] = series_type
    frame["series_role"] = series_role("US", series_type, component)
    frame["component"] = component
    frame["source"] = "BLS"
    frame["source_dataset"] = "BLS Public Data API"
    frame["source_series_id"] = series_id
    frame["frequency"] = "monthly"
    frame["unit"] = "index"
    frame["index_base"] = "1982-1984=100"
    frame["seasonal_adjustment"] = (
        "seasonally_adjusted" if series_id.startswith("CUSR") else "not_seasonally_adjusted"
    )
    frame["observation_status"] = "not_provided"

    frame = frame.loc[:, CPI_SCHEMA_COLUMNS].sort_values("date").reset_index(drop=True)
    validate_cpi_schema(frame)
    return frame


def fetch_bls_cpi_series(
    series_registry: Mapping[str, Mapping[str, str]],
    api_key: str | None = None,
    *,
    session: requests.Session | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Fetch every series in ``series_registry`` from BLS's public API,
    batching 25 series x a fixed set of <= 20-year windows per call."""
    resolved_key = get_bls_api_key(api_key)
    resolved_end_year = end_year or pd.Timestamp.today().year
    series_ids = list(series_registry)
    frames = []

    for batch_start in range(0, len(series_ids), BLS_SERIES_BATCH_SIZE):
        batch = series_ids[batch_start : batch_start + BLS_SERIES_BATCH_SIZE]
        by_series: dict[str, list[Mapping[str, Any]]] = {series_id: [] for series_id in batch}
        for start_year, window_end_year in _bls_year_windows(resolved_end_year):
            window = fetch_bls_series_batch(
                batch,
                resolved_key,
                start_year=start_year,
                end_year=window_end_year,
                session=session,
            )
            for series_id, observations in window.items():
                by_series[series_id].extend(observations)

        for series_id, observations in by_series.items():
            frame = normalize_bls_series(
                observations,
                series_id=series_id,
                **series_registry[series_id],
            )
            if not frame.empty:
                frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=CPI_SCHEMA_COLUMNS)
    combined = pd.concat(frames, ignore_index=True).sort_values(["date", "series_type"]).reset_index(
        drop=True
    )
    validate_cpi_schema(combined)
    return combined


def fetch_us_cpi(
    api_key: str | None = None,
    *,
    session: requests.Session | None = None,
    bls_api_key: str | None = None,
    bls_session: requests.Session | None = None,
    fetch_bls_fallback: bool = True,
) -> pd.DataFrame:
    """Fetch US headline CPI, core CPI, and their Table A components.

    A series ID resolved from the BLS item map but not actually mirrored on
    FRED (FRED curates a subset of BLS's published CPI series) falls back to
    BLS's own public API, which is authoritative for anything in cu.series.
    Set ``fetch_bls_fallback=False`` to skip that fallback and drop those
    series instead -- the item decomposition's reconciliation row already
    accounts for any tree label that has no matched data."""
    resolved_key = get_fred_api_key(api_key)
    frames = []
    missing_series: list[str] = []

    for series_id, metadata in US_CPI_SERIES.items():
        try:
            observations = fetch_fred_observations(
                series_id,
                resolved_key,
                session=session,
            )
        except FredSeriesNotFoundError:
            missing_series.append(series_id)
            continue
        frames.append(
            normalize_fred_series(
                observations,
                series_id=series_id,
                **metadata,
            )
        )

    if missing_series:
        if fetch_bls_fallback:
            print(
                f"{len(missing_series)} series are not published on FRED; "
                "fetching them from BLS's public API instead."
            )
            bls_registry = {
                series_id: US_CPI_SERIES[series_id] for series_id in missing_series
            }
            bls_frame = fetch_bls_cpi_series(
                bls_registry, bls_api_key, session=bls_session
            )
            if not bls_frame.empty:
                frames.append(bls_frame)
            still_missing = set(missing_series) - set(bls_frame["source_series_id"])
            if still_missing:
                print(
                    f"{len(still_missing)} series had no data from either source: "
                    f"{', '.join(sorted(still_missing))}"
                )
        else:
            print(
                f"Skipped {len(missing_series)} series not published on FRED: "
                f"{', '.join(missing_series)}"
            )

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "series_type"])
        .reset_index(drop=True)
    )
    validate_cpi_schema(combined)
    return combined


def save_us_cpi(
    frame: pd.DataFrame,
    output_path: str | Path = FRED_US_CPI_PATH,
) -> Path:
    """Validate and save normalized US CPI data as Parquet."""
    validate_cpi_schema(frame)
    return atomic_write_parquet(frame, output_path, columns=CPI_SCHEMA_COLUMNS)


def persist_us_cpi(
    frame: pd.DataFrame,
    output_path: str | Path = FRED_US_CPI_PATH,
) -> Path:
    destination = save_us_cpi(frame, output_path)
    update_ingestion_manifest(
        FRED_DATASET_ID,
        frame,
        destination,
        source_updated_at=frame.attrs.get("source_updated_at"),
    )
    return destination


def run(output_path: str | Path = FRED_US_CPI_PATH) -> Path:
    frame = fetch_us_cpi()
    return persist_us_cpi(frame, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch US headline and core CPI data from FRED."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FRED_US_CPI_PATH,
        help=f"Parquet output path (default: {FRED_US_CPI_PATH})",
    )
    args = parser.parse_args()

    destination = run(args.output)
    print(f"Saved US CPI data to {destination}")


if __name__ == "__main__":
    main()
