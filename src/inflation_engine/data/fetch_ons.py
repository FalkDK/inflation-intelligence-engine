import argparse
import csv
import re
from collections.abc import Mapping
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from inflation_engine.utils.components import unique_component_slugs
from inflation_engine.utils.config import COUNTRIES, series_role
from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.manifest import update_ingestion_manifest
from inflation_engine.utils.paths import ONS_UK_CPI_PATH
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS, validate_cpi_schema


ONS_GENERATOR_URL = "https://www.ons.gov.uk/generator"
ONS_MM23_CSV_URL = (
    "https://www.ons.gov.uk/file?uri=%2Feconomy%2Finflationandpriceindices"
    "%2Fdatasets%2Fconsumerpriceindices%2Fcurrent%2Fmm23.csv"
)
ONS_DATASET = "MM23"
ONS_UK_DATASET_ID = "ons_uk_cpi"

UK_CPI_SERIES = {
    "D7BT": {
        "series_type": "headline",
        "component": "all_items",
    },
    "DKC6": {
        "series_type": "core",
        "component": "all_items_less_food_energy_alcohol_tobacco",
    },
    "DK9O": {
        "series_type": "headline",
        "component": "food_including_alcohol_tobacco",
    },
    "DK9T": {
        "series_type": "headline",
        "component": "energy",
    },
    "DK9J": {
        "series_type": "headline",
        "component": "non_energy_industrial_goods",
    },
    "D7F5": {
        "series_type": "headline",
        "component": "services",
    },
    "D7CE": {
        "series_type": "headline",
        "component": "actual_rentals_for_housing",
    },
}

# Detailed COICOP-coded CPI INDEX series (e.g. "01.1.1.1 Rice") live in the same
# MM23 database alongside the seven special aggregates above. Titles are parsed
# to discover them so the fetch stays in sync with whatever ONS currently
# publishes, rather than hand-maintaining ~300 series IDs.
ONS_ITEM_CODE_PATTERN = re.compile(
    r"^CPI INDEX\s*([0-9]+(?:\.[0-9]+)*(?:/[0-9]+)?[A-Za-z]?)\s*[:]?\s*(.*?)\s*2015=100\s*$",
    re.IGNORECASE,
)


def fetch_ons_series_csv(
    series_id: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> str:
    """Fetch one current ONS MM23 time-series CSV."""
    if series_id not in UK_CPI_SERIES:
        raise ValueError(f"Unsupported ONS CPI series: {series_id}")

    client = session or requests.Session()
    try:
        response = client.get(
            ONS_GENERATOR_URL,
            params={
                "format": "csv",
                "uri": (
                    "/economy/inflationandpriceindices/timeseries/"
                    f"{series_id.lower()}/{ONS_DATASET.lower()}"
                ),
            },
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            f"ONS CPI request failed for series {series_id}"
        ) from error
    return response.text


def fetch_ons_database_csv(
    *,
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> str:
    """Fetch the full current MM23 CSV once for all required CPI series."""
    client = session or requests.Session()
    try:
        response = client.get(ONS_MM23_CSV_URL, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError("ONS MM23 database request failed") from error
    return response.text


def discover_ons_item_series(database_csv: str) -> dict[str, tuple[str, str]]:
    """Return {coicop_code: (cdid, description)} for every detailed CPI INDEX
    series with a parseable COICOP code in the MM23 database."""
    rows = list(csv.reader(StringIO(database_csv)))
    if len(rows) < 2 or rows[0][0] != "Title" or rows[1][0] != "CDID":
        raise ValueError("ONS MM23 database has an unexpected layout")
    titles, cdids = rows[0][1:], rows[1][1:]
    codes: dict[str, tuple[str, str]] = {}
    for cdid, title in zip(cdids, titles):
        if not title.startswith("CPI INDEX") or cdid in UK_CPI_SERIES:
            continue
        match = ONS_ITEM_CODE_PATTERN.match(title.strip())
        if not match:
            continue
        code = match.group(1)
        if code in codes:
            continue
        codes[code] = (cdid, match.group(2).strip())
    if not codes:
        raise ValueError("Discovered no detailed ONS CPI INDEX series")
    return codes


def fetch_ons_item_series(
    database_csv: str,
    item_codes: Mapping[str, tuple[str, str]],
) -> pd.DataFrame:
    """Normalize every discovered detailed CPI INDEX series into CPI schema rows.

    ``component`` is a human-readable slug; ``source_series_id`` is set to the
    COICOP code (e.g. "01.1.1.1"), not the ONS CDID, because ONS uses unrelated
    CDID namespaces for the index and weight tables (e.g. index CDID L78H and
    weight CDID L83B both describe "01.1.1.1 Rice"). Using the COICOP code lets
    the item decomposition module join index values to weights directly, the
    same way Denmark's source_series_id already doubles as its COICOP code.
    """
    labels = {code: description for code, (_, description) in item_codes.items()}
    slugs = unique_component_slugs(list(item_codes), labels)
    frames = []
    for code, (cdid, _) in item_codes.items():
        series_csv = extract_ons_database_series_csv(database_csv, series_id=cdid)
        frame = normalize_ons_series(
            series_csv,
            expected_series_id=cdid,
            series_type="headline",
            component=slugs[code],
        )
        frame["source_series_id"] = code
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def extract_ons_database_series_csv(
    database_csv: str,
    *,
    series_id: str,
) -> str:
    """Extract one conventional two-column series from transposed MM23 CSV."""
    rows = list(csv.reader(StringIO(database_csv)))
    if len(rows) < 8 or rows[0][0] != "Title" or rows[1][0] != "CDID":
        raise ValueError("ONS MM23 database has an unexpected layout")
    try:
        column = rows[1].index(series_id)
    except ValueError:
        raise ValueError(f"ONS MM23 database is missing series {series_id}") from None
    output = StringIO()
    writer = csv.writer(output)
    metadata_labels = {
        "Title": "Title",
        "CDID": "CDID",
        "PreUnit": "PreUnit",
        "Unit": "Unit",
        "Release Date": "Release date",
        "Next release": "Next release",
        "Important Notes": "Important notes",
    }
    for row in rows[:7]:
        writer.writerow([metadata_labels[row[0]], row[column]])
    writer.writerow(["Source dataset ID", ONS_DATASET])
    for row in rows[7:]:
        if len(row) > column and row[column].strip():
            writer.writerow([row[0], row[column]])
    return output.getvalue()


def normalize_ons_series(
    csv_data: str,
    *,
    expected_series_id: str,
    series_type: str,
    component: str,
) -> pd.DataFrame:
    """Normalize an ONS time-series CSV into the common CPI schema."""
    rows = list(csv.reader(StringIO(csv_data)))
    metadata = {
        row[0].strip(): row[1].strip()
        for row in rows
        if len(row) >= 2 and row[0].strip()
    }
    returned_series_id = metadata.get("CDID")
    if returned_series_id != expected_series_id:
        raise ValueError(
            "ONS response series ID does not match request: "
            f"expected {expected_series_id}, received {returned_series_id}"
        )
    if metadata.get("Source dataset ID") != ONS_DATASET:
        raise ValueError(
            f"Unexpected ONS source dataset: {metadata.get('Source dataset ID')}"
        )
    # ONS leaves "Unit" blank for most detailed item-level series (unlike the
    # seven special aggregates), so only reject an explicit mismatch; the
    # "2015=100" check below still confirms the index basis independently.
    unit = metadata.get("Unit", "")
    if unit and unit != "Index, base year = 100":
        raise ValueError(f"Unexpected ONS CPI unit: {unit}")
    if "2015=100" not in metadata.get("Title", ""):
        raise ValueError(f"Unexpected ONS CPI index base: {metadata.get('Title')}")

    observations = pd.DataFrame(
        [
            {"period": row[0].strip(), "value": row[1].strip()}
            for row in rows
            if len(row) >= 2
            and pd.notna(
                pd.to_datetime(
                    row[0].strip(),
                    format="%Y %b",
                    errors="coerce",
                )
            )
        ]
    )
    if observations.empty:
        raise ValueError(f"ONS response contains no monthly data: {expected_series_id}")

    observations["date"] = pd.to_datetime(
        observations["period"],
        format="%Y %b",
        errors="coerce",
    )
    observations["value"] = pd.to_numeric(
        observations["value"],
        errors="coerce",
    )
    observations = observations.dropna(subset=["date", "value"])

    country = COUNTRIES["GB"]
    observations["country"] = country["name"]
    observations["country_code"] = "GB"
    observations["region"] = country["region"]
    observations["series_type"] = series_type
    observations["series_role"] = series_role("GB", series_type, component)
    observations["component"] = component
    observations["source"] = "Office for National Statistics"
    observations["source_dataset"] = ONS_DATASET
    observations["source_series_id"] = expected_series_id
    observations["frequency"] = "monthly"
    observations["unit"] = "index"
    observations["index_base"] = "2015=100"
    observations["seasonal_adjustment"] = "not_seasonally_adjusted"
    observations["observation_status"] = "not_provided"

    frame = (
        observations.loc[:, CPI_SCHEMA_COLUMNS]
        .sort_values("date")
        .reset_index(drop=True)
    )
    validate_cpi_schema(frame)
    release_date = pd.to_datetime(
        metadata.get("Release date"),
        format="%d-%m-%Y",
        errors="coerce",
    )
    if pd.notna(release_date):
        frame.attrs["source_updated_at"] = release_date.isoformat()
    return frame


def fetch_uk_cpi(
    *,
    session: requests.Session | None = None,
    include_detailed_items: bool = True,
) -> pd.DataFrame:
    """Fetch and normalize UK CPI index histories.

    Always includes the seven special aggregates (headline, core, and the
    five HICP-comparable buckets). When ``include_detailed_items`` is set,
    also includes every detailed COICOP-coded CPI INDEX series discovered in
    the same MM23 database, for item-level breadth analysis.
    """
    frames = []
    source_updates = []
    database_csv = fetch_ons_database_csv(session=session)
    for series_id, metadata in UK_CPI_SERIES.items():
        csv_data = extract_ons_database_series_csv(
            database_csv, series_id=series_id
        )
        frame = normalize_ons_series(
            csv_data,
            expected_series_id=series_id,
            **metadata,
        )
        frames.append(frame)
        if frame.attrs.get("source_updated_at"):
            source_updates.append(frame.attrs["source_updated_at"])

    if include_detailed_items:
        item_codes = discover_ons_item_series(database_csv)
        frames.append(fetch_ons_item_series(database_csv, item_codes))

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["series_type", "date"])
        .reset_index(drop=True)
    )
    validate_cpi_schema(combined)
    if source_updates:
        combined.attrs["source_updated_at"] = max(
            pd.Timestamp(value) for value in source_updates
        ).isoformat()
    return combined


def save_uk_cpi(
    frame: pd.DataFrame,
    output_path: str | Path = ONS_UK_CPI_PATH,
) -> Path:
    """Validate and save normalized UK CPI data as Parquet."""
    validate_cpi_schema(frame)
    return atomic_write_parquet(frame, output_path, columns=CPI_SCHEMA_COLUMNS)


def persist_uk_cpi(
    frame: pd.DataFrame,
    output_path: str | Path = ONS_UK_CPI_PATH,
) -> Path:
    destination = save_uk_cpi(frame, output_path)
    update_ingestion_manifest(
        ONS_UK_DATASET_ID,
        frame,
        destination,
        source_updated_at=frame.attrs.get("source_updated_at"),
    )
    return destination


def latest_uk_cpi_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the latest observation for every downloaded UK CPI series."""
    validate_cpi_schema(frame)
    latest = (
        frame.sort_values("date")
        .groupby(["series_type", "component"], as_index=False)
        .tail(1)
    )
    return (
        latest.loc[:, ["series_type", "component", "date", "value"]]
        .sort_values("series_type")
        .reset_index(drop=True)
    )


def run(output_path: str | Path = ONS_UK_CPI_PATH) -> Path:
    frame = fetch_uk_cpi()
    return persist_uk_cpi(frame, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch UK headline and core CPI indices from ONS."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ONS_UK_CPI_PATH,
        help=f"Parquet output path (default: {ONS_UK_CPI_PATH})",
    )
    args = parser.parse_args()

    frame = fetch_uk_cpi()
    destination = persist_uk_cpi(frame, args.output)
    print(f"Saved UK CPI data to {destination}")
    print("\nLatest observations:")
    print(
        latest_uk_cpi_observations(frame).to_string(
            index=False,
            formatters={
                "date": lambda value: value.strftime("%Y-%m-%d"),
                "value": lambda value: f"{value:.2f}",
            },
        )
    )


if __name__ == "__main__":
    main()
