"""Fetch annual UK CPI weights for the five-bucket decomposition."""

import argparse
import csv
import re
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from inflation_engine.data.fetch_bls_weights import (
    BUCKETS,
    WEIGHT_COLUMNS,
    validate_weights,
)
from inflation_engine.data.fetch_ons import (
    ONS_GENERATOR_URL,
    fetch_ons_database_csv,
)
from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.paths import ONS_UK_CPI_WEIGHTS_PATH


ONS_WEIGHT_SERIES = {
    "food": "A9EW",
    "energy": "A9F3",
    "core_goods": "A9ER",
    "shelter": "CJVC",
}


def fetch_ons_weight_csv(
    series_id: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> str:
    client = session or requests.Session()
    try:
        response = client.get(
            ONS_GENERATOR_URL,
            params={
                "format": "csv",
                "uri": (
                    "/economy/inflationandpriceindices/timeseries/"
                    f"{series_id.lower()}/mm23"
                ),
            },
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            f"ONS CPI weight request failed for series {series_id}"
        ) from error
    return response.text


def parse_ons_annual_weight(csv_data: str, *, series_id: str) -> pd.Series:
    rows = list(csv.reader(StringIO(csv_data)))
    metadata = {
        row[0].strip(): row[1].strip()
        for row in rows
        if len(row) >= 2 and row[0].strip()
    }
    if metadata.get("CDID") != series_id:
        raise ValueError(f"ONS weight response does not match {series_id}")
    if metadata.get("Unit") != "Parts per 1000":
        raise ValueError(f"Unexpected ONS weight unit for {series_id}")
    observations = {
        int(row[0]): float(row[1])
        for row in rows
        if len(row) >= 2
        and re.fullmatch(r"\d{4}", row[0].strip())
        and row[1].strip()
    }
    if not observations:
        raise ValueError(f"ONS weight response contains no annual data: {series_id}")
    return pd.Series(observations, dtype=float).sort_index()


def parse_ons_database_weight(
    database_csv: str, *, series_id: str
) -> pd.Series:
    rows = list(csv.reader(StringIO(database_csv)))
    if len(rows) < 8 or rows[1][0] != "CDID":
        raise ValueError("ONS MM23 database has an unexpected layout")
    try:
        column = rows[1].index(series_id)
    except ValueError:
        raise ValueError(f"ONS MM23 database is missing series {series_id}") from None
    if rows[3][column] != "Parts per 1000":
        raise ValueError(f"Unexpected ONS weight unit for {series_id}")
    observations = {
        int(row[0]): float(row[column])
        for row in rows[7:]
        if len(row) > column
        and re.fullmatch(r"\d{4}", row[0].strip())
        and row[column].strip()
    }
    if not observations:
        raise ValueError(f"ONS weight response contains no annual data: {series_id}")
    return pd.Series(observations, dtype=float).sort_index()


def build_uk_weights(series: dict[str, pd.Series]) -> pd.DataFrame:
    missing = set(ONS_WEIGHT_SERIES).difference(series)
    if missing:
        raise ValueError(f"UK CPI weight series are missing: {sorted(missing)}")
    years = sorted(set.intersection(*(set(values.index) for values in series.values())))
    rows = []
    for year in years:
        effective_month = 2 if year >= 2017 else 1
        published = {
            bucket: float(values.loc[year]) / 10.0
            for bucket, values in series.items()
        }
        services = 100.0 - (
            published["food"] + published["energy"] + published["core_goods"]
        )
        weights = {
            **published,
            "core_services_ex_housing": services - published["shelter"],
        }
        for bucket in BUCKETS:
            rows.append(
                {
                    "reference_date": pd.Timestamp(year, effective_month, 1),
                    "effective_start": pd.Timestamp(year, effective_month, 1),
                    "effective_end": pd.Timestamp(year, 12, 1),
                    "country_code": "GB",
                    "index_family": "CPI",
                    "bucket": bucket,
                    "weight_pct": weights[bucket],
                    "source": "Office for National Statistics",
                    "source_url": (
                        "https://www.ons.gov.uk/economy/"
                        "inflationandpriceindices/datasets/"
                        "consumerpriceindices/current"
                    ),
                    "weight_basis_year": int(year),
                    "estimation_method": (
                        "all_items_less_food_energy_goods_and_rent"
                        if bucket == "core_services_ex_housing"
                        else (
                            "published_feb_dec_special_aggregate_weight"
                            if year >= 2017
                            else "published_annual_special_aggregate_weight"
                        )
                    ),
                }
            )
    frame = pd.DataFrame(rows, columns=WEIGHT_COLUMNS)
    validate_weights(frame)
    return frame


def fetch_uk_weights(
    *,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    database_csv = fetch_ons_database_csv(session=session)
    series = {
        bucket: parse_ons_database_weight(
            database_csv, series_id=series_id
        )
        for bucket, series_id in ONS_WEIGHT_SERIES.items()
    }
    return build_uk_weights(series)


def save_uk_weights(
    frame: pd.DataFrame,
    output_path: str | Path = ONS_UK_CPI_WEIGHTS_PATH,
) -> Path:
    validate_weights(frame)
    return atomic_write_parquet(frame, output_path, columns=WEIGHT_COLUMNS)


def run(
    output_path: str | Path = ONS_UK_CPI_WEIGHTS_PATH,
) -> Path:
    return save_uk_weights(fetch_uk_weights(), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch annual UK CPI weights for the decomposition."
    )
    parser.add_argument("--output", type=Path, default=ONS_UK_CPI_WEIGHTS_PATH)
    args = parser.parse_args()
    destination = run(args.output)
    frame = pd.read_parquet(destination)
    print(f"Saved ONS CPI weights to {destination}")
    print(
        frame.groupby("country_code")["weight_basis_year"]
        .agg(["min", "max", "nunique"])
        .to_string()
    )


if __name__ == "__main__":
    main()
