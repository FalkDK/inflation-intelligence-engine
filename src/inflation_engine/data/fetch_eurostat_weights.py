"""Fetch annual HICP weights for the harmonised five-bucket decomposition."""

import argparse
from pathlib import Path

import pandas as pd
import requests

from inflation_engine.data.fetch_bls_weights import (
    BUCKETS,
    WEIGHT_COLUMNS,
    validate_weights,
)
from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.jsonstat import parse_jsonstat_observations
from inflation_engine.utils.paths import EUROSTAT_HICP_WEIGHTS_PATH


EUROSTAT_HICP_WEIGHT_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "prc_hicp_iw"
)
EUROSTAT_WEIGHT_GEOGRAPHIES = {
    "AT": "AT",
    "BE": "BE",
    "CY": "CY",
    "DE": "DE",
    "DK": "DK",
    "EA20": "EA20",
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
    "SE": "SE",
    "SI": "SI",
    "SK": "SK",
}
REQUIRED_WEIGHT_ITEMS = ("TOTAL", "FOOD", "NRG", "IGD_NNRG", "SERV", "CP041")


def fetch_eurostat_weight_dataset(
    geography: str,
    *,
    since_year: int = 1996,
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> dict:
    """Fetch the small set of official annual weights needed for one geography."""
    client = session or requests.Session()
    params = [
        ("lang", "en"),
        ("geo", geography),
        *[("coicop18", item) for item in REQUIRED_WEIGHT_ITEMS],
        ("sinceTimePeriod", str(since_year)),
    ]
    try:
        response = client.get(
            EUROSTAT_HICP_WEIGHT_URL,
            params=params,
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError(
            f"Eurostat HICP weight request failed for {geography}"
        ) from error


def normalize_eurostat_weights(
    datasets: dict[str, dict],
) -> pd.DataFrame:
    """Map exhaustive HICP aggregates into five non-overlapping macro buckets."""
    rows = []
    source_url = EUROSTAT_HICP_WEIGHT_URL
    for country_code, dataset in datasets.items():
        observations = parse_jsonstat_observations(dataset)
        required_columns = {"coicop18", "geo", "time", "value"}
        missing = required_columns.difference(observations.columns)
        if missing:
            raise ValueError(
                f"Eurostat HICP weights are missing columns: {sorted(missing)}"
            )
        observations["value"] = pd.to_numeric(
            observations["value"], errors="coerce"
        )
        for year, annual in observations.groupby("time", sort=True):
            values = annual.set_index("coicop18")["value"]
            missing_items = set(REQUIRED_WEIGHT_ITEMS).difference(values.index)
            if missing_items:
                continue
            if abs(float(values["TOTAL"]) - 1000.0) > 0.5:
                raise ValueError(
                    f"Unexpected HICP total weight for {country_code} in {year}"
                )
            # FOOD, NRG, IGD_NNRG, and SERV are meant to be an exhaustive,
            # non-overlapping split of TOTAL; Greece's first published year
            # (1996) doesn't reconcile (sums to ~1015 instead of 1000), a
            # genuine data-quality gap in that earliest vintage rather than a
            # parsing bug -- skip a year here rather than build five buckets
            # that silently don't add up to the real basket.
            four_bucket_total = (
                float(values["FOOD"])
                + float(values["NRG"])
                + float(values["IGD_NNRG"])
                + float(values["SERV"])
            )
            if abs(four_bucket_total - float(values["TOTAL"])) > 0.5:
                continue
            weights = {
                "food": float(values["FOOD"]) / 10.0,
                "energy": float(values["NRG"]) / 10.0,
                "shelter": float(values["CP041"]) / 10.0,
                "core_goods": float(values["IGD_NNRG"]) / 10.0,
                "core_services_ex_housing": (
                    float(values["SERV"]) - float(values["CP041"])
                )
                / 10.0,
            }
            basis_year = int(year)
            for bucket in BUCKETS:
                rows.append(
                    {
                        "reference_date": pd.Timestamp(basis_year, 1, 1),
                        "effective_start": pd.Timestamp(basis_year, 1, 1),
                        "effective_end": pd.Timestamp(basis_year, 12, 1),
                        "country_code": country_code,
                        "index_family": "HICP",
                        "bucket": bucket,
                        "weight_pct": weights[bucket],
                        "source": "Eurostat",
                        "source_url": source_url,
                        "weight_basis_year": basis_year,
                        "estimation_method": (
                            "services_minus_actual_rentals"
                            if bucket == "core_services_ex_housing"
                            else "published_item_weight"
                        ),
                    }
                )
    frame = pd.DataFrame(rows, columns=WEIGHT_COLUMNS)
    if frame.empty:
        raise ValueError("Eurostat returned no complete annual HICP weights")
    validate_weights(frame)
    return frame.sort_values(
        ["country_code", "effective_start", "bucket"], kind="stable"
    ).reset_index(drop=True)


def fetch_eurostat_weights(
    *,
    since_year: int = 1996,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    datasets = {
        country_code: fetch_eurostat_weight_dataset(
            geography,
            since_year=since_year,
            session=session,
        )
        for country_code, geography in EUROSTAT_WEIGHT_GEOGRAPHIES.items()
    }
    return normalize_eurostat_weights(datasets)


def save_eurostat_weights(
    frame: pd.DataFrame,
    output_path: str | Path = EUROSTAT_HICP_WEIGHTS_PATH,
) -> Path:
    validate_weights(frame)
    return atomic_write_parquet(frame, output_path, columns=WEIGHT_COLUMNS)


def run(
    output_path: str | Path = EUROSTAT_HICP_WEIGHTS_PATH,
    *,
    since_year: int = 1996,
) -> Path:
    return save_eurostat_weights(
        fetch_eurostat_weights(since_year=since_year), output_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch annual HICP weights for the European decomposition."
    )
    parser.add_argument("--since-year", type=int, default=1996)
    parser.add_argument("--output", type=Path, default=EUROSTAT_HICP_WEIGHTS_PATH)
    args = parser.parse_args()
    destination = run(args.output, since_year=args.since_year)
    frame = pd.read_parquet(destination)
    print(f"Saved Eurostat HICP weights to {destination}")
    print(
        frame.groupby("country_code")["weight_basis_year"]
        .agg(["min", "max", "nunique"])
        .to_string()
    )


if __name__ == "__main__":
    main()
