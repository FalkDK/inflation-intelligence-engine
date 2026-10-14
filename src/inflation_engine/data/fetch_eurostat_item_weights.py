"""Fetch annual COICOP-level HICP item weights for the euro-area item
decomposition (see inflation_engine.features.eurozone_item_decomposition).

A sibling to fetch_eurostat_weights.py, which instead collapses the same
underlying Eurostat weight table (``prc_hicp_inw``) into five coarse
harmonised buckets for decomposition.py.
"""

import argparse
from pathlib import Path

import pandas as pd
import requests

from inflation_engine.data.fetch_eurostat_items import HICP_ITEM_GEOGRAPHIES
from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.jsonstat import parse_jsonstat_observations
from inflation_engine.utils.paths import EUROSTAT_HICP_ITEM_WEIGHTS_PATH


EUROSTAT_HICP_ITEM_WEIGHT_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "prc_hicp_inw"
)
# Only the true additive COICOP hierarchy -- see fetch_eurostat_items.py for
# why the "coicop" dimension needs this filter.
_COICOP_HIERARCHY_PATTERN = r"CP\d+"
_ROOT_CODE = "CP00"
_ROOT_WEIGHT_TOLERANCE = 0.5

WEIGHT_COLUMNS = [
    "country_code",
    "coicop_code",
    "label",
    "weight_pct",
    "weight_reference_date",
    "weight_basis_year",
]


def fetch_eurostat_item_weight_dataset(
    geography: str,
    *,
    since_year: int = 2015,
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> dict:
    """Fetch one geography's full annual COICOP weight history."""
    client = session or requests.Session()
    params = [
        ("lang", "en"),
        ("geo", geography),
        ("sinceTimePeriod", str(since_year)),
    ]
    try:
        response = client.get(
            EUROSTAT_HICP_ITEM_WEIGHT_URL,
            params=params,
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError(
            f"Eurostat HICP item weight request failed for {geography}"
        ) from error


def _validate_item_weights(frame: pd.DataFrame) -> None:
    roots = frame.loc[frame["coicop_code"] == _ROOT_CODE]
    totals = roots.set_index(["country_code", "weight_basis_year"])["weight_pct"]
    bad = totals[(totals - 100.0).abs() > _ROOT_WEIGHT_TOLERANCE]
    if not bad.empty:
        raise ValueError(
            f"Eurostat HICP item weights for {_ROOT_CODE} are not ~100%: "
            f"{bad.to_dict()}"
        )


def normalize_eurostat_item_weights(datasets: dict[str, dict]) -> pd.DataFrame:
    """Map each geography's JSON-stat weight response into a long frame."""
    frames = []
    for country_code, dataset in datasets.items():
        observations = parse_jsonstat_observations(dataset)
        required_columns = {"coicop", "geo", "time", "value"}
        missing_columns = required_columns.difference(observations.columns)
        if missing_columns:
            raise ValueError(
                "Eurostat HICP item weight response is missing required "
                f"columns: {sorted(missing_columns)}"
            )
        observations = observations.loc[
            observations["coicop"].str.fullmatch(_COICOP_HIERARCHY_PATTERN)
        ].copy()
        observations["value"] = pd.to_numeric(
            observations["value"], errors="coerce"
        )
        observations = observations.dropna(subset=["value"])
        labels = dataset["dimension"]["coicop"]["category"]["label"]
        observations["label"] = observations["coicop"].map(labels)
        observations["country_code"] = country_code
        observations["weight_pct"] = observations["value"] / 10.0
        observations["weight_basis_year"] = observations["time"].astype(int)
        observations["weight_reference_date"] = pd.to_datetime(
            observations["weight_basis_year"].astype(str) + "-01-01"
        )
        frames.append(observations)

    if not frames:
        raise ValueError("No Eurostat HICP item weight geographies were requested")
    frame = pd.concat(frames, ignore_index=True).rename(
        columns={"coicop": "coicop_code"}
    )
    frame = (
        frame.loc[:, WEIGHT_COLUMNS]
        .sort_values(["country_code", "weight_basis_year", "coicop_code"])
        .reset_index(drop=True)
    )
    _validate_item_weights(frame)
    return frame


def fetch_eurostat_item_weights(
    *,
    since_year: int = 2015,
    session: requests.Session | None = None,
    geographies: dict[str, str] = HICP_ITEM_GEOGRAPHIES,
) -> pd.DataFrame:
    datasets = {
        country_code: fetch_eurostat_item_weight_dataset(
            geography, since_year=since_year, session=session
        )
        for country_code, geography in geographies.items()
    }
    return normalize_eurostat_item_weights(datasets)


def save_eurostat_item_weights(
    frame: pd.DataFrame,
    output_path: str | Path = EUROSTAT_HICP_ITEM_WEIGHTS_PATH,
) -> Path:
    return atomic_write_parquet(frame, output_path, columns=WEIGHT_COLUMNS)


def run(
    output_path: str | Path = EUROSTAT_HICP_ITEM_WEIGHTS_PATH,
    *,
    since_year: int = 2015,
) -> Path:
    return save_eurostat_item_weights(
        fetch_eurostat_item_weights(since_year=since_year), output_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch annual COICOP-level HICP item weights from Eurostat."
    )
    parser.add_argument("--since-year", type=int, default=2015)
    parser.add_argument("--output", type=Path, default=EUROSTAT_HICP_ITEM_WEIGHTS_PATH)
    args = parser.parse_args()
    destination = run(args.output, since_year=args.since_year)
    frame = pd.read_parquet(destination)
    print(f"Saved Eurostat HICP item weights to {destination}")
    print(
        frame.groupby("country_code")["weight_basis_year"]
        .agg(["min", "max", "nunique"])
        .to_string()
    )


if __name__ == "__main__":
    main()
