"""Fetch annual US CPI-U relative-importance weights from BLS."""

import argparse
import re
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.paths import (
    BLS_US_CPI_WEIGHTS_PATH,
    BLS_US_CPI_WEIGHT_SNAPSHOT_PATH,
)


BLS_RELATIVE_IMPORTANCE_URL = (
    "https://www.bls.gov/cpi/tables/relative-importance/{year}.htm"
)
WEIGHT_COLUMNS = [
    "reference_date",
    "effective_start",
    "effective_end",
    "country_code",
    "index_family",
    "bucket",
    "weight_pct",
    "source",
    "source_url",
    "weight_basis_year",
    "estimation_method",
]
BUCKETS = [
    "food",
    "energy",
    "shelter",
    "core_goods",
    "core_services_ex_housing",
]
SOURCE_LABELS = {
    "food": "Food",
    "energy": "Energy",
    "shelter": "Shelter",
    "core_goods": "Commodities less food and energy commodities",
    "services_less_energy": "Services less energy services",
}


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._in_row = False
        self._in_cell = False
        self._row: list[str] = []
        self._cell: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._in_row = True
            self._row = []
        elif self._in_row and tag in {"th", "td"}:
            self._in_cell = True
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._in_row and self._in_cell and tag in {"th", "td"}:
            value = " ".join("".join(self._cell).split())
            self._row.append(value)
            self._in_cell = False
        elif self._in_row and tag == "tr":
            if self._row:
                self.rows.append(self._row)
            self._in_row = False


def _first_number(values: list[str]) -> float:
    for value in values:
        cleaned = value.replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            continue
    raise ValueError(f"BLS relative-importance row contains no number: {values}")


def validate_weights(frame: pd.DataFrame) -> None:
    missing = [column for column in WEIGHT_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"CPI weight data is missing columns: {missing}")
    if frame.empty:
        raise ValueError("CPI weight data cannot be empty")
    if frame[WEIGHT_COLUMNS].isnull().any().any():
        raise ValueError("CPI weight data cannot contain null values")
    if set(frame["bucket"]) != set(BUCKETS):
        raise ValueError("CPI weights must contain each canonical bucket once")
    if frame.duplicated(["effective_start", "country_code", "bucket"]).any():
        raise ValueError("CPI weight data contains duplicate bucket periods")
    if not np.isfinite(frame["weight_pct"]).all():
        raise ValueError("CPI weights must be finite")
    if (frame["weight_pct"] <= 0).any():
        raise ValueError("CPI weights must be greater than zero")
    totals = frame.groupby(["country_code", "effective_start"])["weight_pct"].sum()
    if not np.allclose(totals, 100.0, atol=0.05):
        raise ValueError("Canonical CPI bucket weights must sum to 100 percent")
    if (frame["effective_start"] > frame["effective_end"]).any():
        raise ValueError("CPI weight effective dates are invalid")


def parse_bls_relative_importance(
    html: str,
    *,
    reference_year: int,
    source_url: str | None = None,
) -> pd.DataFrame:
    """Parse the first (new-weight) CPI-U table on an annual BLS page."""
    parser = _TableParser()
    parser.feed(html)

    values: dict[str, float] = {}
    label_to_key = {label.casefold(): key for key, label in SOURCE_LABELS.items()}
    for row in parser.rows:
        if not row:
            continue
        key = label_to_key.get(row[0].casefold())
        if key is not None and key not in values:
            values[key] = _first_number(row[1:])

    missing = sorted(set(SOURCE_LABELS) - set(values))
    if missing:
        raise ValueError(f"BLS weight table is missing required rows: {missing}")

    basis_match = re.search(
        r"Table\s+1\s*\((\d{4})\s+Weights\)",
        html,
        flags=re.IGNORECASE,
    )
    if basis_match is None:
        raise ValueError("BLS weight table does not identify its weight basis year")

    weights = {
        "food": values["food"],
        "energy": values["energy"],
        "shelter": values["shelter"],
        "core_goods": values["core_goods"],
        "core_services_ex_housing": (
            values["services_less_energy"] - values["shelter"]
        ),
    }
    reference_date = pd.Timestamp(reference_year, 12, 1)
    effective_start = pd.Timestamp(reference_year + 1, 1, 1)
    effective_end = pd.Timestamp(reference_year + 1, 12, 1)
    url = source_url or BLS_RELATIVE_IMPORTANCE_URL.format(year=reference_year)
    frame = pd.DataFrame(
        [
            {
                "reference_date": reference_date,
                "effective_start": effective_start,
                "effective_end": effective_end,
                "country_code": "US",
                "index_family": "CPI-U",
                "bucket": bucket,
                "weight_pct": weight,
                "source": "U.S. Bureau of Labor Statistics",
                "source_url": url,
                "weight_basis_year": int(basis_match.group(1)),
                "estimation_method": (
                    "published_relative_importance"
                    if bucket != "core_services_ex_housing"
                    else "services_less_energy_minus_shelter"
                ),
            }
            for bucket, weight in weights.items()
        ],
        columns=WEIGHT_COLUMNS,
    )
    validate_weights(frame)
    return frame


def fetch_bls_relative_importance(
    reference_year: int,
    *,
    session: requests.Session | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    url = BLS_RELATIVE_IMPORTANCE_URL.format(year=reference_year)
    client = session or requests.Session()
    try:
        response = client.get(
            url,
            headers={"User-Agent": "inflation-intelligence-engine/0.1"},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            f"BLS relative-importance request failed for {reference_year}"
        ) from error
    return parse_bls_relative_importance(
        response.text,
        reference_year=reference_year,
        source_url=url,
    )


def save_bls_weights(
    frame: pd.DataFrame,
    output_path: str | Path = BLS_US_CPI_WEIGHTS_PATH,
) -> Path:
    validate_weights(frame)
    return atomic_write_parquet(frame, output_path, columns=WEIGHT_COLUMNS)


def load_weight_snapshot(
    path: str | Path = BLS_US_CPI_WEIGHT_SNAPSHOT_PATH,
) -> pd.DataFrame:
    """Load a source-controlled official snapshot when BLS blocks automation."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"BLS CPI weight snapshot does not exist: {source}")
    frame = pd.read_csv(
        source,
        parse_dates=["reference_date", "effective_start", "effective_end"],
    )
    validate_weights(frame)
    return frame.loc[:, WEIGHT_COLUMNS]


def run(
    output_path: str | Path = BLS_US_CPI_WEIGHTS_PATH,
    *,
    reference_year: int,
    snapshot_path: str | Path | None = None,
) -> Path:
    frame = (
        load_weight_snapshot(snapshot_path)
        if snapshot_path is not None
        else fetch_bls_relative_importance(reference_year)
    )
    return save_bls_weights(frame, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch annual US CPI-U relative-importance weights."
    )
    parser.add_argument("--reference-year", type=int, required=True)
    parser.add_argument(
        "--snapshot",
        type=Path,
        help=(
            "Use a reviewed local BLS snapshot when automated access is blocked."
        ),
    )
    parser.add_argument("--output", type=Path, default=BLS_US_CPI_WEIGHTS_PATH)
    args = parser.parse_args()
    destination = run(
        args.output,
        reference_year=args.reference_year,
        snapshot_path=args.snapshot,
    )
    print(f"Saved US CPI-U weights to {destination}")


if __name__ == "__main__":
    main()
