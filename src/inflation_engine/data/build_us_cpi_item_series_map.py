"""Build the US CPI item -> BLS/FRED series-ID crosswalk from BLS's own
published reference tables.

This replaces the old ``us_cpi_item_series_map.csv``, which was a ~50-row
map built once via manual, one-off matching against FRED's series-search
API. Two official BLS bulk-download tables make the same join deterministic
and far more complete:

- ``cu.item``: every CPI item code and its published name.
- ``cu.series``: every published CPI series, keyed by area code, item code,
  and seasonal-adjustment flag -- so for each item we can tell whether a
  seasonally adjusted series exists at the U.S. city average level (area
  ``0000``), and fall back to the not-seasonally-adjusted one when it
  doesn't.

The weight tree itself is unchanged: it still comes from BLS's Table 1
relative-importance workbook via ``load_us_weight_tree`` in
``us_item_decomposition.py``. This module only replaces how each tree label
gets matched to a series ID, lifting weighted-basket coverage of the US item
frontier from the ~76% the hand-curated map reached to effectively the
entire tree BLS discloses a standalone published index for.
"""

import argparse
import io
import re
from pathlib import Path

import pandas as pd
import requests

from inflation_engine.features.us_item_decomposition import load_us_weight_tree
from inflation_engine.utils.paths import (
    BLS_CPI_RELATIVE_IMPORTANCE_PATH,
    US_CPI_ITEM_SERIES_MAP_PATH,
)


CU_ITEM_URL = "https://download.bls.gov/pub/time.series/cu/cu.item"
CU_SERIES_URL = "https://download.bls.gov/pub/time.series/cu/cu.series"
REQUEST_HEADERS = {
    # BLS's bulk-download endpoint 403s any User-Agent that doesn't look like
    # a browser -- a bare tool/contact string (the more typical, more
    # identifiable convention) is rejected outright.
    "User-Agent": (
        "Mozilla/5.0 (compatible; inflation-intelligence-engine research fetch)"
    )
}
US_CITY_AVERAGE_AREA_CODE = "0000"
MONTHLY_PERIODICITY_CODE = "R"
SEASONAL_PREFERENCE_RANK = {"S": 0, "U": 1}
SERIES_MAP_COLUMNS = ["level", "label", "weight_pct", "series_id"]


def _bls_get(url: str, *, session: requests.Session | None = None, timeout: float = 30.0) -> str:
    client = session or requests.Session()
    response = client.get(url, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_cu_item(text: str) -> pd.DataFrame:
    """Parse BLS's ``cu.item`` reference table: item_code -> item_name."""
    frame = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    frame.columns = [column.strip() for column in frame.columns]
    frame["item_code"] = frame["item_code"].str.strip()
    frame["item_name"] = frame["item_name"].str.strip()
    return frame.loc[:, ["item_code", "item_name"]]


def parse_cu_series(text: str) -> pd.DataFrame:
    """Parse BLS's ``cu.series`` catalog of every published CPI series."""
    frame = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    frame.columns = [column.strip() for column in frame.columns]
    for column in frame.columns:
        frame[column] = frame[column].str.strip()
    return frame


def fetch_cu_item(*, session: requests.Session | None = None) -> pd.DataFrame:
    return parse_cu_item(_bls_get(CU_ITEM_URL, session=session))


def fetch_cu_series(*, session: requests.Session | None = None) -> pd.DataFrame:
    return parse_cu_series(_bls_get(CU_SERIES_URL, session=session))


def normalize_label(label: str) -> str:
    """Normalize cosmetic differences between the weight-tree label and
    BLS's own item name (curly apostrophes, an Oxford comma before "and")."""
    label = label.replace("’", "'").replace("‘", "'")
    label = re.sub(r"\s*,\s*and\s+", " and ", label)
    return re.sub(r"\s+", " ", label).strip()


def select_preferred_series(cu_series: pd.DataFrame, *, min_end_year: int) -> dict[str, str]:
    """One series ID per item code: the U.S.-city-average, monthly series
    still being published (``end_year`` at or after ``min_end_year``),
    preferring seasonally adjusted (S) over not seasonally adjusted (U)
    when both exist."""
    candidates = cu_series.loc[
        (cu_series["area_code"] == US_CITY_AVERAGE_AREA_CODE)
        & (cu_series["periodicity_code"] == MONTHLY_PERIODICITY_CODE)
        & (cu_series["end_year"].astype(int) >= min_end_year)
    ].copy()
    candidates["pref_rank"] = candidates["seasonal"].map(SEASONAL_PREFERENCE_RANK)
    best = candidates.sort_values(["item_code", "pref_rank"]).drop_duplicates(
        "item_code", keep="first"
    )
    return dict(zip(best["item_code"], best["series_id"]))


def slugify_label(label: str) -> str:
    """A stable, human-readable identifier for the raw panel's ``component``
    column -- e.g. "Owners' equivalent rent of primary residence" ->
    "owners_equivalent_rent_of_primary_residence"."""
    slug = label.lower().replace("'", "")
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug


def build_us_cpi_item_series_map(
    tree: pd.DataFrame,
    cu_item: pd.DataFrame,
    cu_series: pd.DataFrame,
    *,
    min_end_year: int,
) -> pd.DataFrame:
    """Join the BLS weight tree to a published series ID per label.

    Labels with no matching item code, or whose item has no currently
    published U.S.-city-average monthly series (BLS's "Unsampled ..."
    placeholders and a handful of other unindexed line items), are dropped;
    ``select_us_frontier`` in ``us_item_decomposition.py`` already knows how
    to cascade their weight up to the nearest matched ancestor."""
    name_to_code = dict(zip(cu_item["item_name"], cu_item["item_code"]))
    normalized_to_code = {
        normalize_label(name): code for name, code in name_to_code.items()
    }
    code_to_series = select_preferred_series(cu_series, min_end_year=min_end_year)

    rows = []
    for row in tree.itertuples(index=False):
        code = name_to_code.get(row.label) or normalized_to_code.get(
            normalize_label(row.label)
        )
        series_id = code_to_series.get(code) if code else None
        if series_id:
            rows.append(
                {
                    "level": int(row.level),
                    "label": row.label,
                    "weight_pct": float(row.weight_pct),
                    "series_id": series_id,
                }
            )
    if not rows:
        raise ValueError("No BLS weight-tree labels matched a published series ID")
    return (
        pd.DataFrame(rows, columns=SERIES_MAP_COLUMNS)
        .sort_values(["level", "label"], kind="stable")
        .reset_index(drop=True)
    )


def save_us_cpi_item_series_map(
    frame: pd.DataFrame,
    output_path: str | Path = US_CPI_ITEM_SERIES_MAP_PATH,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, SERIES_MAP_COLUMNS].to_csv(destination, index=False)
    return destination


def run(
    *,
    weights_path: str | Path = BLS_CPI_RELATIVE_IMPORTANCE_PATH,
    output_path: str | Path = US_CPI_ITEM_SERIES_MAP_PATH,
    session: requests.Session | None = None,
) -> Path:
    tree = load_us_weight_tree(weights_path)
    # A series still published as of the year before the weight table's own
    # reference date is safely "currently active" without hardcoding a year.
    min_end_year = int(tree["weight_reference_date"].dt.year.max()) - 1
    cu_item = fetch_cu_item(session=session)
    cu_series = fetch_cu_series(session=session)
    frame = build_us_cpi_item_series_map(
        tree, cu_item, cu_series, min_end_year=min_end_year
    )
    return save_us_cpi_item_series_map(frame, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the US CPI item series-ID map from BLS's cu.item/"
            "cu.series reference tables."
        )
    )
    parser.add_argument("--weights", type=Path, default=BLS_CPI_RELATIVE_IMPORTANCE_PATH)
    parser.add_argument("--output", type=Path, default=US_CPI_ITEM_SERIES_MAP_PATH)
    args = parser.parse_args()

    destination = run(weights_path=args.weights, output_path=args.output)
    print(f"Saved US CPI item series map to {destination}")


if __name__ == "__main__":
    main()
