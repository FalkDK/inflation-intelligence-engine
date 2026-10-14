"""Additive US item-level CPI decomposition.

This module deliberately sits beside the existing five-bucket decomposition,
the same way danish_item_decomposition.py / uk_item_decomposition.py do for
Denmark and the UK. Two committed reference files drive it:

- ``bls_cpi_relative_importance_2024weights.xlsx``: BLS's own published
  "Relative Importance of Components in the CPI" table (Table 1), which
  gives an explicit additive hierarchy via an "Indent Level" column, but no
  machine-readable series ID for any row.
- ``us_cpi_item_series_map.csv``: a curated mapping from each weight-table
  label to the FRED/BLS series ID that publishes its actual price index
  (there is no official code linking the two BLS releases). This was built
  once via FRED's series-search API, restricted to exact-title matches
  against monthly (CUSR/CUUR) CPI-U series only -- average-price series
  (APU...) and semiannual-only series (CUUS...) are deliberately excluded.
  BLS does not publish a standalone index for every weighted line item (for
  example "Uncooked ground beef" has a weight but no published index of its
  own), so this mapping is necessarily partial; the frontier below falls
  back to the nearest matched ancestor wherever a subtree isn't fully
  covered, and whatever remains lands in the reconciliation row.
"""

import re
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from inflation_engine.features.calendar_yoy import calendar_yoy
from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.paths import (
    BLS_CPI_RELATIVE_IMPORTANCE_PATH,
    FRED_US_CPI_PATH,
    US_CPI_ITEM_SERIES_MAP_PATH,
    US_ITEM_DECOMPOSITION_PATH,
)


OUTPUT_COLUMNS = [
    "date",
    "country",
    "country_code",
    "source_component_code",
    "label",
    "hierarchy_level",
    "series_role",
    "weight_pct",
    "weight_reference_date",
    "weight_basis_year",
    "index_value",
    "inflation_yoy",
    "estimated_contribution_yoy_pp",
    "headline_yoy",
    "reconciliation_row",
]

_TITLE_PATTERN = re.compile(
    r"Table\s+1\s*\((\d{4})\s+Weights\).*?(\w+)\s+(\d{4})", re.IGNORECASE
)
_SECTION_BOUNDARY_LABEL = "Special aggregate indexes"
_WEIGHT_TOLERANCE = 1e-9


# Two rows in BLS's own published workbook are indented at the wrong depth
# relative to where their weight arithmetic says they belong -- verified
# against the actual declared weights, not guessed:
#   "Alcoholic beverages" (0.840) is published one level too deep: it reads
#   as a third child of "Food" alongside "Food at home" (8.325) and "Food
#   away from home" (5.373), which alone already sum to Food's full declared
#   13.698. It belongs one level shallower, as "Food"'s own sibling under
#   "Food and beverages" (14.539 = 13.698 + 0.840, an exact match).
#   "Information technology, hardware and services" (1.714) is published
#   one level too shallow: as a sibling of "Information and information
#   processing" it inflates "Communication"'s children to 4.959 against a
#   declared 3.244, but as that node's child it closes exactly (1.466
#   "Telephone services" + 1.714 = 3.180, matching the declared 3.181).
# Each correction shifts the named row -1/+1 levels, carrying along
# whatever is already nested beneath it (identified by original level, so
# the subtree's own internal shape is preserved) -- and closes what would
# otherwise be a ~1.7-point overcount once matched deeply enough to reach
# both branches.
_MISINDENTED_LABEL_DELTAS = {
    "Alcoholic beverages": -1,
    "Information technology, hardware and services": 1,
}


def _correct_misindented_branches(records: list[dict]) -> list[dict]:
    corrected = []
    shifting_original_level: int | None = None
    shift_delta = 0
    for record in records:
        record = dict(record)
        if shifting_original_level is not None:
            if record["level"] > shifting_original_level:
                record["level"] += shift_delta
            else:
                shifting_original_level = None
        if record["label"] in _MISINDENTED_LABEL_DELTAS:
            shift_delta = _MISINDENTED_LABEL_DELTAS[record["label"]]
            shifting_original_level = record["level"]
            record["level"] += shift_delta
        corrected.append(record)
    return corrected


def load_us_weight_tree(
    path: str | Path = BLS_CPI_RELATIVE_IMPORTANCE_PATH,
) -> pd.DataFrame:
    """Parse BLS's Table 1 into the real additive hierarchy only.

    The same sheet appends a second, unrelated "Special aggregate indexes"
    section after the real tree (overlapping combos like "All items less
    shelter"), which is cut off; "Expenditure category" (a header, not an
    item) and "All items" (the root -- already the Tier 1 headline) are
    dropped too, since neither is a component contribution. Two specific
    branches are also re-indented to match their own declared weights --
    see ``_MISINDENTED_LABELS``.
    """
    workbook = openpyxl.load_workbook(path, data_only=True)
    sheet = workbook["Table 1"]
    rows = list(sheet.iter_rows(min_row=1, max_row=sheet.max_row, values_only=True))

    title = str(rows[0][1])
    match = _TITLE_PATTERN.search(title)
    if not match:
        raise ValueError(f"Could not parse weight basis year from title: {title!r}")
    weight_basis_year = int(match.group(1))
    reference_date = pd.to_datetime(f"{match.group(2)} {match.group(3)}", format="%B %Y")

    records = []
    for row in rows:
        level, label, cpi_u = row[0], row[1], row[2]
        if not isinstance(level, int) or not isinstance(label, str) or not label.strip():
            continue
        label = label.strip()
        if label == _SECTION_BOUNDARY_LABEL:
            break
        if label in {"Expenditure category", "All items"}:
            continue
        weight_pct = pd.to_numeric(cpi_u, errors="coerce")
        if pd.isna(weight_pct):
            continue
        records.append({"level": level, "label": label, "weight_pct": float(weight_pct)})

    records = _correct_misindented_branches(records)

    tree = pd.DataFrame(records)
    if tree.empty:
        raise ValueError("Parsed no BLS relative-importance item rows")
    tree["weight_reference_date"] = reference_date
    tree["weight_basis_year"] = weight_basis_year
    return tree


def select_us_frontier(
    tree: pd.DataFrame,
    series_map: pd.DataFrame,
) -> pd.DataFrame:
    """Select the deepest available weight nodes with a matched series ID,
    falling back to the nearest matched ancestor wherever a subtree has no
    fully matched descendants (BLS does not publish a standalone index for
    every weighted line item)."""
    label_to_series = series_map.set_index("label")["series_id"].to_dict()

    records = tree.to_dict("records")
    for record in records:
        record["series_id"] = label_to_series.get(record["label"])

    def build_forest(records: list[dict]) -> list[dict]:
        stack: list[tuple[int, dict]] = []
        roots = []
        for record in records:
            node = {**record, "children": []}
            while stack and stack[-1][0] >= node["level"]:
                stack.pop()
            if stack:
                stack[-1][1]["children"].append(node)
            else:
                roots.append(node)
            stack.append((node["level"], node))
        return roots

    def resolve(node: dict) -> list[dict]:
        """Return parts partitioning this node's own weight, going as deep as
        matched. BLS frequently publishes a weight (and sometimes a series)
        for a parent whose listed children only partially cover it -- e.g.
        "Information and information processing" (weight 3.181) has only one
        matched child, "Telephone services" (1.466); the other 1.715pp has
        no separate line item, but the parent itself has a published series
        (CUSR0000SAE21) that can stand in for that remainder, cascading up
        only as far as needed rather than dropping straight to the single
        top-level reconciliation row."""
        parts: list[dict] = []
        covered = 0.0
        for child in node["children"]:
            child_parts = resolve(child)
            parts.extend(child_parts)
            covered += sum(part["weight_pct"] for part in child_parts)
        remainder = node["weight_pct"] - covered
        if remainder > _WEIGHT_TOLERANCE and node["series_id"]:
            residual = {**node, "weight_pct": remainder}
            if covered > _WEIGHT_TOLERANCE:
                residual["label"] = f"{node['label']} (unitemized remainder only)"
            parts.append(residual)
        return parts

    frontier_nodes = []
    for root in build_forest(records):
        frontier_nodes.extend(resolve(root))

    frontier = pd.DataFrame(frontier_nodes)
    if frontier.empty:
        raise ValueError("No BLS weight rows matched a usable series ID")
    return frontier.loc[
        :,
        ["level", "label", "weight_pct", "series_id", "weight_reference_date", "weight_basis_year"],
    ].rename(columns={"level": "hierarchy_level", "series_id": "source_component_code"})


def build_us_item_decomposition(
    raw_cpi: pd.DataFrame,
    weights_path: str | Path = BLS_CPI_RELATIVE_IMPORTANCE_PATH,
    series_map_path: str | Path = US_CPI_ITEM_SERIES_MAP_PATH,
) -> pd.DataFrame:
    """Build US item contributions while leaving the canonical buckets untouched."""
    required = {
        "date", "country", "country_code", "series_role", "component",
        "source_series_id", "value",
    }
    missing = required.difference(raw_cpi.columns)
    if missing:
        raise ValueError(f"US item panel is missing columns: {sorted(missing)}")

    tree = load_us_weight_tree(weights_path)
    series_map = pd.read_csv(series_map_path)
    frontier = select_us_frontier(tree, series_map)

    source = raw_cpi.loc[
        (raw_cpi["country_code"] == "US")
        & raw_cpi["source_series_id"].isin(frontier["source_component_code"])
    ].copy()
    source["date"] = pd.to_datetime(source["date"])
    source["index_value"] = pd.to_numeric(source["value"], errors="coerce")
    source = source.sort_values(["source_series_id", "date"]).reset_index(drop=True)
    source["inflation_yoy"] = calendar_yoy(source, group_column="source_series_id")

    headline = raw_cpi.loc[
        (raw_cpi["country_code"] == "US") & (raw_cpi["source_series_id"] == "CPIAUCSL")
    ].copy()
    headline["date"] = pd.to_datetime(headline["date"])
    headline["index_value"] = pd.to_numeric(headline["value"], errors="coerce")
    headline = headline.sort_values("date").reset_index(drop=True)
    headline["source_series_id"] = "CPIAUCSL"
    headline["headline_yoy"] = calendar_yoy(headline, group_column="source_series_id")
    headline = headline[["date", "headline_yoy"]]

    selected = source.merge(
        frontier,
        left_on="source_series_id",
        right_on="source_component_code",
        how="inner",
        validate="many_to_one",
    )
    selected["estimated_contribution_yoy_pp"] = (
        selected["weight_pct"] * selected["inflation_yoy"] / 100.0
    )
    selected = selected.merge(headline, on="date", how="left", validate="many_to_one")
    selected["series_role"] = "component"
    selected["reconciliation_row"] = False

    sums = selected.groupby("date")["estimated_contribution_yoy_pp"].sum(min_count=1)
    target = headline.set_index("date")["headline_yoy"]
    reconciliation = (target - sums).rename("estimated_contribution_yoy_pp")
    reconciliation = reconciliation.dropna().reset_index()
    reconciliation["country"] = "United States"
    reconciliation["country_code"] = "US"
    reconciliation["source_component_code"] = "__reconciliation__"
    reconciliation["label"] = "Unallocated reconciliation to headline CPI"
    reconciliation["hierarchy_level"] = 0
    reconciliation["series_role"] = "reconciliation"
    reconciliation["weight_pct"] = 0.0
    reconciliation["weight_reference_date"] = reconciliation["date"]
    reconciliation["weight_basis_year"] = tree["weight_basis_year"].iloc[0]
    reconciliation["index_value"] = np.nan
    reconciliation["inflation_yoy"] = np.nan
    reconciliation["headline_yoy"] = reconciliation["date"].map(target)
    reconciliation["reconciliation_row"] = True

    output = pd.concat([selected, reconciliation], ignore_index=True, sort=False)
    output["country"] = output["country"].fillna("United States")
    output["country_code"] = "US"
    return output.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "reconciliation_row", "label"], kind="stable"
    ).reset_index(drop=True)


def save_us_item_decomposition(
    frame: pd.DataFrame,
    output_path: str | Path = US_ITEM_DECOMPOSITION_PATH,
) -> Path:
    return atomic_write_parquet(frame, output_path, columns=OUTPUT_COLUMNS)


def run(
    input_path: str | Path = FRED_US_CPI_PATH,
    weights_path: str | Path = BLS_CPI_RELATIVE_IMPORTANCE_PATH,
    series_map_path: str | Path = US_CPI_ITEM_SERIES_MAP_PATH,
    output_path: str | Path = US_ITEM_DECOMPOSITION_PATH,
) -> Path:
    frame = pd.read_parquet(input_path)
    return save_us_item_decomposition(
        build_us_item_decomposition(frame, weights_path, series_map_path), output_path
    )


if __name__ == "__main__":
    destination = run()
    print(f"Saved US item decomposition to {destination}")
