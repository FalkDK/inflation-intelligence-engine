"""Additive Norwegian item-level CPI decomposition.

This module deliberately sits beside nordic_item_contributions.py's existing
Norway output, the same way danish_item_decomposition.py sits beside the
five-bucket decomposition. That existing output reuses
fetch_ssb_weights.py's ``select_norway_component_frontier``, which is built
specifically for five-bucket macro classification -- it deliberately stops
at shallow "durability-marked" nodes (see UNMARKED_NODE_BUCKETS there) once
a node can be classified, well short of SSB table 14700's actual published
detail. That frontier stays untouched here.

This module instead selects a maximally granular, non-overlapping frontier
from every weighted node SSB publishes (fetch_norway_full_component_weights),
cascading a residual entry at every level of the tree -- using a node's own
rate for whatever its resolved children don't cover -- rather than only
reconciling once at the very top (SSB does not publish a standalone index
for every weighted line item, the same limitation seen in the UK and US
builds, but it frequently does publish one for the parent that a partially
itemized child set sits under).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.paths import (
    NORWAY_ITEM_DECOMPOSITION_PATH,
    SSB_NORWAY_CPI_PATH,
    SSB_NORWAY_FULL_WEIGHTS_PATH,
)


OUTPUT_COLUMNS = [
    "date",
    "country",
    "country_code",
    "source_component_code",
    "description",
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


def _parent_code(code: str) -> str | None:
    """SSB nests table 14700 codes with dots (01.1.1) and, past the normal
    COICOP depth, with a trailing "_N" split (01.1.1.1_1 = Rice, a child of
    01.1.1.1 = Cereals)."""
    if "_" in code:
        return code.rsplit("_", 1)[0]
    if "." in code:
        return code.rsplit(".", 1)[0]
    return None


_WEIGHT_TOLERANCE = 1e-9


def select_norway_item_frontier(
    weights: pd.DataFrame,
    available_component_codes: set[str],
) -> pd.DataFrame:
    """Select the deepest available weight nodes with a matched index series.

    SSB frequently publishes a weight (and index) for a parent node whose
    listed children only partially cover it -- e.g. division "09" (weight
    8.45%) has published children summing to only 6.40%, because some of its
    economic content (09.4, 09.5, 09.9) is never broken out as its own line
    item. Rather than silently dropping that 2.05pp to the single top-level
    reconciliation row, this cascades a residual entry at *every* level: use
    a node's own rate for whatever its resolved descendants don't cover,
    provided the node itself has an index. Only a subtree where no node
    anywhere (from the leaf up to the nearest ancestor with an index) is
    matched actually falls through to headline-level reconciliation.
    """
    codes = set(weights["source_component_code"])
    weight_by_code = weights.set_index("source_component_code")["weight_pct"]
    description_by_code = weights.set_index("source_component_code")["description"]

    def children_of(code: str) -> list[str]:
        return [c for c in codes if c != code and _parent_code(c) == code]

    def resolve(code: str) -> list[tuple[str, float]]:
        parts: list[tuple[str, float]] = []
        covered = 0.0
        for child in children_of(code):
            child_parts = resolve(child)
            parts.extend(child_parts)
            covered += sum(weight for _, weight in child_parts)

        node_weight = weight_by_code.get(code)
        if node_weight is None:
            return parts
        remainder = node_weight - covered
        if remainder > _WEIGHT_TOLERANCE and code in available_component_codes:
            parts.append((code, remainder))
        return parts

    top_level = [code for code in codes if _parent_code(code) not in codes]
    parts: list[tuple[str, float]] = []
    for top in top_level:
        parts.extend(resolve(top))
    if not parts:
        raise ValueError("No Norway weight rows matched an available index series")

    frontier = pd.DataFrame(parts, columns=["source_component_code", "weight_pct"])
    full_weight = frontier["source_component_code"].map(weight_by_code)
    is_partial = (full_weight - frontier["weight_pct"]) > _WEIGHT_TOLERANCE
    frontier["description"] = frontier["source_component_code"].map(description_by_code)
    frontier.loc[is_partial, "description"] += " (unitemized remainder only)"
    frontier["hierarchy_level"] = frontier["source_component_code"].map(
        lambda code: code.count(".") + code.count("_") + 1
    )
    return frontier


def build_norway_item_decomposition(
    raw_cpi: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Build Norway item contributions while leaving the existing five-bucket
    contract (nordic_item_contributions.py) untouched."""
    required = {"date", "country", "country_code", "source_series_id", "value"}
    missing = required.difference(raw_cpi.columns)
    if missing:
        raise ValueError(f"Norway item panel is missing columns: {sorted(missing)}")

    source = raw_cpi.loc[
        (raw_cpi["country_code"] == "NO")
        & raw_cpi["source_series_id"].str.startswith("14700:", na=False)
    ].copy()
    source["date"] = pd.to_datetime(source["date"])
    source["source_component_code"] = source["source_series_id"].str.removeprefix(
        "14700:"
    )
    source["index_value"] = pd.to_numeric(source["value"], errors="coerce")
    source = source.sort_values(["source_component_code", "date"])
    source["inflation_yoy"] = source.groupby("source_component_code")[
        "index_value"
    ].pct_change(12) * 100.0

    frontier = select_norway_item_frontier(
        weights, set(source["source_component_code"].unique())
    )

    headline = raw_cpi.loc[
        (raw_cpi["country_code"] == "NO") & (raw_cpi["component"] == "all_items")
    ].copy()
    headline["date"] = pd.to_datetime(headline["date"])
    headline["index_value"] = pd.to_numeric(headline["value"], errors="coerce")
    headline = headline.sort_values("date")
    headline["headline_yoy"] = headline["index_value"].pct_change(12) * 100.0
    headline = headline[["date", "headline_yoy"]]

    selected = source.merge(
        frontier,
        on="source_component_code",
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
    reconciliation["country"] = "Norway"
    reconciliation["country_code"] = "NO"
    reconciliation["source_component_code"] = "__reconciliation__"
    reconciliation["description"] = "Unallocated reconciliation to headline CPI"
    reconciliation["hierarchy_level"] = 0
    reconciliation["series_role"] = "reconciliation"
    reconciliation["weight_pct"] = 0.0
    reconciliation["weight_reference_date"] = reconciliation["date"]
    reconciliation["weight_basis_year"] = weights["weight_basis_year"].iloc[0]
    reconciliation["index_value"] = np.nan
    reconciliation["inflation_yoy"] = np.nan
    reconciliation["headline_yoy"] = reconciliation["date"].map(target)
    reconciliation["reconciliation_row"] = True

    output = pd.concat([selected, reconciliation], ignore_index=True, sort=False)
    output["country"] = output["country"].fillna("Norway")
    output["country_code"] = "NO"
    return output.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "reconciliation_row", "source_component_code"], kind="stable"
    ).reset_index(drop=True)


def save_norway_item_decomposition(
    frame: pd.DataFrame,
    output_path: str | Path = NORWAY_ITEM_DECOMPOSITION_PATH,
) -> Path:
    return atomic_write_parquet(frame, output_path, columns=OUTPUT_COLUMNS)


def run(
    input_path: str | Path = SSB_NORWAY_CPI_PATH,
    weights_path: str | Path = SSB_NORWAY_FULL_WEIGHTS_PATH,
    output_path: str | Path = NORWAY_ITEM_DECOMPOSITION_PATH,
) -> Path:
    raw_cpi = pd.read_parquet(input_path)
    weights = pd.read_parquet(weights_path)
    return save_norway_item_decomposition(
        build_norway_item_decomposition(raw_cpi, weights), output_path
    )


if __name__ == "__main__":
    destination = run()
    print(f"Saved Norway item decomposition to {destination}")
