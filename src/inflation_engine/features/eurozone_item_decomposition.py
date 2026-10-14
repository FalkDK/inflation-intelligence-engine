"""Additive euro-area item-level HICP decomposition.

This module deliberately sits beside decomposition.py's five coarse
harmonised buckets, the same way us_item_decomposition.py /
uk_item_decomposition.py / danish_item_decomposition.py do for their own
countries. Unlike those three, which each need a bespoke, manually curated
weight-to-series crosswalk (BLS's relative-importance workbook, ONS's Annex A
workbook, Statistics Denmark's own hierarchy), the euro area's COICOP
classification is regular and code-driven: a code's parent is simply itself
with its last digit dropped (CP01111 -> CP0111 -> CP011 -> CP01 -> CP00), and
Eurostat publishes both the annual weight and the monthly index under that
same code -- so the additive hierarchy and the frontier's available series
both come straight from fetch_eurostat_item_weights.py / fetch_eurostat_items.py
with no separate crosswalk needed.

Covers all 20 euro-area members (see fetch_eurostat_items.py's
HICP_ITEM_GEOGRAPHIES). Each country's own CP00 ("all-items HICP") row
doubles as its headline reconciliation target -- it comes from the same
dataset and vintage as every other component here, rather than depending on
fetch_eurostat.py's separately sourced headline HICP.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from inflation_engine.features.calendar_yoy import calendar_yoy
from inflation_engine.utils.config import COUNTRIES
from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.paths import (
    EUROSTAT_HICP_ITEM_WEIGHTS_PATH,
    EUROZONE_ITEM_DECOMPOSITION_PATH,
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

_ROOT_CODE = "CP00"
_WEIGHT_TOLERANCE = 1e-6


def _parent_code(code: str) -> str | None:
    if code == _ROOT_CODE:
        return None
    if len(code) == 4:
        return _ROOT_CODE
    return code[:-1]


def select_eurozone_frontier(
    weights: pd.DataFrame,
    available_component_codes: set[str],
) -> pd.DataFrame:
    """Select the deepest available weight nodes with a matched index series,
    for one country's one weight-basis-year vintage.

    Eurostat frequently publishes a weight and index for a parent node (e.g.
    "CP08" Communications) whose listed children only partially cover it --
    the remainder has no separate line item, but the parent's own index can
    stand in for that gap. This cascades a residual at every level of the
    tree, using a node's own rate for whatever its resolved children don't
    cover, rather than dropping straight to the single top-level
    reconciliation row.

    CP00 (all-items) itself is never a candidate frontier node -- it is
    this same module's headline reconciliation target (see
    build_eurozone_item_decomposition), so treating it as a component here
    would double-count the headline rather than fall through to the
    reconciliation row wherever the true divisions (CP01 .. CP12) don't
    fully cover it.
    """
    codes = set(weights["coicop_code"])
    if _ROOT_CODE not in codes:
        raise ValueError("Eurostat item weights are missing the CP00 root")
    weight_by_code = weights.set_index("coicop_code")["weight_pct"]
    row_by_code = weights.set_index("coicop_code")

    children_of: dict[str, list[str]] = {code: [] for code in codes}
    for code in codes:
        parent = _parent_code(code)
        if parent in children_of:
            children_of[parent].append(code)

    def resolve(code: str) -> list[tuple[str, float]]:
        parts: list[tuple[str, float]] = []
        covered = 0.0
        for child in children_of.get(code, []):
            child_parts = resolve(child)
            parts.extend(child_parts)
            covered += sum(weight for _, weight in child_parts)
        remainder = weight_by_code[code] - covered
        if remainder > _WEIGHT_TOLERANCE and code in available_component_codes:
            parts.append((code, remainder))
        return parts

    parts = []
    for division in children_of.get(_ROOT_CODE, []):
        parts.extend(resolve(division))
    if not parts:
        raise ValueError("No euro-area weight rows survived frontier selection")

    frontier = pd.DataFrame(parts, columns=["coicop_code", "weight_pct"])
    full_weight = frontier["coicop_code"].map(weight_by_code)
    is_partial = (full_weight - frontier["weight_pct"]) > _WEIGHT_TOLERANCE
    frontier["label"] = frontier["coicop_code"].map(row_by_code["label"])
    frontier.loc[is_partial, "label"] = (
        frontier.loc[is_partial, "label"] + " (unitemized remainder only)"
    )
    frontier["weight_reference_date"] = frontier["coicop_code"].map(
        row_by_code["weight_reference_date"]
    )
    frontier["weight_basis_year"] = frontier["coicop_code"].map(
        row_by_code["weight_basis_year"]
    )
    frontier["hierarchy_level"] = frontier["coicop_code"].str.len() - 3
    return frontier


def _build_country_decomposition(
    country_code: str,
    source: pd.DataFrame,
    headline: pd.DataFrame,
    country_weights: pd.DataFrame,
) -> pd.DataFrame | None:
    headline = headline[["date", "headline_yoy"]]
    if headline.empty:
        return None

    available_codes = set(source["source_series_id"].unique())
    selected_frames = []
    for year, year_weights in country_weights.groupby("weight_basis_year"):
        frontier = select_eurozone_frontier(year_weights, available_codes)
        year_source = source.loc[source["date"].dt.year == year]
        merged = year_source.merge(
            frontier,
            left_on="source_series_id",
            right_on="coicop_code",
            how="inner",
            validate="many_to_one",
        )
        selected_frames.append(merged)
    if not selected_frames:
        return None

    selected = pd.concat(selected_frames, ignore_index=True)
    if selected.empty:
        return None
    selected["estimated_contribution_yoy_pp"] = (
        selected["weight_pct"] * selected["inflation_yoy"] / 100.0
    )
    selected = selected.merge(headline, on="date", how="left", validate="many_to_one")
    selected["series_role"] = "component"
    selected["reconciliation_row"] = False
    selected["source_component_code"] = selected["coicop_code"]

    sums = selected.groupby("date")["estimated_contribution_yoy_pp"].sum(min_count=1)
    target = headline.set_index("date")["headline_yoy"]
    reconciliation = (target - sums).rename("estimated_contribution_yoy_pp")
    reconciliation = reconciliation.dropna().reset_index()
    reconciliation["country_code"] = country_code
    reconciliation["source_component_code"] = "__reconciliation__"
    reconciliation["label"] = "Unallocated reconciliation to headline HICP"
    reconciliation["hierarchy_level"] = 0
    reconciliation["series_role"] = "reconciliation"
    reconciliation["weight_pct"] = 0.0
    reconciliation["weight_reference_date"] = reconciliation["date"]
    reconciliation["weight_basis_year"] = reconciliation["date"].dt.year
    reconciliation["index_value"] = np.nan
    reconciliation["inflation_yoy"] = np.nan
    reconciliation["headline_yoy"] = reconciliation["date"].map(target)
    reconciliation["reconciliation_row"] = True

    output = pd.concat([selected, reconciliation], ignore_index=True, sort=False)
    output["country_code"] = country_code
    return output


def build_eurozone_item_decomposition(
    raw_cpi: pd.DataFrame,
    item_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Build euro-area item contributions while leaving decomposition.py's
    five-bucket harmonised model untouched."""
    required = {"date", "country_code", "source_series_id", "value"}
    missing = required.difference(raw_cpi.columns)
    if missing:
        raise ValueError(f"Euro-area item panel is missing columns: {sorted(missing)}")
    required_weights = {
        "country_code", "coicop_code", "label", "weight_pct",
        "weight_reference_date", "weight_basis_year",
    }
    missing_weights = required_weights.difference(item_weights.columns)
    if missing_weights:
        raise ValueError(
            f"Euro-area item weights are missing columns: {sorted(missing_weights)}"
        )

    item_index = raw_cpi.loc[
        raw_cpi["source_series_id"].astype(str).str.fullmatch(r"CP\d+", na=False)
    ]
    countries = sorted(
        set(item_index["country_code"]) & set(item_weights["country_code"])
    )

    country_outputs = []
    for country_code in countries:
        source = item_index.loc[item_index["country_code"] == country_code].copy()
        source["date"] = pd.to_datetime(source["date"])
        source["index_value"] = pd.to_numeric(source["value"], errors="coerce")
        source = source.sort_values(["source_series_id", "date"]).reset_index(drop=True)
        source["inflation_yoy"] = calendar_yoy(source, group_column="source_series_id")

        headline = source.loc[source["source_series_id"] == _ROOT_CODE].copy()
        headline["headline_yoy"] = headline["inflation_yoy"]

        country_weights = item_weights.loc[item_weights["country_code"] == country_code]
        output = _build_country_decomposition(
            country_code, source, headline, country_weights
        )
        if output is not None:
            country_outputs.append(output)

    if not country_outputs:
        raise ValueError("No euro-area country produced an item decomposition")

    combined = pd.concat(country_outputs, ignore_index=True)
    combined["country"] = combined["country_code"].map(
        {code: COUNTRIES[code]["name"] for code in combined["country_code"].unique()}
    )
    return (
        combined.loc[:, OUTPUT_COLUMNS]
        .sort_values(
            ["country_code", "date", "reconciliation_row", "source_component_code"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def save_eurozone_item_decomposition(
    frame: pd.DataFrame,
    output_path: str | Path = EUROZONE_ITEM_DECOMPOSITION_PATH,
) -> Path:
    return atomic_write_parquet(frame, output_path, columns=OUTPUT_COLUMNS)


def run(
    raw_cpi: pd.DataFrame,
    item_weights_path: str | Path = EUROSTAT_HICP_ITEM_WEIGHTS_PATH,
    output_path: str | Path = EUROZONE_ITEM_DECOMPOSITION_PATH,
) -> Path:
    item_weights = pd.read_parquet(item_weights_path)
    return save_eurozone_item_decomposition(
        build_eurozone_item_decomposition(raw_cpi, item_weights), output_path
    )
