"""Additive Danish item-level CPI decomposition.

This module deliberately sits beside the existing five-bucket decomposition.
It uses Statistics Denmark's 2026 weight basis and selects a non-overlapping
COICOP frontier from the detailed PRIS01 component panel.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.paths import (
    DANISH_CPI_WEIGHTS_2026_PATH,
    DANISH_CPI_WEIGHTS_BACKCAST_PATH,
    DANISH_ITEM_AGGREGATION_VALIDATION_PATH,
    DANISH_ITEM_CONTRIBUTIONS_PATH,
    DANISH_ITEM_DECOMPOSITION_PATH,
    STATBANK_DK_CPI_PATH,
)


OUTPUT_COLUMNS = [
    "date",
    "country",
    "country_code",
    "source_component_code",
    "coicop_code",
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

VALIDATION_COLUMNS = [
    "date",
    "country_code",
    "weight_basis_year",
    "matched_weight_sum_pct",
    "computed_headline_index",
    "official_headline_index",
    "headline_index_gap",
    "computed_headline_yoy",
    "official_headline_yoy",
    "headline_yoy_gap",
    "computed_core_index",
    "official_core_index",
    "core_index_gap",
    "computed_core_yoy",
    "official_core_yoy",
    "core_yoy_gap",
    "core_weight_pct",
    "official_core_weight_pct",
    "core_weight_gap_pct",
    "core_validation_status",
]

CONTRIBUTION_COLUMNS = [
    "date", "scope", "source_component_code", "coicop_code", "description",
    "weight_current_pct", "weight_prior_pct", "current_index_value",
    "prior_index_value", "contribution_yoy_pp", "computed_scope_yoy",
    "official_scope_yoy", "official_reconciliation_yoy_pp", "is_reconciliation",
]

# Statistics Denmark's published core aggregate is CPI excluding energy and
# unprocessed food. These COICOP prefixes are the ECOICOP selection used for
# that special aggregate; parent/child selection is still performed by the
# frontier algorithm before applying this exclusion.
DANISH_CORE_EXCLUSION_PREFIXES = (
    "0112",  # meat
    "0113",  # fish and seafood
    "0116",  # fruit and nuts
    "0117",  # vegetables
    "045",   # electricity, gas and other fuels
    "0722",  # fuels and lubricants for personal transport
)


def _normalize_code(value: object) -> str:
    return str(value).strip().replace(".", "")


def load_danish_weight_basis(
    path: str | Path = DANISH_CPI_WEIGHTS_2026_PATH,
) -> pd.DataFrame:
    """Read Statistics Denmark's official 2026 CPI weight workbook."""
    raw = pd.read_excel(path, header=None, skiprows=2)
    if raw.shape[1] < 3:
        raise ValueError("Danish weight workbook has fewer than three columns")
    weights = raw.iloc[:, :3].copy()
    weights.columns = ["coicop_code", "description", "weight_pct"]
    weights = weights.loc[weights["coicop_code"].notna()].copy()
    weights["coicop_code"] = weights["coicop_code"].map(_normalize_code)
    weights["weight_pct"] = pd.to_numeric(weights["weight_pct"], errors="coerce")
    weights = weights.dropna(subset=["weight_pct"])
    weights = weights.loc[weights["coicop_code"].str.fullmatch(r"\d+")].copy()
    # PRIS01 contains special aggregate codes (for example 151 for core CPI)
    # alongside ECOICOP item codes. Only divisions 01–13 are valid item nodes.
    weights = weights.loc[
        weights["coicop_code"].str[:2].astype(int).between(1, 13)
        | weights["coicop_code"].eq("00")
    ].copy()
    if weights.empty:
        raise ValueError("Danish weight workbook contains no usable COICOP rows")
    total = float(weights.loc[weights["coicop_code"] == "00", "weight_pct"].iloc[0])
    if not np.isclose(total, 100.0, atol=0.05):
        raise ValueError(f"Danish CPI weight total is {total}, expected 100")
    weights["weight_reference_date"] = pd.Timestamp("2026-01-01")
    weights["weight_basis_year"] = 2026
    return weights.reset_index(drop=True)


def load_danish_weights_for_year(year: int) -> pd.DataFrame:
    """Load the basket used for a calendar year in the official backcast files."""
    if year == 2026:
        return load_danish_weight_basis(DANISH_CPI_WEIGHTS_2026_PATH)
    if 2001 <= year <= 2025:
        raw = pd.read_excel(
            DANISH_CPI_WEIGHTS_BACKCAST_PATH,
            sheet_name=str(year),
            header=None,
            skiprows=2,
        )
        weights = raw.iloc[:, :3].copy()
        weights.columns = ["coicop_code", "description", "weight_pct"]
        weights = weights.loc[weights["coicop_code"].notna()].copy()
        weights["coicop_code"] = weights["coicop_code"].map(_normalize_code)
        weights["weight_pct"] = pd.to_numeric(
            weights["weight_pct"], errors="coerce"
        )
        weights = weights.dropna(subset=["weight_pct"])
        weights = weights.loc[
            weights["coicop_code"].str.fullmatch(r"\d+")
            & (
                weights["coicop_code"].str[:2].astype(int).between(1, 13)
                | weights["coicop_code"].eq("00")
            )
        ].copy()
        weights["weight_reference_date"] = pd.Timestamp(year, 1, 1)
        weights["weight_basis_year"] = year
        return weights.reset_index(drop=True)
    raise ValueError(f"No Danish CPI weight basis available for {year}")


def load_danish_core_weight(year: int) -> float:
    """Read Statistics Denmark's published core-basket weight."""
    if year == 2026:
        path, sheet = DANISH_CPI_WEIGHTS_2026_PATH, 0
    elif 2001 <= year <= 2025:
        path, sheet = DANISH_CPI_WEIGHTS_BACKCAST_PATH, str(year)
    else:
        raise ValueError(f"No Danish CPI core weight available for {year}")
    raw = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=2).iloc[:, :3]
    raw.columns = ["code", "description", "weight_pct"]
    raw["code"] = raw["code"].map(_normalize_code)
    raw["weight_pct"] = pd.to_numeric(raw["weight_pct"], errors="coerce")
    match = raw.loc[raw["code"] == "151", "weight_pct"].dropna()
    if match.empty:
        match = raw.loc[
            raw["description"].astype(str).str.contains(
                "Eksklusiv energi og ikke-forarbejdede fødevarer",
                case=False,
                na=False,
            ),
            "weight_pct",
        ].dropna()
    if match.empty:
        raise ValueError(f"Danish core weight 151 missing for {year}")
    return float(match.iloc[0])


def select_non_overlapping_frontier(
    weights: pd.DataFrame,
    available_component_codes: set[str],
) -> pd.DataFrame:
    """Select the deepest available weight nodes without parent/child overlap."""
    matched = weights.loc[
        weights["coicop_code"].isin(available_component_codes)
        & weights["coicop_code"].ne("00")
    ].copy()
    available = set(matched["coicop_code"])
    is_parent = matched["coicop_code"].map(
        lambda code: any(
            child.startswith(code) and len(child) > len(code)
            for child in available
        )
    )
    frontier = matched.loc[~is_parent].copy()
    frontier["hierarchy_level"] = frontier["coicop_code"].str.len()
    if frontier.empty:
        raise ValueError("No Danish weight rows match detailed CPI components")
    if not np.isclose(frontier["weight_pct"].sum(), 100.0, atol=0.1):
        raise ValueError(
            "Selected Danish component frontier does not sum to 100%; "
            "refusing to double-count or silently renormalize"
        )
    return frontier


def build_danish_item_decomposition(
    raw_cpi: pd.DataFrame,
    weights_path: str | Path = DANISH_CPI_WEIGHTS_2026_PATH,
) -> pd.DataFrame:
    """Build item contributions while leaving the canonical buckets untouched."""
    required = {
        "date",
        "country",
        "country_code",
        "series_role",
        "component",
        "source_series_id",
        "value",
    }
    missing = required.difference(raw_cpi.columns)
    if missing:
        raise ValueError(f"Danish item panel is missing columns: {sorted(missing)}")

    source = raw_cpi.loc[
        (raw_cpi["country_code"] == "DK")
        & (raw_cpi["series_role"] == "component")
    ].copy()
    source["date"] = pd.to_datetime(source["date"])
    source["source_component_code"] = source["source_series_id"].map(_normalize_code)
    source["index_value"] = pd.to_numeric(source["value"], errors="coerce")
    source = source.loc[source["date"].dt.year >= 2001].copy()
    source = source.sort_values(["source_component_code", "date"])
    source["inflation_yoy"] = source.groupby("source_component_code")[
        "index_value"
    ].pct_change(12) * 100.0

    weights = load_danish_weight_basis(weights_path)
    frontier = select_non_overlapping_frontier(
        weights,
        set(source["source_component_code"].unique()),
    )
    selected = source.merge(
        frontier,
        left_on="source_component_code",
        right_on="coicop_code",
        how="inner",
        validate="many_to_one",
    )
    # The workbook is the 2026 basket. Do not back-cast those weights into
    # earlier baskets; historical expansion will use the archived annual
    # weight workbooks as they are added.
    selected = selected.loc[
        selected["date"] >= pd.Timestamp("2026-01-01")
    ].copy()
    selected["estimated_contribution_yoy_pp"] = (
        selected["weight_pct"] * selected["inflation_yoy"] / 100.0
    )

    headline = raw_cpi.loc[
        (raw_cpi["country_code"] == "DK")
        & (raw_cpi["source_series_id"].map(_normalize_code) == "000000")
    ].copy()
    headline["date"] = pd.to_datetime(headline["date"])
    headline["index_value"] = pd.to_numeric(headline["value"], errors="coerce")
    headline = headline.sort_values("date")
    headline["headline_yoy"] = headline["index_value"].pct_change(12) * 100.0
    headline = headline[["date", "headline_yoy"]]
    selected = selected.merge(headline, on="date", how="left", validate="many_to_one")
    selected["reconciliation_row"] = False

    sums = selected.groupby("date")["estimated_contribution_yoy_pp"].sum(min_count=1)
    target = headline.set_index("date")["headline_yoy"]
    reconciliation = (target - sums).rename("estimated_contribution_yoy_pp")
    reconciliation = reconciliation.dropna().reset_index()
    reconciliation["country"] = "Denmark"
    reconciliation["country_code"] = "DK"
    reconciliation["source_component_code"] = "__reconciliation__"
    reconciliation["coicop_code"] = "__reconciliation__"
    reconciliation["description"] = "Unallocated reconciliation to headline CPI"
    reconciliation["hierarchy_level"] = 0
    reconciliation["series_role"] = "reconciliation"
    reconciliation["weight_pct"] = 0.0
    reconciliation["weight_reference_date"] = pd.Timestamp("2026-01-01")
    reconciliation["weight_basis_year"] = 2026
    reconciliation["index_value"] = np.nan
    reconciliation["inflation_yoy"] = np.nan
    reconciliation["headline_yoy"] = reconciliation["date"].map(target)
    reconciliation["reconciliation_row"] = True

    selected = selected.rename(columns={"component": "_component"})
    selected["description"] = selected["description"].astype(str)
    output = pd.concat([selected, reconciliation], ignore_index=True, sort=False)
    output["country"] = output["country"].fillna("Denmark")
    output["country_code"] = "DK"
    return output.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "reconciliation_row", "coicop_code"], kind="stable"
    ).reset_index(drop=True)


def save_danish_item_decomposition(
    frame: pd.DataFrame,
    output_path: str | Path = DANISH_ITEM_DECOMPOSITION_PATH,
) -> Path:
    return atomic_write_parquet(frame, output_path, columns=OUTPUT_COLUMNS)


def build_danish_aggregation_validation(raw_cpi: pd.DataFrame) -> pd.DataFrame:
    """Validate item-level weighted index levels against official headline CPI.

    Each calendar year uses its own official basket. This is the important
    distinction from summing current-year item YoY rates against a prior-year
    index built with a different basket.
    """
    source = raw_cpi.loc[
        (raw_cpi["country_code"] == "DK")
        & (raw_cpi["series_role"] == "component")
    ].copy()
    source["date"] = pd.to_datetime(source["date"])
    source["source_component_code"] = source["source_series_id"].map(_normalize_code)
    source["index_value"] = pd.to_numeric(source["value"], errors="coerce")
    source = source.loc[source["date"].dt.year >= 2001].copy()
    official = raw_cpi.loc[
        (raw_cpi["country_code"] == "DK")
        & (raw_cpi["source_series_id"].map(_normalize_code) == "000000")
    ].copy()
    official["date"] = pd.to_datetime(official["date"])
    official["official_headline_index"] = pd.to_numeric(
        official["value"], errors="coerce"
    )
    official_core = raw_cpi.loc[
        (raw_cpi["country_code"] == "DK")
        & (raw_cpi["source_series_id"].map(_normalize_code) == "151")
    ].copy()
    official_core["date"] = pd.to_datetime(official_core["date"])
    official_core["official_core_index"] = pd.to_numeric(
        official_core["value"], errors="coerce"
    )
    official = official[["date", "official_headline_index"]].merge(
        official_core[["date", "official_core_index"]],
        on="date",
        how="left",
        validate="one_to_one",
    )

    rows = []
    for year, year_source in source.groupby(source["date"].dt.year):
        weights = load_danish_weights_for_year(int(year))
        frontier = select_non_overlapping_frontier(
            weights,
            set(year_source["source_component_code"]),
        )
        frontier["is_core"] = ~frontier["coicop_code"].str.startswith(
            DANISH_CORE_EXCLUSION_PREFIXES
        )
        core_weight = float(frontier.loc[frontier["is_core"], "weight_pct"].sum())
        official_core_weight = load_danish_core_weight(int(year))
        matched = year_source.merge(
            frontier[["coicop_code", "weight_pct", "is_core"]],
            left_on="source_component_code",
            right_on="coicop_code",
            how="inner",
            validate="many_to_one",
        )
        aggregate = (
            matched.assign(weighted_index=matched["weight_pct"] * matched["index_value"])
            .groupby("date")["weighted_index"]
            .sum()
            / 100.0
        )
        core_aggregate = (
            matched.loc[matched["is_core"]]
            .assign(
                weighted_index=lambda frame: frame["weight_pct"]
                * frame["index_value"]
            )
            .groupby("date")["weighted_index"]
            .sum()
            / core_weight
        )
        rows.append(
            pd.DataFrame(
                {
                    "date": aggregate.index,
                    "country_code": "DK",
                    "weight_basis_year": int(year),
                    "matched_weight_sum_pct": float(frontier["weight_pct"].sum()),
                    "computed_headline_index": aggregate.to_numpy(),
                    "computed_core_index": core_aggregate.reindex(
                        aggregate.index
                    ).to_numpy(),
                    "core_weight_pct": core_weight,
                    "official_core_weight_pct": official_core_weight,
                }
            )
        )
    result = pd.concat(rows, ignore_index=True).merge(official, on="date", how="left")
    result = result.sort_values("date").reset_index(drop=True)
    result["headline_index_gap"] = (
        result["computed_headline_index"] - result["official_headline_index"]
    )
    result["computed_headline_yoy"] = (
        result["computed_headline_index"]
        / result["computed_headline_index"].shift(12)
        - 1.0
    ) * 100.0
    result["official_headline_yoy"] = (
        result["official_headline_index"]
        / result["official_headline_index"].shift(12)
        - 1.0
    ) * 100.0
    result["headline_yoy_gap"] = (
        result["computed_headline_yoy"] - result["official_headline_yoy"]
    )
    result["core_index_gap"] = (
        result["computed_core_index"] - result["official_core_index"]
    )
    result["computed_core_yoy"] = (
        result["computed_core_index"]
        / result["computed_core_index"].shift(12)
        - 1.0
    ) * 100.0
    result["official_core_yoy"] = (
        result["official_core_index"]
        / result["official_core_index"].shift(12)
        - 1.0
    ) * 100.0
    result["core_yoy_gap"] = (
        result["computed_core_yoy"] - result["official_core_yoy"]
    )
    result["core_weight_gap_pct"] = (
        result["core_weight_pct"] - result["official_core_weight_pct"]
    )
    result["core_validation_status"] = np.where(
        result["core_weight_gap_pct"].abs() <= 0.05,
        "passed_weight_mapping",
        "review_weight_mapping",
    )
    return result.loc[:, VALIDATION_COLUMNS]


def save_danish_aggregation_validation(
    frame: pd.DataFrame,
    output_path: str | Path = DANISH_ITEM_AGGREGATION_VALIDATION_PATH,
) -> Path:
    return atomic_write_parquet(frame, output_path, columns=VALIDATION_COLUMNS)


def build_danish_item_contributions(raw_cpi: pd.DataFrame) -> pd.DataFrame:
    """Decompose Danish headline/core YoY using consecutive official baskets.

    The contribution is calculated from weighted index levels (rather than
    weight times each item's YoY), so the item rows add to the reconstructed
    aggregate exactly. Any difference to the published aggregate is retained
    as an explicit reconciliation row.
    """
    required = {"date", "country_code", "series_role", "source_series_id", "value"}
    missing = required.difference(raw_cpi.columns)
    if missing:
        raise ValueError(f"Danish CPI panel is missing columns: {sorted(missing)}")
    source = raw_cpi.loc[
        (raw_cpi["country_code"] == "DK")
        & (raw_cpi["series_role"] == "component")
    ].copy()
    source["date"] = pd.to_datetime(source["date"])
    source["source_component_code"] = source["source_series_id"].map(_normalize_code)
    source["index_value"] = pd.to_numeric(source["value"], errors="coerce")
    source = source.dropna(subset=["date", "index_value"])
    source = source.loc[source["date"].dt.year >= 2001]
    available = set(source["source_component_code"])
    official = raw_cpi.loc[raw_cpi["country_code"] == "DK"].copy()
    official["date"] = pd.to_datetime(official["date"])
    official["code"] = official["source_series_id"].map(_normalize_code)
    official["value"] = pd.to_numeric(official["value"], errors="coerce")
    official = official.dropna(subset=["date", "value"])
    rows = []
    for date in sorted(source.loc[source["date"].dt.year >= 2026, "date"].unique()):
        date = pd.Timestamp(date)
        prior_date = date - pd.DateOffset(years=1)
        current_year, prior_year = date.year, prior_date.year
        if not ((source["date"] == date).any() and (source["date"] == prior_date).any()):
            continue
        current_frontier = select_non_overlapping_frontier(
            load_danish_weights_for_year(current_year), available
        ).rename(columns={"weight_pct": "weight_current_pct", "description": "description_current"})
        prior_frontier = select_non_overlapping_frontier(
            load_danish_weights_for_year(prior_year), available
        ).rename(columns={"weight_pct": "weight_prior_pct", "description": "description_prior"})
        current = source.loc[source["date"] == date, ["source_component_code", "index_value"]].rename(
            columns={"index_value": "current_index_value"}
        )
        prior = source.loc[source["date"] == prior_date, ["source_component_code", "index_value"]].rename(
            columns={"index_value": "prior_index_value"}
        )
        items = current_frontier[["coicop_code", "weight_current_pct", "description_current"]].merge(
            prior_frontier[["coicop_code", "weight_prior_pct", "description_prior"]],
            on="coicop_code", how="outer"
        )
        items["source_component_code"] = items["coicop_code"]
        items = items.merge(current, on="source_component_code", how="left").merge(
            prior, on="source_component_code", how="left"
        )
        items["description"] = items["description_current"].fillna(items["description_prior"])
        for scope in ("headline", "core"):
            items_scope = items.copy()
            if scope == "core":
                items_scope = items_scope.loc[~items_scope["coicop_code"].str.startswith(DANISH_CORE_EXCLUSION_PREFIXES)].copy()
                current_weight = load_danish_core_weight(current_year)
                prior_weight = load_danish_core_weight(prior_year)
            else:
                current_weight = prior_weight = 100.0
            items_scope["weight_current_pct"] = items_scope["weight_current_pct"].fillna(0.0)
            items_scope["weight_prior_pct"] = items_scope["weight_prior_pct"].fillna(0.0)
            items_scope["current_index_value"] = items_scope["current_index_value"].fillna(0.0)
            items_scope["prior_index_value"] = items_scope["prior_index_value"].fillna(0.0)
            current_aggregate = (items_scope["weight_current_pct"] * items_scope["current_index_value"]).sum() / current_weight
            prior_aggregate = (items_scope["weight_prior_pct"] * items_scope["prior_index_value"]).sum() / prior_weight
            items_scope["contribution_yoy_pp"] = (
                (items_scope["weight_current_pct"] * items_scope["current_index_value"] / current_weight)
                - (items_scope["weight_prior_pct"] * items_scope["prior_index_value"] / prior_weight)
            ) / prior_aggregate * 100.0
            official_code = "000000" if scope == "headline" else "151"
            official_rows = official.loc[(official["date"].isin([date, prior_date])) & (official["code"] == official_code)]
            official_map = official_rows.set_index("date")["value"]
            if date not in official_map.index or prior_date not in official_map.index:
                continue
            official_yoy = (official_map.loc[date] / official_map.loc[prior_date] - 1.0) * 100.0
            computed_yoy = (current_aggregate / prior_aggregate - 1.0) * 100.0
            items_scope["date"] = date
            items_scope["scope"] = scope
            items_scope["computed_scope_yoy"] = computed_yoy
            items_scope["official_scope_yoy"] = official_yoy
            items_scope["official_reconciliation_yoy_pp"] = np.nan
            items_scope["is_reconciliation"] = False
            rows.extend(items_scope.loc[:, CONTRIBUTION_COLUMNS].to_dict("records"))
            rows.append({
                "date": date, "scope": scope, "source_component_code": "__official_reconciliation__",
                "coicop_code": "__official_reconciliation__",
                "description": "Official aggregate minus reconstructed index-level contributions",
                "weight_current_pct": 0.0, "weight_prior_pct": 0.0,
                "current_index_value": np.nan, "prior_index_value": np.nan,
                "contribution_yoy_pp": official_yoy - computed_yoy,
                "computed_scope_yoy": computed_yoy, "official_scope_yoy": official_yoy,
                "official_reconciliation_yoy_pp": official_yoy - computed_yoy,
                "is_reconciliation": True,
            })
    return pd.DataFrame(rows, columns=CONTRIBUTION_COLUMNS).sort_values(["date", "scope", "is_reconciliation", "coicop_code"]).reset_index(drop=True)


def save_danish_item_contributions(
    frame: pd.DataFrame,
    output_path: str | Path = DANISH_ITEM_CONTRIBUTIONS_PATH,
) -> Path:
    return atomic_write_parquet(frame, output_path, columns=CONTRIBUTION_COLUMNS)


def run(
    input_path: str | Path = STATBANK_DK_CPI_PATH,
    weights_path: str | Path = DANISH_CPI_WEIGHTS_2026_PATH,
    output_path: str | Path = DANISH_ITEM_DECOMPOSITION_PATH,
) -> Path:
    frame = pd.read_parquet(input_path)
    return save_danish_item_decomposition(
        build_danish_item_decomposition(frame, weights_path), output_path
    )
