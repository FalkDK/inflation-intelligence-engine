import pandas as pd
import pytest

from inflation_engine.features.norway_item_decomposition import (
    _parent_code,
    build_norway_item_decomposition,
    select_norway_item_frontier,
)


def test_frontier_prefers_deepest_fully_matched_subtree():
    weights = pd.DataFrame(
        {
            "source_component_code": ["01", "01.1", "01.1.1", "01.1.2"],
            "description": ["Food and beverages", "Food", "Bread", "Meat"],
            "weight_pct": [10.0, 8.0, 4.0, 4.0],
            "weight_reference_date": pd.Timestamp("2026-01-01"),
            "weight_basis_year": 2026,
        }
    )
    frontier = select_norway_item_frontier(weights, {"01.1.1", "01.1.2"})
    assert set(frontier["source_component_code"]) == {"01.1.1", "01.1.2"}


def test_frontier_handles_underscore_split_below_normal_coicop_depth():
    weights = pd.DataFrame(
        {
            "source_component_code": ["01.1.1.1", "01.1.1.1_1", "01.1.1.1_2"],
            "description": ["Cereals (ND)", "Rice", "Grains and groats"],
            "weight_pct": [1.0, 0.6, 0.4],
            "weight_reference_date": pd.Timestamp("2026-01-01"),
            "weight_basis_year": 2026,
        }
    )
    frontier = select_norway_item_frontier(
        weights, {"01.1.1.1_1", "01.1.1.1_2"}
    )
    assert set(frontier["source_component_code"]) == {"01.1.1.1_1", "01.1.1.1_2"}


def test_frontier_falls_back_to_matched_ancestor_when_children_unmatched():
    weights = pd.DataFrame(
        {
            "source_component_code": ["04.4", "04.4.1", "04.4.2"],
            "description": ["Water supply etc.", "Water supply", "Sewerage"],
            "weight_pct": [1.0, 0.6, 0.4],
            "weight_reference_date": pd.Timestamp("2026-01-01"),
            "weight_basis_year": 2026,
        }
    )
    # Only the parent "04.4" has a published index in this scenario.
    frontier = select_norway_item_frontier(weights, {"04.4"})
    assert list(frontier["source_component_code"]) == ["04.4"]
    assert list(frontier["weight_pct"]) == [1.0]


def test_frontier_adds_residual_for_partially_itemized_parent():
    # Mirrors real SSB division "09": the parent has its own published
    # weight AND index, but its listed children only cover part of it (the
    # rest of its economic content is never broken out as its own line
    # item). The parent's own rate should be used for that remainder,
    # instead of losing it to the single top-level reconciliation row.
    weights = pd.DataFrame(
        {
            "source_component_code": ["09", "09.1", "09.2"],
            "description": ["Recreation", "Recreational durables", "Other recreational goods"],
            "weight_pct": [8.45, 0.42, 0.85],
            "weight_reference_date": pd.Timestamp("2026-01-01"),
            "weight_basis_year": 2026,
        }
    )
    frontier = select_norway_item_frontier(weights, {"09", "09.1", "09.2"})
    assert set(frontier["source_component_code"]) == {"09", "09.1", "09.2"}
    residual = frontier.loc[frontier["source_component_code"] == "09"].iloc[0]
    assert abs(residual["weight_pct"] - (8.45 - 0.42 - 0.85)) < 1e-9
    assert "unitemized remainder only" in residual["description"]
    assert abs(frontier["weight_pct"].sum() - 8.45) < 1e-9


def test_frontier_residual_is_skipped_when_children_fully_cover_parent():
    weights = pd.DataFrame(
        {
            "source_component_code": ["01", "01.1", "01.2"],
            "description": ["Food and beverages", "Food", "Non-alcoholic beverages"],
            "weight_pct": [10.0, 7.0, 3.0],
            "weight_reference_date": pd.Timestamp("2026-01-01"),
            "weight_basis_year": 2026,
        }
    )
    frontier = select_norway_item_frontier(weights, {"01", "01.1", "01.2"})
    assert set(frontier["source_component_code"]) == {"01.1", "01.2"}
    assert not frontier["description"].str.contains("unitemized remainder only").any()


def test_parent_code_prefers_underscore_over_dot():
    assert _parent_code("01.1.1.1_1") == "01.1.1.1"
    assert _parent_code("01.1.1") == "01.1"
    assert _parent_code("01") is None


@pytest.mark.integration
def test_norway_item_layer_reconciles_to_headline_in_recent_history():
    raw = pd.read_parquet("data/raw/ssb_norway_cpi.parquet")
    weights = pd.read_parquet("data/raw/ssb_norway_full_weights.parquet")
    result = build_norway_item_decomposition(raw, weights)
    assert result["reconciliation_row"].any()
    # 2000-2001 has a handful of dates with a real, isolated reconciliation
    # gap (consistent with a known Norwegian CPI base-year rebasing in that
    # window); recent history should reconcile essentially exactly.
    recent = result.loc[result["date"] >= pd.Timestamp("2005-01-01")]
    for date, group in recent.groupby("date"):
        headline = group["headline_yoy"].dropna()
        if headline.empty:
            continue
        assert abs(
            group["estimated_contribution_yoy_pp"].sum() - headline.iloc[0]
        ) < 1e-6
    latest = result[result["date"] == result["date"].max()]
    non_reconciliation = latest.loc[~latest["reconciliation_row"]]
    # Cascading residuals recover nearly all of the basket, unlike the
    # single-top-level-reconciliation approach this replaced (~91%).
    assert non_reconciliation["weight_pct"].sum() > 99.0


def test_norway_item_layer_requires_no_component_rows():
    with pytest.raises(ValueError, match="missing columns"):
        build_norway_item_decomposition(pd.DataFrame({"date": []}), pd.DataFrame())
