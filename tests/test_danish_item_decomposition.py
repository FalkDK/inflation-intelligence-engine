import pytest
import pandas as pd

from inflation_engine.features.danish_item_decomposition import (
    build_danish_aggregation_validation,
    build_danish_item_contributions,
    build_danish_item_decomposition,
    load_danish_weight_basis,
    select_non_overlapping_frontier,
)


def test_frontier_does_not_double_count_parent_and_child():
    weights = pd.DataFrame(
        {
            "coicop_code": ["03", "031", "03121", "032"],
            "description": ["Clothing and footwear", "Clothing", "Men", "Footwear"],
            "weight_pct": [100.0, 80.0, 50.0, 50.0],
        }
    )
    # The fixture intentionally omits children under 032, so 032 remains the
    # valid frontier node while 03/031 are replaced by 03121.
    frontier = select_non_overlapping_frontier(weights, {"03121", "032"})
    assert set(frontier["coicop_code"]) == {"03121", "032"}
    assert frontier["weight_pct"].sum() == 100.0


@pytest.mark.integration
def test_danish_item_layer_reconciles_to_headline():
    raw = pd.read_parquet("data/raw/statbank_dk_cpi.parquet")
    result = build_danish_item_decomposition(raw)
    assert result["date"].min() >= pd.Timestamp("2026-01-01")
    assert result["reconciliation_row"].any()
    for date, group in result.groupby("date"):
        headline = group["headline_yoy"].dropna().iloc[0]
        assert abs(group["estimated_contribution_yoy_pp"].sum() - headline) < 1e-9
    latest = result[result["date"] == result["date"].max()]
    assert "03122" in set(latest["coicop_code"])
    assert load_danish_weight_basis()["weight_pct"].loc[lambda s: s > 0].sum() > 99


@pytest.mark.integration
def test_year_specific_weights_reproduce_headline_with_rounding_tolerance():
    raw = pd.read_parquet("data/raw/statbank_dk_cpi.parquet")
    validation = build_danish_aggregation_validation(raw)
    recent = validation.loc[validation["date"] >= pd.Timestamp("2026-01-01")]
    assert recent["matched_weight_sum_pct"].between(99.9, 100.1).all()
    assert recent["headline_index_gap"].abs().max() < 0.25
    assert recent["headline_yoy_gap"].abs().max() < 0.10
    assert recent["core_weight_gap_pct"].abs().max() < 0.05
    assert recent["core_index_gap"].abs().max() < 0.25
    assert recent["core_yoy_gap"].abs().max() < 0.10
    assert (recent["core_validation_status"] == "passed_weight_mapping").all()


@pytest.mark.integration
def test_index_level_contributions_sum_with_explicit_official_reconciliation():
    raw = pd.read_parquet("data/raw/statbank_dk_cpi.parquet")
    result = build_danish_item_contributions(raw)
    assert not result.empty
    assert result["is_reconciliation"].any()
    for (_, _), group in result.groupby(["date", "scope"]):
        computed = group["computed_scope_yoy"].iloc[0]
        official = group["official_scope_yoy"].iloc[0]
        assert abs(group.loc[~group["is_reconciliation"], "contribution_yoy_pp"].sum() - computed) < 1e-9
        assert abs(group["contribution_yoy_pp"].sum() - official) < 1e-9
    latest_clothing = result.loc[
        (result["date"] == result["date"].max())
        & (result["scope"] == "headline")
        & result["coicop_code"].eq("03122")
    ]
    assert len(latest_clothing) == 1
