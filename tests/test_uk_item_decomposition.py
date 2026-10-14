import pandas as pd
import pytest

from inflation_engine.features.uk_item_decomposition import (
    _implied_prefixes,
    build_uk_item_decomposition,
    load_uk_weight_basis,
    select_uk_frontier,
)


def _weights(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["weight_reference_date"] = pd.Timestamp("2026-02-01")
    frame["weight_basis_year"] = 2026
    return frame


def test_frontier_does_not_double_count_parent_and_child():
    weights = _weights(
        [
            {"cdid": "A", "coicop_code": "03", "description": "Clothing and footwear", "weight_pct": 100.0},
            {"cdid": "B", "coicop_code": "03.1", "description": "Clothing", "weight_pct": 80.0},
            {"cdid": "C", "coicop_code": "03.1.2.1", "description": "Men", "weight_pct": 50.0},
            {"cdid": "D", "coicop_code": "03.2", "description": "Footwear", "weight_pct": 20.0},
        ]
    )
    frontier = select_uk_frontier(weights, {"03.1.2.1", "03.2"})
    assert set(frontier["coicop_code"]) == {"03.1.2.1", "03.2"}


def test_frontier_adds_residual_for_partially_itemized_parent():
    # Mirrors real ONS division "10" (Education): the parent has its own
    # published weight AND index, but its only listed child covers just
    # part of it. The parent's own rate should cover the remainder.
    weights = _weights(
        [
            {"cdid": "X", "coicop_code": "10", "description": "Education", "weight_pct": 3.42466},
            {"cdid": "Y", "coicop_code": "10.4", "description": "Tertiary education", "weight_pct": 2.35335},
        ]
    )
    frontier = select_uk_frontier(weights, {"10", "10.4"})
    assert set(frontier["coicop_code"]) == {"10", "10.4"}
    residual = frontier.loc[frontier["coicop_code"] == "10"].iloc[0]
    assert abs(residual["weight_pct"] - (3.42466 - 2.35335)) < 1e-9
    assert "unitemized remainder only" in residual["description"]
    assert abs(frontier["weight_pct"].sum() - 3.42466) < 1e-9


def test_frontier_residual_skipped_when_parent_itself_unavailable():
    weights = _weights(
        [
            {"cdid": "X", "coicop_code": "10", "description": "Education", "weight_pct": 3.42466},
            {"cdid": "Y", "coicop_code": "10.4", "description": "Tertiary education", "weight_pct": 2.35335},
        ]
    )
    # "10" itself has no matched index in this scenario -- only "10.4" does.
    frontier = select_uk_frontier(weights, {"10.4"})
    assert set(frontier["coicop_code"]) == {"10.4"}


def test_compressed_range_code_is_excluded_when_finer_detail_exists():
    # "08.2/3" combines groups 08.2 and 08.3; ONS separately publishes finer
    # detail under both, so the compressed row must not survive alongside it.
    weights = _weights(
        [
            {"cdid": "A", "coicop_code": "08.2/3", "description": "Telephone equipment & services", "weight_pct": 2.3},
            {"cdid": "B", "coicop_code": "08.2.0.1", "description": "Fixed phones", "weight_pct": 0.4},
            {"cdid": "C", "coicop_code": "08.3.0.1", "description": "Wired services", "weight_pct": 0.2},
        ]
    )
    frontier = select_uk_frontier(weights, {"08.2.0.1", "08.3.0.1"})
    assert set(frontier["coicop_code"]) == {"08.2.0.1", "08.3.0.1"}


def test_compressed_range_code_gets_residual_when_also_available():
    weights = _weights(
        [
            {"cdid": "A", "coicop_code": "08.2/3", "description": "Telephone equipment & services", "weight_pct": 2.3},
            {"cdid": "B", "coicop_code": "08.2.0.1", "description": "Fixed phones", "weight_pct": 0.4},
            {"cdid": "C", "coicop_code": "08.3.0.1", "description": "Wired services", "weight_pct": 0.2},
        ]
    )
    frontier = select_uk_frontier(weights, {"08.2/3", "08.2.0.1", "08.3.0.1"})
    assert set(frontier["coicop_code"]) == {"08.2/3", "08.2.0.1", "08.3.0.1"}
    residual = frontier.loc[frontier["coicop_code"] == "08.2/3"].iloc[0]
    assert abs(residual["weight_pct"] - (2.3 - 0.4 - 0.2)) < 1e-9


def test_compressed_range_code_survives_as_leaf_without_finer_detail():
    weights = _weights(
        [
            {"cdid": "A", "coicop_code": "01.1.1.7/8", "description": "Breakfast cereals and other cereal products", "weight_pct": 0.34},
        ]
    )
    frontier = select_uk_frontier(weights, {"01.1.1.7/8"})
    assert set(frontier["coicop_code"]) == {"01.1.1.7/8"}


def test_implied_prefixes_expands_compressed_codes():
    assert _implied_prefixes("08.2/3") == ["08.2", "08.3"]
    assert _implied_prefixes("01.1.9.1/2") == ["01.1.9.1", "01.1.9.2"]
    assert _implied_prefixes("07.1.1A") == ["07.1.1A"]


def test_load_uk_weight_basis_covers_essentially_the_whole_basket_when_all_available():
    weights = load_uk_weight_basis(2026)
    all_codes = set(weights.loc[weights["coicop_code"] != "00", "coicop_code"])
    frontier = select_uk_frontier(weights, all_codes)
    # With every published code treated as available, cascading residuals
    # should recover essentially the full basket (some residual dust from
    # the compressed-range/duplicate-header edge cases is expected).
    assert 99.5 < frontier["weight_pct"].sum() <= 100.5
    assert weights.loc[weights["coicop_code"] == "00", "weight_pct"].iloc[0] == 100.0


def test_validate_full_basket_coverage_rejects_a_broken_parse():
    from inflation_engine.features.uk_item_decomposition import (
        _validate_full_basket_coverage,
    )

    tiny_frontier = pd.DataFrame({"coicop_code": ["01.1"], "weight_pct": [0.5]})
    with pytest.raises(ValueError, match="outside the expected"):
        _validate_full_basket_coverage(tiny_frontier, 2026)


@pytest.mark.integration
def test_uk_item_layer_reconciles_to_headline():
    raw = pd.read_parquet("data/raw/ons_uk_cpi.parquet")
    result = build_uk_item_decomposition(raw)
    assert result["reconciliation_row"].any()
    for date, group in result.groupby("date"):
        headline = group["headline_yoy"].dropna()
        if headline.empty:
            continue
        assert abs(
            group["estimated_contribution_yoy_pp"].sum() - headline.iloc[0]
        ) < 1e-9
    latest = result[result["date"] == result["date"].max()]
    non_reconciliation = latest.loc[~latest["reconciliation_row"]]
    # Cascading residuals recover nearly all of the basket now, including
    # the Education gap the single top-level reconciliation used to absorb.
    assert non_reconciliation["weight_pct"].sum() > 99.0


def test_uk_item_layer_requires_gb_component_rows():
    with pytest.raises(ValueError, match="missing columns"):
        build_uk_item_decomposition(pd.DataFrame({"date": []}))
