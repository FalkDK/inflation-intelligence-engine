import pandas as pd
import pytest

from inflation_engine.data.fetch_eurostat_items import HICP_ITEM_GEOGRAPHIES
from inflation_engine.features.eurozone_item_decomposition import (
    build_eurozone_item_decomposition,
    select_eurozone_frontier,
)


def _weights(rows):
    frame = pd.DataFrame(
        rows, columns=["coicop_code", "weight_pct", "label"]
    )
    frame["weight_reference_date"] = pd.Timestamp("2025-01-01")
    frame["weight_basis_year"] = 2025
    return frame


def test_frontier_descends_when_children_fully_cover_the_parent():
    weights = _weights(
        [
            ("CP00", 100.0, "All-items HICP"),
            ("CP01", 20.0, "Food"),
            ("CP011", 12.0, "Food (detail)"),
            ("CP012", 8.0, "Non-alcoholic beverages"),
        ]
    )

    frontier = select_eurozone_frontier(
        weights, available_component_codes={"CP00", "CP01", "CP011", "CP012"}
    )

    assert set(frontier["coicop_code"]) == {"CP011", "CP012"}
    assert frontier.set_index("coicop_code")["weight_pct"].to_dict() == {
        "CP011": 12.0,
        "CP012": 8.0,
    }
    assert not frontier["label"].str.contains("unitemized").any()


def test_frontier_cascades_a_residual_when_children_only_partially_cover_it():
    """CP08's real-world shape: a parent with a weight and its own index,
    whose one published child only covers a small slice of it."""
    weights = _weights(
        [
            ("CP00", 100.0, "All-items HICP"),
            ("CP08", 24.4, "Communications"),
            ("CP081", 1.6, "Postal services"),
        ]
    )

    frontier = select_eurozone_frontier(
        weights, available_component_codes={"CP00", "CP08", "CP081"}
    )

    result = frontier.set_index("coicop_code")["weight_pct"].to_dict()
    assert result == {"CP081": 1.6, "CP08": pytest.approx(22.8)}
    residual_label = frontier.set_index("coicop_code")["label"]["CP08"]
    assert "unitemized remainder only" in residual_label


def test_frontier_bubbles_up_when_a_child_has_no_available_index():
    weights = _weights(
        [
            ("CP00", 100.0, "All-items HICP"),
            ("CP01", 20.0, "Food"),
            ("CP011", 20.0, "Food (detail)"),
        ]
    )

    frontier = select_eurozone_frontier(
        weights, available_component_codes={"CP00", "CP01"}
    )

    assert frontier.set_index("coicop_code")["weight_pct"].to_dict() == {"CP01": 20.0}


def test_frontier_requires_the_cp00_root():
    weights = _weights([("CP01", 20.0, "Food")])

    with pytest.raises(ValueError, match="CP00 root"):
        select_eurozone_frontier(weights, available_component_codes={"CP01"})


def test_eurozone_item_layer_requires_expected_columns():
    with pytest.raises(ValueError, match="missing columns"):
        build_eurozone_item_decomposition(pd.DataFrame({"date": []}), pd.DataFrame())


@pytest.mark.integration
def test_eurozone_item_layer_reconciles_to_headline():
    items = pd.read_parquet("data/raw/eurostat_hicp_items.parquet")
    weights = pd.read_parquet("data/raw/eurostat_hicp_item_weights.parquet")

    result = build_eurozone_item_decomposition(items, weights)

    assert set(result["country_code"]) == set(HICP_ITEM_GEOGRAPHIES)
    assert result["reconciliation_row"].any()
    for _, group in result.groupby(["country_code", "date"]):
        headline = group["headline_yoy"].dropna()
        if headline.empty:
            continue
        assert abs(
            group["estimated_contribution_yoy_pp"].sum() - headline.iloc[0]
        ) < 1e-9

    latest = result[result["date"] == result["date"].max()]
    for _, group in latest.groupby("country_code"):
        non_reconciliation = group.loc[~group["reconciliation_row"]]
        assert 95.0 < non_reconciliation["weight_pct"].sum() < 101.0
