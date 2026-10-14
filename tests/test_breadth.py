import numpy as np
import pandas as pd
import pytest

from inflation_engine.data.fetch_eurostat_items import HICP_ITEM_GEOGRAPHIES
from inflation_engine.features.breadth import (
    build_bucket_breadth,
    build_item_breadth,
    build_item_breadth_denmark,
    build_item_breadth_eurozone,
    build_item_breadth_norway,
    build_item_breadth_sweden,
    build_item_breadth_uk,
    build_item_breadth_us,
    build_item_breadth_us_pce,
)


@pytest.mark.integration
def test_bucket_breadth_covers_both_horizons_for_every_geography():
    cpi = pd.read_parquet("data/processed/cpi_panel.parquet")
    bls_weights = pd.read_parquet("data/raw/bls_us_cpi_weights.parquet")
    no_weights = pd.read_parquet("data/raw/ssb_norway_cpi_weights.parquet")
    no_contract = pd.read_parquet("data/raw/ssb_norway_component_contract.parquet")
    hicp_weights = pd.read_parquet("data/raw/eurostat_hicp_weights.parquet")
    uk_weights = pd.read_parquet("data/raw/ons_uk_cpi_weights.parquet")
    comparable_weights = pd.concat([hicp_weights, uk_weights], ignore_index=True)

    result = build_bucket_breadth(
        cpi,
        bls_weights=bls_weights,
        no_weights=no_weights,
        no_contract=no_contract,
        comparable_weights=comparable_weights,
    )

    assert set(result["horizon"]) == {"12m_yoy", "6m_ann"}
    counts = result.groupby("country_code")["horizon"].nunique()
    assert (counts == 2).all()
    assert result["weighted_share_above_2pct"].between(0, 100).all()
    assert result["weighted_share_above_3pct"].between(0, 100).all()
    # A bucket above the 3% threshold is, by construction, also above 2%.
    assert (
        result["weighted_share_above_2pct"] >= result["weighted_share_above_3pct"] - 1e-9
    ).all()

    # Unweighted is a plain fraction of the five buckets, so it only ever
    # lands on multiples of 20%.
    for column in (
        "unweighted_share_above_2pct",
        "unweighted_share_above_3pct",
        "unweighted_share_above_4pct",
    ):
        assert result[column].between(0, 100).all()
        assert np.isclose(result[column] % 20, 0, atol=1e-6).all()
    assert (
        result["unweighted_share_above_2pct"]
        >= result["unweighted_share_above_3pct"] - 1e-9
    ).all()


@pytest.mark.integration
def test_us_item_breadth_is_weighted_and_reasonable():
    frame = pd.read_parquet("data/processed/us_item_decomposition.parquet")
    result = build_item_breadth_us(frame)
    assert (result["n_items"] > 0).all()
    assert set(result["horizon"]) == {"12m_yoy", "6m_ann"}
    assert result["n_above_2pct"].le(result["n_items"]).all()
    assert result["weighted_share_above_2pct"].between(0, 100).all()
    assert result["unweighted_share_above_2pct"].between(0, 100).all()
    assert result["unweighted_share_above_3pct"].between(0, 100).all()
    # Unweighted share is exactly the plain count fraction, independent of weight.
    expected_unweighted_2pct = result["n_above_2pct"] / result["n_items"] * 100.0
    pd.testing.assert_series_equal(
        result["unweighted_share_above_2pct"],
        expected_unweighted_2pct,
        check_names=False,
    )


@pytest.mark.integration
def test_uk_item_breadth_is_weighted_and_reasonable():
    frame = pd.read_parquet("data/processed/uk_item_decomposition.parquet")
    result = build_item_breadth_uk(frame)
    assert (result["n_items"] > 0).all()
    assert result["weighted_share_above_2pct"].between(0, 100).all()
    assert result["unweighted_share_above_2pct"].between(0, 100).all()


@pytest.mark.integration
def test_danish_item_breadth_is_weighted_and_reasonable():
    frame = pd.read_parquet("data/processed/danish_item_decomposition.parquet")
    result = build_item_breadth_denmark(frame)
    assert (result["n_items"] > 0).all()
    assert result["weighted_share_above_2pct"].between(0, 100).all()
    assert result["unweighted_share_above_2pct"].between(0, 100).all()


@pytest.mark.integration
def test_norway_item_breadth_is_weighted_and_reasonable():
    frame = pd.read_parquet("data/processed/norway_item_decomposition.parquet")
    result = build_item_breadth_norway(frame)
    assert (result["n_items"] > 0).all()
    assert result["weighted_share_above_2pct"].between(0, 100).all()
    assert result["unweighted_share_above_2pct"].between(0, 100).all()


@pytest.mark.integration
def test_sweden_item_breadth_covers_sweden_only():
    frame = pd.read_parquet("data/processed/nordic_item_contributions.parquet")
    result = build_item_breadth_sweden(frame)
    assert set(result["country_code"]) == {"SE"}
    assert (result["n_items"] > 0).all()


@pytest.mark.integration
def test_eurozone_item_breadth_covers_every_member_separately():
    frame = pd.read_parquet("data/processed/eurozone_item_decomposition.parquet")
    result = build_item_breadth_eurozone(frame)
    assert set(result["country_code"]) == set(HICP_ITEM_GEOGRAPHIES)
    counts = result.groupby("country_code")["horizon"].nunique()
    assert (counts == 2).all()
    assert (result["n_items"] > 0).all()
    assert result["weighted_share_above_2pct"].between(0, 100).all()
    assert result["unweighted_share_above_2pct"].between(0, 100).all()


@pytest.mark.integration
def test_build_item_breadth_combines_all_available_countries():
    us = pd.read_parquet("data/processed/us_item_decomposition.parquet")
    uk = pd.read_parquet("data/processed/uk_item_decomposition.parquet")
    dk = pd.read_parquet("data/processed/danish_item_decomposition.parquet")
    no = pd.read_parquet("data/processed/norway_item_decomposition.parquet")
    nordic = pd.read_parquet("data/processed/nordic_item_contributions.parquet")
    eurozone = pd.read_parquet("data/processed/eurozone_item_decomposition.parquet")

    result = build_item_breadth(
        us_item_decomposition=us,
        uk_item_decomposition=uk,
        danish_item_decomposition=dk,
        norway_item_decomposition=no,
        nordic_item_contributions=nordic,
        eurozone_item_decomposition=eurozone,
    )

    assert set(result["country_code"]) == {
        "US", "GB", "DK", "NO", "SE",
    } | set(HICP_ITEM_GEOGRAPHIES)


@pytest.mark.integration
def test_build_item_breadth_skips_missing_countries_gracefully():
    us = pd.read_parquet("data/processed/us_item_decomposition.parquet")
    result = build_item_breadth(us_item_decomposition=us)
    assert set(result["country_code"]) == {"US"}


def test_us_pce_item_breadth_collapses_npish_and_drops_negative_lines():
    dates = pd.date_range("2024-01-01", "2025-08-01", freq="MS")
    npish_label = "Final consumption expenditures of NPISHs"
    decomposition_rows = []
    for date in dates[12:]:
        for line, group, sign, weight, rate, index in [
            (10, "Durable goods", 1, 40.0, 5.0, 110.0),
            (11, "Nondurable goods", 1, 50.0, 1.0, 101.0),
            (20, npish_label, 1, 6.0, 9.0, 120.0),     # NPISH gross output
            (21, npish_label, -1, -2.0, 9.0, 120.0),   # Less: receipts
            (30, "Durable goods", 1, -1.0, 9.0, 120.0),  # net transactions < 0
        ]:
            decomposition_rows.append(
                {
                    "date": date, "table_line": line, "product_group": group,
                    "aggregation_sign": sign, "weight_pct": weight,
                    "index_value": index * (1 + 0.001 * dates.get_loc(date)),
                    "inflation_yoy": rate, "reconciliation_row": False,
                }
            )
    decomposition = pd.DataFrame(decomposition_rows)
    detail = pd.concat(
        [
            pd.DataFrame(
                {"date": dates, "table_line": 1, "label": "PCE",
                 "nominal_series_id": "DPCERC", "additive_tree": True,
                 "price_index": 100.0, "nominal_spending_musd": 1000.0}
            ),
            pd.DataFrame(
                {"date": dates, "table_line": 19, "label": npish_label,
                 "nominal_series_id": "DNPIRC", "additive_tree": True,
                 "price_index": np.linspace(100.0, 104.0, len(dates)),
                 "nominal_spending_musd": 40.0}
            ),
        ],
        ignore_index=True,
    )

    result = build_item_breadth_us_pce(decomposition, detail).set_index("horizon")

    yoy = result.loc["12m_yoy"]
    assert yoy["country_code"] == "US_PCE"
    assert yoy["n_items"] == 3  # lines 10, 11 and the collapsed NPISH line 19
    assert yoy["n_above_2pct"] == 2  # line 10 (5%) and NPISH (~2.5%)
    # Weighted: (40 + 4) / (40 + 50 + 4), NPISH weighing 40 / 1000 = 4%.
    assert yoy["weighted_share_above_2pct"] == pytest.approx(44 / 94 * 100)
    assert result.loc["6m_ann", "n_items"] == 3

