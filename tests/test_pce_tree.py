import numpy as np
import pandas as pd
import pytest

from inflation_engine.data.fetch_bea_pce import SUPERCORE_COMPONENT, build_pce_supercore
from inflation_engine.utils.pce_tree import PceTree, chained_fisher_index
from inflation_engine.utils.schema import validate_cpi_schema


# 1 PCE = 2 Services; Services = 3 Housing (4 rent) + 5 Utilities (6
# electricity and gas) + 7 other + 8 restaurants.
STRUCTURE = pd.DataFrame(
    {
        "table_line": [1, 2, 3, 4, 5, 6, 7, 8],
        "label": ["PCE", "Services", "Housing", "Rent", "Utilities", "Electricity and gas",
                  "Other", "Restaurants"],
        "parent_line": pd.array([None, 1, 2, 3, 2, 5, 2, 2], dtype="Int64"),
        "aggregation_sign": [1] * 8,
        "nominal_series_id": ["DPCERC", "DSERRC", "DHSGRC", "XRENTC", "XUTILC", "DELGRC",
                              "XOTHRC", "XRESTC"],
    }
)
LEAVES = [4, 6, 7, 8]


def _fisher_relative(p0, p1, y0, y1):
    q0, q1 = y0 / p0, y1 / p1
    return np.sqrt(((p1 * q0).sum() / (p0 * q0).sum()) * ((p1 * q1).sum() / (p0 * q1).sum()))


def _data(months: int = 36):
    rng = np.random.default_rng(3)
    dates = pd.date_range("2016-01-01", periods=months, freq="MS")
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.003, 0.01, (months, len(LEAVES))), axis=0),
        index=dates, columns=LEAVES,
    )
    nominal = pd.DataFrame(rng.uniform(50, 150, (months, len(LEAVES))), index=dates, columns=LEAVES)
    return prices, nominal


def test_frontier_falls_back_to_covered_ancestor():
    tree = PceTree(STRUCTURE)
    assert set(tree.frontier(frozenset({4, 6, 7, 8}))) == {4, 6, 7, 8}
    assert set(tree.frontier(frozenset({3, 6, 7, 8}))) == {3, 6, 7, 8}
    assert set(tree.frontier(frozenset({2}))) == {2}


def test_chained_fisher_excludes_whole_subtrees_and_rebases():
    tree = PceTree(STRUCTURE)
    prices, nominal = _data()
    index = chained_fisher_index(
        tree, prices.reindex(columns=range(2, 9)), nominal.reindex(columns=range(2, 9)),
        start=2, excluded_lines={3, 6},
    )
    expected = [1.0]
    for position in range(1, len(prices)):
        expected.append(
            expected[-1]
            * _fisher_relative(
                prices.iloc[position - 1][[7, 8]].to_numpy(),
                prices.iloc[position][[7, 8]].to_numpy(),
                nominal.iloc[position - 1][[7, 8]].to_numpy(),
                nominal.iloc[position][[7, 8]].to_numpy(),
            )
        )
    expected = pd.Series(expected, index=prices.index)
    expected = expected / expected.loc["2017"].mean() * 100
    pd.testing.assert_series_equal(index, expected, check_names=False, atol=1e-10)
    assert index.loc["2017"].mean() == pytest.approx(100.0)


def test_supercore_panel_series():
    prices, nominal = _data()
    rows = []
    for line in STRUCTURE["table_line"]:
        meta = STRUCTURE.set_index("table_line").loc[line]
        rows.append(
            pd.DataFrame(
                {
                    "date": prices.index,
                    "table_line": line,
                    "label": meta["label"],
                    "parent_line": meta["parent_line"],
                    "hierarchy_level": 0,
                    "aggregation_sign": 1,
                    "additive_tree": True,
                    "price_series_id": meta["nominal_series_id"][:-1] + "G",
                    "nominal_series_id": meta["nominal_series_id"],
                    "price_index": prices[line].to_numpy() if line in LEAVES else np.nan,
                    "nominal_spending_musd": nominal[line].to_numpy() if line in LEAVES else np.nan,
                }
            )
        )
    frame = build_pce_supercore(pd.concat(rows, ignore_index=True))
    validate_cpi_schema(frame)
    assert set(frame["component"]) == {SUPERCORE_COMPONENT}
    assert set(frame["country_code"]) == {"US_PCE"}
    assert set(frame["series_role"]) == {"component"}
    assert len(frame) == len(prices)
