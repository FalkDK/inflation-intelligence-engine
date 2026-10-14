import numpy as np
import pandas as pd
import pytest

from inflation_engine.features.us_pce_item_decomposition import (
    build_us_pce_item_decomposition,
    fisher_link_contributions,
)


def _fisher_relative(p0, p1, y0, y1):
    q0, q1 = y0 / p0, y1 / p1
    laspeyres = (p1 * q0).sum() / (p0 * q0).sum()
    paasche = (p1 * q1).sum() / (p0 * q1).sum()
    return np.sqrt(laspeyres * paasche)


def test_fisher_link_contributions_add_up_exactly():
    rng = np.random.default_rng(0)
    p0, p1 = rng.uniform(80, 120, 5), rng.uniform(80, 120, 5)
    y0, y1 = rng.uniform(10, 50, 5), rng.uniform(10, 50, 5)
    y0[-1], y1[-1] = -5.0, -6.0  # a subtracted ("Less:") line
    relative = _fisher_relative(p0, p1, y0, y1)
    contributions = fisher_link_contributions(
        p0,
        p1,
        y0,
        y1,
        aggregate_price_relative=relative,
        aggregate_nominal_relative=y1.sum() / y0.sum(),
    )
    assert contributions.sum() == pytest.approx(relative - 1.0, abs=1e-12)


# Tree (by table line): 1 PCE = 2 food + 3 energy goods + 4 electricity
# and gas + 5 services, where 5 = 6 item A, less 7 receipts (subtracted).
# Line 8 is BEA's "PCE excluding food and energy" addendum.
LEAVES = {
    2: ("DFXARC", 1),
    3: ("DGOERC", 1),
    4: ("DELGRC", 1),
    6: ("XAAARC", 1),
    7: ("XBBBRC", -1),
}
PARENTS = {1: None, 2: 1, 3: 1, 4: 1, 5: 1, 6: 5, 7: 5}


def _chained_fisher(prices: pd.DataFrame, nominal: pd.DataFrame) -> pd.Series:
    level = [100.0]
    for position in range(1, len(prices)):
        level.append(
            level[-1]
            * _fisher_relative(
                prices.iloc[position - 1].to_numpy(),
                prices.iloc[position].to_numpy(),
                nominal.iloc[position - 1].to_numpy(),
                nominal.iloc[position].to_numpy(),
            )
        )
    return pd.Series(level, index=prices.index)


def _synthetic_detail(months: int = 30, *, drop_services_split_before: int = 0):
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-01", periods=months, freq="MS")
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rng.normal(0.003, 0.01, (months, len(LEAVES))), axis=0),
        index=dates,
        columns=list(LEAVES),
    )
    nominal = pd.DataFrame(
        rng.uniform(50, 150, (months, len(LEAVES))), index=dates, columns=list(LEAVES)
    )
    nominal[7] = nominal[6] * 0.2  # receipts stay smaller than gross output
    signs = pd.Series({line: sign for line, (_, sign) in LEAVES.items()})
    signed = nominal * signs

    services_prices = _chained_fisher(prices[[6, 7]], signed[[6, 7]])
    core_lines = [6, 7]
    series = {
        1: (_chained_fisher(prices, signed), signed.sum(axis=1), "DPCERG", "DPCERC", "PCE", 1),
        5: (services_prices, signed[[6, 7]].sum(axis=1), "XSVCRG", "XSVCRC", "Services", 1),
        8: (
            _chained_fisher(prices[core_lines], signed[core_lines]),
            signed[core_lines].sum(axis=1),
            "DPCCRG",
            "DPCCRC",
            "PCE excluding food and energy",
            1,
        ),
    }
    for line, (code, sign) in LEAVES.items():
        label = f"Less: item {line}" if sign < 0 else f"Item {line}"
        series[line] = (prices[line], nominal[line], code[:-1] + "G", code, label, sign)

    rows = []
    for line, (price, spend, price_code, nominal_code, label, sign) in series.items():
        price = price.copy()
        if line in (6, 7) and drop_services_split_before:
            price.iloc[:drop_services_split_before] = np.nan
        in_tree = line != 8
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "table_line": line,
                    "label": label,
                    "parent_line": PARENTS.get(line) if in_tree else None,
                    "hierarchy_level": 0 if line == 1 else (2 if line in (6, 7) else 1),
                    "aggregation_sign": sign,
                    "additive_tree": in_tree,
                    "price_series_id": price_code,
                    "nominal_series_id": nominal_code,
                    "price_index": price.to_numpy(),
                    "nominal_spending_musd": spend.to_numpy(),
                }
            )
        )
    detail = pd.concat(rows, ignore_index=True)
    detail["parent_line"] = detail["parent_line"].astype("Int64")
    return detail


def test_contributions_reconcile_exactly_to_chained_fisher_headline_and_core():
    output = build_us_pce_item_decomposition(_synthetic_detail())
    items = output.loc[~output["reconciliation_row"]]
    reconciliation = output.loc[output["reconciliation_row"]]

    assert output["date"].min() == pd.Timestamp("2021-01-01")
    assert set(items["table_line"]) == {2, 3, 4, 6, 7}
    assert reconciliation["estimated_contribution_yoy_pp"].abs().max() < 1e-9
    assert reconciliation["estimated_core_contribution_yoy_pp"].abs().max() < 1e-9

    totals = output.groupby("date")["estimated_contribution_yoy_pp"].sum()
    headline = output.groupby("date")["headline_yoy"].first()
    pd.testing.assert_series_equal(totals, headline, check_names=False, atol=1e-12)


def test_core_flags_and_signed_weights():
    output = build_us_pce_item_decomposition(_synthetic_detail())
    latest = output.loc[
        (output["date"] == output["date"].max()) & ~output["reconciliation_row"]
    ].set_index("table_line")
    assert not latest.loc[[2, 3, 4], "in_core"].any()
    assert latest.loc[[6, 7], "in_core"].all()
    assert latest.loc[[2, 3, 4], "estimated_core_contribution_yoy_pp"].isna().all()
    assert latest.at[7, "aggregation_sign"] == -1
    assert latest.at[7, "weight_pct"] < 0
    assert latest["weight_pct"].sum() == pytest.approx(100.0)
    assert latest.loc[[6, 7], "core_weight_pct"].sum() == pytest.approx(100.0)


def test_frontier_falls_back_to_ancestor_when_split_is_unpublished():
    output = build_us_pce_item_decomposition(
        _synthetic_detail(drop_services_split_before=15)
    )
    items = output.loc[~output["reconciliation_row"]]
    early = items.loc[items["date"] == pd.Timestamp("2021-01-01"), "table_line"]
    late = items.loc[items["date"] == output["date"].max(), "table_line"]
    assert set(early) == {2, 3, 4, 5}
    assert set(late) == {2, 3, 4, 6, 7}
    # Chained Fisher isn't consistent in aggregation, so treating the
    # services aggregate as one item leaves a small (not zero) residual,
    # which vanishes once the full split is on the frontier all year.
    reconciliation = output.loc[output["reconciliation_row"]].set_index("date")
    assert reconciliation["estimated_contribution_yoy_pp"].abs().max() < 0.01
    assert abs(reconciliation["estimated_contribution_yoy_pp"].iloc[-1]) < 1e-9


def test_rejects_detail_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        build_us_pce_item_decomposition(pd.DataFrame({"date": []}))
