"""Additive US PCE item-level decomposition from BEA's underlying detail.

Sits beside us_item_decomposition.py (US CPI) and never reads or writes
any of its inputs or outputs. The input is ``bea_us_pce.parquet`` from
fetch_bea_pce.py: every line of NIPA Tables 2.4.4U (Fisher price indexes)
and 2.4.5U (nominal spending), with the additive product tree.

Unlike CPI, PCE is a chained Fisher index whose weights move every month,
so a fixed-weight ``weight x rate`` approximation leaves a sizeable gap.
Instead, contributions are built the way BEA builds its own contributions
to percent change:

1. Each month-to-month link gets exact additive Fisher contributions
   (Reinsdorf-Diewert-Ehemann): for frontier item i with prices p0/p1 and
   real quantities q0/q1 (nominal / price),
   ``c_i = (q0_i + q1_i / Q_F) (p1_i - p0_i) / sum_j (q0_j + q1_j / Q_F) p0_j``,
   where ``Q_F`` is the aggregate's Fisher quantity relative.
2. The twelve links ending at t are summed into a YoY contribution, each
   link scaled by the aggregate's level at the start of that link relative
   to t-12, so they add up exactly: ``P_t / P_{t-12} - 1 =
   sum_m (P_{m-1} / P_{t-12}) g_m``.

The frontier is the deepest set of tree lines with a price and spending
in both months of a link. BEA has discontinued some splits over time
(e.g. tenant-occupied stationary homes vs. landlord durables), so the
frontier falls back to the nearest covered ancestor wherever a subtree
isn't fully covered. Lines labelled "Less: ..." enter with a negative sign,
so their contributions are signed too. Whatever the frontier doesn't explain
(published-index rounding, BEA's own aggregation details) lands in one
reconciliation row per date: typically ~0.01pp, up to ~0.2pp in 2020-21.

Core contributions use the same links against BEA's published "PCE
excluding food and energy" aggregate; food and energy are the subtrees
under food and beverages purchased for off-premises consumption, gasoline
and other energy goods, and electricity and gas (which together match BEA's
own "PCE food and energy" nominal line exactly).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from inflation_engine.utils.config import COUNTRIES
from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.paths import BEA_US_PCE_PATH, US_PCE_ITEM_DECOMPOSITION_PATH
from inflation_engine.utils.pce_tree import ROOT_LINE, PceTree, link_coverage


PCE_COUNTRY_CODE = "US_PCE"
HEADLINE_PRICE_SERIES = "DPCERG"
CORE_PRICE_SERIES = "DPCCRG"
FOOD_AND_ENERGY_NOMINAL_SERIES = ("DFXARC", "DGOERC", "DELGRC")
RECONCILIATION_CODE = "__reconciliation__"

OUTPUT_COLUMNS = [
    "date",
    "country",
    "country_code",
    "table_line",
    "source_component_code",
    "label",
    "parent_line",
    "hierarchy_level",
    "product_group",
    "aggregation_sign",
    "in_core",
    "series_role",
    "weight_pct",
    "core_weight_pct",
    "weight_reference_date",
    "index_value",
    "inflation_yoy",
    "estimated_contribution_yoy_pp",
    "estimated_core_contribution_yoy_pp",
    "headline_yoy",
    "core_yoy",
    "contribution_method",
    "reconciliation_row",
]

REQUIRED_DETAIL_COLUMNS = {
    "date",
    "table_line",
    "label",
    "parent_line",
    "hierarchy_level",
    "aggregation_sign",
    "additive_tree",
    "price_series_id",
    "nominal_series_id",
    "price_index",
    "nominal_spending_musd",
}


def fisher_link_contributions(
    p0: np.ndarray,
    p1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    *,
    aggregate_price_relative: float,
    aggregate_nominal_relative: float,
) -> np.ndarray:
    """Exact additive Fisher contributions (as fractions, not percent) to
    one link's aggregate price change. ``y0``/``y1`` are signed nominal
    spending; they sum to the aggregate whose relatives are given."""
    q0 = y0 / p0
    q1 = y1 / p1
    quantity_relative = aggregate_nominal_relative / aggregate_price_relative
    numerator = (q0 + q1 / quantity_relative) * (p1 - p0)
    denominator = (q0 * p0).sum() + (q1 * p0).sum() / quantity_relative
    return numerator / denominator


class _DecompositionTree(PceTree):
    """The shared PCE tree plus the core and product-group labels the
    decomposition output carries."""

    def __init__(self, structure: pd.DataFrame):
        super().__init__(structure)
        food_energy_roots = self.lines_for_nominal_series(FOOD_AND_ENERGY_NOMINAL_SERIES)
        if len(food_energy_roots) != len(FOOD_AND_ENERGY_NOMINAL_SERIES):
            raise ValueError("PCE tree is missing a food or energy subtree")
        # A line straddling the core/non-core boundary (an ancestor of a
        # food/energy subtree) is neither; it only reaches the frontier when
        # its subtree lacks data, and then lands in core's reconciliation.
        non_core = self.subtree(food_energy_roots) | self.strict_ancestors(food_energy_roots)
        self.in_core = {line: line not in non_core for line in self.parent}
        self.product_group = {line: self._product_group(line) for line in self.parent}

    def _product_group(self, line: int) -> str:
        """The level-2 ancestor, e.g. "Durable goods" or "Household
        consumption expenditures (for services)"."""
        chain = self.ancestry(line)
        group = chain[-3] if len(chain) >= 3 else line
        return str(self.structure.at[group, "label"])


def _wide(detail: pd.DataFrame, column: str, lines) -> pd.DataFrame:
    wide = detail.pivot(index="date", columns="table_line", values=column)
    calendar = pd.date_range(wide.index.min(), wide.index.max(), freq="MS")
    return wide.reindex(index=calendar, columns=list(lines))


def _addendum_series(detail: pd.DataFrame, price_series_id: str) -> pd.DataFrame:
    rows = detail.loc[detail["price_series_id"] == price_series_id]
    if rows.empty:
        raise ValueError(f"BEA PCE detail has no {price_series_id} aggregate")
    return rows.drop_duplicates("date").set_index("date")[
        ["price_index", "nominal_spending_musd"]
    ]


def _monthly_contributions(
    tree: PceTree,
    prices: pd.DataFrame,
    nominal: pd.DataFrame,
    aggregate: pd.DataFrame,
    lines: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-link Fisher contributions (fractions) for each month's frontier,
    restricted to ``lines``, plus which lines sat on that frontier; months
    without a full link are left NaN."""
    aggregate = aggregate.reindex(prices.index)
    signs = pd.Series(tree.net_sign)
    link_ok = link_coverage(prices, nominal)
    allowed = set(lines)
    contributions = pd.DataFrame(np.nan, index=prices.index, columns=lines)
    on_frontier = pd.DataFrame(False, index=prices.index, columns=lines)
    frontier_cache: dict[frozenset[int], list[int]] = {}
    for position in range(1, len(prices.index)):
        date, previous = prices.index[position], prices.index[position - 1]
        p_agg = aggregate["price_index"].loc[[previous, date]].to_numpy()
        y_agg = aggregate["nominal_spending_musd"].loc[[previous, date]].to_numpy()
        if np.isnan(p_agg).any() or np.isnan(y_agg).any():
            continue
        covered = frozenset(link_ok.columns[link_ok.loc[date].to_numpy()])
        if covered not in frontier_cache:
            frontier_cache[covered] = tree.frontier(covered)
        frontier = [line for line in frontier_cache[covered] if line in allowed]
        if not frontier:
            continue
        line_signs = signs.loc[frontier].to_numpy()
        link = fisher_link_contributions(
            prices.loc[previous, frontier].to_numpy(dtype=float),
            prices.loc[date, frontier].to_numpy(dtype=float),
            nominal.loc[previous, frontier].to_numpy(dtype=float) * line_signs,
            nominal.loc[date, frontier].to_numpy(dtype=float) * line_signs,
            aggregate_price_relative=p_agg[1] / p_agg[0],
            aggregate_nominal_relative=y_agg[1] / y_agg[0],
        )
        contributions.loc[date] = 0.0
        contributions.loc[date, frontier] = link
        on_frontier.loc[date, frontier] = True
    return contributions, on_frontier


def _yoy_contributions(
    monthly: pd.DataFrame, aggregate_price: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Chain twelve link contributions into YoY percentage points."""
    level = aggregate_price.reindex(monthly.index)
    scaled = monthly.mul(level.shift(1), axis=0)
    window_complete = monthly.notna().all(axis=1).astype(float).rolling(12).sum() == 12
    summed = scaled.fillna(0.0).rolling(12).sum()
    yoy = summed.div(level.shift(12), axis=0) * 100.0
    yoy.loc[~window_complete] = np.nan
    headline = (level / level.shift(12) - 1.0) * 100.0
    return yoy, headline


def build_us_pce_item_decomposition(detail: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_DETAIL_COLUMNS.difference(detail.columns)
    if missing:
        raise ValueError(f"BEA PCE detail is missing columns: {sorted(missing)}")
    detail = detail.copy()
    detail["date"] = pd.to_datetime(detail["date"])

    tree_rows = detail.loc[detail["additive_tree"]]
    structure = tree_rows.drop_duplicates("table_line").loc[
        :,
        [
            "table_line",
            "label",
            "parent_line",
            "hierarchy_level",
            "aggregation_sign",
            "price_series_id",
            "nominal_series_id",
        ],
    ]
    tree = _DecompositionTree(structure)
    all_lines = [line for line in structure["table_line"] if line != ROOT_LINE]
    core_lines = [line for line in all_lines if tree.in_core[line]]

    prices = _wide(tree_rows, "price_index", all_lines)
    nominal = _wide(tree_rows, "nominal_spending_musd", all_lines)
    root = tree_rows.loc[tree_rows["table_line"] == ROOT_LINE].set_index("date")[
        ["price_index", "nominal_spending_musd"]
    ]
    core = _addendum_series(detail, CORE_PRICE_SERIES)

    headline_monthly, on_frontier = _monthly_contributions(
        tree, prices, nominal, root, all_lines
    )
    core_monthly, _ = _monthly_contributions(tree, prices, nominal, core, core_lines)
    headline_contrib, headline_yoy = _yoy_contributions(
        headline_monthly, root["price_index"]
    )
    core_contrib, core_yoy = _yoy_contributions(core_monthly, core["price_index"])
    core_contrib = core_contrib.reindex(columns=all_lines)

    signs = pd.Series(tree.net_sign).reindex(all_lines)
    signed_nominal = nominal.mul(signs, axis=1)
    weight = signed_nominal.div(root["nominal_spending_musd"].reindex(nominal.index), axis=0)
    core_weight = signed_nominal.loc[:, core_lines].div(
        core["nominal_spending_musd"].reindex(nominal.index), axis=0
    ).reindex(columns=all_lines)
    inflation_yoy = (prices / prices.shift(12) - 1.0) * 100.0

    def stack(frame: pd.DataFrame, name: str) -> pd.Series:
        return frame.rename_axis(index="date", columns="table_line").stack(
            future_stack=True
        ).rename(name)

    items = pd.concat(
        [
            stack(headline_contrib, "estimated_contribution_yoy_pp"),
            stack(core_contrib, "estimated_core_contribution_yoy_pp"),
            stack(weight.shift(12) * 100.0, "weight_pct"),
            stack(core_weight.shift(12) * 100.0, "core_weight_pct"),
            stack(prices, "index_value"),
            stack(inflation_yoy, "inflation_yoy"),
        ],
        axis=1,
    ).reset_index()
    # Keep each line only for the dates it actually contributed to the
    # headline over the trailing year (i.e. it sat on the frontier).
    window_active = on_frontier.astype(int).rolling(12).sum() > 0
    active = stack(window_active, "active").reset_index()
    items = items.merge(active, on=["date", "table_line"], how="left")
    items = items.loc[
        items["active"].fillna(False) & items["estimated_contribution_yoy_pp"].notna()
    ].drop(columns="active")
    items.loc[~items["table_line"].map(tree.in_core), "estimated_core_contribution_yoy_pp"] = np.nan

    meta = structure.set_index("table_line")
    items["source_component_code"] = items["table_line"].map(meta["price_series_id"])
    items["label"] = items["table_line"].map(meta["label"])
    items["parent_line"] = items["table_line"].map(meta["parent_line"])
    items["hierarchy_level"] = items["table_line"].map(meta["hierarchy_level"])
    items["product_group"] = items["table_line"].map(tree.product_group)
    items["aggregation_sign"] = items["table_line"].map(tree.net_sign)
    items["in_core"] = items["table_line"].map(tree.in_core)
    items["series_role"] = "component"
    items["contribution_method"] = "chained_fisher_monthly_links"
    items["reconciliation_row"] = False

    headline_target = headline_yoy.loc[headline_contrib.dropna(how="all").index]
    core_target = core_yoy.reindex(headline_target.index)
    sums = items.groupby("date")["estimated_contribution_yoy_pp"].sum()
    core_sums = items.groupby("date")["estimated_core_contribution_yoy_pp"].sum(min_count=1)
    reconciliation = pd.DataFrame({"date": headline_target.index})
    reconciliation["estimated_contribution_yoy_pp"] = (
        headline_target - sums.reindex(headline_target.index).fillna(0.0)
    ).to_numpy()
    reconciliation["estimated_core_contribution_yoy_pp"] = (
        core_target - core_sums.reindex(headline_target.index)
    ).to_numpy()
    reconciliation["table_line"] = 0
    reconciliation["source_component_code"] = RECONCILIATION_CODE
    reconciliation["label"] = "Unallocated reconciliation to headline and core PCE"
    reconciliation["parent_line"] = pd.NA
    reconciliation["hierarchy_level"] = 0
    reconciliation["product_group"] = ""
    reconciliation["aggregation_sign"] = 1
    reconciliation["in_core"] = True
    reconciliation["series_role"] = "reconciliation"
    reconciliation["weight_pct"] = 0.0
    reconciliation["core_weight_pct"] = 0.0
    reconciliation["index_value"] = np.nan
    reconciliation["inflation_yoy"] = np.nan
    reconciliation["contribution_method"] = "headline_accounting_residual"
    reconciliation["reconciliation_row"] = True

    items["parent_line"] = items["parent_line"].astype("Int64")
    reconciliation = reconciliation.astype(
        {column: items[column].dtype for column in reconciliation.columns if column in items}
    )
    output = pd.concat([items, reconciliation], ignore_index=True, sort=False)
    output["country"] = COUNTRIES[PCE_COUNTRY_CODE]["name"]
    output["country_code"] = PCE_COUNTRY_CODE
    output["weight_reference_date"] = output["date"] - pd.DateOffset(months=12)
    output["headline_yoy"] = output["date"].map(headline_yoy)
    output["core_yoy"] = output["date"].map(core_yoy)
    output["parent_line"] = output["parent_line"].astype("Int64")
    output["table_line"] = output["table_line"].astype(int)
    output["hierarchy_level"] = output["hierarchy_level"].astype(int)
    output["aggregation_sign"] = output["aggregation_sign"].astype(int)
    output["in_core"] = output["in_core"].astype(bool)
    return (
        output.loc[:, OUTPUT_COLUMNS]
        .sort_values(["date", "reconciliation_row", "table_line"], kind="stable")
        .reset_index(drop=True)
    )


def save_us_pce_item_decomposition(
    frame: pd.DataFrame,
    output_path: str | Path = US_PCE_ITEM_DECOMPOSITION_PATH,
) -> Path:
    return atomic_write_parquet(frame, output_path, columns=OUTPUT_COLUMNS)


def run(
    input_path: str | Path = BEA_US_PCE_PATH,
    output_path: str | Path = US_PCE_ITEM_DECOMPOSITION_PATH,
) -> Path:
    detail = pd.read_parquet(input_path)
    return save_us_pce_item_decomposition(
        build_us_pce_item_decomposition(detail), output_path
    )


if __name__ == "__main__":
    destination = run()
    print(f"Saved US PCE item decomposition to {destination}")
