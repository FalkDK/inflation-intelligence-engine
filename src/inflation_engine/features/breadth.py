"""Inflation breadth: how much of the CPI basket is running hot, at two
horizons (12-month YoY and 6-month annualized), across two granularities.

Tier 1 (all model geographies) reuses the existing validated five-bucket
decomposition in ``decomposition.py`` unchanged -- the 6-month horizon is
produced by swapping in ``inflation_6m_ann`` for ``inflation_yoy`` before
calling the same build functions, since their bucket math only reads
whatever column is named "inflation_yoy" on the input frame.

Tier 2 (item level) uses whichever validated, non-overlapping, weighted item
frontier exists for a country: the US's BLS-relative-importance-matched FRED
frontier, the US PCE chained-Fisher frontier (US_PCE, from BEA's underlying
detail -- see build_item_breadth_us_pce), the UK/Denmark/Norway/Sweden COICOP
item decompositions, and the
euro area's own COICOP item decomposition (all 20 members -- see
eurozone_item_decomposition.py). EA20 itself (the aggregate) is not included
in Tier 2, since Eurostat does not publish a detailed COICOP item breakdown
for the euro-area aggregate the way it does per member state.

Every threshold share is reported two ways: **weighted** (by basket/expenditure
share -- "how much of the dollar is in categories running hot") and
**unweighted** (a plain count of components/buckets above the threshold --
"how many different things are running hot", independent of their size). The
two can diverge a lot: a handful of heavily-weighted categories running hot
can make the weighted share look broad even when most of the basket, by
count, is not. This is also the convention behind headline-grabbing Fed
breadth figures (e.g. "54% of the 199 components in the PCE basket" is a
component count, not an expenditure-weighted share) -- the unweighted series
here is the directly comparable measure.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from inflation_engine.features.decomposition import (
    build_bucket_decomposition,
    build_decomposition_summary,
    build_hicp_bucket_decomposition,
    build_norway_bucket_decomposition,
)
from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.paths import (
    BLS_US_CPI_WEIGHTS_PATH,
    BREADTH_BUCKETS_PATH,
    BREADTH_ITEMS_PATH,
    CPI_PANEL_PATH,
    DANISH_ITEM_DECOMPOSITION_PATH,
    EUROSTAT_HICP_WEIGHTS_PATH,
    EUROZONE_ITEM_DECOMPOSITION_PATH,
    NORDIC_ITEM_CONTRIBUTIONS_PATH,
    NORWAY_ITEM_DECOMPOSITION_PATH,
    ONS_UK_CPI_WEIGHTS_PATH,
    SSB_NORWAY_COMPONENT_CONTRACT_PATH,
    SSB_NORWAY_CPI_WEIGHTS_PATH,
    BEA_US_PCE_PATH,
    UK_ITEM_DECOMPOSITION_PATH,
    US_ITEM_DECOMPOSITION_PATH,
    US_PCE_ITEM_DECOMPOSITION_PATH,
)


BUCKET_BREADTH_COLUMNS = [
    "date", "country", "country_code", "horizon", "headline_rate",
    "weighted_share_above_2pct", "unweighted_share_above_2pct",
    "weighted_share_above_3pct", "unweighted_share_above_3pct",
    "weighted_share_above_4pct", "unweighted_share_above_4pct",
]
ITEM_BREADTH_COLUMNS = [
    "date", "country_code", "horizon", "n_items",
    "n_above_2pct", "weighted_share_above_2pct", "unweighted_share_above_2pct",
    "n_above_3pct", "weighted_share_above_3pct", "unweighted_share_above_3pct",
]


def build_bucket_breadth(
    cpi: pd.DataFrame,
    *,
    bls_weights: pd.DataFrame,
    no_weights: pd.DataFrame | None = None,
    no_contract: pd.DataFrame | None = None,
    comparable_weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Weighted and unweighted five-bucket breadth for every geography with a
    decomposition, at both the 12-month YoY and 6-month annualized horizon.

    The unweighted share is the plain fraction of the five buckets above each
    threshold (0/20/40/60/80/100%) -- coarse at only five buckets, but kept
    alongside the weighted share for the same reason it matters at item level:
    it doesn't let one large bucket (shelter, at over a third of the US
    basket) make the reading look broader than it is."""

    def _build_all(frame: pd.DataFrame) -> pd.DataFrame:
        frames = [build_bucket_decomposition(frame, bls_weights)]
        if no_weights is not None and no_contract is not None:
            frames.append(
                build_norway_bucket_decomposition(frame, no_weights, no_contract)
            )
        if comparable_weights is not None:
            frames.append(build_hicp_bucket_decomposition(frame, comparable_weights))
        return pd.concat(frames, ignore_index=True)

    rows = []
    for horizon, rate_column in (("12m_yoy", "inflation_yoy"), ("6m_ann", "inflation_6m_ann")):
        horizon_cpi = cpi.copy()
        if horizon != "12m_yoy":
            horizon_cpi["inflation_yoy"] = horizon_cpi[rate_column]
            horizon_cpi = horizon_cpi.dropna(subset=["inflation_yoy"])
        buckets = _build_all(horizon_cpi)
        summary = build_decomposition_summary(buckets)
        latest = summary.loc[summary.groupby("country_code")["date"].idxmax()]
        latest_buckets = buckets.merge(
            latest[["country_code", "date"]], on=["country_code", "date"], how="inner"
        )
        unweighted = (
            latest_buckets.groupby("country_code")[["above_2pct", "above_3pct", "above_4pct"]]
            .mean()
            * 100.0
        )
        for _, row in latest.iterrows():
            country_unweighted = unweighted.loc[row["country_code"]]
            rows.append(
                {
                    "date": row["date"],
                    "country": row["country"],
                    "country_code": row["country_code"],
                    "horizon": horizon,
                    "headline_rate": row["headline_yoy"],
                    "weighted_share_above_2pct": row["weighted_share_above_2pct"],
                    "unweighted_share_above_2pct": float(country_unweighted["above_2pct"]),
                    "weighted_share_above_3pct": row["weighted_share_above_3pct"],
                    "unweighted_share_above_3pct": float(country_unweighted["above_3pct"]),
                    "weighted_share_above_4pct": row["weighted_share_above_4pct"],
                    "unweighted_share_above_4pct": float(country_unweighted["above_4pct"]),
                }
            )
    return pd.DataFrame(rows, columns=BUCKET_BREADTH_COLUMNS).sort_values(
        ["country_code", "horizon"]
    ).reset_index(drop=True)


def _threshold_row(
    country_code: str,
    horizon: str,
    frame: pd.DataFrame,
    *,
    rate_column: str,
    weight_column: str,
) -> dict:
    valid = frame.dropna(subset=[rate_column, weight_column])
    total_weight = float(valid[weight_column].sum())
    n_items = len(valid)
    above2 = valid[rate_column] > 2.0
    above3 = valid[rate_column] > 3.0
    return {
        "date": pd.Timestamp.max,  # overwritten by caller with the real date
        "country_code": country_code,
        "horizon": horizon,
        "n_items": int(n_items),
        "n_above_2pct": int(above2.sum()),
        "weighted_share_above_2pct": (
            float(valid.loc[above2, weight_column].sum() / total_weight * 100.0)
            if total_weight else np.nan
        ),
        "unweighted_share_above_2pct": (
            float(above2.sum() / n_items * 100.0) if n_items else np.nan
        ),
        "n_above_3pct": int(above3.sum()),
        "weighted_share_above_3pct": (
            float(valid.loc[above3, weight_column].sum() / total_weight * 100.0)
            if total_weight else np.nan
        ),
        "unweighted_share_above_3pct": (
            float(above3.sum() / n_items * 100.0) if n_items else np.nan
        ),
    }


def _6m_annualized_from_index(
    frame: pd.DataFrame,
    *,
    code_column: str,
    index_column: str,
    date: pd.Timestamp,
) -> pd.Series:
    """Derive a 6-month annualized rate from index levels 6 months apart,
    for item decompositions that only publish a YoY rate directly."""
    prior_date = date - pd.DateOffset(months=6)
    now = frame.loc[frame["date"] == date, [code_column, index_column]]
    prior = frame.loc[frame["date"] == prior_date, [code_column, index_column]].rename(
        columns={index_column: "index_value_prior6"}
    )
    merged = now.merge(prior, on=code_column, how="inner")
    rate = ((merged[index_column] / merged["index_value_prior6"]) ** 2 - 1.0) * 100.0
    return pd.Series(rate.to_numpy(), index=merged[code_column])


def build_item_breadth_us(us_item_decomposition: pd.DataFrame) -> pd.DataFrame:
    frame = us_item_decomposition.loc[~us_item_decomposition["reconciliation_row"]]
    date = frame["date"].max()
    latest = frame.loc[frame["date"] == date].copy()
    rate_6m = _6m_annualized_from_index(
        frame, code_column="source_component_code", index_column="index_value", date=date
    )
    latest["inflation_6m_ann"] = latest["source_component_code"].map(rate_6m)
    rows = []
    for horizon, rate_column in (("12m_yoy", "inflation_yoy"), ("6m_ann", "inflation_6m_ann")):
        row = _threshold_row(
            "US", horizon, latest,
            rate_column=rate_column, weight_column="weight_pct",
        )
        row["date"] = date
        rows.append(row)
    return pd.DataFrame(rows, columns=ITEM_BREADTH_COLUMNS)


PCE_NPISH_NOMINAL_SERIES = "DNPIRC"
PCE_ROOT_LINE = 1


def build_item_breadth_us_pce(
    us_pce_item_decomposition: pd.DataFrame,
    us_pce_detail: pd.DataFrame,
) -> pd.DataFrame:
    """Item breadth over the PCE frontier, restated as a basket of
    positively weighted household purchases.

    The decomposition's frontier is additive but signed: NPISH final
    consumption is gross output *less* receipts from sales to households
    (those receipts -- nonprofit hospitals, say -- already appear under
    household spending), net foreign travel subtracts nonresidents' US
    spending, and net used-vehicle transactions can go negative. Counting
    those lines as basket items would double-count or invert them, so the
    NPISH branch collapses into BEA's single published NPISH line and only
    lines with a positive net sign and positive spending share are kept."""
    tree_detail = us_pce_detail.loc[us_pce_detail["additive_tree"]]
    npish = tree_detail.loc[tree_detail["nominal_series_id"] == PCE_NPISH_NOMINAL_SERIES]
    if npish.empty:
        raise ValueError("BEA PCE detail has no NPISH final consumption line")
    npish_line = int(npish["table_line"].iloc[0])
    npish_label = str(npish["label"].iloc[0])

    calendar = pd.date_range(tree_detail["date"].min(), tree_detail["date"].max(), freq="MS")
    npish_series = npish.set_index("date").reindex(calendar)
    root_spending = (
        tree_detail.loc[tree_detail["table_line"] == PCE_ROOT_LINE]
        .set_index("date")["nominal_spending_musd"]
        .reindex(calendar)
    )
    npish_frame = pd.DataFrame(
        {
            "date": calendar,
            "table_line": npish_line,
            "aggregation_sign": 1,
            "weight_pct": (
                npish_series["nominal_spending_musd"].shift(12) / root_spending.shift(12) * 100.0
            ).to_numpy(),
            "index_value": npish_series["price_index"].to_numpy(),
            "inflation_yoy": (
                (npish_series["price_index"] / npish_series["price_index"].shift(12) - 1.0)
                * 100.0
            ).to_numpy(),
        }
    )

    items = us_pce_item_decomposition.loc[
        ~us_pce_item_decomposition["reconciliation_row"]
        & (us_pce_item_decomposition["product_group"] != npish_label),
        ["date", "table_line", "aggregation_sign", "weight_pct", "index_value", "inflation_yoy"],
    ]
    universe = pd.concat(
        [items, npish_frame.loc[npish_frame["date"].isin(items["date"].unique())]],
        ignore_index=True,
    )
    universe = universe.loc[(universe["aggregation_sign"] > 0) & (universe["weight_pct"] > 0)]

    date = items["date"].max()
    latest = universe.loc[universe["date"] == date].copy()
    rate_6m = _6m_annualized_from_index(
        universe, code_column="table_line", index_column="index_value", date=date
    )
    latest["inflation_6m_ann"] = latest["table_line"].map(rate_6m)
    rows = []
    for horizon, rate_column in (("12m_yoy", "inflation_yoy"), ("6m_ann", "inflation_6m_ann")):
        row = _threshold_row(
            "US_PCE", horizon, latest,
            rate_column=rate_column, weight_column="weight_pct",
        )
        row["date"] = date
        rows.append(row)
    return pd.DataFrame(rows, columns=ITEM_BREADTH_COLUMNS)


def build_item_breadth_uk(uk_item_decomposition: pd.DataFrame) -> pd.DataFrame:
    frame = uk_item_decomposition.loc[~uk_item_decomposition["reconciliation_row"]]
    date = frame["date"].max()
    latest = frame.loc[frame["date"] == date].copy()
    rate_6m = _6m_annualized_from_index(
        frame, code_column="coicop_code", index_column="index_value", date=date
    )
    latest["inflation_6m_ann"] = latest["coicop_code"].map(rate_6m)
    rows = []
    for horizon, rate_column in (("12m_yoy", "inflation_yoy"), ("6m_ann", "inflation_6m_ann")):
        row = _threshold_row(
            "GB", horizon, latest,
            rate_column=rate_column, weight_column="weight_pct",
        )
        row["date"] = date
        rows.append(row)
    return pd.DataFrame(rows, columns=ITEM_BREADTH_COLUMNS)


def build_item_breadth_denmark(danish_item_decomposition: pd.DataFrame) -> pd.DataFrame:
    frame = danish_item_decomposition.loc[~danish_item_decomposition["reconciliation_row"]]
    date = frame["date"].max()
    latest = frame.loc[frame["date"] == date].copy()
    rate_6m = _6m_annualized_from_index(
        frame, code_column="coicop_code", index_column="index_value", date=date
    )
    latest["inflation_6m_ann"] = latest["coicop_code"].map(rate_6m)
    rows = []
    for horizon, rate_column in (("12m_yoy", "inflation_yoy"), ("6m_ann", "inflation_6m_ann")):
        row = _threshold_row(
            "DK", horizon, latest,
            rate_column=rate_column, weight_column="weight_pct",
        )
        row["date"] = date
        rows.append(row)
    return pd.DataFrame(rows, columns=ITEM_BREADTH_COLUMNS)


def build_item_breadth_norway(norway_item_decomposition: pd.DataFrame) -> pd.DataFrame:
    frame = norway_item_decomposition.loc[~norway_item_decomposition["reconciliation_row"]]
    date = frame["date"].max()
    latest = frame.loc[frame["date"] == date].copy()
    rate_6m = _6m_annualized_from_index(
        frame, code_column="source_component_code", index_column="index_value", date=date
    )
    latest["inflation_6m_ann"] = latest["source_component_code"].map(rate_6m)
    rows = []
    for horizon, rate_column in (("12m_yoy", "inflation_yoy"), ("6m_ann", "inflation_6m_ann")):
        row = _threshold_row(
            "NO", horizon, latest,
            rate_column=rate_column, weight_column="weight_pct",
        )
        row["date"] = date
        rows.append(row)
    return pd.DataFrame(rows, columns=ITEM_BREADTH_COLUMNS)


def build_item_breadth_sweden(nordic_item_contributions: pd.DataFrame) -> pd.DataFrame:
    source = nordic_item_contributions.loc[
        (~nordic_item_contributions["is_reconciliation"])
        & (nordic_item_contributions["scope"] == "headline")
        & (nordic_item_contributions["country_code"] == "SE")
    ].copy()
    if source.empty:
        return pd.DataFrame(columns=ITEM_BREADTH_COLUMNS)
    date = source["date"].max()
    latest = source.loc[source["date"] == date].copy()
    latest["inflation_yoy"] = (
        latest["current_index_value"] / latest["prior_index_value"] - 1.0
    ) * 100.0
    rate_6m = _6m_annualized_from_index(
        source,
        code_column="source_component_code",
        index_column="current_index_value",
        date=date,
    )
    latest["inflation_6m_ann"] = latest["source_component_code"].map(rate_6m)
    rows = []
    for horizon, rate_column in (("12m_yoy", "inflation_yoy"), ("6m_ann", "inflation_6m_ann")):
        row = _threshold_row(
            "SE", horizon, latest,
            rate_column=rate_column, weight_column="weight_current_pct",
        )
        row["date"] = date
        rows.append(row)
    return pd.DataFrame(rows, columns=ITEM_BREADTH_COLUMNS)


def build_item_breadth_eurozone(eurozone_item_decomposition: pd.DataFrame) -> pd.DataFrame:
    """One row per euro-area member per horizon -- unlike the other Tier 2
    builders above, eurozone_item_decomposition.py returns all 20 members in
    a single frame rather than one country at a time."""
    non_reconciliation = eurozone_item_decomposition.loc[
        ~eurozone_item_decomposition["reconciliation_row"]
    ]
    rows = []
    for country_code, frame in non_reconciliation.groupby("country_code"):
        date = frame["date"].max()
        latest = frame.loc[frame["date"] == date].copy()
        rate_6m = _6m_annualized_from_index(
            frame,
            code_column="source_component_code",
            index_column="index_value",
            date=date,
        )
        latest["inflation_6m_ann"] = latest["source_component_code"].map(rate_6m)
        for horizon, rate_column in (("12m_yoy", "inflation_yoy"), ("6m_ann", "inflation_6m_ann")):
            row = _threshold_row(
                country_code, horizon, latest,
                rate_column=rate_column, weight_column="weight_pct",
            )
            row["date"] = date
            rows.append(row)
    return pd.DataFrame(rows, columns=ITEM_BREADTH_COLUMNS)


def build_item_breadth(
    *,
    us_item_decomposition: pd.DataFrame,
    uk_item_decomposition: pd.DataFrame | None = None,
    danish_item_decomposition: pd.DataFrame | None = None,
    norway_item_decomposition: pd.DataFrame | None = None,
    nordic_item_contributions: pd.DataFrame | None = None,
    eurozone_item_decomposition: pd.DataFrame | None = None,
    us_pce_item_decomposition: pd.DataFrame | None = None,
    us_pce_detail: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames = [build_item_breadth_us(us_item_decomposition)]
    if uk_item_decomposition is not None and not uk_item_decomposition.empty:
        frames.append(build_item_breadth_uk(uk_item_decomposition))
    if danish_item_decomposition is not None and not danish_item_decomposition.empty:
        frames.append(build_item_breadth_denmark(danish_item_decomposition))
    if norway_item_decomposition is not None and not norway_item_decomposition.empty:
        frames.append(build_item_breadth_norway(norway_item_decomposition))
    if nordic_item_contributions is not None and not nordic_item_contributions.empty:
        frames.append(build_item_breadth_sweden(nordic_item_contributions))
    if eurozone_item_decomposition is not None and not eurozone_item_decomposition.empty:
        frames.append(build_item_breadth_eurozone(eurozone_item_decomposition))
    if (
        us_pce_item_decomposition is not None
        and not us_pce_item_decomposition.empty
        and us_pce_detail is not None
    ):
        frames.append(build_item_breadth_us_pce(us_pce_item_decomposition, us_pce_detail))
    return pd.concat(frames, ignore_index=True).loc[:, ITEM_BREADTH_COLUMNS]


def save_breadth(
    buckets: pd.DataFrame,
    items: pd.DataFrame,
    *,
    bucket_output: str | Path = BREADTH_BUCKETS_PATH,
    item_output: str | Path = BREADTH_ITEMS_PATH,
) -> tuple[Path, Path]:
    bucket_path = atomic_write_parquet(buckets, bucket_output, columns=BUCKET_BREADTH_COLUMNS)
    item_path = atomic_write_parquet(items, item_output, columns=ITEM_BREADTH_COLUMNS)
    return bucket_path, item_path


def run(
    *,
    cpi_path: str | Path = CPI_PANEL_PATH,
    bls_weights_path: str | Path = BLS_US_CPI_WEIGHTS_PATH,
    no_weights_path: str | Path = SSB_NORWAY_CPI_WEIGHTS_PATH,
    no_contract_path: str | Path = SSB_NORWAY_COMPONENT_CONTRACT_PATH,
    hicp_weights_path: str | Path = EUROSTAT_HICP_WEIGHTS_PATH,
    uk_weights_path: str | Path = ONS_UK_CPI_WEIGHTS_PATH,
    us_item_path: str | Path = US_ITEM_DECOMPOSITION_PATH,
    uk_item_path: str | Path = UK_ITEM_DECOMPOSITION_PATH,
    danish_item_path: str | Path = DANISH_ITEM_DECOMPOSITION_PATH,
    norway_item_path: str | Path = NORWAY_ITEM_DECOMPOSITION_PATH,
    nordic_item_path: str | Path = NORDIC_ITEM_CONTRIBUTIONS_PATH,
    eurozone_item_path: str | Path = EUROZONE_ITEM_DECOMPOSITION_PATH,
    us_pce_item_path: str | Path = US_PCE_ITEM_DECOMPOSITION_PATH,
    us_pce_detail_path: str | Path = BEA_US_PCE_PATH,
    bucket_output: str | Path = BREADTH_BUCKETS_PATH,
    item_output: str | Path = BREADTH_ITEMS_PATH,
) -> tuple[Path, Path]:
    cpi = pd.read_parquet(cpi_path)
    bls_weights = pd.read_parquet(bls_weights_path)
    no_weights = pd.read_parquet(no_weights_path) if Path(no_weights_path).is_file() else None
    no_contract = pd.read_parquet(no_contract_path) if Path(no_contract_path).is_file() else None
    comparable_sources = [
        Path(path) for path in (hicp_weights_path, uk_weights_path) if Path(path).is_file()
    ]
    comparable_weights = (
        pd.concat([pd.read_parquet(source) for source in comparable_sources], ignore_index=True)
        if comparable_sources else None
    )
    buckets = build_bucket_breadth(
        cpi,
        bls_weights=bls_weights,
        no_weights=no_weights,
        no_contract=no_contract,
        comparable_weights=comparable_weights,
    )
    us_item = pd.read_parquet(us_item_path)
    uk_item = pd.read_parquet(uk_item_path) if Path(uk_item_path).is_file() else None
    danish_item = pd.read_parquet(danish_item_path) if Path(danish_item_path).is_file() else None
    norway_item = pd.read_parquet(norway_item_path) if Path(norway_item_path).is_file() else None
    nordic_item = pd.read_parquet(nordic_item_path) if Path(nordic_item_path).is_file() else None
    eurozone_item = (
        pd.read_parquet(eurozone_item_path) if Path(eurozone_item_path).is_file() else None
    )
    items = build_item_breadth(
        us_item_decomposition=us_item,
        uk_item_decomposition=uk_item,
        danish_item_decomposition=danish_item,
        norway_item_decomposition=norway_item,
        nordic_item_contributions=nordic_item,
        eurozone_item_decomposition=eurozone_item,
        us_pce_item_decomposition=(
            pd.read_parquet(us_pce_item_path) if Path(us_pce_item_path).is_file() else None
        ),
        us_pce_detail=(
            pd.read_parquet(us_pce_detail_path) if Path(us_pce_detail_path).is_file() else None
        ),
    )
    return save_breadth(buckets, items, bucket_output=bucket_output, item_output=item_output)


if __name__ == "__main__":
    bucket_path, item_path = run()
    print(f"Saved bucket breadth to {bucket_path}")
    print(f"Saved item breadth to {item_path}")
