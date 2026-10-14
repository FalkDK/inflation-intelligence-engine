from pathlib import Path

import pandas as pd

from inflation_engine.reports.charts import (
    load_processed_panel,
    select_primary_series,
)
from inflation_engine.utils.paths import CPI_PANEL_PATH


METRIC_LABELS = {
    "inflation_mom": "MoM",
    "inflation_yoy": "YoY",
    "inflation_yoy_change_3m": "3m change in YoY",
    "inflation_3m_ann": "3m Ann.",
    "inflation_3m3m_ann": "3m/3m Ann.",
    "inflation_6m_ann": "6m Ann.",
    "inflation_6m6m_ann": "6m/6m Ann.",
    "inflation_3m3m_yoy_spread": "3m/3m - YoY",
}
METRIC_COLUMNS = list(METRIC_LABELS)
DEFAULT_COUNTRY_METRICS = [
    "inflation_yoy",
    "inflation_yoy_change_3m",
]
DEFAULT_HEATMAP_METRICS = [
    "inflation_yoy",
    "inflation_yoy_change_3m",
]
LATEST_KEY = ["country_code", "series_type"]
SEASONAL_METRICS = [
    "inflation_mom",
    "inflation_3m_ann",
    "inflation_3m3m_ann",
    "inflation_6m_ann",
    "inflation_6m6m_ann",
    "inflation_3m3m_yoy_spread",
]
ADJUSTED_METHODS = {"seasonally_adjusted", "x13_arima_seats"}


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing_columns = [column for column in columns if column not in frame]
    if missing_columns:
        raise ValueError(f"CPI panel is missing columns: {missing_columns}")


def load_dashboard_panel(path: str | Path = CPI_PANEL_PATH) -> pd.DataFrame:
    frame = load_processed_panel(path)
    _require_columns(frame, METRIC_COLUMNS)
    return frame


def primary_dashboard_panel(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, [*METRIC_COLUMNS, "inflation_momentum_signal"])
    primary = select_primary_series(frame).copy()
    adjusted = primary["analysis_seasonal_adjustment"].isin(ADJUSTED_METHODS)
    primary.loc[~adjusted, SEASONAL_METRICS] = float("nan")
    return primary


def available_countries(frame: pd.DataFrame) -> pd.DataFrame:
    primary = primary_dashboard_panel(frame)
    return (
        primary[["country", "country_code", "region"]]
        .drop_duplicates()
        .sort_values("country", kind="stable")
        .reset_index(drop=True)
    )


def latest_primary_observations(frame: pd.DataFrame) -> pd.DataFrame:
    primary = primary_dashboard_panel(frame)
    return (
        primary.sort_values("date", kind="stable")
        .groupby(LATEST_KEY, as_index=False)
        .tail(1)
        .sort_values(["country", "series_type"], kind="stable")
        .reset_index(drop=True)
    )


def country_latest_metrics(frame: pd.DataFrame, country_code: str) -> pd.DataFrame:
    latest = latest_primary_observations(frame)
    selected = latest.loc[latest["country_code"] == country_code].copy()
    if selected.empty:
        raise ValueError(f"Unknown country code: {country_code}")
    return selected.sort_values("series_type", kind="stable").reset_index(drop=True)


def country_metric_history(
    frame: pd.DataFrame,
    country_code: str,
    *,
    metrics: list[str] | None = None,
    series_types: list[str] | None = None,
    years: int = 5,
) -> pd.DataFrame:
    if years < 1:
        raise ValueError("History window must include at least one year")
    metrics = metrics or DEFAULT_COUNTRY_METRICS
    series_types = series_types or ["headline", "core"]
    _require_columns(frame, metrics)

    primary = primary_dashboard_panel(frame)
    selected = primary.loc[
        (primary["country_code"] == country_code)
        & (primary["series_type"].isin(series_types))
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No dashboard series found for country code: {country_code}"
        )

    cutoff = selected["date"].max() - pd.DateOffset(years=years)
    selected = selected.loc[selected["date"] >= cutoff]
    history = selected[
        ["date", "country", "country_code", "series_type", *metrics]
    ].melt(
        id_vars=["date", "country", "country_code", "series_type"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    history["metric_label"] = history["metric"].map(METRIC_LABELS)
    history["series_label"] = history["series_type"].str.title()
    return history.dropna(subset=["value"]).reset_index(drop=True)


def country_ranking(
    frame: pd.DataFrame,
    *,
    metric: str = "inflation_yoy",
    series_type: str = "headline",
) -> pd.DataFrame:
    _require_columns(frame, [metric])
    latest = latest_primary_observations(frame)
    ranking = latest.loc[latest["series_type"] == series_type].copy()
    ranking = ranking.sort_values(
        metric,
        ascending=False,
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking["metric"] = metric
    ranking["metric_label"] = METRIC_LABELS.get(metric, metric)
    return ranking[
        [
            "rank",
            "country",
            "country_code",
            "region",
            "date",
            metric,
            "metric",
            "metric_label",
            "inflation_momentum_signal",
        ]
    ]


def heatmap_values(
    frame: pd.DataFrame,
    *,
    metrics: list[str] | None = None,
    series_type: str = "headline",
) -> pd.DataFrame:
    metrics = metrics or DEFAULT_HEATMAP_METRICS
    _require_columns(frame, metrics)
    latest = latest_primary_observations(frame)
    selected = latest.loc[latest["series_type"] == series_type].copy()
    if selected.empty:
        raise ValueError(
            f"No latest observations found for series type: {series_type}"
        )

    heatmap = selected[
        ["country", "country_code", "region", "date", "series_type", *metrics]
    ].melt(
        id_vars=["country", "country_code", "region", "date", "series_type"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    heatmap["metric_label"] = heatmap["metric"].map(METRIC_LABELS)
    country_order = (
        selected.sort_values("country", kind="stable")["country"].drop_duplicates()
    )
    heatmap["country"] = pd.Categorical(
        heatmap["country"],
        categories=country_order,
        ordered=True,
    )
    return heatmap.dropna(subset=["value"]).reset_index(drop=True)
