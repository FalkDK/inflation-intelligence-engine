import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from inflation_engine.features.seasonal_adjustment import (
    X13_METHOD,
    seasonal_adjust_x13,
)
from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.paths import (
    CPI_PANEL_PATH,
    CPI_PANEL_RAW_COMBINED_PATH,
)
from inflation_engine.utils.schema import (
    CPI_SCHEMA_COLUMNS,
    CPI_UNIQUE_KEY,
    validate_cpi_schema,
)


SERIES_KEY = ["country_code", "series_type", "component"]
ANALYSIS_COLUMNS = [
    "analysis_value",
    "analysis_seasonal_adjustment",
]
METRIC_COLUMNS = [
    "inflation_mom",
    "inflation_yoy",
    "inflation_yoy_change_3m",
    "inflation_3m_ann",
    "inflation_3m3m_ann",
    "inflation_6m_ann",
    "inflation_6m6m_ann",
    "inflation_3m3m_yoy_spread",
]
MOMENTUM_SIGNAL_COLUMN = "inflation_momentum_signal"
OUTPUT_COLUMNS = (
    CPI_SCHEMA_COLUMNS
    + ANALYSIS_COLUMNS
    + METRIC_COLUMNS
    + [MOMENTUM_SIGNAL_COLUMN]
)
MOMENTUM_THRESHOLD_PCT_POINTS = 0.25
SEASONAL_ADJUSTMENT_OPTIONS = ("source", "x13")


def _annualized_change(
    current: pd.Series,
    previous: pd.Series,
    periods_per_year: int,
) -> pd.Series:
    return ((current / previous) ** periods_per_year - 1.0) * 100.0


def _calculate_series_metrics(series: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics for one CPI series using contiguous calendar months."""
    ordered = series.sort_values("date", kind="stable").copy()
    monthly_index = pd.date_range(
        ordered["date"].min(),
        ordered["date"].max(),
        freq="MS",
    )
    values = ordered.set_index("date")["analysis_value"].reindex(monthly_index)

    rolling_3m = values.rolling(window=3, min_periods=3).mean()
    rolling_6m = values.rolling(window=6, min_periods=6).mean()
    metrics = pd.DataFrame(
        {
            "inflation_mom": (values / values.shift(1) - 1.0) * 100.0,
            "inflation_yoy": (values / values.shift(12) - 1.0) * 100.0,
            "inflation_3m_ann": _annualized_change(
                values,
                values.shift(3),
                4,
            ),
            "inflation_3m3m_ann": _annualized_change(
                rolling_3m,
                rolling_3m.shift(3),
                4,
            ),
            "inflation_6m_ann": _annualized_change(
                values,
                values.shift(6),
                2,
            ),
            "inflation_6m6m_ann": _annualized_change(
                rolling_6m,
                rolling_6m.shift(6),
                2,
            ),
        },
        index=monthly_index,
    )
    metrics["inflation_3m3m_yoy_spread"] = (
        metrics["inflation_3m3m_ann"] - metrics["inflation_yoy"]
    )
    metrics["inflation_yoy_change_3m"] = (
        metrics["inflation_yoy"] - metrics["inflation_yoy"].shift(3)
    )

    ordered = ordered.join(metrics, on="date")
    direction = ordered["inflation_yoy_change_3m"]
    ordered[MOMENTUM_SIGNAL_COLUMN] = np.select(
        [
            direction > MOMENTUM_THRESHOLD_PCT_POINTS,
            direction < -MOMENTUM_THRESHOLD_PCT_POINTS,
        ],
        ["accelerating", "slowing"],
        default="stable",
    )
    ordered.loc[direction.isna(), MOMENTUM_SIGNAL_COLUMN] = "not_available"
    return ordered


def _prepare_analysis_values(
    frame: pd.DataFrame,
    *,
    seasonal_adjustment: str,
    x13_path: str | Path | None,
) -> pd.DataFrame:
    if seasonal_adjustment not in SEASONAL_ADJUSTMENT_OPTIONS:
        raise ValueError(
            "seasonal_adjustment must be one of: "
            f"{', '.join(SEASONAL_ADJUSTMENT_OPTIONS)}"
        )

    prepared = []
    for series_key, series in frame.groupby(SERIES_KEY, sort=False):
        ordered = series.sort_values("date", kind="stable").copy()
        source_adjustments = ordered["seasonal_adjustment"].unique()
        if len(source_adjustments) != 1:
            raise ValueError(
                f"Series has inconsistent seasonal-adjustment metadata: "
                f"{series_key}"
            )

        source_adjustment = source_adjustments[0]
        ordered["analysis_value"] = ordered["value"].astype(float)
        ordered["analysis_seasonal_adjustment"] = source_adjustment
        if (
            seasonal_adjustment == "x13"
            and source_adjustment != "seasonally_adjusted"
        ):
            values = ordered.set_index("date")["value"]
            adjusted = seasonal_adjust_x13(values, x13_path=x13_path)
            ordered["analysis_value"] = adjusted.reindex(
                ordered["date"]
            ).to_numpy()
            ordered["analysis_seasonal_adjustment"] = X13_METHOD
        prepared.append(ordered)

    return pd.concat(prepared, ignore_index=True)


def calculate_inflation_metrics(
    frame: pd.DataFrame,
    *,
    seasonal_adjustment: str = "source",
    x13_path: str | Path | None = None,
) -> pd.DataFrame:
    """Add inflation rates and momentum measures to a normalized CPI panel.

    Rate columns are expressed in percent, and the momentum spread is in
    percentage points.
    """
    validate_cpi_schema(frame)
    prepared = _prepare_analysis_values(
        frame,
        seasonal_adjustment=seasonal_adjustment,
        x13_path=x13_path,
    )

    calculated = [
        _calculate_series_metrics(series)
        for _, series in prepared.groupby(SERIES_KEY, sort=False)
    ]
    result = (
        pd.concat(calculated, ignore_index=True)
        .sort_values(CPI_UNIQUE_KEY, kind="stable")
        .reset_index(drop=True)
    )

    if result.duplicated(CPI_UNIQUE_KEY).any():
        raise ValueError(
            f"Inflation metrics contain duplicate observations for key: "
            f"{CPI_UNIQUE_KEY}"
        )
    if result[METRIC_COLUMNS].notna().any(axis=None):
        finite_metrics = result[METRIC_COLUMNS].stack()
        if not np.isfinite(finite_metrics).all():
            raise ValueError("Inflation metrics must be finite when available")
    if result["analysis_value"].isna().any():
        raise ValueError("Analysis values cannot contain null values")
    if (result["analysis_value"] <= 0).any():
        raise ValueError("Analysis values must be greater than zero")

    return result.loc[:, OUTPUT_COLUMNS]


def load_combined_panel(
    input_path: str | Path = CPI_PANEL_RAW_COMBINED_PATH,
) -> pd.DataFrame:
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Combined CPI panel does not exist: {source}")
    frame = pd.read_parquet(source)
    validate_cpi_schema(frame)
    return frame


def save_cpi_panel(
    frame: pd.DataFrame,
    output_path: str | Path = CPI_PANEL_PATH,
) -> Path:
    """Atomically save a CPI panel containing calculated inflation metrics."""
    missing_columns = [column for column in OUTPUT_COLUMNS if column not in frame]
    if missing_columns:
        raise ValueError(f"Missing inflation metric columns: {missing_columns}")
    validate_cpi_schema(frame)

    return atomic_write_parquet(frame, output_path, columns=OUTPUT_COLUMNS)


def run(
    input_path: str | Path = CPI_PANEL_RAW_COMBINED_PATH,
    output_path: str | Path = CPI_PANEL_PATH,
    *,
    seasonal_adjustment: str = "source",
    x13_path: str | Path | None = None,
) -> Path:
    frame = load_combined_panel(input_path)
    metrics = calculate_inflation_metrics(
        frame,
        seasonal_adjustment=seasonal_adjustment,
        x13_path=x13_path,
    )
    return save_cpi_panel(metrics, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate inflation metrics from the combined CPI panel."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=CPI_PANEL_RAW_COMBINED_PATH,
        help=f"Combined CPI input path (default: {CPI_PANEL_RAW_COMBINED_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CPI_PANEL_PATH,
        help=f"Processed Parquet output path (default: {CPI_PANEL_PATH})",
    )
    parser.add_argument(
        "--seasonal-adjustment",
        choices=SEASONAL_ADJUSTMENT_OPTIONS,
        default="source",
        help=(
            "Use source index values or seasonally adjust non-adjusted series "
            "with X-13ARIMA-SEATS (default: source)"
        ),
    )
    parser.add_argument(
        "--x13-path",
        type=Path,
        help=(
            "Path to the X-13 executable or its directory. X13PATH and PATH "
            "are searched when omitted."
        ),
    )
    args = parser.parse_args()

    frame = load_combined_panel(args.input)
    metrics = calculate_inflation_metrics(
        frame,
        seasonal_adjustment=args.seasonal_adjustment,
        x13_path=args.x13_path,
    )
    destination = save_cpi_panel(metrics, args.output)
    latest = metrics.loc[metrics["date"] == metrics["date"].max()]

    print(f"Saved CPI panel with inflation metrics to {destination}")
    print(
        f"Rows: {len(metrics):,} | Series: "
        f"{metrics[SERIES_KEY].drop_duplicates().shape[0]} "
        f"| Latest date: {metrics['date'].max().date()} "
        f"| Latest observations: {len(latest)} "
        f"| Adjustment: {args.seasonal_adjustment}"
    )


if __name__ == "__main__":
    main()
