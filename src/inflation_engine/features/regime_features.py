"""Build a compact, model-ready inflation feature panel.

The output deliberately contains features rather than inferred regimes. A
downstream HMM or other state model should estimate regimes jointly with the
growth, liquidity, volatility, and market-price feature blocks.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from inflation_engine.features.availability import (
    attach_information_dates,
    load_release_calendar,
)
from inflation_engine.features.inflation_metrics import OUTPUT_COLUMNS
from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.paths import (
    CPI_PANEL_PATH,
    INFLATION_REGIME_FEATURES_PATH,
)


IDENTIFIER_COLUMNS = ["date", "country", "country_code", "region"]
EXPORT_IDENTIFIER_COLUMNS = [
    "observation_date",
    "publication_date",
    "information_date",
    "availability_status",
    "information_lag_days",
    "feature_contract_version",
    "country",
    "country_code",
    "region",
]
INPUT_VALUE_COLUMNS = [
    "inflation_yoy",
    "inflation_3m3m_ann",
    "analysis_seasonal_adjustment",
]
REGIME_FEATURE_COLUMNS = [
    "headline_yoy",
    "core_yoy",
    "headline_yoy_change_3m",
    "core_yoy_change_3m",
    "headline_yoy_change_12m",
    "core_yoy_change_12m",
    "headline_core_gap",
    "inflation_level",
    "inflation_direction_3m",
    "headline_3m3m_ann_sa",
    "core_3m3m_ann_sa",
]
QUALITY_COLUMNS = [
    "headline_is_seasonally_adjusted",
    "core_is_seasonally_adjusted",
    "short_horizon_usable",
    "feature_complete",
]
REGIME_OUTPUT_COLUMNS = (
    EXPORT_IDENTIFIER_COLUMNS + REGIME_FEATURE_COLUMNS + QUALITY_COLUMNS
)
ADJUSTED_METHODS = {"seasonally_adjusted", "x13_arima_seats"}
FEATURE_CONTRACT_VERSION = "inflation_features_v1"


def _require_processed_columns(frame: pd.DataFrame) -> None:
    required = [
        *IDENTIFIER_COLUMNS,
        "series_type",
        "series_role",
        *INPUT_VALUE_COLUMNS,
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Processed CPI panel is missing columns: {missing}")
    invalid_types = sorted(set(frame["series_type"]) - {"headline", "core"})
    if invalid_types:
        raise ValueError(f"Unsupported inflation series types: {invalid_types}")


def _calendar_change(
    frame: pd.DataFrame,
    column: str,
    periods: int,
) -> pd.Series:
    """Return a lagged change that never bridges missing calendar months."""
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, positions in frame.groupby("country_code", sort=False).groups.items():
        country = frame.loc[positions].sort_values("date", kind="stable")
        monthly_index = pd.date_range(
            country["date"].min(),
            country["date"].max(),
            freq="MS",
        )
        values = country.set_index("date")[column].reindex(monthly_index)
        change = values - values.shift(periods)
        result.loc[country.index] = change.reindex(country["date"]).to_numpy()
    return result


def build_regime_features(
    frame: pd.DataFrame,
    release_calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create one inflation feature row per country and observation month.

    Short-horizon annualized rates are exposed only when the corresponding
    source series is seasonally adjusted. Changes in YoY inflation provide the
    comparable direction signal for non-adjusted cross-country series.
    """
    _require_processed_columns(frame)
    primary = frame.loc[frame["series_role"] == "aggregate"].copy()
    if primary.empty:
        raise ValueError("Processed CPI panel contains no aggregate series")

    duplicate_key = primary.duplicated(
        ["date", "country_code", "series_type"],
        keep=False,
    )
    if duplicate_key.any():
        raise ValueError(
            "Aggregate CPI panel contains duplicate country/type observations"
        )

    metadata = (
        primary[IDENTIFIER_COLUMNS]
        .drop_duplicates(["date", "country_code"])
        .sort_values(["country_code", "date"], kind="stable")
        .reset_index(drop=True)
    )

    values = primary.pivot(
        index=["date", "country_code"],
        columns="series_type",
        values=["inflation_yoy", "inflation_3m3m_ann"],
    )
    adjustments = primary.pivot(
        index=["date", "country_code"],
        columns="series_type",
        values="analysis_seasonal_adjustment",
    )
    keyed = metadata.set_index(["date", "country_code"])

    for series_type in ("headline", "core"):
        keyed[f"{series_type}_yoy"] = values["inflation_yoy"].get(series_type)
        is_adjusted = adjustments.get(series_type, pd.Series(dtype=object)).isin(
            ADJUSTED_METHODS
        )
        keyed[f"{series_type}_is_seasonally_adjusted"] = (
            is_adjusted.reindex(keyed.index, fill_value=False).astype(bool)
        )
        short_rate = values["inflation_3m3m_ann"].get(series_type)
        if short_rate is None:
            keyed[f"{series_type}_3m3m_ann_sa"] = np.nan
        else:
            keyed[f"{series_type}_3m3m_ann_sa"] = short_rate.where(
                keyed[f"{series_type}_is_seasonally_adjusted"]
            )

    result = keyed.reset_index()
    for series_type in ("headline", "core"):
        column = f"{series_type}_yoy"
        result[f"{series_type}_yoy_change_3m"] = _calendar_change(
            result,
            column,
            3,
        )
        result[f"{series_type}_yoy_change_12m"] = _calendar_change(
            result, column, 12
        )

    result["headline_core_gap"] = result["core_yoy"] - result["headline_yoy"]
    result["inflation_level"] = result[["headline_yoy", "core_yoy"]].mean(
        axis=1,
        skipna=False,
    )
    result["inflation_direction_3m"] = result[
        ["headline_yoy_change_3m", "core_yoy_change_3m"]
    ].mean(axis=1, skipna=False)
    result["short_horizon_usable"] = (
        result["headline_is_seasonally_adjusted"]
        & result["core_is_seasonally_adjusted"]
        & result["headline_3m3m_ann_sa"].notna()
        & result["core_3m3m_ann_sa"].notna()
    )
    result["feature_complete"] = result[
        [
            "headline_yoy",
            "core_yoy",
            "headline_yoy_change_3m",
            "core_yoy_change_3m",
        ]
    ].notna().all(axis=1)

    result = result.sort_values(
        ["country_code", "date"],
        kind="stable",
    ).reset_index(drop=True)
    numeric = result[REGIME_FEATURE_COLUMNS].stack()
    if not np.isfinite(numeric).all():
        raise ValueError("Inflation regime features must be finite when available")
    result = attach_information_dates(result, release_calendar)
    result["feature_contract_version"] = FEATURE_CONTRACT_VERSION
    return result.loc[:, REGIME_OUTPUT_COLUMNS]


def load_processed_panel(
    input_path: str | Path = CPI_PANEL_PATH,
) -> pd.DataFrame:
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Processed CPI panel does not exist: {source}")
    frame = pd.read_parquet(source)
    missing = [column for column in OUTPUT_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Processed CPI panel is missing columns: {missing}")
    return frame


def save_regime_features(
    frame: pd.DataFrame,
    output_path: str | Path = INFLATION_REGIME_FEATURES_PATH,
) -> Path:
    missing = [column for column in REGIME_OUTPUT_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Inflation feature panel is missing columns: {missing}")
    if frame.duplicated(["observation_date", "country_code"]).any():
        raise ValueError("Inflation feature panel contains duplicate rows")
    return atomic_write_parquet(
        frame,
        output_path,
        columns=REGIME_OUTPUT_COLUMNS,
    )


def run(
    input_path: str | Path = CPI_PANEL_PATH,
    output_path: str | Path = INFLATION_REGIME_FEATURES_PATH,
) -> Path:
    frame = load_processed_panel(input_path)
    release_calendar = load_release_calendar()
    features = build_regime_features(frame, release_calendar)
    return save_regime_features(features, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build model-ready inflation regime features."
    )
    parser.add_argument("--input", type=Path, default=CPI_PANEL_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=INFLATION_REGIME_FEATURES_PATH,
    )
    args = parser.parse_args()

    frame = load_processed_panel(args.input)
    features = build_regime_features(frame, load_release_calendar())
    destination = save_regime_features(features, args.output)
    complete = int(features["feature_complete"].sum())
    short_horizon = int(features["short_horizon_usable"].sum())
    print(f"Saved inflation regime features to {destination}")
    print(
        f"Rows: {len(features):,} | Complete: {complete:,} | "
        f"Seasonally adjusted short horizon: {short_horizon:,}"
    )


if __name__ == "__main__":
    main()
