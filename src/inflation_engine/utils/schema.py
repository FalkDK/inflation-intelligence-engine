from collections.abc import Sequence

import numpy as np
import pandas as pd

from inflation_engine.utils.config import COUNTRIES, series_role


CPI_SCHEMA_COLUMNS = [
    "date",
    "country",
    "country_code",
    "region",
    "series_type",
    "series_role",
    "component",
    "value",
    "source",
    "source_dataset",
    "source_series_id",
    "frequency",
    "unit",
    "index_base",
    "seasonal_adjustment",
    "observation_status",
]

CPI_UNIQUE_KEY = [
    "date",
    "country_code",
    "series_type",
    "component",
]

ALLOWED_SERIES_TYPES = {"headline", "core"}
ALLOWED_SERIES_ROLES = {"aggregate", "component"}
ALLOWED_FREQUENCIES = {"monthly"}
ALLOWED_UNITS = {"index"}
ALLOWED_SEASONAL_ADJUSTMENTS = {
    "seasonally_adjusted",
    "not_seasonally_adjusted",
    "not_provided",
}
ALLOWED_OBSERVATION_STATUSES = {
    "normal",
    "estimated",
    "provisional",
    "forecast",
    "revised",
    "break",
    "not_provided",
}


def validate_cpi_schema(
    frame: pd.DataFrame,
    *,
    required_columns: Sequence[str] = CPI_SCHEMA_COLUMNS,
) -> None:
    """Raise ValueError when a CPI frame does not satisfy the common schema."""
    missing_columns = [column for column in required_columns if column not in frame]
    if missing_columns:
        raise ValueError(f"Missing required CPI columns: {missing_columns}")

    if frame.empty:
        raise ValueError("CPI data cannot be empty")

    if frame[list(required_columns)].isnull().any().any():
        null_columns = frame[list(required_columns)].columns[
            frame[list(required_columns)].isnull().any()
        ].tolist()
        raise ValueError(f"CPI data contains null values in: {null_columns}")

    if not pd.api.types.is_datetime64_any_dtype(frame["date"]):
        raise ValueError("CPI date column must use a datetime dtype")

    if not pd.api.types.is_numeric_dtype(frame["value"]):
        raise ValueError("CPI value column must use a numeric dtype")

    if not np.isfinite(frame["value"]).all() or (frame["value"] <= 0).any():
        raise ValueError("CPI values must be finite and greater than zero")

    if not frame["date"].dt.is_month_start.all():
        raise ValueError("CPI dates must be normalized to the first day of the month")

    duplicates = frame.duplicated(CPI_UNIQUE_KEY, keep=False)
    if duplicates.any():
        raise ValueError(
            f"CPI data contains duplicate observations for key: {CPI_UNIQUE_KEY}"
        )

    allowed_values = {
        "series_type": ALLOWED_SERIES_TYPES,
        "series_role": ALLOWED_SERIES_ROLES,
        "frequency": ALLOWED_FREQUENCIES,
        "unit": ALLOWED_UNITS,
        "seasonal_adjustment": ALLOWED_SEASONAL_ADJUSTMENTS,
        "observation_status": ALLOWED_OBSERVATION_STATUSES,
    }
    for column, allowed in allowed_values.items():
        invalid = sorted(set(frame[column]) - allowed)
        if invalid:
            raise ValueError(f"Invalid {column} values: {invalid}")

    string_columns = [
        column
        for column in required_columns
        if column not in {"date", "value"}
    ]
    for column in string_columns:
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError(f"CPI {column} values must be non-empty strings")

    invalid_countries = sorted(set(frame["country_code"]) - set(COUNTRIES))
    if invalid_countries:
        raise ValueError(f"Unknown CPI country codes: {invalid_countries}")

    expected_names = frame["country_code"].map(
        {code: metadata["name"] for code, metadata in COUNTRIES.items()}
    )
    expected_regions = frame["country_code"].map(
        {code: metadata["region"] for code, metadata in COUNTRIES.items()}
    )
    if not frame["country"].equals(expected_names):
        raise ValueError("CPI country names do not match country metadata")
    if not frame["region"].equals(expected_regions):
        raise ValueError("CPI regions do not match country metadata")

    expected_roles = pd.Series(
        [
            series_role(country_code, series_type, component)
            for country_code, series_type, component in frame[
                ["country_code", "series_type", "component"]
            ].itertuples(index=False, name=None)
        ],
        index=frame.index,
    )
    if not frame["series_role"].equals(expected_roles):
        raise ValueError("CPI series roles do not match primary-series metadata")
