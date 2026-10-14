"""Publication and information-date handling for point-in-time features."""

from pathlib import Path

import pandas as pd

from inflation_engine.utils.paths import INFLATION_RELEASE_CALENDAR_PATH


RELEASE_CALENDAR_COLUMNS = [
    "observation_date",
    "country_code",
    "publication_date",
    "release_status",
    "source_url",
]
ACTUAL_RELEASE_STATUSES = {"actual", "source_updated_at"}
# Forward-schedule rows: the release has not happened yet, so no observation in
# the panel should match one unless the calendar is stale.
RELEASE_STATUSES = ACTUAL_RELEASE_STATUSES | {"scheduled"}


def validate_release_calendar(frame: pd.DataFrame) -> None:
    missing = [column for column in RELEASE_CALENDAR_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Inflation release calendar is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Inflation release calendar cannot be empty")
    if frame[RELEASE_CALENDAR_COLUMNS].isnull().any().any():
        raise ValueError("Inflation release calendar cannot contain null values")
    if frame.duplicated(["observation_date", "country_code"]).any():
        raise ValueError("Inflation release calendar contains duplicate releases")
    if not pd.api.types.is_datetime64_any_dtype(frame["observation_date"]):
        raise ValueError("Release observation dates must use a datetime dtype")
    if not pd.api.types.is_datetime64_any_dtype(frame["publication_date"]):
        raise ValueError("Publication dates must use a datetime dtype")
    if (frame["publication_date"] <= frame["observation_date"]).any():
        raise ValueError("Publication dates must follow observation dates")
    invalid = sorted(set(frame["release_status"]) - RELEASE_STATUSES)
    if invalid:
        raise ValueError(f"Invalid inflation release statuses: {invalid}")
    if "publication_datetime_utc" in frame:
        timestamps = frame["publication_datetime_utc"]
        if str(timestamps.dtype) != "datetime64[ns, UTC]":
            raise ValueError("Publication timestamps must use timezone-aware UTC datetimes")
        known = timestamps.notna()
        dates = timestamps.loc[known].dt.tz_localize(None).dt.normalize()
        if not dates.eq(frame.loc[known, "publication_date"]).all():
            raise ValueError("Publication timestamps must match publication dates in UTC")


def load_release_calendar(
    path: str | Path = INFLATION_RELEASE_CALENDAR_PATH,
) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Inflation release calendar does not exist: {source}")
    frame = pd.read_csv(
        source,
        parse_dates=["observation_date", "publication_date"],
    )
    optional = []
    if "publication_datetime_utc" in frame:
        frame["publication_datetime_utc"] = pd.to_datetime(
            frame["publication_datetime_utc"], utc=True, errors="raise"
        )
        optional.append("publication_datetime_utc")
    validate_release_calendar(frame)
    return frame.loc[:, RELEASE_CALENDAR_COLUMNS + optional]


def attach_information_dates(
    frame: pd.DataFrame,
    release_calendar: pd.DataFrame | None = None,
    *,
    conservative_month_lag: int = 2,
) -> pd.DataFrame:
    """Attach actual release dates or a conservative no-lookahead fallback."""
    if conservative_month_lag < 1:
        raise ValueError("Conservative information lag must be at least one month")
    required = ["date", "country_code"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Inflation feature panel is missing columns: {missing}")

    result = frame.copy()
    result["observation_date"] = result["date"]
    if release_calendar is None:
        calendar = pd.DataFrame(columns=RELEASE_CALENDAR_COLUMNS)
        calendar["observation_date"] = pd.to_datetime(
            calendar["observation_date"]
        )
        calendar["publication_date"] = pd.to_datetime(
            calendar["publication_date"]
        )
    else:
        validate_release_calendar(release_calendar)
        calendar = release_calendar

    result = result.merge(
        calendar,
        on=["observation_date", "country_code"],
        how="left",
        validate="many_to_one",
    )
    fallback = result["observation_date"] + pd.DateOffset(
        months=conservative_month_lag
    )
    result["information_date"] = result["publication_date"].fillna(fallback)
    result["availability_status"] = result["release_status"].fillna(
        "conservative_month_lag"
    )
    result["information_lag_days"] = (
        result["information_date"] - result["observation_date"]
    ).dt.days.astype(int)
    return result.drop(columns=["date", "release_status", "source_url"])
