"""CPI release calendar (past publication dates and upcoming schedule).

Source: TradingView's public economic-calendar feed, which carries the
agency's release timestamp and the observation month (``referenceDate``) for
every CPI print back to 2016, plus the forward schedule. Official agency
calendars are fragmented (BLS blocks scripted access; Eurostat, ONS, DST, SSB
and SCB each publish in a different format), so one feed is used for all
geographies and every row keeps the agency's own URL as ``source_url``.

Each panel geography is mapped to the release that publishes the value the
panel actually stores -- not the earliest headline print:

- Euro-area members and EA20 use Eurostat's full HICP release ("Final"), as
  the panel's HICP values come from Eurostat, which publishes all member-state
  HICPs on that day. National flash/prelim CPI prints are earlier but are a
  different (often non-harmonised) number.
- Sweden uses the final CPI release; SCB's flash ("Prel") started in 2024.
- US_PCE uses the BEA Personal Income and Outlays release.

Months without a matching event keep the conservative fallback lag applied in
features/availability.py.
"""

import argparse
import os
import tempfile
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd
import requests

from inflation_engine.features.availability import (
    RELEASE_CALENDAR_COLUMNS,
    validate_release_calendar,
)
from inflation_engine.utils.config import COUNTRIES
from inflation_engine.utils.paths import INFLATION_RELEASE_CALENDAR_PATH


TRADINGVIEW_CALENDAR_URL = "https://economic-calendar.tradingview.com/events"
TRADINGVIEW_HEADERS = {
    "Origin": "https://www.tradingview.com",
    "User-Agent": "Mozilla/5.0 (compatible; inflation-intelligence-engine)",
}
# The feed has no history before 2016 and silently truncates large responses
# (quarterly windows dropped every quarter-end month), so request one month
# at a time.
CALENDAR_HISTORY_START = pd.Timestamp("2016-01-01")
DEFAULT_HORIZON_MONTHS = 6
REQUEST_PAUSE_SECONDS = 0.2

# panel country_code -> (TradingView country, accepted event titles).
# Titles changed over time (e.g. SE "Inflation Rate YoY" -> "... Final" in
# 2024), so every historical spelling of the same release is listed.
EUROSTAT_FULL_RELEASE = ("EU", ("Inflation Rate YoY Final",))
RELEASE_EVENTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "US": ("US", ("Inflation Rate YoY",)),
    "US_PCE": ("US", ("PCE Price Index YoY",)),
    "GB": ("GB", ("Inflation Rate YoY",)),
    "DK": ("DK", ("Inflation Rate YoY",)),
    "NO": ("NO", ("Inflation Rate YoY", "Inflation Rate YoY Final")),
    "SE": ("SE", ("Inflation Rate YoY", "Inflation Rate YoY Final")),
    **{
        code: EUROSTAT_FULL_RELEASE
        for code, meta in COUNTRIES.items()
        if meta["source"] == "eurostat"
    },
}
FALLBACK_SOURCE_URL = "https://www.tradingview.com/economic-calendar/"


def _month_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return [
        (month, month + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59))
        for month in pd.date_range(start, end, freq="MS")
    ]


def fetch_tradingview_events(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    countries: Iterable[str] | None = None,
    session: requests.Session | None = None,
    pause_seconds: float = REQUEST_PAUSE_SECONDS,
) -> pd.DataFrame:
    """Download raw calendar events, one request per calendar month."""
    countries = sorted(
        set(countries or (country for country, _ in RELEASE_EVENTS.values()))
    )
    client = session or requests.Session()
    events: list[dict] = []
    for window_start, window_end in _month_windows(start, end):
        response = client.get(
            TRADINGVIEW_CALENDAR_URL,
            params={
                "from": window_start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "to": window_end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "countries": ",".join(countries),
            },
            headers=TRADINGVIEW_HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        if status == "ok":
            events.extend(payload.get("result", []))
        elif status != "no_data":
            raise ValueError(f"Unexpected TradingView calendar status: {status!r}")
        if pause_seconds:
            time.sleep(pause_seconds)
    frame = pd.DataFrame(events)
    if frame.empty:
        raise ValueError("TradingView calendar returned no events")
    return frame.drop_duplicates("id")


def build_release_calendar(
    events: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    release_events: Mapping[str, tuple[str, tuple[str, ...]]] = RELEASE_EVENTS,
) -> pd.DataFrame:
    """Map raw events to one publication date per (observation month, geography)."""
    required = ["country", "title", "date", "referenceDate"]
    missing = [column for column in required if column not in events]
    if missing:
        raise ValueError(f"Calendar events are missing columns: {missing}")

    events = events.loc[events["referenceDate"].fillna("").astype(bool)].copy()
    events["observation_date"] = (
        pd.to_datetime(events["referenceDate"], utc=True)
        .dt.tz_localize(None)
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    events["publication_datetime_utc"] = pd.to_datetime(events["date"], utc=True)
    events["publication_date"] = (
        events["publication_datetime_utc"].dt.tz_localize(None).dt.normalize()
    )
    if "source_url" not in events:
        events["source_url"] = None

    frames = []
    for country_code, (tv_country, titles) in release_events.items():
        matched = events.loc[
            events["country"].eq(tv_country) & events["title"].isin(titles),
            ["observation_date", "publication_date", "publication_datetime_utc", "source_url"],
        ]
        frames.append(matched.assign(country_code=country_code))
    calendar = pd.concat(frames, ignore_index=True)

    # Duplicate releases for one month (e.g. a re-issued "Final") resolve to
    # the latest date: the value is not settled before then.
    calendar = (
        calendar.sort_values("publication_datetime_utc")
        .drop_duplicates(["observation_date", "country_code"], keep="last")
    )
    calendar = calendar.loc[
        calendar["publication_date"] > calendar["observation_date"]
    ]
    as_of_utc = pd.Timestamp(as_of)
    as_of_utc = (as_of_utc.tz_localize("UTC") if as_of_utc.tzinfo is None
                 else as_of_utc.tz_convert("UTC"))
    calendar["release_status"] = calendar["publication_datetime_utc"].le(as_of_utc).map(
        {True: "actual", False: "scheduled"}
    )
    calendar["source_url"] = (
        calendar["source_url"].fillna("").replace("", FALLBACK_SOURCE_URL)
    )
    calendar = calendar.sort_values(
        ["observation_date", "country_code"]
    ).reset_index(drop=True)[RELEASE_CALENDAR_COLUMNS + ["publication_datetime_utc"]]
    validate_release_calendar(calendar)
    return calendar


def upcoming_releases(
    calendar: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    days: int = 45,
) -> pd.DataFrame:
    """Scheduled releases in the next ``days`` days, earliest first."""
    start = pd.Timestamp(as_of).normalize()
    window = calendar.loc[
        calendar["publication_date"].between(start, start + pd.Timedelta(days=days))
    ]
    return window.sort_values(["publication_date", "country_code"]).reset_index(
        drop=True
    )


def save_release_calendar(
    calendar: pd.DataFrame,
    output_path: str | Path = INFLATION_RELEASE_CALENDAR_PATH,
) -> Path:
    validate_release_calendar(calendar)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            serialized = calendar.copy()
            if "publication_datetime_utc" in serialized:
                serialized["publication_datetime_utc"] = serialized[
                    "publication_datetime_utc"
                ].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            serialized.to_csv(temporary_file, index=False, date_format="%Y-%m-%d")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def run(
    output_path: str | Path = INFLATION_RELEASE_CALENDAR_PATH,
    *,
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    as_of: pd.Timestamp | None = None,
    recent_months: int | None = None,
) -> Path:
    today = pd.Timestamp(as_of or pd.Timestamp.now(tz="UTC").tz_localize(None))
    end = (today + pd.DateOffset(months=horizon_months)).to_period("M").to_timestamp()
    start = CALENDAR_HISTORY_START
    if recent_months is not None:
        if recent_months < 1:
            raise ValueError("recent_months must be positive")
        if Path(output_path).is_file():
            start = (today - pd.DateOffset(months=recent_months)).to_period("M").to_timestamp()
    events = fetch_tradingview_events(start, end)
    calendar = build_release_calendar(events, as_of=today)
    if start > CALENDAR_HISTORY_START:
        from inflation_engine.features.availability import load_release_calendar

        previous = load_release_calendar(output_path)
        previous = previous.loc[previous["publication_date"] < start]
        calendar = pd.concat([previous, calendar], ignore_index=True)
        calendar = calendar.sort_values("publication_date").drop_duplicates(
            ["observation_date", "country_code"], keep="last"
        ).sort_values(["observation_date", "country_code"]).reset_index(drop=True)
    return save_release_calendar(calendar, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch past and scheduled CPI release dates (TradingView)."
    )
    parser.add_argument(
        "--output", type=Path, default=INFLATION_RELEASE_CALENDAR_PATH
    )
    parser.add_argument(
        "--horizon-months", type=int, default=DEFAULT_HORIZON_MONTHS
    )
    parser.add_argument("--upcoming-days", type=int, default=45)
    parser.add_argument("--recent-months", type=int, help="Refresh a recent window, retaining older history.")
    parser.add_argument("--timezone", default="UTC", help="Display timezone, e.g. Europe/Copenhagen.")
    args = parser.parse_args()
    destination = run(args.output, horizon_months=args.horizon_months, recent_months=args.recent_months)
    from inflation_engine.features.availability import load_release_calendar

    calendar = load_release_calendar(destination)
    print(f"Saved {len(calendar)} release dates to {destination}")
    upcoming = upcoming_releases(
        calendar,
        as_of=pd.Timestamp.now(tz="UTC").tz_localize(None),
        days=args.upcoming_days,
    )
    # Collapse the euro-area members onto their shared Eurostat release.
    upcoming = (
        upcoming.groupby(["publication_datetime_utc", "observation_date"])["country_code"]
        .agg(lambda codes: ", ".join(sorted(codes)))
        .reset_index()
    )
    for row in upcoming.itertuples(index=False):
        print(
            f"{row.publication_datetime_utc.tz_convert(args.timezone):%Y-%m-%d %H:%M %Z}  "
            f"{row.observation_date:%b %Y}  {row.country_code}"
        )


if __name__ == "__main__":
    main()
