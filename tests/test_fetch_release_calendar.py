import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from inflation_engine.data.fetch_release_calendar import (
    FALLBACK_SOURCE_URL,
    build_release_calendar,
    fetch_tradingview_events,
    save_release_calendar,
    upcoming_releases,
)
from inflation_engine.features.availability import load_release_calendar


def event(country, title, date, reference_date, source_url="https://agency.test/"):
    return {
        "id": f"{country}-{title}-{date}",
        "country": country,
        "title": title,
        "date": date,
        "referenceDate": reference_date,
        "source_url": source_url,
    }


def events():
    return pd.DataFrame(
        [
            event("US", "Inflation Rate YoY", "2026-06-10T12:30:00.000Z", "2026-05-31T00:00:00Z"),
            event("US", "Core Inflation Rate YoY", "2026-06-10T12:30:00.000Z", "2026-05-31T00:00:00Z"),
            event("US", "PCE Price Index YoY", "2026-06-26T12:30:00.000Z", "2026-05-31T00:00:00Z"),
            event("EU", "Inflation Rate YoY Flash", "2026-06-02T09:00:00.000Z", "2026-05-31T00:00:00Z"),
            event("EU", "Inflation Rate YoY Final", "2026-06-17T09:00:00.000Z", "2026-05-31T00:00:00Z"),
            event("DE", "Inflation Rate YoY Prel", "2026-05-29T12:00:00.000Z", "2026-05-31T00:00:00Z"),
            event("NO", "Inflation Rate YoY", "2026-07-10T06:00:00.000Z", "2026-06-30T00:00:00Z"),
            event("NO", "Inflation Rate YoY Final", "2026-07-15T06:00:00.000Z", "2026-06-30T00:00:00Z", ""),
            event("US", "Inflation Rate YoY", "2026-10-14T12:30:00.000Z", "2026-09-30T00:00:00Z"),
            event("US", "Inflation Rate YoY", "2026-01-01T00:00:00.000Z", ""),
        ]
    )


RELEASE_EVENTS = {
    "US": ("US", ("Inflation Rate YoY",)),
    "US_PCE": ("US", ("PCE Price Index YoY",)),
    "NO": ("NO", ("Inflation Rate YoY", "Inflation Rate YoY Final")),
    "DE": ("EU", ("Inflation Rate YoY Final",)),
    "EA20": ("EU", ("Inflation Rate YoY Final",)),
}


class BuildReleaseCalendarTests(unittest.TestCase):
    def build(self):
        return build_release_calendar(
            events(),
            as_of=pd.Timestamp("2026-10-08"),
            release_events=RELEASE_EVENTS,
        ).set_index(["observation_date", "country_code"])

    def test_maps_reference_month_to_publication_date(self):
        calendar = self.build()
        may = pd.Timestamp("2026-05-01")

        self.assertEqual(calendar.loc[(may, "US"), "publication_date"], pd.Timestamp("2026-06-10"))
        self.assertEqual(calendar.loc[(may, "US_PCE"), "publication_date"], pd.Timestamp("2026-06-26"))

    def test_preserves_utc_release_time_and_local_conversion(self):
        stamp = self.build().loc[(pd.Timestamp("2026-05-01"), "US"), "publication_datetime_utc"]
        self.assertEqual(stamp, pd.Timestamp("2026-06-10T12:30:00Z"))
        self.assertEqual(stamp.tz_convert("Europe/Copenhagen").hour, 14)

    def test_release_today_stays_scheduled_until_release_time(self):
        for instant, expected in [("2026-10-14T12:29:59Z", "scheduled"),
                                  ("2026-10-14T12:30:00Z", "actual")]:
            calendar = build_release_calendar(events(), as_of=pd.Timestamp(instant),
                                              release_events=RELEASE_EVENTS)
            row = calendar.loc[calendar.observation_date.eq(pd.Timestamp("2026-09-01"))].iloc[0]
            self.assertEqual(row.release_status, expected)

    def test_same_day_duplicates_use_latest_time(self):
        raw = pd.DataFrame([
            event("US", "Inflation Rate YoY", "2026-06-10T15:00:00Z", "2026-05-31T00:00:00Z"),
            event("US", "Inflation Rate YoY", "2026-06-10T12:30:00Z", "2026-05-31T00:00:00Z"),
        ])
        calendar = build_release_calendar(raw, as_of=pd.Timestamp("2026-06-11"),
                                          release_events={"US": RELEASE_EVENTS["US"]})
        self.assertEqual(calendar.iloc[0].publication_datetime_utc, pd.Timestamp("2026-06-10T15:00:00Z"))

    def test_legacy_calendar_without_times_still_loads(self):
        calendar = self.build().reset_index().drop(columns="publication_datetime_utc")
        with tempfile.TemporaryDirectory() as directory:
            path = save_release_calendar(calendar, Path(directory) / "calendar.csv")
            loaded = load_release_calendar(path)
        pd.testing.assert_frame_equal(loaded, calendar)

    def test_euro_members_use_eurostat_full_release_not_flash_or_national(self):
        calendar = self.build()
        may = pd.Timestamp("2026-05-01")

        for code in ("DE", "EA20"):
            self.assertEqual(calendar.loc[(may, code), "publication_date"], pd.Timestamp("2026-06-17"))

    def test_duplicate_release_resolves_to_latest_date(self):
        row = self.build().loc[(pd.Timestamp("2026-06-01"), "NO")]

        self.assertEqual(row["publication_date"], pd.Timestamp("2026-07-15"))
        self.assertEqual(row["source_url"], FALLBACK_SOURCE_URL)

    def test_future_releases_are_scheduled(self):
        calendar = self.build()

        self.assertEqual(calendar.loc[(pd.Timestamp("2026-09-01"), "US"), "release_status"], "scheduled")
        self.assertEqual(calendar.loc[(pd.Timestamp("2026-05-01"), "US"), "release_status"], "actual")

    def test_events_without_reference_date_are_dropped(self):
        self.assertEqual(len(self.build()), 6)

    def test_upcoming_releases_window(self):
        calendar = self.build().reset_index()

        upcoming = upcoming_releases(calendar, as_of=pd.Timestamp("2026-10-08"), days=30)

        self.assertEqual(upcoming["country_code"].tolist(), ["US"])
        self.assertEqual(upcoming.iloc[0]["publication_date"], pd.Timestamp("2026-10-14"))

    def test_saved_calendar_round_trips_through_loader(self):
        calendar = self.build().reset_index()
        with tempfile.TemporaryDirectory() as directory:
            path = save_release_calendar(calendar, Path(directory) / "calendar.csv")
            loaded = load_release_calendar(path)
            self.assertIn("T12:30:00Z", path.read_text())

        pd.testing.assert_frame_equal(loaded, calendar)


class FetchTradingViewEventsTests(unittest.TestCase):
    def test_requests_one_month_per_window_and_skips_no_data(self):
        session = mock.Mock()
        ok = mock.Mock()
        ok.json.return_value = {"status": "ok", "result": events().iloc[:2].to_dict("records")}
        empty = mock.Mock()
        empty.json.return_value = {"status": "no_data"}
        session.get.side_effect = [ok, empty, ok]

        frame = fetch_tradingview_events(
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-03-01"),
            countries=["US"],
            session=session,
            pause_seconds=0,
        )

        self.assertEqual(session.get.call_count, 3)
        first = session.get.call_args_list[0].kwargs["params"]
        self.assertEqual(first["from"], "2026-01-01T00:00:00.000Z")
        self.assertEqual(first["to"], "2026-01-31T23:59:00.000Z")
        self.assertEqual(len(frame), 2)

    def test_rejects_unexpected_status(self):
        session = mock.Mock()
        session.get.return_value.json.return_value = {"status": "error"}

        with self.assertRaisesRegex(ValueError, "Unexpected"):
            fetch_tradingview_events(
                pd.Timestamp("2026-01-01"),
                pd.Timestamp("2026-01-01"),
                session=session,
                pause_seconds=0,
            )


if __name__ == "__main__":
    unittest.main()
