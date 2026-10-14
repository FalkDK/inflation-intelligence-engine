import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import pandas as pd

from inflation_engine.automation import (
    exclusive_lock, next_action, rebuild_with_rollback, refresh_denmark,
)
from inflation_engine.data.fetch_release_calendar import (
    build_release_calendar, run as refresh_calendar, save_release_calendar,
)
from inflation_engine.features.availability import load_release_calendar


def frame(month="2026-09-01"):
    rows = []
    for kind, code, component in [
        ("headline", "000000", "all_items"),
        ("core", "151", "all_items_less_energy_unprocessed_food"),
    ]:
        rows.append(dict(date=pd.Timestamp(month), country="Denmark", country_code="DK",
                         region="Nordics", series_type=kind, series_role="aggregate",
                         component=component, value=102.6, source="Statistics Denmark",
                         source_dataset="PRIS01", source_series_id=code, frequency="monthly",
                         unit="index", index_base="2025=100",
                         seasonal_adjustment="not_seasonally_adjusted", observation_status="normal"))
    return pd.DataFrame(rows)


def calendar():
    raw = pd.DataFrame([
        dict(country="DK", title="Inflation Rate YoY", date="2026-10-12T06:00:00Z",
             referenceDate="2026-09-01"),
        dict(country="DK", title="Inflation Rate YoY", date="2026-11-11T07:00:00Z",
             referenceDate="2026-10-01"),
    ])
    return build_release_calendar(raw, as_of=pd.Timestamp("2026-10-13"))


class ReleasePlanningTests(unittest.TestCase):
    def test_normal_day_exits_without_fetch(self):
        self.assertIsNone(next_action(calendar(), frame(), frame(), {}, "2026-10-13T08:00Z"))

    def test_release_day_waits_for_timestamp_plus_fifteen_minutes(self):
        action = next_action(calendar(), frame(), frame(), {}, "2026-11-11T07:00Z")
        self.assertEqual(action, (pd.Timestamp("2026-11-11T07:15Z"), pd.Timestamp("2026-10-01")))

    def test_missed_release_is_fetched_next_day(self):
        wake, month = next_action(calendar(), frame(), frame(), {}, "2026-11-12T07:00Z")
        self.assertEqual(wake, pd.Timestamp("2026-11-12T07:00Z"))
        self.assertEqual(month, pd.Timestamp("2026-10-01"))

    def test_hourly_retry_is_respected(self):
        action = next_action(calendar(), frame(), frame(),
                             {"next_attempt_utc": "2026-11-11T09:00Z"}, "2026-11-11T08:10Z")
        self.assertEqual(action[0], pd.Timestamp("2026-11-11T09:00Z"))

    def test_missing_core_is_not_success(self):
        partial = frame("2026-10-01").iloc[:1]
        action = next_action(calendar(), partial, partial, {}, "2026-11-11T08:10Z")
        self.assertIsNotNone(action)

    def test_pending_build_retries_even_with_complete_observations(self):
        action = next_action(calendar(), frame(), frame(), {"build_pending": True}, "2026-10-13T08:00Z")
        self.assertIsNotNone(action)

    def test_legacy_calendar_does_not_invent_midnight_release(self):
        with self.assertRaisesRegex(ValueError, "no release times"):
            next_action(calendar().drop(columns="publication_datetime_utc"), frame(), frame(), {}, "2026-10-13")

    def test_overlapping_daily_run_cannot_acquire_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            with exclusive_lock(Path(temp) / "lock") as first:
                with exclusive_lock(Path(temp) / "lock") as second:
                    self.assertTrue(first)
                    self.assertFalse(second)


class ReleaseExecutionTests(unittest.TestCase):
    def test_delayed_api_does_not_replace_raw_or_rebuild(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw.parquet"
            frame().to_parquet(raw)
            original = raw.read_bytes()
            persist, rebuild = mock.Mock(), mock.Mock()
            ok = refresh_denmark(pd.Timestamp("2026-10-01"), {}, root / "state.json",
                                 raw_path=raw, fetcher=mock.Mock(return_value=frame()),
                                 persist=persist, rebuild=rebuild)
            self.assertFalse(ok)
            persist.assert_not_called()
            rebuild.assert_not_called()
            self.assertEqual(raw.read_bytes(), original)

    def test_failed_build_remains_pending_and_does_not_refetch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, panel = root / "raw.parquet", root / "panel.parquet"
            frame("2026-10-01").to_parquet(raw)
            state, fetcher = {}, mock.Mock()
            with self.assertRaisesRegex(RuntimeError, "failed"):
                refresh_denmark(pd.Timestamp("2026-10-01"), state, root / "state.json",
                                raw_path=raw, panel_path=panel, fetcher=fetcher,
                                rebuild=mock.Mock(side_effect=RuntimeError("failed")))
            fetcher.assert_not_called()
            self.assertTrue(json.loads((root / "state.json").read_text())["build_pending"])

    def test_success_records_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, panel = root / "raw.parquet", root / "panel.parquet"
            frame().to_parquet(raw)
            fetched = pd.concat([frame(), frame("2026-10-01")], ignore_index=True)
            state = {"next_attempt_utc": "2026-11-11T09:00Z"}
            ok = refresh_denmark(
                pd.Timestamp("2026-10-01"), state, root / "state.json",
                raw_path=raw, panel_path=panel, fetcher=mock.Mock(return_value=fetched),
                persist=lambda f, p: f.to_parquet(p), rebuild=lambda: fetched.to_parquet(panel),
            )
            self.assertTrue(ok)
            self.assertFalse(state["build_pending"])
            self.assertNotIn("next_attempt_utc", state)
            self.assertEqual(state["last_observation"], "2026-10-01")

    def test_rebuild_failure_restores_outputs_and_removes_partial_new_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "data/processed"
            folder.mkdir(parents=True)
            existing = folder / "old.parquet"
            existing.write_bytes(b"last good output")

            def failed_run(*args, **kwargs):
                existing.write_bytes(b"partial overwrite")
                (folder / "new.parquet").write_bytes(b"partial new file")
                return subprocess.CompletedProcess(args, 1, stdout="failed")

            with mock.patch("inflation_engine.automation.subprocess.run", side_effect=failed_run):
                with self.assertRaises(RuntimeError):
                    rebuild_with_rollback(root)
            self.assertEqual(existing.read_bytes(), b"last good output")
            self.assertFalse((folder / "new.parquet").exists())

    def test_weekly_calendar_refresh_retains_history_and_replaces_rescheduled_time(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "calendar.csv"
            previous = pd.DataFrame([
                dict(country="DK", title="Inflation Rate YoY", date="2020-02-10T07:00Z", referenceDate="2020-01-01"),
                dict(country="DK", title="Inflation Rate YoY", date="2026-11-11T07:00Z", referenceDate="2026-10-01"),
            ])
            save_release_calendar(build_release_calendar(previous, as_of=pd.Timestamp("2026-10-13")), path)
            upcoming = pd.DataFrame([
                dict(country="DK", title="Inflation Rate YoY", date="2026-11-12T08:00Z", referenceDate="2026-10-01"),
            ])
            with mock.patch("inflation_engine.data.fetch_release_calendar.fetch_tradingview_events", return_value=upcoming) as fetch:
                refresh_calendar(path, recent_months=2, as_of=pd.Timestamp("2026-10-13"))
            self.assertEqual(fetch.call_args.args[0], pd.Timestamp("2026-08-01"))
            loaded = load_release_calendar(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded.iloc[-1].publication_datetime_utc, pd.Timestamp("2026-11-12T08:00Z"))


if __name__ == "__main__":
    unittest.main()
