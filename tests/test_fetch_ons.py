import unittest
import csv
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import requests

from inflation_engine.data.fetch_ons import (
    ONS_GENERATOR_URL,
    UK_CPI_SERIES,
    discover_ons_item_series,
    fetch_ons_database_csv,
    fetch_ons_series_csv,
    fetch_uk_cpi,
    latest_uk_cpi_observations,
    normalize_ons_series,
    save_uk_cpi,
)
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS


def ons_csv(series_id, title, values):
    rows = [
        f'"Title","{title}"',
        f'"CDID","{series_id}"',
        '"Source dataset ID","MM23"',
        '"PreUnit",""',
        '"Unit","Index, base year = 100"',
        '"Release date","20-05-2026"',
        '"Next release","17 June 2026"',
        '"Important notes",',
        '"2025","136.0"',
    ]
    rows.extend(f'"{period}","{value}"' for period, value in values)
    return "\n".join(rows)


HEADLINE_CSV = ons_csv(
    "D7BT",
    "CPI INDEX 00: ALL ITEMS 2015=100",
    [("2026 MAR", "141.0"), ("2026 APR", "142.1")],
)
CORE_CSV = ons_csv(
    "DKC6",
    (
        "CPI INDEX: Excluding Energy, food, alcoholic beverages "
        "& tobacco 2015=100"
    ),
    [("2026 MAR", "137.2"), ("2026 APR", "138.2")],
)


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, *, params=None, timeout):
        self.calls.append((url, params, timeout))
        if params is None:
            output = StringIO()
            writer = csv.writer(output)
            series_ids = list(UK_CPI_SERIES) + ["L83B"]
            titles = [f"CPI {code} 2015=100" for code in UK_CPI_SERIES] + [
                "CPI INDEX 01.1.1.1 Rice 2015=100"
            ]
            writer.writerow(["Title", *titles])
            writer.writerow(["CDID", *series_ids])
            writer.writerow(["PreUnit", *([""] * len(series_ids))])
            # ONS leaves "Unit" blank for detailed item series in the real
            # database; the trailing column exercises that relaxed-check path.
            writer.writerow(
                ["Unit", *(["Index, base year = 100"] * (len(series_ids) - 1)), ""]
            )
            writer.writerow(
                ["Release Date", *(["20-05-2026"] * len(series_ids))]
            )
            writer.writerow(
                ["Next release", *(["17 June 2026"] * len(series_ids))]
            )
            writer.writerow(["Important Notes", *([""] * len(series_ids))])
            for period, headline, core, component, item in [
                ("2026 MAR", "141.0", "137.2", "120.0", "110.0"),
                ("2026 APR", "142.1", "138.2", "121.0", "111.0"),
            ]:
                writer.writerow(
                    [
                        period,
                        *[
                            headline
                            if code == "D7BT"
                            else core
                            if code == "DKC6"
                            else component
                            for code in UK_CPI_SERIES
                        ],
                        item,
                    ]
                )
            return FakeResponse(output.getvalue())
        series_id = params["uri"].split("/")[-2].upper()
        if series_id == "D7BT":
            return FakeResponse(HEADLINE_CSV)
        if series_id == "DKC6":
            return FakeResponse(CORE_CSV)
        return FakeResponse(
            ons_csv(
                series_id,
                f"CPI component {series_id} 2015=100",
                [("2026 MAR", "120.0"), ("2026 APR", "121.0")],
            )
        )


class FailingSession:
    def get(self, url, *, params, timeout):
        raise requests.ConnectionError("network unavailable")


class FetchOnsTests(unittest.TestCase):
    def test_fetch_series_uses_generator_csv(self):
        session = FakeSession()

        result = fetch_ons_series_csv("D7BT", session=session)

        self.assertEqual(result, HEADLINE_CSV)
        url, params, _ = session.calls[0]
        self.assertEqual(url, ONS_GENERATOR_URL)
        self.assertEqual(params["format"], "csv")
        self.assertTrue(params["uri"].endswith("/d7bt/mm23"))

    def test_normalize_ons_series_uses_standard_schema(self):
        frame = normalize_ons_series(
            HEADLINE_CSV,
            expected_series_id="D7BT",
            **UK_CPI_SERIES["D7BT"],
        )

        self.assertEqual(frame.columns.tolist(), CPI_SCHEMA_COLUMNS)
        self.assertEqual(frame["country_code"].unique().tolist(), ["GB"])
        self.assertEqual(frame["source_series_id"].unique().tolist(), ["D7BT"])
        self.assertEqual(frame["index_base"].unique().tolist(), ["2015=100"])
        self.assertEqual(len(frame), 2)

    def test_fetch_uk_cpi_returns_headline_and_core(self):
        frame = fetch_uk_cpi(session=FakeSession(), include_detailed_items=False)

        self.assertEqual(len(frame), 14)
        self.assertEqual(set(frame["series_type"]), {"headline", "core"})
        self.assertEqual(set(frame["source_series_id"]), set(UK_CPI_SERIES))

    def test_discover_ons_item_series_finds_coded_items_only(self):
        database_csv = fetch_ons_database_csv(session=FakeSession())

        codes = discover_ons_item_series(database_csv)

        self.assertEqual(codes, {"01.1.1.1": ("L83B", "Rice")})

    def test_fetch_uk_cpi_includes_detailed_items_by_default(self):
        frame = fetch_uk_cpi(session=FakeSession())

        self.assertEqual(len(frame), 16)
        item_rows = frame.loc[frame["source_series_id"] == "01.1.1.1"]
        self.assertEqual(len(item_rows), 2)
        self.assertEqual(item_rows["component"].unique().tolist(), ["rice"])
        self.assertEqual(item_rows["series_type"].unique().tolist(), ["headline"])
        self.assertEqual(
            item_rows.set_index("date")["value"].loc[pd.Timestamp("2026-04-01")],
            111.0,
        )

    def test_latest_observations_returns_april_values(self):
        frame = fetch_uk_cpi(session=FakeSession(), include_detailed_items=False)

        latest = latest_uk_cpi_observations(frame)

        self.assertTrue((latest["date"] == pd.Timestamp("2026-04-01")).all())
        values = latest.set_index("component")["value"]
        self.assertEqual(values["all_items"], 142.1)
        self.assertEqual(
            values["all_items_less_food_energy_alcohol_tobacco"], 138.2
        )
        self.assertEqual(values["energy"], 121.0)

    def test_mismatched_series_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            normalize_ons_series(
                HEADLINE_CSV,
                expected_series_id="DKC6",
                **UK_CPI_SERIES["DKC6"],
            )

    def test_fetch_error_is_concise(self):
        with self.assertRaisesRegex(RuntimeError, "ONS CPI request failed"):
            fetch_ons_series_csv("D7BT", session=FailingSession())

    def test_save_uk_cpi_writes_parquet(self):
        frame = fetch_uk_cpi(session=FakeSession(), include_detailed_items=False)

        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "ons_uk_cpi.parquet"
            destination = save_uk_cpi(frame, output)
            reloaded = pd.read_parquet(destination)

        self.assertEqual(destination, output)
        self.assertEqual(reloaded.columns.tolist(), CPI_SCHEMA_COLUMNS)


if __name__ == "__main__":
    unittest.main()
