import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import requests

from inflation_engine.data.fetch_ssb import (
    NORWAY_CPI_TABLES,
    SSB_API_BASE_URL,
    fetch_norwegian_cpi,
    fetch_ssb_dataset,
    latest_norwegian_cpi_observations,
    save_norwegian_cpi,
)
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS


def headline_dataset():
    return {
        "version": "2.0",
        "class": "dataset",
        "id": ["ContentsCode", "Tid"],
        "size": [1, 2],
        "dimension": {
            "ContentsCode": {
                "category": {"index": {"KpiIndMnd": 0}},
            },
            "Tid": {
                "category": {"index": {"2026M04": 0, "2026M05": 1}},
            },
        },
        "value": [102.8, 103.0],
    }


def headline_dataset_may_only():
    return {
        **headline_dataset(),
        "dimension": {
            **headline_dataset()["dimension"],
            "Tid": {
                "category": {"index": {"2026M05": 0}},
            },
        },
        "size": [1, 1],
        "value": [103.0],
    }


def core_dataset():
    return {
        "version": "2.0",
        "class": "dataset",
        "id": ["KPIavledetSerie", "ContentsCode", "Tid"],
        "size": [1, 1, 2],
        "dimension": {
            "KPIavledetSerie": {
                "category": {"index": {"KPI-JAE": 0}},
            },
            "ContentsCode": {
                "category": {"index": {"KPIJustIndMnd": 0}},
            },
            "Tid": {
                "category": {"index": {"2026M04": 0, "2026M05": 1}},
            },
        },
        "value": [102.7, 103.1],
    }


def core_dataset_may_only():
    return {
        **core_dataset(),
        "dimension": {
            **core_dataset()["dimension"],
            "Tid": {
                "category": {"index": {"2026M05": 0}},
            },
        },
        "size": [1, 1, 1],
        "value": [103.1],
    }


def component_dataset():
    return {
        "version": "2.0",
        "class": "dataset",
        "id": ["VareTjenesteGrp", "ContentsCode", "Tid"],
        "size": [3, 1, 2],
        "dimension": {
            "VareTjenesteGrp": {
                "category": {
                    "index": {"00": 0, "01": 1, "06": 2},
                    "label": {
                        "00": "Total",
                        "01": "Food and non-alcoholic beverages",
                        "06": "Health",
                    },
                },
            },
            "ContentsCode": {
                "category": {"index": {"KpiIndMnd": 0}},
            },
            "Tid": {
                "category": {"index": {"2026M04": 0, "2026M05": 1}},
            },
        },
        "value": [103.0, 103.1, 110.0, 111.0, 120.0, 121.0],
    }


def component_dataset_may_only():
    return {
        **component_dataset(),
        "dimension": {
            **component_dataset()["dimension"],
            "Tid": {
                "category": {"index": {"2026M05": 0}},
            },
        },
        "size": [3, 1, 1],
        "value": [103.1, 111.0, 121.0],
    }


def metadata_dataset():
    return {
        "version": "2.0",
        "class": "dataset",
        "id": ["Tid"],
        "size": [2],
        "updated": "2026-06-10T08:00:00",
        "dimension": {
            "Tid": {
                "category": {"index": {"2026M04": 0, "2026M05": 1}},
            },
        },
        "value": [0, 0],
    }


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, *, params, json, timeout):
        self.calls.append((url, params, json, timeout))
        variables = {
            item["variableCode"]: item["valueCodes"]
            for item in json["selection"]
        }
        may_only = variables["Tid"] == ["2026M05"]
        if "14710" in url:
            if may_only:
                return FakeResponse(headline_dataset_may_only())
            return FakeResponse(headline_dataset())
        if "14700" in url:
            if may_only:
                return FakeResponse(component_dataset_may_only())
            return FakeResponse(component_dataset())
        if may_only:
            return FakeResponse(core_dataset_may_only())
        return FakeResponse(core_dataset())

    def get(self, url, *, params, timeout):
        self.calls.append((url, params, timeout))
        return FakeResponse(metadata_dataset())


class FailingSession:
    def post(self, url, *, params, json, timeout):
        raise requests.ConnectionError("network unavailable")


class FetchSsbTests(unittest.TestCase):
    def test_fetch_dataset_uses_pxweb_v2_selection(self):
        session = FakeSession()
        selection = NORWAY_CPI_TABLES["core"]["selection"]

        fetch_ssb_dataset("14706", selection, session=session)

        url, params, payload, _ = session.calls[0]
        self.assertEqual(url, f"{SSB_API_BASE_URL}/14706/data")
        self.assertEqual(params["outputFormat"], "json-stat2")
        self.assertEqual(payload["selection"], selection)

    def test_fetch_norwegian_cpi_normalizes_both_series(self):
        frame = fetch_norwegian_cpi(session=FakeSession())

        self.assertEqual(frame.columns.tolist(), CPI_SCHEMA_COLUMNS)
        self.assertEqual(len(frame), 8)
        self.assertEqual(frame["country_code"].unique().tolist(), ["NO"])
        self.assertEqual(set(frame["series_type"]), {"headline", "core"})
        self.assertEqual(
            set(frame["component"]),
            {
                "all_items",
                "cpi_adjusted_taxes_excluding_energy",
                "food_and_non_alcoholic_beverages",
                "health",
            },
        )
        self.assertEqual(
            frame.loc[frame["component"].eq("health"), "series_role"]
            .unique()
            .tolist(),
            ["component"],
        )

    def test_fetch_norwegian_cpi_reuses_current_existing_file(self):
        existing = fetch_norwegian_cpi(session=FakeSession())
        session = FakeSession()

        frame = fetch_norwegian_cpi(session=session, existing_frame=existing)

        post_calls = [
            call
            for call in session.calls
            if call[0].endswith("/data")
        ]
        self.assertEqual(post_calls, [])
        self.assertEqual(len(frame), 8)

    def test_fetch_norwegian_cpi_appends_only_new_periods(self):
        existing = fetch_norwegian_cpi(session=FakeSession())
        existing = existing.loc[
            ~existing["date"].eq(pd.Timestamp("2026-05-01"))
        ].reset_index(drop=True)
        session = FakeSession()

        frame = fetch_norwegian_cpi(session=session, existing_frame=existing)

        post_calls = [
            call
            for call in session.calls
            if call[0].endswith("/data")
        ]
        self.assertEqual(len(post_calls), 3)
        for _, _, payload, _ in post_calls:
            variables = {
                item["variableCode"]: item["valueCodes"]
                for item in payload["selection"]
            }
            self.assertEqual(variables["Tid"], ["2026M05"])
        self.assertEqual(len(frame), 8)

    def test_latest_observations_returns_may_values(self):
        latest = latest_norwegian_cpi_observations(
            fetch_norwegian_cpi(session=FakeSession())
        )

        self.assertTrue((latest["date"] == pd.Timestamp("2026-05-01")).all())
        self.assertEqual(
            latest.set_index("series_type")["value"].to_dict(),
            {"core": 103.1, "headline": 103.0},
        )

    def test_table_mapping_is_current(self):
        self.assertEqual(NORWAY_CPI_TABLES["headline"]["table"], "14710")
        self.assertEqual(NORWAY_CPI_TABLES["core"]["table"], "14706")

    def test_fetch_error_is_concise(self):
        with self.assertRaisesRegex(RuntimeError, "Statistics Norway request failed"):
            fetch_ssb_dataset(
                "14710",
                NORWAY_CPI_TABLES["headline"]["selection"],
                session=FailingSession(),
            )

    def test_save_norwegian_cpi_writes_parquet(self):
        frame = fetch_norwegian_cpi(session=FakeSession())
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "norway.parquet"
            destination = save_norwegian_cpi(frame, output)
            reloaded = pd.read_parquet(destination)

        self.assertEqual(destination, output)
        self.assertEqual(len(reloaded), 8)


if __name__ == "__main__":
    unittest.main()
