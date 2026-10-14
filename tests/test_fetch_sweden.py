import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import requests

from inflation_engine.data.fetch_sweden import (
    SCB_API_BASE_URL,
    SCB_DATA_QUERY_BASE_URL,
    SWEDEN_CPI_TABLES,
    fetch_scb_component_dataset,
    fetch_scb_dataset,
    fetch_swedish_cpi,
    latest_swedish_cpi_observations,
    save_swedish_cpi,
)
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS


def dataset(content_code, values):
    return {
        "version": "2.0",
        "class": "dataset",
        "id": ["ContentsCode", "Tid"],
        "size": [1, 2],
        "dimension": {
            "ContentsCode": {
                "category": {"index": {content_code: 0}},
            },
            "Tid": {
                "category": {"index": {"2026M04": 0, "2026M05": 1}},
            },
        },
        "value": values,
    }


def component_dataset():
    return {
        "version": "2.0",
        "class": "dataset",
        "id": ["VaruTjanstegrupp", "ContentsCode", "Tid"],
        "size": [3, 1, 2],
        "dimension": {
            "VaruTjanstegrupp": {
                "category": {
                    "index": {"00": 0, "01": 1, "06": 2},
                    "label": {
                        "00": "TOTAL",
                        "01": "FOOD AND NON-ALCOHOLIC BEVERAGES",
                        "06": "HEALTH",
                    },
                },
            },
            "ContentsCode": {
                "category": {"index": {"0000083S": 0}},
            },
            "Tid": {
                "category": {"index": {"2026M04": 0, "2026M05": 1}},
            },
        },
        "value": [125.09, 125.40, 110.0, 111.0, 120.0, 121.0],
    }


def component_dataset_may_only():
    return {
        "version": "2.0",
        "class": "dataset",
        "id": ["VaruTjanstegrupp", "ContentsCode", "Tid"],
        "size": [3, 1, 1],
        "dimension": {
            "VaruTjanstegrupp": {
                "category": {
                    "index": {"00": 0, "01": 1, "06": 2},
                    "label": {
                        "00": "TOTAL",
                        "01": "FOOD AND NON-ALCOHOLIC BEVERAGES",
                        "06": "HEALTH",
                    },
                },
            },
            "ContentsCode": {
                "category": {"index": {"0000083S": 0}},
            },
            "Tid": {
                "category": {"index": {"2026M05": 0}},
            },
        },
        "value": [125.40, 111.0, 121.0],
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

    def get(self, url, *, params=None, timeout):
        self.calls.append((url, params, timeout))
        if url.startswith(SCB_DATA_QUERY_BASE_URL) and "2026M04" not in url:
            return FakeResponse(component_dataset_may_only())
        return FakeResponse(component_dataset())

    def post(self, url, *, params, json, timeout):
        self.calls.append((url, params, json, timeout))
        if "TAB6590" in url:
            return FakeResponse(dataset("000007ZN", [123.86, 125.09]))
        if "TAB6613" in url:
            return FakeResponse(component_dataset())
        return FakeResponse(dataset("000007ZW", [121.58, 122.45]))


class FailingSession:
    def post(self, url, *, params, json, timeout):
        raise requests.ConnectionError("network unavailable")


class FailingMetadataSession(FakeSession):
    def get(self, url, *, params=None, timeout):
        if url.startswith(SCB_API_BASE_URL):
            raise requests.ConnectionError("metadata unavailable")
        return super().get(url, params=params, timeout=timeout)


class FailingComponentSession(FakeSession):
    def get(self, url, *, params=None, timeout):
        if url.startswith(SCB_DATA_QUERY_BASE_URL):
            raise requests.ConnectionError("component data unavailable")
        return super().get(url, params=params, timeout=timeout)


class FetchSwedenTests(unittest.TestCase):
    def test_fetch_dataset_uses_pxweb_v2_selection(self):
        session = FakeSession()

        fetch_scb_dataset("TAB6590", "000007ZN", session=session)

        url, params, payload, _ = session.calls[0]
        self.assertEqual(url, f"{SCB_API_BASE_URL}/TAB6590/data")
        self.assertEqual(params["outputFormat"], "json-stat2")
        self.assertEqual(
            payload["selection"],
            [
                {"variableCode": "ContentsCode", "valueCodes": ["000007ZN"]},
                {"variableCode": "Tid", "valueCodes": ["*"]},
            ],
        )

    def test_component_fetch_uses_encoded_get_selection(self):
        session = FakeSession()

        fetch_scb_component_dataset(
            "TAB6613",
            "0000083S",
            component_dimension="VaruTjanstegrupp",
            component_codes=["01", "06"],
            periods=["2026M05"],
            session=session,
        )

        url, params, _ = session.calls[0]
        self.assertTrue(url.startswith(f"{SCB_DATA_QUERY_BASE_URL}/TAB6613/data?"))
        self.assertIsNone(params)
        self.assertIn("outputFormat=json-stat2", url)
        self.assertIn("valueCodes%5BContentsCode%5D=0000083S", url)
        self.assertIn("valueCodes%5BVaruTjanstegrupp%5D=01,06", url)
        self.assertIn("valueCodes%5BTid%5D=2026M05", url)

    def test_fetch_swedish_cpi_normalizes_both_series(self):
        frame = fetch_swedish_cpi(session=FakeSession())

        self.assertEqual(frame.columns.tolist(), CPI_SCHEMA_COLUMNS)
        self.assertEqual(len(frame), 8)
        self.assertEqual(frame["country_code"].unique().tolist(), ["SE"])
        self.assertEqual(set(frame["series_type"]), {"headline", "core"})
        self.assertEqual(
            set(frame["component"]),
            {
                "cpif_fixed_interest_rate",
                "cpif_fixed_interest_rate_excluding_energy",
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

    def test_fetch_swedish_cpi_reuses_current_components(self):
        existing = fetch_swedish_cpi(session=FakeSession())
        session = FakeSession()

        frame = fetch_swedish_cpi(session=session, existing_frame=existing)

        component_calls = [
            call
            for call in session.calls
            if call[0].startswith(SCB_DATA_QUERY_BASE_URL)
        ]
        self.assertEqual(component_calls, [])
        self.assertEqual(len(frame), 8)

    def test_refreshes_aggregates_when_component_metadata_is_unavailable(self):
        existing = fetch_swedish_cpi(session=FakeSession())

        frame = fetch_swedish_cpi(
            session=FailingMetadataSession(), existing_frame=existing
        )

        self.assertEqual(len(frame), 8)
        self.assertEqual(
            len(frame.loc[frame["series_role"].eq("component")]), 4
        )

    def test_preserves_complete_components_when_incremental_batch_fails(self):
        existing = fetch_swedish_cpi(session=FakeSession())
        existing = existing.loc[
            ~(
                existing["source_dataset"].eq("TAB6613")
                & existing["date"].eq(pd.Timestamp("2026-05-01"))
            )
        ].reset_index(drop=True)

        frame = fetch_swedish_cpi(
            session=FailingComponentSession(), existing_frame=existing
        )

        component_dates = frame.loc[
            frame["series_role"].eq("component"), "date"
        ]
        self.assertEqual(component_dates.max(), pd.Timestamp("2026-04-01"))

    def test_fetch_swedish_cpi_appends_only_new_component_periods(self):
        existing = fetch_swedish_cpi(session=FakeSession())
        existing = existing.loc[
            ~(
                existing["source_dataset"].eq("TAB6613")
                & existing["date"].eq(pd.Timestamp("2026-05-01"))
            )
        ].reset_index(drop=True)
        session = FakeSession()

        fetch_swedish_cpi(session=session, existing_frame=existing)

        component_calls = [
            call
            for call in session.calls
            if call[0].startswith(SCB_DATA_QUERY_BASE_URL)
        ]
        self.assertEqual(len(component_calls), 1)
        self.assertIn("valueCodes%5BTid%5D=2026M05", component_calls[0][0])
        self.assertNotIn("2026M04", component_calls[0][0])

    def test_latest_observations_returns_may_values(self):
        latest = latest_swedish_cpi_observations(
            fetch_swedish_cpi(session=FakeSession())
        )

        self.assertTrue((latest["date"] == pd.Timestamp("2026-05-01")).all())
        self.assertEqual(
            latest.set_index("series_type")["value"].to_dict(),
            {"core": 122.45, "headline": 125.09},
        )

    def test_table_mapping_is_current(self):
        self.assertEqual(SWEDEN_CPI_TABLES["headline"]["table"], "TAB6590")
        self.assertEqual(
            SWEDEN_CPI_TABLES["headline"]["content_code"],
            "000007ZN",
        )
        self.assertEqual(SWEDEN_CPI_TABLES["core"]["table"], "TAB6593")

    def test_fetch_error_is_concise(self):
        with self.assertRaisesRegex(RuntimeError, "Statistics Sweden request failed"):
            fetch_scb_dataset(
                "TAB6590",
                "000007ZN",
                session=FailingSession(),
            )

    def test_save_swedish_cpi_writes_parquet(self):
        frame = fetch_swedish_cpi(session=FakeSession())
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "sweden.parquet"
            destination = save_swedish_cpi(frame, output)
            reloaded = pd.read_parquet(destination)

        self.assertEqual(destination, output)
        self.assertEqual(len(reloaded), 8)


if __name__ == "__main__":
    unittest.main()
