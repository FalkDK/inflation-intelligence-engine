import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import requests

from inflation_engine.data.fetch_statbank_dk import (
    STATBANK_DATA_URL,
    fetch_danish_cpi,
    fetch_statbank_payload,
    latest_danish_cpi_observations,
    normalize_statbank_cpi,
    save_danish_cpi,
)
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS


DATASET = {
    "dimension": {
        "VAREGR": {
            "category": {
                "index": {"000000": 0, "151": 1},
                "label": {
                    "000000": "00 Consumer price index, total",
                    "151": (
                        "15.1 Overall Consumer Price Index - excl. energy "
                        "and unprocessed food (core inflation)"
                    ),
                },
            }
        },
        "ENHED": {
            "category": {
                "index": {"100": 0},
            }
        },
        "ContentsCode": {
            "category": {
                "index": {"PRIS01": 0},
            }
        },
        "Tid": {
            "category": {
                "index": {"2026M04": 0, "2026M05": 1},
            }
        },
    },
    "id": ["VAREGR", "ENHED", "ContentsCode", "Tid"],
    "size": [2, 1, 1, 2],
    "value": [100.74, 101.32, 101.20, 101.74],
}


TABLEINFO = {
    "variables": [
        {
            "id": "Tid",
            "values": [
                {"id": "2026M04", "text": "2026M04"},
                {"id": "2026M05", "text": "2026M05"},
            ],
        }
    ],
    "updated": "2026-06-10T08:00:00",
}


def dataset_may_only():
    return {
        **DATASET,
        "dimension": {
            **DATASET["dimension"],
            "Tid": {
                "category": {
                    "index": {"2026M05": 0},
                }
            },
        },
        "size": [2, 1, 1, 1],
        "value": [101.32, 101.74],
    }


class FakeResponse:
    def __init__(self, payload=None):
        self.payload = payload or {"dataset": DATASET}

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, *, json, timeout):
        self.calls.append((url, json, timeout))
        periods = {
            item["code"]: item["values"] for item in json["variables"]
        }["Tid"]
        if periods == ["2026M05"]:
            return FakeResponse({"dataset": dataset_may_only()})
        return FakeResponse()

    def get(self, url, *, params, timeout):
        self.calls.append((url, params, timeout))
        return FakeResponse(TABLEINFO)


class FailingSession:
    def post(self, url, *, json, timeout):
        raise requests.ConnectionError("network unavailable")


class FetchStatbankDkTests(unittest.TestCase):
    def test_fetch_payload_requests_pris01_series(self):
        session = FakeSession()

        dataset = fetch_statbank_payload(session=session)

        self.assertEqual(dataset, DATASET)
        self.assertEqual(len(session.calls), 1)
        url, payload, _ = session.calls[0]
        self.assertEqual(url, STATBANK_DATA_URL)
        self.assertEqual(payload["table"], "PRIS01")
        self.assertEqual(payload["format"], "JSONSTAT")
        variables = {item["code"]: item["values"] for item in payload["variables"]}
        self.assertEqual(variables["VAREGR"], ["*"])
        self.assertEqual(variables["ENHED"], ["100"])
        self.assertEqual(variables["Tid"], ["*"])

    def test_normalize_statbank_cpi_uses_standard_schema(self):
        frame = normalize_statbank_cpi(DATASET)

        self.assertEqual(frame.columns.tolist(), CPI_SCHEMA_COLUMNS)
        self.assertEqual(len(frame), 4)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["date"]))
        self.assertEqual(frame["country_code"].unique().tolist(), ["DK"])
        self.assertEqual(set(frame["series_type"]), {"headline", "core"})
        self.assertEqual(frame["source"].unique().tolist(), ["Statistics Denmark"])

    def test_normalize_derives_dimensions_when_full_response_omits_metadata(self):
        full_response_shape = {
            key: value
            for key, value in DATASET.items()
            if key not in {"id", "size"}
        }
        full_response_shape["dimension"] = {
            **DATASET["dimension"],
            "id": DATASET["id"],
            "size": DATASET["size"],
            "role": {"metric": ["ContentsCode"], "time": ["Tid"]},
        }

        frame = normalize_statbank_cpi(full_response_shape)

        self.assertEqual(len(frame), 4)
        self.assertEqual(set(frame["series_type"]), {"headline", "core"})

    def test_fetch_danish_cpi_uses_payload_and_normalization(self):
        component_dataset = {
            **DATASET,
            "dimension": {
                **DATASET["dimension"],
                "VAREGR": {
                    "category": {
                        "index": {"000000": 0, "151": 1, "01": 2},
                        "label": {
                            "000000": "00 Consumer price index, total",
                            "151": (
                                "15.1 Overall Consumer Price Index - excl. energy "
                                "and unprocessed food (core inflation)"
                            ),
                            "01": "01 Food and non-alcoholic beverages",
                        },
                    }
                },
            },
            "size": [3, 1, 1, 2],
            "value": [100.74, 101.32, 101.20, 101.74, 98.1, 98.7],
        }

        class ComponentResponse(FakeResponse):
            def json(self):
                return {"dataset": component_dataset}

        class ComponentSession(FakeSession):
            def post(self, url, *, json, timeout):
                self.calls.append((url, json, timeout))
                return ComponentResponse()

        frame = fetch_danish_cpi(session=ComponentSession())

        self.assertEqual(len(frame), 6)
        self.assertEqual(
            set(frame["component"]),
            {
                "all_items",
                "all_items_less_energy_unprocessed_food",
                "food_and_non_alcoholic_beverages",
            },
        )
        self.assertEqual(
            frame.loc[
                frame["component"].eq("food_and_non_alcoholic_beverages"),
                "series_role",
            ].unique().tolist(),
            ["component"],
        )

    def test_fetch_danish_cpi_reuses_current_existing_file(self):
        existing = fetch_danish_cpi(session=FakeSession())
        session = FakeSession()

        frame = fetch_danish_cpi(session=session, existing_frame=existing)

        post_calls = [
            call for call in session.calls if call[0] == STATBANK_DATA_URL
        ]
        self.assertEqual(post_calls, [])
        self.assertEqual(len(frame), 4)

    def test_fetch_danish_cpi_appends_only_new_periods(self):
        existing = fetch_danish_cpi(session=FakeSession())
        existing = existing.loc[
            ~existing["date"].eq(pd.Timestamp("2026-05-01"))
        ].reset_index(drop=True)
        session = FakeSession()

        frame = fetch_danish_cpi(session=session, existing_frame=existing)

        post_calls = [
            call for call in session.calls if call[0] == STATBANK_DATA_URL
        ]
        self.assertEqual(len(post_calls), 1)
        variables = {item["code"]: item["values"] for item in post_calls[0][1]["variables"]}
        self.assertEqual(variables["Tid"], ["2026M05"])
        self.assertEqual(len(frame), 4)

    def test_latest_observations_returns_headline_and_core(self):
        frame = normalize_statbank_cpi(DATASET)

        latest = latest_danish_cpi_observations(frame)

        self.assertEqual(len(latest), 2)
        self.assertTrue((latest["date"] == pd.Timestamp("2026-05-01")).all())
        self.assertEqual(
            latest.set_index("series_type")["value"].to_dict(),
            {"core": 101.74, "headline": 101.32},
        )

    def test_missing_series_is_rejected(self):
        incomplete_dataset = {
            **DATASET,
            "dimension": {
                **DATASET["dimension"],
                "VAREGR": {"category": {"index": {"000000": 0}}},
            },
            "size": [1, 1, 1, 2],
            "value": [100.74, 101.32],
        }

        with self.assertRaisesRegex(ValueError, "missing CPI series"):
            normalize_statbank_cpi(incomplete_dataset)

    def test_fetch_error_is_concise(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "Statistics Denmark CPI request failed",
        ):
            fetch_statbank_payload(session=FailingSession())

    def test_save_danish_cpi_writes_parquet(self):
        frame = normalize_statbank_cpi(DATASET)

        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "statbank_dk_cpi.parquet"
            destination = save_danish_cpi(frame, output)
            reloaded = pd.read_parquet(destination)

        self.assertEqual(destination, output)
        self.assertEqual(reloaded.columns.tolist(), CPI_SCHEMA_COLUMNS)
        self.assertEqual(len(reloaded), 4)


if __name__ == "__main__":
    unittest.main()
