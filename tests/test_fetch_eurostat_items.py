import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import requests

from inflation_engine.data.fetch_eurostat_items import (
    EUROSTAT_HICP_ITEM_INDEX_URL,
    fetch_eurostat_item_index_dataset,
    normalize_eurostat_item_index,
    save_hicp_items,
)
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS


def index_dataset(geo="FR"):
    """coicop=[CP01, FOOD], geo=[geo], time=[2025-01, 2025-02]. FOOD is a
    non-hierarchical special aggregate and must be filtered out."""
    return {
        "id": ["coicop", "geo", "time"],
        "size": [2, 1, 2],
        "dimension": {
            "coicop": {"category": {"index": {"CP01": 0, "FOOD": 1}}},
            "geo": {"category": {"index": {geo: 0}}},
            "time": {"category": {"index": {"2025-01": 0, "2025-02": 1}}},
        },
        "value": [100.0, 101.0, 110.0, 111.0],
    }


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, *, params, headers, timeout):
        self.calls.append((url, params, headers, timeout))
        return FakeResponse(self.payload)


class FailingSession:
    def get(self, url, *, params, headers, timeout):
        raise requests.ConnectionError("network unavailable")


class FetchEurostatItemsTests(unittest.TestCase):
    def test_fetch_dataset_requests_expected_params(self):
        session = FakeSession(index_dataset())

        dataset = fetch_eurostat_item_index_dataset("FR", session=session)

        self.assertEqual(dataset, index_dataset())
        self.assertEqual(len(session.calls), 1)
        url, params, headers, _ = session.calls[0]
        self.assertEqual(url, EUROSTAT_HICP_ITEM_INDEX_URL)
        self.assertEqual(
            params,
            [
                ("lang", "en"),
                ("geo", "FR"),
                ("unit", "I15"),
                ("sinceTimePeriod", "1996"),
            ],
        )
        self.assertEqual(headers, {"Accept": "application/json"})

    def test_fetch_error_is_concise(self):
        with self.assertRaisesRegex(RuntimeError, "Eurostat HICP item index request failed"):
            fetch_eurostat_item_index_dataset("FR", session=FailingSession())

    def test_normalize_keeps_only_the_true_coicop_hierarchy(self):
        frame = normalize_eurostat_item_index({"FR": index_dataset("FR")})

        self.assertEqual(frame.columns.tolist(), CPI_SCHEMA_COLUMNS)
        self.assertEqual(set(frame["source_series_id"]), {"CP01"})
        self.assertEqual(len(frame), 2)
        self.assertEqual(set(frame["component"]), {"hicp_item_cp01"})
        self.assertEqual(frame["country_code"].unique().tolist(), ["FR"])
        self.assertEqual(frame["country"].unique().tolist(), ["France"])
        self.assertEqual(frame["series_type"].unique().tolist(), ["headline"])
        self.assertEqual(frame["series_role"].unique().tolist(), ["component"])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["date"]))
        self.assertEqual(
            sorted(frame["date"].dt.strftime("%Y-%m-%d")),
            ["2025-01-01", "2025-02-01"],
        )

    def test_normalize_merges_multiple_geographies(self):
        frame = normalize_eurostat_item_index(
            {"FR": index_dataset("FR"), "DE": index_dataset("DE")}
        )

        self.assertEqual(set(frame["country_code"]), {"FR", "DE"})
        self.assertEqual(len(frame), 4)

    def test_normalize_drops_a_published_zero_as_though_it_were_missing(self):
        """Some country/item combinations are genuinely published as zero
        rather than withheld -- e.g. Ireland's water-supply component reads 0
        for its entire history because domestic water charges were
        suspended/abolished there. An index can't legitimately be zero, and
        the common CPI schema rejects non-positive values outright, so these
        must be dropped like a missing observation rather than kept."""
        dataset = {
            "id": ["coicop", "geo", "time"],
            "size": [1, 1, 2],
            "dimension": {
                "coicop": {"category": {"index": {"CP0441": 0}}},
                "geo": {"category": {"index": {"IE": 0}}},
                "time": {"category": {"index": {"2025-01": 0, "2025-02": 1}}},
            },
            "value": [0.0, 101.0],
        }

        frame = normalize_eurostat_item_index({"IE": dataset})

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame["date"].iloc[0].strftime("%Y-%m-%d"), "2025-02-01")

    def test_save_hicp_items_writes_parquet(self):
        frame = normalize_eurostat_item_index({"FR": index_dataset("FR")})

        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "eurostat_hicp_items.parquet"
            destination = save_hicp_items(frame, output)
            reloaded = pd.read_parquet(destination)

        self.assertEqual(destination, output)
        self.assertEqual(reloaded.columns.tolist(), CPI_SCHEMA_COLUMNS)
        self.assertEqual(len(reloaded), 2)


if __name__ == "__main__":
    unittest.main()
