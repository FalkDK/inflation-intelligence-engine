import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import requests

from inflation_engine.data.fetch_eurostat_item_weights import (
    EUROSTAT_HICP_ITEM_WEIGHT_URL,
    fetch_eurostat_item_weight_dataset,
    normalize_eurostat_item_weights,
    save_eurostat_item_weights,
)


def weight_dataset(geo="FR", *, root_weight=1000.0):
    """coicop=[CP00, CP01, FOOD], geo=[geo], time=[2025]. FOOD is a
    non-hierarchical special aggregate and must be filtered out."""
    return {
        "id": ["coicop", "geo", "time"],
        "size": [3, 1, 1],
        "dimension": {
            "coicop": {
                "category": {
                    "index": {"CP00": 0, "CP01": 1, "FOOD": 2},
                    "label": {
                        "CP00": "All-items HICP",
                        "CP01": "Food and non-alcoholic beverages",
                        "FOOD": "Food including alcohol and tobacco",
                    },
                }
            },
            "geo": {"category": {"index": {geo: 0}}},
            "time": {"category": {"index": {"2025": 0}}},
        },
        "value": [root_weight, 150.0, 180.0],
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


class FetchEurostatItemWeightsTests(unittest.TestCase):
    def test_fetch_dataset_requests_expected_params(self):
        session = FakeSession(weight_dataset())

        dataset = fetch_eurostat_item_weight_dataset("FR", session=session)

        self.assertEqual(dataset, weight_dataset())
        url, params, headers, _ = session.calls[0]
        self.assertEqual(url, EUROSTAT_HICP_ITEM_WEIGHT_URL)
        self.assertEqual(
            params, [("lang", "en"), ("geo", "FR"), ("sinceTimePeriod", "2015")]
        )
        self.assertEqual(headers, {"Accept": "application/json"})

    def test_fetch_error_is_concise(self):
        with self.assertRaisesRegex(
            RuntimeError, "Eurostat HICP item weight request failed"
        ):
            fetch_eurostat_item_weight_dataset("FR", session=FailingSession())

    def test_normalize_keeps_only_the_true_coicop_hierarchy(self):
        frame = normalize_eurostat_item_weights({"FR": weight_dataset("FR")})

        self.assertEqual(set(frame["coicop_code"]), {"CP00", "CP01"})
        self.assertEqual(
            frame.set_index("coicop_code")["weight_pct"].to_dict(),
            {"CP00": 100.0, "CP01": 15.0},
        )
        self.assertEqual(
            frame.set_index("coicop_code")["label"].to_dict(),
            {"CP00": "All-items HICP", "CP01": "Food and non-alcoholic beverages"},
        )
        self.assertEqual(frame["weight_basis_year"].unique().tolist(), [2025])
        self.assertEqual(
            frame["weight_reference_date"].unique().tolist(),
            [pd.Timestamp("2025-01-01")],
        )

    def test_normalize_merges_multiple_geographies(self):
        frame = normalize_eurostat_item_weights(
            {"FR": weight_dataset("FR"), "DE": weight_dataset("DE")}
        )

        self.assertEqual(set(frame["country_code"]), {"FR", "DE"})
        self.assertEqual(len(frame), 4)

    def test_normalize_rejects_a_root_weight_far_from_100_percent(self):
        with self.assertRaisesRegex(ValueError, "CP00.*not ~100%"):
            normalize_eurostat_item_weights(
                {"FR": weight_dataset("FR", root_weight=800.0)}
            )

    def test_save_eurostat_item_weights_writes_parquet(self):
        frame = normalize_eurostat_item_weights({"FR": weight_dataset("FR")})

        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "eurostat_hicp_item_weights.parquet"
            destination = save_eurostat_item_weights(frame, output)
            reloaded = pd.read_parquet(destination)

        self.assertEqual(destination, output)
        self.assertEqual(len(reloaded), 2)


if __name__ == "__main__":
    unittest.main()
