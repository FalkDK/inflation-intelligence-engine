import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import requests

from inflation_engine.data.fetch_eurostat import (
    ECB_HICP_API_URL,
    ECB_HICP_SERIES_KEY,
    HICP_GEOGRAPHIES,
    fetch_ecb_hicp_csv,
    fetch_hicp,
    latest_hicp_observations,
    normalize_ecb_hicp,
    save_hicp,
)
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS


CSV_DATA = """\
REF_AREA,ADJUSTMENT,ICP_ITEM,DATA_PROVIDER,ICP_SUFFIX,TIME_PERIOD,OBS_VALUE,OBS_STATUS,UNIT_INDEX_BASE
DE,N,000000,4D0,INX,2026-01,100.5,A,2025 = 100
DE,N,000000,4D0,INX,2026-02,101.0,E,2025 = 100
DE,N,XEF000,4D0,INX,2026-01,100.2,A,2025 = 100
DE,N,XEF000,4D0,INX,2026-02,100.8,E,2025 = 100
U2,N,000000,4D0,INX,2026-01,100.4,A,2025 = 100
U2,N,000000,4D0,INX,2026-02,100.9,E,2025 = 100
U2,N,XEF000,4D0,INX,2026-01,100.1,A,2025 = 100
U2,N,XEF000,4D0,INX,2026-02,100.7,E,2025 = 100
"""


class FakeResponse:
    text = CSV_DATA

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, *, params, headers, timeout):
        self.calls.append((url, params, headers, timeout))
        return FakeResponse()


class FailingSession:
    def get(self, url, *, params, headers, timeout):
        raise requests.ConnectionError("network unavailable")


class FetchEurostatTests(unittest.TestCase):
    def test_fetch_csv_requests_replacement_hicp_flow(self):
        session = FakeSession()

        csv_data = fetch_ecb_hicp_csv(session=session)

        self.assertEqual(csv_data, CSV_DATA)
        self.assertEqual(len(session.calls), 1)
        url, params, headers, _ = session.calls[0]
        self.assertEqual(url, f"{ECB_HICP_API_URL}/{ECB_HICP_SERIES_KEY}")
        self.assertEqual(params, {"format": "csvdata"})
        self.assertEqual(headers, {"Accept": "text/csv"})
        self.assertIn("+".join(HICP_GEOGRAPHIES), ECB_HICP_SERIES_KEY)
        self.assertIn("000000+XEF000", ECB_HICP_SERIES_KEY)
        self.assertIn("FOOD00+NRGY00+IGXE00+SERV00+041000", ECB_HICP_SERIES_KEY)
        self.assertIn("4D0.INX", ECB_HICP_SERIES_KEY)

    def test_normalize_ecb_hicp_uses_standard_schema(self):
        frame = normalize_ecb_hicp(CSV_DATA)

        self.assertEqual(frame.columns.tolist(), CPI_SCHEMA_COLUMNS)
        self.assertEqual(len(frame), 8)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["date"]))
        self.assertEqual(set(frame["country_code"]), {"DE", "EA20"})
        self.assertEqual(set(frame["series_type"]), {"headline", "core"})
        self.assertEqual(frame["source"].unique().tolist(), ["ECB Data Portal"])
        self.assertEqual(set(frame["observation_status"]), {"normal", "estimated"})
        self.assertEqual(
            frame["seasonal_adjustment"].unique().tolist(),
            ["not_seasonally_adjusted"],
        )

    def test_fetch_hicp_uses_csv_and_normalization(self):
        frame = fetch_hicp(session=FakeSession())

        self.assertEqual(len(frame), 8)
        self.assertEqual(
            set(frame["component"]),
            {
                "all_items",
                "all_items_less_food_energy_alcohol_tobacco",
            },
        )

    def test_latest_hicp_observations_returns_each_latest_series(self):
        frame = normalize_ecb_hicp(CSV_DATA)

        latest = latest_hicp_observations(frame)

        self.assertEqual(len(latest), 4)
        self.assertTrue((latest["date"] == pd.Timestamp("2026-02-01")).all())
        self.assertEqual(
            latest.set_index(["country_code", "series_type"])["value"].to_dict(),
            {
                ("DE", "core"): 100.8,
                ("DE", "headline"): 101.0,
                ("EA20", "core"): 100.7,
                ("EA20", "headline"): 100.9,
            },
        )

    def test_fetch_error_is_concise(self):
        with self.assertRaisesRegex(RuntimeError, "ECB HICP request failed"):
            fetch_ecb_hicp_csv(session=FailingSession())

    def test_unexpected_index_base_is_rejected(self):
        invalid_csv = CSV_DATA.replace("2025 = 100", "2015 = 100", 1)

        with self.assertRaisesRegex(ValueError, "Unexpected ECB HICP index bases"):
            normalize_ecb_hicp(invalid_csv)

    def test_save_hicp_writes_parquet(self):
        frame = normalize_ecb_hicp(CSV_DATA)

        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "eurostat_hicp.parquet"
            destination = save_hicp(frame, output)
            reloaded = pd.read_parquet(destination)

        self.assertEqual(destination, output)
        self.assertEqual(reloaded.columns.tolist(), CPI_SCHEMA_COLUMNS)
        self.assertEqual(len(reloaded), 8)


if __name__ == "__main__":
    unittest.main()
