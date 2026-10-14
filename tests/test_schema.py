import unittest

import pandas as pd

from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS, validate_cpi_schema


def valid_frame():
    row = {
        "date": pd.Timestamp("2026-05-01"),
        "country": "Denmark",
        "country_code": "DK",
        "region": "Nordics",
        "series_type": "headline",
        "series_role": "aggregate",
        "component": "all_items",
        "value": 101.32,
        "source": "Statistics Denmark",
        "source_dataset": "PRIS01",
        "source_series_id": "000000",
        "frequency": "monthly",
        "unit": "index",
        "index_base": "2025=100",
        "seasonal_adjustment": "not_seasonally_adjusted",
        "observation_status": "not_provided",
    }
    return pd.DataFrame([row], columns=CPI_SCHEMA_COLUMNS)


class SchemaTests(unittest.TestCase):
    def test_valid_frame_passes(self):
        validate_cpi_schema(valid_frame())

    def test_duplicate_observation_is_rejected(self):
        frame = pd.concat([valid_frame(), valid_frame()], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "duplicate observations"):
            validate_cpi_schema(frame)

    def test_invalid_observation_status_is_rejected(self):
        frame = valid_frame()
        frame["observation_status"] = "maybe"

        with self.assertRaisesRegex(ValueError, "Invalid observation_status"):
            validate_cpi_schema(frame)

    def test_series_role_must_match_primary_series_registry(self):
        frame = valid_frame()
        frame["series_role"] = "component"

        with self.assertRaisesRegex(ValueError, "series roles"):
            validate_cpi_schema(frame)

    def test_non_month_start_date_is_rejected(self):
        frame = valid_frame()
        frame["date"] = pd.Timestamp("2026-05-15")

        with self.assertRaisesRegex(ValueError, "first day of the month"):
            validate_cpi_schema(frame)


if __name__ == "__main__":
    unittest.main()
