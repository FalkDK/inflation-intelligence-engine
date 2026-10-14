import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from inflation_engine.data.combine_sources import (
    combine_cpi_sources,
    load_cpi_source,
    save_combined_panel,
)
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS


def source_frame(country_code, country, component, source_series_id, value):
    row = {
        "date": pd.Timestamp("2026-05-01"),
        "country": country,
        "country_code": country_code,
        "region": "Nordics",
        "series_type": "headline",
        "series_role": "aggregate",
        "component": component,
        "value": value,
        "source": f"Source {country_code}",
        "source_dataset": f"dataset_{country_code}",
        "source_series_id": source_series_id,
        "frequency": "monthly",
        "unit": "index",
        "index_base": "2025=100",
        "seasonal_adjustment": "not_seasonally_adjusted",
        "observation_status": "not_provided",
    }
    return pd.DataFrame([row], columns=CPI_SCHEMA_COLUMNS)


class CombineSourcesTests(unittest.TestCase):
    def test_load_source_validates_and_preserves_schema(self):
        frame = source_frame("DK", "Denmark", "all_items", "000000", 101.3)

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "denmark.parquet"
            frame.to_parquet(path, index=False)
            loaded = load_cpi_source(path)

        self.assertEqual(loaded.columns.tolist(), CPI_SCHEMA_COLUMNS)
        self.assertEqual(len(loaded), 1)

    def test_load_source_migrates_legacy_file_without_series_role(self):
        frame = source_frame("DK", "Denmark", "all_items", "000000", 101.3)

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "legacy.parquet"
            frame.drop(columns="series_role").to_parquet(path, index=False)
            loaded = load_cpi_source(path)

        self.assertEqual(loaded["series_role"].tolist(), ["aggregate"])

    def test_combine_sources_returns_sorted_valid_panel(self):
        denmark = source_frame("DK", "Denmark", "all_items", "000000", 101.3)
        norway = source_frame("NO", "Norway", "all_items", "KpiIndMnd", 103.0)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            denmark_path = root / "denmark.parquet"
            norway_path = root / "norway.parquet"
            denmark.to_parquet(denmark_path, index=False)
            norway.to_parquet(norway_path, index=False)

            combined = combine_cpi_sources(
                {
                    "norway": norway_path,
                    "denmark": denmark_path,
                }
            )

        self.assertEqual(combined.columns.tolist(), CPI_SCHEMA_COLUMNS)
        self.assertEqual(combined["country_code"].tolist(), ["DK", "NO"])
        self.assertEqual(len(combined), 2)

    def test_combine_sources_rejects_cross_source_duplicates(self):
        frame = source_frame("DK", "Denmark", "all_items", "000000", 101.3)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.parquet"
            second = root / "second.parquet"
            frame.to_parquet(first, index=False)
            frame.to_parquet(second, index=False)

            with self.assertRaisesRegex(ValueError, "duplicate observations"):
                combine_cpi_sources({"first": first, "second": second})

    def test_combine_sources_reports_missing_dataset(self):
        with self.assertRaisesRegex(FileNotFoundError, "missing_source"):
            combine_cpi_sources({"missing_source": "does-not-exist.parquet"})

    def test_save_combined_panel_writes_parquet(self):
        frame = source_frame("DK", "Denmark", "all_items", "000000", 101.3)

        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "combined.parquet"
            destination = save_combined_panel(frame, output)
            reloaded = pd.read_parquet(destination)

        self.assertEqual(destination, output)
        self.assertEqual(reloaded.columns.tolist(), CPI_SCHEMA_COLUMNS)
        self.assertEqual(len(reloaded), 1)


if __name__ == "__main__":
    unittest.main()
