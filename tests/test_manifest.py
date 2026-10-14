import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from inflation_engine.utils.manifest import update_ingestion_manifest
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS


def valid_frame(country_code="DK", country="Denmark"):
    row = {
        "date": pd.Timestamp("2026-05-01"),
        "country": country,
        "country_code": country_code,
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


class ManifestTests(unittest.TestCase):
    def test_manifest_records_provenance_and_checksum(self):
        frame = valid_frame()

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "dataset.parquet"
            manifest = root / "ingestion_manifest.json"
            frame.to_parquet(output, index=False)

            update_ingestion_manifest(
                "statbank_dk_cpi",
                frame,
                output,
                source_updated_at="2026-06-10T08:00:00+02:00",
                retrieved_at="2026-06-15T12:00:00Z",
                manifest_path=manifest,
            )
            document = json.loads(manifest.read_text())
            expected_hash = hashlib.sha256(output.read_bytes()).hexdigest()

        entry = document["datasets"]["statbank_dk_cpi"]
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(entry["retrieved_at"], "2026-06-15T12:00:00Z")
        self.assertEqual(entry["source_updated_at"], "2026-06-10T06:00:00Z")
        self.assertEqual(entry["row_count"], 1)
        self.assertEqual(entry["sha256"], expected_hash)
        self.assertEqual(entry["latest_observation_date"], "2026-05-01")

    def test_manifest_preserves_existing_dataset_entries(self):
        frame = valid_frame()

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "ingestion_manifest.json"
            first_output = root / "first.parquet"
            second_output = root / "second.parquet"
            frame.to_parquet(first_output, index=False)
            frame.to_parquet(second_output, index=False)

            update_ingestion_manifest(
                "first",
                frame,
                first_output,
                retrieved_at="2026-06-15T12:00:00Z",
                manifest_path=manifest,
            )
            update_ingestion_manifest(
                "second",
                frame,
                second_output,
                retrieved_at="2026-06-15T12:01:00Z",
                manifest_path=manifest,
            )
            document = json.loads(manifest.read_text())

        self.assertEqual(set(document["datasets"]), {"first", "second"})
        self.assertEqual(document["updated_at"], "2026-06-15T12:01:00Z")

    def test_manifest_requires_existing_output_file(self):
        frame = valid_frame()

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaises(FileNotFoundError):
                update_ingestion_manifest(
                    "missing",
                    frame,
                    root / "missing.parquet",
                    manifest_path=root / "manifest.json",
                )


if __name__ == "__main__":
    unittest.main()
