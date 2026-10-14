import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from inflation_engine.utils.io import atomic_write_parquet


class AtomicParquetTests(unittest.TestCase):
    def test_writes_readable_parquet(self):
        frame = pd.DataFrame({"value": [1.0, 2.0]})

        with TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "data.parquet"
            result = atomic_write_parquet(frame, destination)
            reloaded = pd.read_parquet(destination)

        self.assertEqual(result, destination)
        pd.testing.assert_frame_equal(reloaded, frame)

    def test_failed_write_preserves_existing_destination(self):
        frame = pd.DataFrame({"value": [1.0]})

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            destination = root / "data.parquet"
            destination.write_bytes(b"existing")

            with patch.object(
                pd.DataFrame,
                "to_parquet",
                side_effect=RuntimeError("write failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "write failed"):
                    atomic_write_parquet(frame, destination)

            self.assertEqual(destination.read_bytes(), b"existing")
            self.assertEqual(list(root.iterdir()), [destination])


if __name__ == "__main__":
    unittest.main()
