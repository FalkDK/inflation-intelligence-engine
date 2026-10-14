import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from inflation_engine.features.seasonal_adjustment import (
    build_x13_seats_spec,
    resolve_x13_path,
    seasonal_adjust_x13,
)


def monthly_series(periods=60):
    return pd.Series(
        100.0 + np.arange(periods),
        index=pd.date_range("2020-01-01", periods=periods, freq="MS"),
        name="Test CPI",
    )


class SeasonalAdjustmentTests(unittest.TestCase):
    def test_build_spec_requests_seats_adjustment(self):
        specification = build_x13_seats_spec(monthly_series())

        self.assertIn("start = 2020.1", specification)
        self.assertIn("period = 12", specification)
        self.assertIn("transform { function = auto }", specification)
        self.assertIn("automdl { }", specification)
        self.assertIn("outlier { }", specification)
        self.assertIn("forecast { maxlead = 12 }", specification)
        self.assertIn("seats { save = s11 }", specification)

    def test_short_series_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 60"):
            build_x13_seats_spec(monthly_series(periods=59))

    def test_series_with_calendar_gap_is_rejected(self):
        values = monthly_series(periods=61)
        values = values.drop(values.index[12])

        with self.assertRaisesRegex(ValueError, "contiguous"):
            build_x13_seats_spec(values)

    def test_resolve_explicit_executable(self):
        with TemporaryDirectory() as temporary_directory:
            executable = Path(temporary_directory) / "x13as"
            executable.touch()
            executable.chmod(
                executable.stat().st_mode | stat.S_IXUSR
            )

            resolved = resolve_x13_path(executable)

        self.assertEqual(resolved, executable.resolve())

    def test_runs_x13_and_reads_seats_output(self):
        values = monthly_series()

        def create_output(command, **kwargs):
            base_path = Path(command[1])
            output = "date\tvalue\n----\t-----\n"
            output += "\n".join(
                f"{date.year}.{date.month}\t{value + 1.0}"
                for date, value in values.items()
            )
            base_path.with_suffix(".s11").write_text(
                output,
                encoding="ascii",
            )
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "", "stderr": ""},
            )()

        with TemporaryDirectory() as temporary_directory:
            executable = Path(temporary_directory) / "x13as"
            executable.touch()
            executable.chmod(
                executable.stat().st_mode | stat.S_IXUSR
            )
            adjusted = seasonal_adjust_x13(
                values,
                x13_path=executable,
                _runner=create_output,
            )

        np.testing.assert_allclose(
            adjusted.to_numpy(),
            values.to_numpy() + 1.0,
        )
        self.assertTrue(adjusted.index.equals(values.index))


if __name__ == "__main__":
    unittest.main()
