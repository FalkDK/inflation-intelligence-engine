import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from inflation_engine.reports.charts import (
    generate_charts,
    plot_yoy_by_country,
    select_primary_series,
)
from inflation_engine.reports.generate_report import (
    build_report,
    generate_report,
)
from inflation_engine.utils.config import COUNTRIES, primary_series_keys


COUNTRY_NAMES = {code: metadata["name"] for code, metadata in COUNTRIES.items()}


def report_frame():
    dates = pd.date_range("2024-01-01", periods=24, freq="MS")
    rows = []
    for series_number, (country_code, series_type, component) in enumerate(
        primary_series_keys()
    ):
        for date_number, date in enumerate(dates):
            if country_code == "GB" and date == dates[-1]:
                continue
            yoy = 1.0 + series_number / 10 + date_number / 100
            momentum = yoy + (0.8 if series_type == "headline" else -0.8)
            rows.append(
                {
                    "date": date,
                    "country": COUNTRY_NAMES[country_code],
                    "country_code": country_code,
                    "series_type": series_type,
                    "series_role": "aggregate",
                    "component": component,
                    "analysis_seasonal_adjustment": (
                        "seasonally_adjusted"
                        if country_code == "US"
                        else "not_seasonally_adjusted"
                    ),
                    "inflation_yoy": yoy,
                    "inflation_yoy_change_3m": (
                        0.8 if series_type == "headline" else -0.8
                    ),
                    "inflation_3m3m_ann": momentum,
                    "inflation_3m3m_yoy_spread": momentum - yoy,
                    "inflation_momentum_signal": (
                        "accelerating"
                        if series_type == "headline"
                        else "slowing"
                    ),
                }
            )
    return pd.DataFrame(rows)


class ReportTests(unittest.TestCase):
    def test_primary_series_excludes_other_components(self):
        frame = report_frame()
        extra = frame.iloc[[0]].copy()
        extra["component"] = "shelter"
        extra["series_role"] = "component"

        selected = select_primary_series(
            pd.concat([frame, extra], ignore_index=True)
        )

        self.assertEqual(len(selected), len(frame))
        self.assertNotIn("shelter", selected["component"].tolist())

    def test_generate_charts_writes_png_files(self):
        frame = report_frame()

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            yoy_path, momentum_path = generate_charts(
                frame,
                yoy_path=root / "yoy.png",
                momentum_path=root / "momentum.png",
                years=1,
            )

            for path in (yoy_path, momentum_path):
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 10_000)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_chart_labels_latest_values(self):
        frame = report_frame()

        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "yoy.png"
            destination = plot_yoy_by_country(frame, output, years=1)

        self.assertEqual(destination, output)

    def test_report_contains_required_sections_and_units(self):
        report = build_report(report_frame())

        self.assertIn("# Inflation Intelligence Report", report)
        self.assertIn("## Latest Observations", report)
        self.assertIn("## Country Ranking by YoY Inflation", report)
        self.assertIn(
            "## Country Ranking by Three-Month Change in YoY Inflation",
            report,
        )
        self.assertIn("## Momentum Signal", report)
        self.assertIn("## Data Coverage", report)
        self.assertIn("## Methodology Notes", report)
        self.assertIn("0.8 pp", report)
        self.assertIn("Release months vary by country", report)
        self.assertIn("| United Kingdom | 2025-11 |", report)
        self.assertIn("| Observation |", report)

    def test_generate_report_writes_markdown(self):
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "report.md"
            destination = generate_report(report_frame(), output)

            self.assertEqual(destination, output)
            self.assertTrue(output.read_text().endswith("\n"))


if __name__ == "__main__":
    unittest.main()
