import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS
from run_pipeline import run_pipeline


def us_source_frame():
    rows = []
    dates = pd.date_range("2023-01-01", periods=36, freq="MS")
    series = [
        ("headline", "all_items", "CPIAUCSL", 300.0),
        (
            "core",
            "all_items_less_food_and_energy",
            "CPILFESL",
            310.0,
        ),
    ]
    for series_type, component, series_id, base_value in series:
        for number, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "country": "United States",
                    "country_code": "US",
                    "region": "North America",
                    "series_type": series_type,
                    "series_role": "aggregate",
                    "component": component,
                    "value": base_value + number,
                    "source": "FRED",
                    "source_dataset": "BLS CPI",
                    "source_series_id": series_id,
                    "frequency": "monthly",
                    "unit": "index",
                    "index_base": "1982-84=100",
                    "seasonal_adjustment": "seasonally_adjusted",
                    "observation_status": "normal",
                }
            )
    return pd.DataFrame(rows, columns=CPI_SCHEMA_COLUMNS)


class PipelineTests(unittest.TestCase):
    def test_pipeline_generates_all_outputs(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "raw.parquet"
            us_source_frame().to_parquet(source, index=False)

            outputs = run_pipeline(
                source_paths={"us": source},
                combined_output=root / "interim.parquet",
                panel_output=root / "processed.parquet",
                regime_features_output=root / "regime_features.parquet",
                yoy_chart_output=root / "charts" / "yoy.png",
                momentum_chart_output=root / "charts" / "momentum.png",
                report_output=root / "reports" / "report.md",
                chart_years=2,
            )

            self.assertEqual(
                set(outputs),
                {
                    "combined_panel",
                    "processed_panel",
                    "regime_features",
                    "yoy_chart",
                    "momentum_chart",
                    "report",
                },
            )
            for path in outputs.values():
                self.assertTrue(path.is_file())

    def test_pipeline_can_fetch_before_building_outputs(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "raw.parquet"

            def fetch_source() -> Path:
                us_source_frame().to_parquet(source, index=False)
                return source

            outputs = run_pipeline(
                source_paths={"us": source},
                fetch_raw=True,
                fetchers={"us": fetch_source},
                combined_output=root / "interim.parquet",
                panel_output=root / "processed.parquet",
                regime_features_output=root / "regime_features.parquet",
                yoy_chart_output=root / "charts" / "yoy.png",
                momentum_chart_output=root / "charts" / "momentum.png",
                report_output=root / "reports" / "report.md",
                chart_years=2,
            )

            self.assertEqual(outputs["raw_us"], source)
            self.assertTrue(outputs["combined_panel"].is_file())
            self.assertTrue(outputs["processed_panel"].is_file())


if __name__ == "__main__":
    unittest.main()
