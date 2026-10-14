import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from unittest.mock import patch

from inflation_engine.features.inflation_metrics import (
    OUTPUT_COLUMNS,
    calculate_inflation_metrics,
    load_combined_panel,
    save_cpi_panel,
)
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS


def cpi_frame(
    values,
    *,
    dates=None,
    country_code="DK",
    country="Denmark",
    component="all_items",
):
    if dates is None:
        dates = pd.date_range("2024-01-01", periods=len(values), freq="MS")

    rows = []
    for date, value in zip(dates, values):
        rows.append(
            {
                "date": date,
                "country": country,
                "country_code": country_code,
                "region": "Nordics",
                "series_type": "headline",
                "series_role": (
                    "aggregate" if component == "all_items" else "component"
                ),
                "component": component,
                "value": value,
                "source": "Test source",
                "source_dataset": "test_dataset",
                "source_series_id": f"{country_code}_{component}",
                "frequency": "monthly",
                "unit": "index",
                "index_base": "2025=100",
                "seasonal_adjustment": "seasonally_adjusted",
                "observation_status": "normal",
            }
        )
    return pd.DataFrame(rows, columns=CPI_SCHEMA_COLUMNS)


class InflationMetricsTests(unittest.TestCase):
    def test_calculates_expected_rates_and_momentum(self):
        monthly_growth = 1.01
        values = 100.0 * monthly_growth ** np.arange(16)

        result = calculate_inflation_metrics(cpi_frame(values))
        latest = result.iloc[-1]

        expected_annual = (monthly_growth**12 - 1.0) * 100.0
        self.assertAlmostEqual(latest["inflation_mom"], 1.0)
        self.assertAlmostEqual(latest["inflation_yoy"], expected_annual)
        self.assertAlmostEqual(latest["inflation_yoy_change_3m"], 0.0)
        self.assertAlmostEqual(latest["inflation_3m_ann"], expected_annual)
        self.assertAlmostEqual(latest["inflation_3m3m_ann"], expected_annual)
        self.assertAlmostEqual(latest["inflation_6m_ann"], expected_annual)
        self.assertAlmostEqual(latest["inflation_6m6m_ann"], expected_annual)
        self.assertAlmostEqual(latest["inflation_3m3m_yoy_spread"], 0.0)
        self.assertEqual(latest["inflation_momentum_signal"], "stable")

    def test_true_6m6m_uses_six_month_averages(self):
        values = np.concatenate([np.full(6, 100.0), np.full(6, 106.0)])

        result = calculate_inflation_metrics(cpi_frame(values))
        latest = result.iloc[-1]

        self.assertAlmostEqual(latest["inflation_6m6m_ann"], 12.36)

    def test_missing_month_does_not_create_false_mom_or_rolling_metrics(self):
        dates = pd.to_datetime(
            [
                "2024-01-01",
                "2024-02-01",
                "2024-04-01",
                "2024-05-01",
                "2024-06-01",
            ]
        )
        result = calculate_inflation_metrics(
            cpi_frame([100.0, 101.0, 103.0, 104.0, 105.0], dates=dates)
        )

        april = result.loc[result["date"] == pd.Timestamp("2024-04-01")].iloc[0]
        june = result.loc[result["date"] == pd.Timestamp("2024-06-01")].iloc[0]
        self.assertTrue(pd.isna(april["inflation_mom"]))
        self.assertTrue(pd.isna(june["inflation_3m_ann"]))
        self.assertEqual(june["inflation_momentum_signal"], "not_available")

    def test_metrics_are_isolated_by_series(self):
        denmark = cpi_frame([100.0, 101.0], country_code="DK", country="Denmark")
        norway = cpi_frame(
            [200.0, 220.0],
            country_code="NO",
            country="Norway",
        )

        result = calculate_inflation_metrics(
            pd.concat([norway, denmark], ignore_index=True)
        )
        latest = result.groupby("country_code").tail(1).set_index("country_code")

        self.assertAlmostEqual(latest.loc["DK", "inflation_mom"], 1.0)
        self.assertAlmostEqual(latest.loc["NO", "inflation_mom"], 10.0)

    @patch(
        "inflation_engine.features.inflation_metrics.seasonal_adjust_x13"
    )
    def test_x13_adjusts_only_non_adjusted_source_series(self, adjust):
        dates = pd.date_range("2020-01-01", periods=60, freq="MS")
        denmark = cpi_frame(
            np.arange(100.0, 160.0),
            dates=dates,
        )
        denmark["seasonal_adjustment"] = "not_seasonally_adjusted"
        us = cpi_frame(
            np.arange(200.0, 260.0),
            dates=dates,
            country_code="US",
            country="United States",
        )
        us["region"] = "North America"
        adjust.return_value = pd.Series(
            np.arange(110.0, 170.0),
            index=dates,
        )

        result = calculate_inflation_metrics(
            pd.concat([denmark, us], ignore_index=True),
            seasonal_adjustment="x13",
            x13_path="/tmp/x13as",
        )

        denmark_result = result.loc[result["country_code"] == "DK"]
        us_result = result.loc[result["country_code"] == "US"]
        self.assertEqual(adjust.call_count, 1)
        self.assertTrue(
            denmark_result["analysis_seasonal_adjustment"]
            .eq("x13_arima_seats")
            .all()
        )
        self.assertTrue(
            us_result["analysis_seasonal_adjustment"]
            .eq("seasonally_adjusted")
            .all()
        )
        self.assertAlmostEqual(
            denmark_result.iloc[0]["analysis_value"],
            110.0,
        )
        self.assertAlmostEqual(us_result.iloc[0]["analysis_value"], 200.0)

    def test_load_and_save_processed_panel(self):
        metrics = calculate_inflation_metrics(cpi_frame([100.0, 101.0]))

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "combined.parquet"
            output = root / "processed" / "cpi_panel.parquet"
            cpi_frame([100.0, 101.0]).to_parquet(source, index=False)

            loaded = load_combined_panel(source)
            destination = save_cpi_panel(metrics, output)
            reloaded = pd.read_parquet(destination)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(destination, output)
        self.assertEqual(reloaded.columns.tolist(), OUTPUT_COLUMNS)


if __name__ == "__main__":
    unittest.main()
