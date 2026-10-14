import unittest

import numpy as np
import pandas as pd

from inflation_engine.dashboard.data import (
    DEFAULT_HEATMAP_METRICS,
    available_countries,
    country_metric_history,
    country_ranking,
    heatmap_values,
    latest_primary_observations,
)
from inflation_engine.utils.config import COUNTRIES, primary_series_keys


def dashboard_frame():
    dates = pd.date_range("2025-01-01", periods=4, freq="MS")
    rows = []
    country_codes = ["DK", "NO", "US"]
    primary_by_country = {
        country_code: {
            series_type: component
            for key_country_code, series_type, component in primary_series_keys()
            if key_country_code == country_code
        }
        for country_code in country_codes
    }
    for country_number, country_code in enumerate(country_codes):
        country = COUNTRIES[country_code]
        for series_type, component in primary_by_country[country_code].items():
            for date_number, date in enumerate(dates):
                base = country_number + date_number / 10
                if series_type == "core":
                    base += 0.25
                rows.append(
                    {
                        "date": date,
                        "country": country["name"],
                        "country_code": country_code,
                        "region": country["region"],
                        "series_type": series_type,
                        "series_role": "aggregate",
                        "component": component,
                        "value": 100.0 + base,
                        "analysis_seasonal_adjustment": "seasonally_adjusted",
                        "inflation_mom": base + 0.1,
                        "inflation_yoy": base + 1.0,
                        "inflation_yoy_change_3m": base + 0.2,
                        "inflation_3m_ann": base + 2.0,
                        "inflation_3m3m_ann": base + 3.0,
                        "inflation_6m_ann": base + 4.0,
                        "inflation_6m6m_ann": base + 5.0,
                        "inflation_3m3m_yoy_spread": base + 0.5,
                        "inflation_momentum_signal": "stable",
                    }
                )
    frame = pd.DataFrame(rows)
    extra = frame.iloc[[0]].copy()
    extra["component"] = "other_component"
    extra["series_role"] = "component"
    extra["inflation_yoy"] = 99.0
    return pd.concat([frame, extra], ignore_index=True)


class DashboardDataTests(unittest.TestCase):
    def test_available_countries_are_sorted_and_primary_only(self):
        countries = available_countries(dashboard_frame())

        self.assertEqual(
            countries["country"].tolist(),
            ["Denmark", "Norway", "United States"],
        )
        self.assertEqual(countries["country_code"].tolist(), ["DK", "NO", "US"])

    def test_latest_primary_observations_returns_one_row_per_country_series(self):
        latest = latest_primary_observations(dashboard_frame())

        self.assertEqual(len(latest), 6)
        self.assertTrue(latest["date"].eq(pd.Timestamp("2025-04-01")).all())
        self.assertNotIn("other_component", latest["component"].tolist())

    def test_country_history_melts_selected_metrics(self):
        history = country_metric_history(
            dashboard_frame(),
            "DK",
            metrics=["inflation_yoy", "inflation_3m3m_ann"],
            series_types=["headline"],
            years=1,
        )

        self.assertEqual(
            set(history["metric"]),
            {"inflation_yoy", "inflation_3m3m_ann"},
        )
        self.assertEqual(history["series_type"].unique().tolist(), ["headline"])
        self.assertEqual(len(history), 8)

    def test_country_ranking_sorts_latest_values_descending(self):
        ranking = country_ranking(
            dashboard_frame(),
            metric="inflation_yoy",
            series_type="headline",
        )

        self.assertEqual(ranking.iloc[0]["country_code"], "US")
        self.assertEqual(ranking["rank"].tolist(), [1, 2, 3])
        self.assertTrue(np.isclose(ranking.iloc[0]["inflation_yoy"], 3.3))

    def test_heatmap_values_use_latest_metric_grid(self):
        heatmap = heatmap_values(
            dashboard_frame(),
            metrics=DEFAULT_HEATMAP_METRICS,
            series_type="headline",
        )

        self.assertEqual(len(heatmap), 3 * len(DEFAULT_HEATMAP_METRICS))
        self.assertEqual(
            set(heatmap["metric"]),
            set(DEFAULT_HEATMAP_METRICS),
        )
        self.assertTrue(heatmap["date"].eq(pd.Timestamp("2025-04-01")).all())


if __name__ == "__main__":
    unittest.main()
