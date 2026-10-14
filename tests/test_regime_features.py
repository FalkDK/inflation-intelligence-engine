import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from inflation_engine.features.inflation_metrics import (
    calculate_inflation_metrics,
)
from inflation_engine.features.regime_features import (
    REGIME_OUTPUT_COLUMNS,
    build_regime_features,
    save_regime_features,
)
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS


COUNTRY_METADATA = {
    "DK": ("Denmark", "Nordics"),
    "US": ("United States", "North America"),
}


def processed_frame(
    *,
    country_code="DK",
    seasonal_adjustment="not_seasonally_adjusted",
    periods=30,
):
    country, region = COUNTRY_METADATA[country_code]
    dates = pd.date_range("2020-01-01", periods=periods, freq="MS")
    rows = []
    for series_type, component, monthly_growth in (
        ("headline", "all_items", 1.002),
        (
            "core",
            "all_items_less_food_and_energy"
            if country_code == "US"
            else "all_items_less_energy_unprocessed_food",
            1.0015,
        ),
    ):
        for number, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "country": country,
                    "country_code": country_code,
                    "region": region,
                    "series_type": series_type,
                    "series_role": "aggregate",
                    "component": component,
                    "value": 100.0 * monthly_growth**number,
                    "source": "test",
                    "source_dataset": "test",
                    "source_series_id": f"{country_code}_{series_type}",
                    "frequency": "monthly",
                    "unit": "index",
                    "index_base": "2020=100",
                    "seasonal_adjustment": seasonal_adjustment,
                    "observation_status": "normal",
                }
            )
    return calculate_inflation_metrics(
        pd.DataFrame(rows, columns=CPI_SCHEMA_COLUMNS)
    )


class RegimeFeatureTests(unittest.TestCase):
    def test_builds_one_row_per_country_month(self):
        frame = pd.concat(
            [
                processed_frame(),
                processed_frame(
                    country_code="US",
                    seasonal_adjustment="seasonally_adjusted",
                ),
            ],
            ignore_index=True,
        )

        result = build_regime_features(frame)

        self.assertEqual(result.columns.tolist(), REGIME_OUTPUT_COLUMNS)
        self.assertEqual(len(result), 60)
        self.assertFalse(
            result.duplicated(["observation_date", "country_code"]).any()
        )
        latest = result.groupby("country_code").tail(1)
        self.assertTrue(latest["feature_complete"].all())

    def test_masks_unadjusted_short_horizon_rates(self):
        result = build_regime_features(processed_frame())

        self.assertTrue(result["headline_3m3m_ann_sa"].isna().all())
        self.assertTrue(result["core_3m3m_ann_sa"].isna().all())
        self.assertFalse(result["short_horizon_usable"].any())
        self.assertTrue(result["inflation_direction_3m"].notna().any())

    def test_keeps_adjusted_short_horizon_rates(self):
        result = build_regime_features(
            processed_frame(
                country_code="US",
                seasonal_adjustment="seasonally_adjusted",
            )
        )
        latest = result.iloc[-1]

        self.assertTrue(latest["short_horizon_usable"])
        self.assertTrue(np.isfinite(latest["headline_3m3m_ann_sa"]))
        self.assertTrue(np.isfinite(latest["core_3m3m_ann_sa"]))

    def test_calendar_change_does_not_bridge_missing_month(self):
        frame = processed_frame()
        frame = frame.loc[frame["date"] != pd.Timestamp("2021-01-01")]

        result = build_regime_features(frame)
        april = result.loc[
            result["observation_date"] == pd.Timestamp("2021-04-01")
        ].iloc[0]

        self.assertTrue(pd.isna(april["headline_yoy_change_3m"]))
        self.assertFalse(april["feature_complete"])

    def test_future_observations_do_not_change_past_features(self):
        frame = processed_frame()
        cutoff = pd.Timestamp("2021-12-01")

        full = build_regime_features(frame)
        truncated = build_regime_features(frame.loc[frame["date"] <= cutoff])
        expected = full.loc[
            full["observation_date"] <= cutoff
        ].reset_index(drop=True)

        pd.testing.assert_frame_equal(truncated, expected)

    def test_save_writes_the_feature_contract(self):
        features = build_regime_features(processed_frame())
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "features.parquet"
            destination = save_regime_features(features, output)
            reloaded = pd.read_parquet(destination)

        self.assertEqual(destination, output)
        self.assertEqual(reloaded.columns.tolist(), REGIME_OUTPUT_COLUMNS)


if __name__ == "__main__":
    unittest.main()
