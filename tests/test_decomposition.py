import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from inflation_engine.data.fetch_bls_weights import (
    WEIGHT_COLUMNS,
)
from inflation_engine.features.decomposition import (
    BUCKET_OUTPUT_COLUMNS,
    SUMMARY_OUTPUT_COLUMNS,
    build_bucket_decomposition,
    build_decomposition_summary,
    build_hicp_bucket_decomposition,
    build_norway_bucket_decomposition,
    save_decomposition,
)
from inflation_engine.data.fetch_ssb_weights import CONTRACT_COLUMNS


def weight_frame():
    weights = {
        "food": 14.0,
        "energy": 6.0,
        "shelter": 35.0,
        "core_goods": 20.0,
        "core_services_ex_housing": 25.0,
    }
    return pd.DataFrame(
        [
            {
                "reference_date": pd.Timestamp("2020-12-01"),
                "effective_start": pd.Timestamp("2021-01-01"),
                "effective_end": pd.Timestamp("2021-12-01"),
                "country_code": "US",
                "index_family": "CPI-U",
                "bucket": bucket,
                "weight_pct": weight,
                "source": "test",
                "source_url": "https://example.test",
                "weight_basis_year": 2019,
                "estimation_method": "test",
            }
            for bucket, weight in weights.items()
        ],
        columns=WEIGHT_COLUMNS,
    )


def cpi_frame():
    components = {
        "all_items": 3.0,
        "food": 4.0,
        "energy": 5.0,
        "shelter": 3.5,
        "commodities_less_food_and_energy_commodities": 1.0,
    }
    rows = []
    for month, date in enumerate(
        pd.date_range("2021-01-01", periods=12, freq="MS")
    ):
        for component, starting_rate in components.items():
            rows.append(
                {
                    "date": date,
                    "country": "United States",
                    "country_code": "US",
                    "series_role": (
                        "aggregate" if component == "all_items" else "component"
                    ),
                    "component": component,
                    "inflation_yoy": starting_rate + month * 0.1,
                }
            )
    return pd.DataFrame(rows)


def norway_inputs():
    weights = weight_frame().assign(country_code="NO", index_family="CPI")
    codes = {
        "food": "01",
        "energy": "04.5",
        "shelter": "04.1",
        "core_goods": "03",
        "core_services_ex_housing": "10",
    }
    contract_rows = []
    for row in weights.itertuples(index=False):
        contract_rows.append(
            {
                "country_code": "NO",
                "index_family": "CPI",
                "source_dataset": "14700",
                "source_component_code": codes[row.bucket],
                "source_component": row.bucket,
                "rate_source_component_code": codes[row.bucket],
                "classification_system": "test",
                "hierarchy_level": 1,
                "canonical_macro_bucket": row.bucket,
                "durability_class": "test",
                "effective_start": row.effective_start,
                "effective_end": row.effective_end,
                "annual_effective_weight_pct": row.weight_pct,
                "weight_reference_date": row.reference_date,
                "weight_basis_year": row.weight_basis_year,
                "weight_source": "test",
                "mapping_method": "test",
            }
        )
    contract = pd.DataFrame(contract_rows, columns=CONTRACT_COLUMNS)
    rows = []
    for month, date in enumerate(pd.date_range("2021-01-01", periods=12, freq="MS")):
        rows.append(
            {
                "date": date,
                "country": "Norway",
                "country_code": "NO",
                "series_role": "aggregate",
                "component": "all_items",
                "source_series_id": "KpiIndMnd",
                "inflation_yoy": 3.0 + month * 0.1,
            }
        )
        for number, code in enumerate(codes.values(), start=1):
            rows.append(
                {
                    "date": date,
                    "country": "Norway",
                    "country_code": "NO",
                    "series_role": "component",
                    "component": f"component_{code}",
                    "source_series_id": f"14700:{code}",
                    "inflation_yoy": 1.0 + number + month * 0.1,
                }
            )
    return pd.DataFrame(rows), weights, contract


def hicp_inputs():
    country_codes = {"DE": "Germany", "DK": "Denmark"}
    components = {
        "food_including_alcohol_tobacco": 4.0,
        "energy": 5.0,
        "actual_rentals_for_housing": 3.5,
        "non_energy_industrial_goods": 1.0,
        "services": 3.2,
    }
    rows = []
    weights = []
    for country_code, country in country_codes.items():
        for month, date in enumerate(
            pd.date_range("2021-01-01", periods=12, freq="MS")
        ):
            rows.append(
                {
                    "date": date,
                    "country": country,
                    "country_code": country_code,
                    "series_role": (
                        "component" if country_code == "DK" else "aggregate"
                    ),
                    "component": (
                        "hicp_all_items" if country_code == "DK" else "all_items"
                    ),
                    "inflation_yoy": 3.0 + month * 0.1,
                }
            )
            for component, rate in components.items():
                rows.append(
                    {
                        "date": date,
                        "country": country,
                        "country_code": country_code,
                        "series_role": "component",
                        "component": component,
                        "inflation_yoy": rate + month * 0.1,
                    }
                )
        for row in weight_frame().itertuples(index=False):
            weights.append(
                {
                    **row._asdict(),
                    "country_code": country_code,
                    "index_family": "HICP",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(weights, columns=WEIGHT_COLUMNS)


class DecompositionTests(unittest.TestCase):
    def test_bucket_contributions_reconcile_to_headline(self):
        buckets = build_bucket_decomposition(cpi_frame(), weight_frame())

        sums = buckets.groupby("date")["estimated_contribution_yoy_pp"].sum()
        expected = (
            cpi_frame()
            .loc[lambda frame: frame["component"] == "all_items"]
            .set_index("date")["inflation_yoy"]
        )
        pd.testing.assert_series_equal(
            sums,
            expected,
            check_names=False,
        )
        self.assertEqual(buckets.columns.tolist(), BUCKET_OUTPUT_COLUMNS)

    def test_residual_bucket_is_derived_from_headline(self):
        buckets = build_bucket_decomposition(cpi_frame(), weight_frame())
        january = buckets.loc[buckets["date"] == pd.Timestamp("2021-01-01")]
        residual = january.set_index("bucket").loc["core_services_ex_housing"]

        direct_contribution = 0.14 * 4.0 + 0.06 * 5.0 + 0.35 * 3.5 + 0.20
        expected_contribution = 3.0 - direct_contribution
        self.assertAlmostEqual(
            residual["estimated_contribution_yoy_pp"],
            expected_contribution,
        )
        self.assertAlmostEqual(
            residual["inflation_yoy"],
            expected_contribution / 0.25,
        )
        self.assertEqual(
            residual["contribution_method"],
            "headline_accounting_residual",
        )

    def test_summary_contains_requested_regime_features(self):
        buckets = build_bucket_decomposition(cpi_frame(), weight_frame())
        summary = build_decomposition_summary(buckets)
        latest = summary.iloc[-1]

        self.assertEqual(summary.columns.tolist(), SUMMARY_OUTPUT_COLUMNS)
        self.assertTrue(latest["feature_complete"])
        self.assertTrue(np.isfinite(latest["services_goods_pressure"]))
        self.assertTrue(np.isfinite(latest["cyclical_supply_pressure"]))
        self.assertGreaterEqual(latest["weighted_share_above_2pct"], 0.0)
        self.assertLessEqual(latest["weighted_share_above_2pct"], 100.0)
        self.assertGreaterEqual(latest["contribution_concentration_hhi"], 0.2)
        self.assertLessEqual(latest["contribution_concentration_hhi"], 1.0)

    def test_calendar_momentum_requires_three_months(self):
        buckets = build_bucket_decomposition(cpi_frame(), weight_frame())
        january = buckets.loc[buckets["date"] == pd.Timestamp("2021-01-01")]
        april = buckets.loc[buckets["date"] == pd.Timestamp("2021-04-01")]

        self.assertTrue(january["inflation_yoy_change_3m"].isna().all())
        self.assertTrue(april["inflation_yoy_change_3m"].notna().all())

    def test_save_writes_both_outputs(self):
        buckets = build_bucket_decomposition(cpi_frame(), weight_frame())
        summary = build_decomposition_summary(buckets)
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bucket_path, summary_path = save_decomposition(
                buckets,
                summary,
                bucket_output=root / "buckets.parquet",
                summary_output=root / "summary.parquet",
            )

        self.assertEqual(bucket_path.name, "buckets.parquet")
        self.assertEqual(summary_path.name, "summary.parquet")

    def test_norway_frontier_reconciles_without_parent_child_double_counting(self):
        cpi, weights, contract = norway_inputs()
        buckets = build_norway_bucket_decomposition(cpi, weights, contract)

        self.assertEqual(set(buckets["country_code"]), {"NO"})
        self.assertEqual(set(buckets["bucket"]), set(weight_frame()["bucket"]))
        sums = buckets.groupby("date")["estimated_contribution_yoy_pp"].sum()
        expected = (
            cpi.loc[cpi["component"].eq("all_items")]
            .set_index("date")["inflation_yoy"]
        )
        pd.testing.assert_series_equal(sums, expected, check_names=False)

    def test_summary_keeps_countries_separate_on_the_same_date(self):
        us = build_bucket_decomposition(cpi_frame(), weight_frame())
        no_cpi, no_weights, contract = norway_inputs()
        norway = build_norway_bucket_decomposition(
            no_cpi, no_weights, contract
        )
        summary = build_decomposition_summary(
            pd.concat([us, norway], ignore_index=True)
        )

        self.assertEqual(
            summary.groupby("country_code").size().to_dict(),
            {"NO": 12, "US": 12},
        )

    def test_hicp_decomposition_reconciles_each_country(self):
        cpi, weights = hicp_inputs()
        buckets = build_hicp_bucket_decomposition(cpi, weights)

        self.assertEqual(set(buckets["country_code"]), {"DE", "DK"})
        sums = buckets.groupby(["country_code", "date"])[
            "estimated_contribution_yoy_pp"
        ].sum()
        expected = (
            cpi.loc[cpi["component"].isin({"all_items", "hicp_all_items"})]
            .set_index(["country_code", "date"])["inflation_yoy"]
        )
        pd.testing.assert_series_equal(sums, expected, check_names=False)
        april = buckets.loc[buckets["date"].eq(pd.Timestamp("2021-04-01"))]
        self.assertTrue(april["inflation_yoy_change_3m"].notna().all())


if __name__ == "__main__":
    unittest.main()
