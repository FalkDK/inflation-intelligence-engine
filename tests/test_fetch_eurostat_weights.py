import unittest

import pandas as pd

from inflation_engine.data.fetch_eurostat_weights import (
    normalize_eurostat_weights,
)


def weight_dataset(geo, *, year="2026", values=None):
    codes = ["TOTAL", "CP041", "FOOD", "IGD_NNRG", "NRG", "SERV"]
    if values is None:
        values = [1000.0, 10.0, 140.0, 200.0, 60.0, 600.0]
    return {
        "id": ["freq", "coicop18", "statinfo", "geo", "time"],
        "size": [1, len(codes), 1, 1, 1],
        "dimension": {
            "freq": {"category": {"index": {"A": 0}}},
            "coicop18": {
                "category": {
                    "index": {code: index for index, code in enumerate(codes)}
                }
            },
            "statinfo": {"category": {"index": {"IW": 0}}},
            "geo": {"category": {"index": {geo: 0}}},
            "time": {"category": {"index": {year: 0}}},
        },
        "value": {str(index): value for index, value in enumerate(values)},
    }


class FetchEurostatWeightsTests(unittest.TestCase):
    def test_normalize_maps_exhaustive_buckets_without_double_counting(self):
        frame = normalize_eurostat_weights(
            {"DE": weight_dataset("DE"), "DK": weight_dataset("DK")}
        )

        self.assertEqual(set(frame["country_code"]), {"DE", "DK"})
        totals = frame.groupby("country_code")["weight_pct"].sum()
        pd.testing.assert_series_equal(
            totals, pd.Series({"DE": 100.0, "DK": 100.0}), check_names=False
        )
        residual = frame.loc[
            frame["bucket"].eq("core_services_ex_housing")
        ].set_index("country_code")["weight_pct"]
        self.assertEqual(residual.to_dict(), {"DE": 59.0, "DK": 59.0})

    def test_normalize_skips_a_year_where_the_four_buckets_dont_reconcile(self):
        """Greece's first published year (1996) sums FOOD+NRG+IGD_NNRG+SERV to
        ~1015 against a TOTAL of 1000 -- a genuine data-quality gap in that
        earliest vintage. That year must be skipped, not silently built into
        five buckets that don't actually add up to the real basket, while a
        good year for the same country still comes through fine."""
        codes = ["TOTAL", "CP041", "FOOD", "IGD_NNRG", "NRG", "SERV"]
        bad_year_values = [1000.0, 10.0, 150.0, 210.0, 65.0, 615.0]  # sums to 1015
        good_year_values = [1000.0, 10.0, 140.0, 200.0, 60.0, 600.0]  # sums to 1000
        dataset = {
            "id": ["coicop18", "geo", "time"],
            "size": [len(codes), 1, 2],
            "dimension": {
                "coicop18": {
                    "category": {
                        "index": {code: index for index, code in enumerate(codes)}
                    }
                },
                "geo": {"category": {"index": {"GR": 0}}},
                "time": {"category": {"index": {"1996": 0, "2026": 1}}},
            },
            # time varies fastest: [code0/1996, code0/2026, code1/1996, ...]
            "value": [
                value
                for pair in zip(bad_year_values, good_year_values)
                for value in pair
            ],
        }

        frame = normalize_eurostat_weights({"GR": dataset})

        self.assertEqual(frame["weight_basis_year"].unique().tolist(), [2026])


if __name__ == "__main__":
    unittest.main()
