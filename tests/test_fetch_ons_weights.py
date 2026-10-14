import unittest

import pandas as pd

from inflation_engine.data.fetch_ons_weights import (
    build_uk_weights,
    parse_ons_annual_weight,
)


def annual(values, series_id="TEST"):
    return "\n".join(
        [
            '"Title","CPI weight"',
            f'"CDID","{series_id}"',
            '"Unit","Parts per 1000"',
            *[f'"{year}","{value}"' for year, value in values.items()],
        ]
    )


class FetchOnsWeightsTests(unittest.TestCase):
    def test_parse_annual_weight(self):
        result = parse_ons_annual_weight(
            annual({2025: 140, 2026: 150}, "A9EW"), series_id="A9EW"
        )
        self.assertEqual(result.to_dict(), {2025: 140.0, 2026: 150.0})

    def test_build_uk_weights_is_exhaustive(self):
        frame = build_uk_weights(
            {
                "food": pd.Series({2026: 150.0}),
                "energy": pd.Series({2026: 60.0}),
                "core_goods": pd.Series({2026: 290.0}),
                "shelter": pd.Series({2026: 80.0}),
            }
        )
        self.assertAlmostEqual(frame["weight_pct"].sum(), 100.0)
        residual = frame.set_index("bucket").loc[
            "core_services_ex_housing", "weight_pct"
        ]
        self.assertAlmostEqual(residual, 42.0)


if __name__ == "__main__":
    unittest.main()
