import unittest

import pandas as pd

from inflation_engine.features.availability import (
    attach_information_dates,
    validate_release_calendar,
)


def calendar():
    return pd.DataFrame(
        {
            "observation_date": [pd.Timestamp("2026-05-01")],
            "country_code": ["US"],
            "publication_date": [pd.Timestamp("2026-06-10")],
            "release_status": ["actual"],
            "source_url": ["https://example.test"],
        }
    )


class AvailabilityTests(unittest.TestCase):
    def test_uses_actual_publication_date_when_known(self):
        features = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-05-01")],
                "country_code": ["US"],
            }
        )

        result = attach_information_dates(features, calendar())

        self.assertEqual(result.iloc[0]["information_date"], pd.Timestamp("2026-06-10"))
        self.assertEqual(result.iloc[0]["availability_status"], "actual")
        self.assertEqual(result.iloc[0]["information_lag_days"], 40)

    def test_uses_conservative_lag_when_release_is_unknown(self):
        features = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-01-01")],
                "country_code": ["US"],
            }
        )

        result = attach_information_dates(features, calendar())

        self.assertTrue(pd.isna(result.iloc[0]["publication_date"]))
        self.assertEqual(result.iloc[0]["information_date"], pd.Timestamp("2020-03-01"))
        self.assertEqual(
            result.iloc[0]["availability_status"],
            "conservative_month_lag",
        )

    def test_rejects_release_before_observation(self):
        invalid = calendar()
        invalid["publication_date"] = pd.Timestamp("2026-04-30")

        with self.assertRaisesRegex(ValueError, "must follow"):
            validate_release_calendar(invalid)


if __name__ == "__main__":
    unittest.main()
