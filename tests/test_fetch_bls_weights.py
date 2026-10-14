import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from inflation_engine.data.fetch_bls_weights import (
    BUCKETS,
    fetch_bls_relative_importance,
    load_weight_snapshot,
    parse_bls_relative_importance,
    save_bls_weights,
)


HTML = """
<html>
  <body>
    <h2>Table 1 (2024 Weights).</h2>
    <table>
      <tr><th>Item and Group</th><th>CPI-U</th><th>CPI-W</th></tr>
      <tr><td>Food</td><td>13.698</td><td>14.889</td></tr>
      <tr><td>Energy</td><td>6.383</td><td>8.052</td></tr>
      <tr><td>Shelter</td><td>35.625</td><td>33.594</td></tr>
      <tr>
        <td>Commodities less food and energy commodities</td>
        <td>19.176</td><td>19.966</td>
      </tr>
      <tr>
        <td>Services less energy services</td>
        <td>60.744</td><td>57.093</td>
      </tr>
    </table>
    <h2>Table 1 (2023 Weights).</h2>
    <table>
      <tr><td>Food</td><td>99.000</td></tr>
    </table>
  </body>
</html>
"""


class FakeResponse:
    text = HTML

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse()


class FetchBlsWeightsTests(unittest.TestCase):
    def test_parses_new_weight_table_into_exhaustive_buckets(self):
        result = parse_bls_relative_importance(HTML, reference_year=2025)

        self.assertEqual(result["bucket"].tolist(), BUCKETS)
        self.assertAlmostEqual(result["weight_pct"].sum(), 100.001)
        services = result.set_index("bucket").loc[
            "core_services_ex_housing",
            "weight_pct",
        ]
        self.assertAlmostEqual(services, 25.119)
        self.assertEqual(set(result["weight_basis_year"]), {2024})
        self.assertEqual(
            set(result["effective_start"]),
            {pd.Timestamp("2026-01-01")},
        )

    def test_fetch_uses_the_requested_bls_page(self):
        session = FakeSession()

        result = fetch_bls_relative_importance(2025, session=session)

        self.assertEqual(len(result), 5)
        self.assertIn("/2025.htm", session.calls[0][0])
        self.assertIn("User-Agent", session.calls[0][1]["headers"])

    def test_missing_required_row_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing required rows"):
            parse_bls_relative_importance(
                HTML.replace(
                    "<tr><td>Energy</td><td>6.383</td><td>8.052</td></tr>",
                    "",
                ),
                reference_year=2025,
            )

    def test_save_writes_weight_schema(self):
        weights = parse_bls_relative_importance(HTML, reference_year=2025)
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "weights.parquet"
            destination = save_bls_weights(weights, output)
            reloaded = pd.read_parquet(destination)

        self.assertEqual(destination, output)
        self.assertEqual(len(reloaded), 5)

    def test_loads_reviewed_csv_snapshot(self):
        weights = parse_bls_relative_importance(HTML, reference_year=2025)
        with TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "weights.csv"
            weights.to_csv(source, index=False)
            loaded = load_weight_snapshot(source)

        self.assertEqual(loaded["bucket"].tolist(), BUCKETS)


if __name__ == "__main__":
    unittest.main()
