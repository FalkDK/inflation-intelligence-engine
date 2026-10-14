import unittest

import pandas as pd

from inflation_engine.data.fetch_bls_weights import load_weight_snapshot
from inflation_engine.utils.component_taxonomy import (
    build_weighted_component_contract,
    load_component_taxonomy,
    validate_component_taxonomy,
)


class ComponentTaxonomyTests(unittest.TestCase):
    def test_reference_taxonomy_selects_one_node_per_bucket(self):
        taxonomy = load_component_taxonomy()
        selected = taxonomy.loc[taxonomy["included_in_decomposition"]]

        self.assertEqual(len(selected), 5)
        self.assertEqual(selected["canonical_macro_bucket"].nunique(), 5)
        self.assertIn(
            "overlapping_aggregate",
            taxonomy["mapping_status"].tolist(),
        )

    def test_weighted_contract_joins_effective_annual_weights(self):
        contract = build_weighted_component_contract(
            load_component_taxonomy(),
            load_weight_snapshot(),
        )

        self.assertEqual(len(contract), 5)
        self.assertAlmostEqual(contract["weight_pct"].sum(), 100.001)
        residual = contract.set_index("canonical_macro_bucket").loc[
            "core_services_ex_housing"
        ]
        self.assertEqual(residual["mapping_status"], "derived")

    def test_overlapping_bucket_selection_is_rejected(self):
        taxonomy = load_component_taxonomy()
        duplicate = taxonomy.loc[
            taxonomy["canonical_macro_bucket"] == "food"
        ].copy()
        duplicate["source_component_code"] = "OTHER_FOOD"
        overlapping = pd.concat([taxonomy, duplicate], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "overlapping components"):
            validate_component_taxonomy(overlapping)

    def test_missing_weight_is_rejected(self):
        weights = load_weight_snapshot()
        weights = weights.loc[weights["bucket"] != "energy"]

        with self.assertRaisesRegex(ValueError, "each canonical bucket"):
            build_weighted_component_contract(
                load_component_taxonomy(),
                weights,
            )


if __name__ == "__main__":
    unittest.main()
