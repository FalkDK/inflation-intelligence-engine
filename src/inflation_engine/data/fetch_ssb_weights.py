"""Build a hierarchy-safe Norwegian CPI component and weight contract."""

import argparse
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inflation_engine.data.fetch_bls_weights import (
    BUCKETS,
    WEIGHT_COLUMNS,
    validate_weights,
)
from inflation_engine.data.fetch_ssb import (
    SSB_CPI_COMPONENT_TABLE,
    fetch_ssb_dataset,
    fetch_ssb_metadata,
)
from inflation_engine.utils.components import category_codes, category_labels
from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.jsonstat import parse_jsonstat_observations
from inflation_engine.utils.paths import (
    SSB_NORWAY_COMPONENT_CONTRACT_PATH,
    SSB_NORWAY_CPI_WEIGHTS_PATH,
    SSB_NORWAY_FULL_WEIGHTS_PATH,
)


CLASSIFICATION_SYSTEM = "ECOICOP_v2_COICOP2018"
WEIGHT_CONTENT_CODE = "KpiVektMnd"
DIRECT_BUCKET_PREFIXES = {
    "food": ("01",),
    "shelter": ("04.1", "04.2"),
    "energy": ("04.5", "07.2.2"),
}
UNMARKED_NODE_BUCKETS = {
    # These published groups contain branches that are not separately exposed
    # in table 14700. Select the parent once instead of pretending the visible
    # children exhaust its weight.
    "04.3": "core_services_ex_housing",
    "04.4": "core_services_ex_housing",
    "02.1": "core_goods",
    "03.1": "core_goods",
    "03.2": "core_goods",
    "05.5": "core_goods",
    "05.6": "core_goods",
    "06.1": "core_goods",
    "06.2": "core_services_ex_housing",
    "06.4": "core_services_ex_housing",
    "07.1": "core_goods",
    "07.2.4": "core_services_ex_housing",
    "08.1": "core_goods",
    "08.3": "core_services_ex_housing",
    # Division 09 contains published weight not exposed through every child
    # node. Keep the published parent as one mixed recreation/services block.
    "09": "core_services_ex_housing",
    "10": "core_services_ex_housing",
    "11.1": "core_services_ex_housing",
    "11.2": "core_services_ex_housing",
    "12.1": "core_services_ex_housing",
    "12.2": "core_services_ex_housing",
    "13.2": "core_goods",
    "13.1": "core_goods",
    "13.3": "core_services_ex_housing",
    "13.9": "core_services_ex_housing",
}
RATE_PROXY_CODES = {
    # The weight is published for 07.2.4 but its index is unavailable. Borrow
    # the nearest published parent rate without borrowing the parent's weight.
    "07.2.4": "07.2",
}
CONTRACT_COLUMNS = [
    "country_code",
    "index_family",
    "source_dataset",
    "source_component_code",
    "source_component",
    "rate_source_component_code",
    "classification_system",
    "hierarchy_level",
    "canonical_macro_bucket",
    "durability_class",
    "effective_start",
    "effective_end",
    "annual_effective_weight_pct",
    "weight_reference_date",
    "weight_basis_year",
    "weight_source",
    "mapping_method",
]


def _is_descendant(code: str, parent: str) -> bool:
    return code == parent or code.startswith(parent + ".")


def _hierarchy_level(code: str) -> int:
    return len(code.split("."))


def _durability(label: str) -> str | None:
    match = re.search(r"\((ND|SD|D|S)\)\s*$", label)
    return match.group(1) if match else None


def select_norway_component_frontier(
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    """Select one exhaustive, non-overlapping classification frontier."""
    labels = category_labels(metadata, SSB_CPI_COMPONENT_TABLE["component_dimension"])
    selected: list[dict[str, Any]] = []

    for bucket, prefixes in DIRECT_BUCKET_PREFIXES.items():
        for code in prefixes:
            selected.append(
                {
                    "source_component_code": code,
                    "source_component": labels[code],
                    "canonical_macro_bucket": bucket,
                    "durability_class": "mixed",
                    "mapping_method": "reviewed_direct_macro_node",
                }
            )

    for code, bucket in UNMARKED_NODE_BUCKETS.items():
        selected.append(
            {
                "source_component_code": code,
                "source_component": labels[code],
                "canonical_macro_bucket": bucket,
                "durability_class": (
                    "S" if bucket == "core_services_ex_housing" else "mixed_goods"
                ),
                "mapping_method": "reviewed_unmarked_macro_node",
            }
        )

    excluded = tuple(
        prefix
        for prefixes in DIRECT_BUCKET_PREFIXES.values()
        for prefix in prefixes
    ) + tuple(UNMARKED_NODE_BUCKETS)
    marked_candidates = []
    for code, label in labels.items():
        durability = _durability(label)
        if (
            code == "00"
            or durability is None
            or any(_is_descendant(code, prefix) for prefix in excluded)
        ):
            continue
        marked_candidates.append((code, label, durability))

    # Keep the shallowest durability-tagged node on each branch.
    for code, label, durability in sorted(
        marked_candidates,
        key=lambda item: (_hierarchy_level(item[0]), item[0]),
    ):
        if any(
            _is_descendant(code, row["source_component_code"])
            for row in selected
        ):
            continue
        selected.append(
            {
                "source_component_code": code,
                "source_component": label,
                "canonical_macro_bucket": (
                    "core_services_ex_housing"
                    if durability == "S"
                    else "core_goods"
                ),
                "durability_class": durability,
                "mapping_method": "official_durability_marker",
            }
        )

    result = pd.DataFrame(selected)
    result["rate_source_component_code"] = result["source_component_code"].replace(
        RATE_PROXY_CODES
    )
    duplicate_or_overlap = []
    codes = result["source_component_code"].tolist()
    for index, code in enumerate(codes):
        for other in codes[index + 1 :]:
            if _is_descendant(code, other) or _is_descendant(other, code):
                duplicate_or_overlap.append((code, other))
    if duplicate_or_overlap:
        raise ValueError(
            f"Norway component frontier overlaps: {duplicate_or_overlap[:3]}"
        )
    if set(result["canonical_macro_bucket"]) != set(BUCKETS):
        raise ValueError("Norway component frontier does not cover all macro buckets")
    return result.sort_values("source_component_code").reset_index(drop=True)


def normalize_norway_weights(
    dataset: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    reference_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frontier = select_norway_component_frontier(metadata)
    observations = parse_jsonstat_observations(dataset)
    dimension = SSB_CPI_COMPONENT_TABLE["component_dimension"]
    required = {dimension, "ContentsCode", "Tid", "value"}
    if not required.issubset(observations):
        raise ValueError("SSB weight response is missing required dimensions")
    weights = observations.loc[
        observations["ContentsCode"].eq(WEIGHT_CONTENT_CODE),
        [dimension, "Tid", "value"],
    ].copy()
    weights["value"] = pd.to_numeric(weights["value"], errors="coerce")
    weights = weights.dropna(subset=["value"])
    weights = weights.groupby(dimension, as_index=False)["value"].last()
    contract = frontier.merge(
        weights,
        left_on="source_component_code",
        right_on=dimension,
        how="left",
        validate="one_to_one",
    )
    if contract["value"].isna().any():
        missing = contract.loc[
            contract["value"].isna(), "source_component_code"
        ].tolist()
        raise ValueError(f"SSB component frontier has missing weights: {missing}")

    # SSB publishes weights in per mille.
    contract["annual_effective_weight_pct"] = contract["value"] / 10.0
    frontier_total = float(contract["annual_effective_weight_pct"].sum())
    if np.isclose(frontier_total, 100.0, atol=0.3):
        contract["annual_effective_weight_pct"] *= 100.0 / frontier_total
        contract["mapping_method"] = (
            contract["mapping_method"] + "_rounded_weight_normalized"
        )
        frontier_total = float(contract["annual_effective_weight_pct"].sum())
    if not np.isclose(frontier_total, 100.0, atol=0.05):
        bucket_totals = (
            contract.groupby("canonical_macro_bucket")[
                "annual_effective_weight_pct"
            ]
            .sum()
            .round(3)
            .to_dict()
        )
        published = weights.set_index(dimension)["value"].div(10.0)
        division_gaps = {}
        for division in [f"{number:02d}" for number in range(1, 14)]:
            if division not in published:
                continue
            covered = contract.loc[
                contract["source_component_code"].map(
                    lambda code: _is_descendant(code, division)
                ),
                "annual_effective_weight_pct",
            ].sum()
            gap = float(published[division] - covered)
            if abs(gap) > 0.05:
                division_gaps[division] = round(gap, 3)
        raise ValueError(
            "Norway component frontier weights do not sum to 100 percent: "
            f"{frontier_total:.3f}; buckets={bucket_totals}; "
            f"division_gaps={division_gaps}"
        )
    effective_start = pd.Timestamp(reference_year, 1, 1)
    effective_end = pd.Timestamp(reference_year, 12, 1)
    reference_date = pd.Timestamp(reference_year - 1, 12, 1)
    contract = contract.assign(
        country_code="NO",
        index_family="CPI",
        source_dataset=SSB_CPI_COMPONENT_TABLE["table"],
        classification_system=CLASSIFICATION_SYSTEM,
        hierarchy_level=contract["source_component_code"].map(_hierarchy_level),
        effective_start=effective_start,
        effective_end=effective_end,
        weight_reference_date=reference_date,
        weight_basis_year=reference_year - 1,
        weight_source="Statistics Norway table 14700",
    )
    contract = contract.loc[:, CONTRACT_COLUMNS]

    macro = (
        contract.groupby("canonical_macro_bucket", as_index=False)[
            "annual_effective_weight_pct"
        ]
        .sum()
        .rename(
            columns={
                "canonical_macro_bucket": "bucket",
                "annual_effective_weight_pct": "weight_pct",
            }
        )
    )
    macro = macro.assign(
        reference_date=reference_date,
        effective_start=effective_start,
        effective_end=effective_end,
        country_code="NO",
        index_family="CPI",
        source="Statistics Norway",
        source_url="https://www.ssb.no/en/statbank/table/14700",
        weight_basis_year=reference_year - 1,
        estimation_method="official_frontier_weights_aggregated",
    ).loc[:, WEIGHT_COLUMNS]
    validate_weights(macro)
    return macro, contract


def fetch_norway_weights(
    reference_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = fetch_ssb_metadata(SSB_CPI_COMPONENT_TABLE["table"])
    frontier = select_norway_component_frontier(metadata)
    diagnostic_divisions = [f"{number:02d}" for number in range(1, 14)]
    dataset = fetch_ssb_dataset(
        SSB_CPI_COMPONENT_TABLE["table"],
        [
            {
                "variableCode": SSB_CPI_COMPONENT_TABLE["component_dimension"],
                "valueCodes": (
                    frontier["source_component_code"].tolist()
                    + diagnostic_divisions
                ),
            },
            {"variableCode": "ContentsCode", "valueCodes": [WEIGHT_CONTENT_CODE]},
            {
                "variableCode": "Tid",
                "valueCodes": [f"{reference_year}M01"],
            },
        ],
    )
    return normalize_norway_weights(
        dataset,
        metadata,
        reference_year=reference_year,
    )


FULL_WEIGHT_COLUMNS = [
    "source_component_code",
    "description",
    "weight_pct",
    "weight_reference_date",
    "weight_basis_year",
]


def fetch_norway_full_component_weights(reference_year: int) -> pd.DataFrame:
    """Fetch every published table-14700 component weight, not just the ~40
    nodes select_norway_component_frontier needs for five-bucket
    classification. Used to build a separate, maximally granular item
    frontier for breadth analysis (norway_item_decomposition.py); the
    five-bucket contract above is untouched by this."""
    metadata = fetch_ssb_metadata(SSB_CPI_COMPONENT_TABLE["table"])
    codes = category_codes(metadata, SSB_CPI_COMPONENT_TABLE["component_dimension"])
    labels = category_labels(metadata, SSB_CPI_COMPONENT_TABLE["component_dimension"])
    dataset = fetch_ssb_dataset(
        SSB_CPI_COMPONENT_TABLE["table"],
        [
            {
                "variableCode": SSB_CPI_COMPONENT_TABLE["component_dimension"],
                "valueCodes": codes,
            },
            {"variableCode": "ContentsCode", "valueCodes": [WEIGHT_CONTENT_CODE]},
            {"variableCode": "Tid", "valueCodes": [f"{reference_year}M01"]},
        ],
    )
    dimension = SSB_CPI_COMPONENT_TABLE["component_dimension"]
    observations = parse_jsonstat_observations(dataset)
    weights = observations.loc[
        observations["ContentsCode"].eq(WEIGHT_CONTENT_CODE), [dimension, "value"]
    ].copy()
    weights["value"] = pd.to_numeric(weights["value"], errors="coerce")
    weights = weights.dropna(subset=["value"])
    weights = weights.loc[weights[dimension] != SSB_CPI_COMPONENT_TABLE["total_code"]]
    weights["weight_pct"] = weights["value"] / 10.0  # SSB publishes per mille
    weights["source_component_code"] = weights[dimension]
    weights["description"] = weights["source_component_code"].map(labels)
    weights["weight_reference_date"] = pd.Timestamp(reference_year, 1, 1)
    weights["weight_basis_year"] = reference_year
    return weights.loc[:, FULL_WEIGHT_COLUMNS].sort_values(
        "source_component_code"
    ).reset_index(drop=True)


def save_norway_full_weights(
    frame: pd.DataFrame,
    output_path: str | Path = SSB_NORWAY_FULL_WEIGHTS_PATH,
) -> Path:
    return atomic_write_parquet(frame, output_path, columns=FULL_WEIGHT_COLUMNS)


def save_norway_weights(
    weights: pd.DataFrame,
    contract: pd.DataFrame,
    *,
    weights_path: str | Path = SSB_NORWAY_CPI_WEIGHTS_PATH,
    contract_path: str | Path = SSB_NORWAY_COMPONENT_CONTRACT_PATH,
) -> tuple[Path, Path]:
    validate_weights(weights)
    weight_destination = atomic_write_parquet(
        weights, weights_path, columns=WEIGHT_COLUMNS
    )
    contract_destination = atomic_write_parquet(
        contract, contract_path, columns=CONTRACT_COLUMNS
    )
    return weight_destination, contract_destination


def run(reference_year: int = 2026) -> tuple[Path, Path, Path]:
    weight_path, contract_path = save_norway_weights(*fetch_norway_weights(reference_year))
    full_weights_path = save_norway_full_weights(
        fetch_norway_full_component_weights(reference_year)
    )
    return weight_path, contract_path, full_weights_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Norwegian CPI weights.")
    parser.add_argument("--reference-year", type=int, default=2026)
    args = parser.parse_args()
    weights, contract, full_weights = run(args.reference_year)
    print(f"Saved Norwegian macro weights to {weights}")
    print(f"Saved Norwegian component contract to {contract}")
    print(f"Saved Norwegian full component weights to {full_weights}")


if __name__ == "__main__":
    main()
