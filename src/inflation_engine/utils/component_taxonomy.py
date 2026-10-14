"""Validated component hierarchy and canonical macro-bucket mappings."""

from pathlib import Path

import pandas as pd

from inflation_engine.data.fetch_bls_weights import BUCKETS, validate_weights
from inflation_engine.utils.paths import COMPONENT_TAXONOMY_PATH


TAXONOMY_COLUMNS = [
    "country_code",
    "index_family",
    "source_component_code",
    "source_component",
    "classification_system",
    "hierarchy_level",
    "parent_component_code",
    "canonical_macro_bucket",
    "included_in_decomposition",
    "effective_start",
    "effective_end",
    "mapping_status",
]
MAPPING_STATUSES = {"published", "derived", "overlapping_aggregate"}
NON_BUCKET_LABELS = {"all_items", "unassigned"}
WEIGHTED_CONTRACT_COLUMNS = TAXONOMY_COLUMNS + [
    "weight_pct",
    "weight_reference_date",
    "weight_basis_year",
    "weight_source",
    "weight_estimation_method",
]


def validate_component_taxonomy(frame: pd.DataFrame) -> None:
    missing = [column for column in TAXONOMY_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Component taxonomy is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Component taxonomy cannot be empty")
    required_non_null = [
        column for column in TAXONOMY_COLUMNS if column != "parent_component_code"
    ]
    if frame[required_non_null].isnull().any().any():
        raise ValueError("Component taxonomy contains null required values")
    if frame.duplicated(
        ["country_code", "index_family", "source_component_code", "effective_start"]
    ).any():
        raise ValueError("Component taxonomy contains duplicate effective mappings")
    if (frame["hierarchy_level"] < 0).any():
        raise ValueError("Component hierarchy levels cannot be negative")
    roots = frame["hierarchy_level"] == 0
    if frame.loc[roots, "parent_component_code"].notna().any():
        raise ValueError("Root components cannot have a parent")
    if frame.loc[~roots, "parent_component_code"].isna().any():
        raise ValueError("Non-root components must identify a parent")
    if (frame["effective_start"] > frame["effective_end"]).any():
        raise ValueError("Component mapping effective dates are invalid")
    invalid_status = sorted(set(frame["mapping_status"]) - MAPPING_STATUSES)
    if invalid_status:
        raise ValueError(f"Invalid component mapping statuses: {invalid_status}")
    allowed_buckets = set(BUCKETS) | NON_BUCKET_LABELS
    invalid_buckets = sorted(set(frame["canonical_macro_bucket"]) - allowed_buckets)
    if invalid_buckets:
        raise ValueError(f"Invalid canonical macro buckets: {invalid_buckets}")
    included = frame.loc[frame["included_in_decomposition"]]
    if not included["canonical_macro_bucket"].isin(BUCKETS).all():
        raise ValueError("Included components must map to a canonical bucket")
    duplicate_bucket = included.duplicated(
        [
            "country_code",
            "index_family",
            "canonical_macro_bucket",
            "effective_start",
        ],
        keep=False,
    )
    if duplicate_bucket.any():
        raise ValueError(
            "Decomposition selects overlapping components for one macro bucket"
        )


def load_component_taxonomy(
    path: str | Path = COMPONENT_TAXONOMY_PATH,
) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Component taxonomy does not exist: {source}")
    frame = pd.read_csv(
        source,
        parse_dates=["effective_start", "effective_end"],
        dtype={"parent_component_code": "string"},
    )
    frame["included_in_decomposition"] = frame[
        "included_in_decomposition"
    ].astype(bool)
    validate_component_taxonomy(frame)
    return frame.loc[:, TAXONOMY_COLUMNS]


def build_weighted_component_contract(
    taxonomy: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Join selected, non-overlapping component nodes to effective weights."""
    validate_component_taxonomy(taxonomy)
    validate_weights(weights)
    selected = taxonomy.loc[taxonomy["included_in_decomposition"]].copy()
    weight_fields = weights.rename(
        columns={
            "bucket": "canonical_macro_bucket",
            "reference_date": "weight_reference_date",
            "source": "weight_source",
            "estimation_method": "weight_estimation_method",
        }
    )
    contract = selected.merge(
        weight_fields[
            [
                "country_code",
                "index_family",
                "canonical_macro_bucket",
                "effective_start",
                "effective_end",
                "weight_pct",
                "weight_reference_date",
                "weight_basis_year",
                "weight_source",
                "weight_estimation_method",
            ]
        ],
        on=[
            "country_code",
            "index_family",
            "canonical_macro_bucket",
            "effective_start",
            "effective_end",
        ],
        how="left",
        validate="one_to_one",
    )
    if contract["weight_pct"].isna().any():
        missing = contract.loc[
            contract["weight_pct"].isna(), "canonical_macro_bucket"
        ].tolist()
        raise ValueError(f"Selected component mappings have no weights: {missing}")
    grouped = contract.groupby(
        ["country_code", "index_family", "effective_start"]
    )
    for key, group in grouped:
        if set(group["canonical_macro_bucket"]) != set(BUCKETS):
            raise ValueError(f"Weighted component contract is incomplete: {key}")
    return contract.loc[:, WEIGHTED_CONTRACT_COLUMNS]
