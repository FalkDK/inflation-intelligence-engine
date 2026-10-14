from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

import pandas as pd


def _ordered_dimension_codes(
    dataset: Mapping[str, Any],
    dimension: str,
) -> list[str]:
    category_index = dataset["dimension"][dimension]["category"]["index"]
    if isinstance(category_index, Mapping):
        return [
            code
            for code, _ in sorted(category_index.items(), key=lambda item: item[1])
        ]
    if isinstance(category_index, Sequence) and not isinstance(
        category_index, (str, bytes)
    ):
        return list(category_index)
    raise ValueError(f"Invalid JSON-stat category index for dimension {dimension}")


def parse_jsonstat_observations(dataset: Mapping[str, Any]) -> pd.DataFrame:
    """Convert JSON-stat v1 or v2 data into dimension/value rows."""
    dimensions = dataset.get("id")
    sizes = dataset.get("size")
    values = dataset.get("value")
    dimension_metadata = dataset.get("dimension")

    if dimensions is None and isinstance(dimension_metadata, Mapping):
        nested_dimensions = dimension_metadata.get("id")
        if isinstance(nested_dimensions, list):
            dimensions = nested_dimensions
        else:
            dimensions = [
                dimension
                for dimension in dimension_metadata
                if dimension not in {"id", "size", "role"}
            ]
    if sizes is None and isinstance(dimension_metadata, Mapping):
        nested_sizes = dimension_metadata.get("size")
        if isinstance(nested_sizes, list):
            sizes = nested_sizes
    if sizes is None and isinstance(dimensions, list):
        sizes = [
            len(_ordered_dimension_codes(dataset, dimension))
            for dimension in dimensions
        ]

    if not isinstance(dimensions, list) or not isinstance(sizes, list):
        raise ValueError("JSON-stat response is missing dimensions")
    if len(dimensions) != len(sizes):
        raise ValueError("JSON-stat dimension and size metadata do not match")
    if not isinstance(values, (Mapping, list)):
        raise ValueError("JSON-stat response is missing observation values")

    codes_by_dimension = [
        _ordered_dimension_codes(dataset, dimension) for dimension in dimensions
    ]
    rows = []

    for positions in product(*(range(size) for size in sizes)):
        flat_index = 0
        for position, size in zip(positions, sizes):
            flat_index = flat_index * size + position

        if isinstance(values, Mapping):
            value = values.get(str(flat_index), values.get(flat_index))
        else:
            value = values[flat_index] if flat_index < len(values) else None
        if value is None:
            continue

        row = {
            dimension: codes_by_dimension[index][position]
            for index, (dimension, position) in enumerate(zip(dimensions, positions))
        }
        row["value"] = value
        rows.append(row)

    if not rows:
        raise ValueError("JSON-stat response contains no observations")
    return pd.DataFrame(rows)
