import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def category_labels(
    dataset: Mapping[str, Any],
    dimension: str,
) -> dict[str, str]:
    """Return JSON-stat category labels for a dimension."""
    category = dataset.get("dimension", {}).get(dimension, {}).get("category", {})
    labels = category.get("label", {})
    return dict(labels) if isinstance(labels, Mapping) else {}


def category_codes(
    dataset: Mapping[str, Any],
    dimension: str,
) -> list[str]:
    """Return JSON-stat category codes in source order."""
    category = dataset.get("dimension", {}).get(dimension, {}).get("category", {})
    category_index = category.get("index", {})
    if isinstance(category_index, Mapping):
        return [
            code
            for code, _ in sorted(category_index.items(), key=lambda item: item[1])
        ]
    if isinstance(category_index, Sequence) and not isinstance(
        category_index, (str, bytes)
    ):
        return list(category_index)
    return []


def normalize_component_label(label: str, *, fallback: str) -> str:
    """Convert source component labels into stable snake_case component names."""
    normalized = unicodedata.normalize("NFKD", label).encode("ascii", "ignore")
    text = normalized.decode("ascii").lower()
    text = re.sub(r"^\s*\d+(?:[._]\d+)*\s*", "", text)
    text = re.sub(r"\b(?:d|nd|s|sd)\b", " ", text)
    text = re.sub(r"\bn\.e\.c\.\b", "not_elsewhere_classified", text)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    text = re.sub(r"_+", "_", text)
    return text or fallback.lower().replace(".", "_")


def unique_component_slugs(
    codes: list[str],
    labels: Mapping[str, str],
    *,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return component slugs, suffixing source codes only for duplicate labels."""
    override_map = dict(overrides or {})
    base_slugs = {
        code: override_map.get(
            code,
            normalize_component_label(labels.get(code, ""), fallback=code),
        )
        for code in codes
    }
    counts = Counter(base_slugs.values())
    slugs = {}
    for code, slug in base_slugs.items():
        if counts[slug] == 1 or code in override_map:
            slugs[code] = slug
            continue
        suffix = normalize_component_label(code, fallback=code)
        slugs[code] = f"{slug}_{suffix}"
    return slugs
