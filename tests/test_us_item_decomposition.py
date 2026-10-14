import pandas as pd
import pytest

from inflation_engine.features.us_item_decomposition import (
    build_us_item_decomposition,
    load_us_weight_tree,
    select_us_frontier,
)


def test_frontier_prefers_deepest_fully_matched_subtree():
    tree = pd.DataFrame(
        {
            "level": [1, 2, 3, 3],
            "label": ["Clothing and footwear", "Clothing", "Men's apparel", "Women's apparel"],
            "weight_pct": [10.0, 8.0, 4.0, 4.0],
            "weight_reference_date": pd.Timestamp("2026-01-01"),
            "weight_basis_year": 2024,
        }
    )
    series_map = pd.DataFrame(
        {
            "label": ["Men's apparel", "Women's apparel"],
            "series_id": ["CUSR0000SAA1", "CUSR0000SAA2"],
        }
    )
    frontier = select_us_frontier(tree, series_map)
    assert set(frontier["source_component_code"]) == {"CUSR0000SAA1", "CUSR0000SAA2"}


def test_frontier_falls_back_to_matched_ancestor_when_children_unmatched():
    tree = pd.DataFrame(
        {
            "level": [1, 2, 3],
            "label": ["Meats", "Beef and veal", "Uncooked ground beef"],
            "weight_pct": [1.2, 0.6, 0.2],
            "weight_reference_date": pd.Timestamp("2026-01-01"),
            "weight_basis_year": 2024,
        }
    )
    # Only the top-level "Meats" node has a published standalone index --
    # "Uncooked ground beef" only has an average-price series, not a CPI index.
    series_map = pd.DataFrame({"label": ["Meats"], "series_id": ["CUSR0000SAF11"]})
    frontier = select_us_frontier(tree, series_map)
    assert list(frontier["source_component_code"]) == ["CUSR0000SAF11"]
    assert list(frontier["weight_pct"]) == [1.2]


def test_frontier_leaves_a_gap_when_nothing_in_a_subtree_matches():
    tree = pd.DataFrame(
        {
            "level": [1],
            "label": ["Unmatched category"],
            "weight_pct": [2.0],
            "weight_reference_date": pd.Timestamp("2026-01-01"),
            "weight_basis_year": 2024,
        }
    )
    series_map = pd.DataFrame({"label": [], "series_id": []})
    with pytest.raises(ValueError, match="No BLS weight rows matched"):
        select_us_frontier(tree, series_map)


def test_load_us_weight_tree_excludes_special_aggregate_section():
    tree = load_us_weight_tree()
    labels = set(tree["label"])
    assert "Food and beverages" in labels
    assert "Expenditure category" not in labels
    assert "All items" not in labels
    # "All items less shelter" lives in the unrelated, overlapping special
    # aggregate section appended after the real tree and must not appear.
    assert "All items less shelter" not in labels
    assert tree["weight_basis_year"].iloc[0] == 2024


def _build_forest(records: list[dict]) -> list[dict]:
    stack: list[tuple[int, dict]] = []
    roots: list[dict] = []
    for record in records:
        node = {**record, "children": []}
        while stack and stack[-1][0] >= node["level"]:
            stack.pop()
        if stack:
            stack[-1][1]["children"].append(node)
        else:
            roots.append(node)
        stack.append((node["level"], node))
    return roots


def test_load_us_weight_tree_children_never_exceed_their_parents_weight():
    """A node's direct children summing to more than its own declared weight
    means two rows that should nest one way are numbered as if they nest a
    different way (see the Alcoholic beverages / Information technology
    correction in load_us_weight_tree) -- and that overcounts every item
    frontier built from the tree. This is a general regression guard, not
    specific to those two branches, in case a future BLS weight vintage
    reintroduces the same kind of indent mismatch elsewhere."""
    tree = load_us_weight_tree()
    forest = _build_forest(tree.to_dict("records"))

    violations = []

    def walk(node: dict) -> None:
        if node["children"]:
            child_sum = sum(child["weight_pct"] for child in node["children"])
            if child_sum > node["weight_pct"] + 0.01:
                violations.append((node["label"], node["weight_pct"], child_sum))
            for child in node["children"]:
                walk(child)

    for root in forest:
        walk(root)

    assert violations == []


def test_load_us_weight_tree_corrects_the_two_known_misindented_branches():
    tree = load_us_weight_tree()
    forest = _build_forest(tree.to_dict("records"))

    def find(node: dict, label: str) -> dict | None:
        if node["label"] == label:
            return node
        for child in node["children"]:
            found = find(child, label)
            if found:
                return found
        return None

    def find_anywhere(label: str) -> dict:
        for root in forest:
            found = find(root, label)
            if found:
                return found
        raise AssertionError(f"{label!r} not found in the parsed tree")

    food_and_beverages = find_anywhere("Food and beverages")
    assert {child["label"] for child in food_and_beverages["children"]} == {
        "Food",
        "Alcoholic beverages",
    }

    information_processing = find_anywhere("Information and information processing")
    assert {child["label"] for child in information_processing["children"]} == {
        "Telephone services",
        "Information technology, hardware and services",
    }


@pytest.mark.integration
def test_us_item_layer_reconciles_to_headline():
    raw = pd.read_parquet("data/raw/fred_us_cpi.parquet")
    result = build_us_item_decomposition(raw)
    assert result["reconciliation_row"].any()
    for date, group in result.groupby("date"):
        headline = group["headline_yoy"].dropna()
        if headline.empty:
            continue
        assert abs(
            group["estimated_contribution_yoy_pp"].sum() - headline.iloc[0]
        ) < 1e-9
    latest = result[result["date"] == result["date"].max()]
    non_reconciliation = latest.loc[~latest["reconciliation_row"]]
    # The series map is now built from BLS's own cu.item/cu.series catalog
    # (see build_us_cpi_item_series_map.py) rather than a ~50-row hand
    # curated subset, so coverage is close to complete -- the remainder is
    # just whichever handful of granular series haven't posted the latest
    # month yet, not a structural gap the way it was before.
    assert 95.0 < non_reconciliation["weight_pct"].sum() < 101.0


def test_us_item_layer_requires_us_component_rows():
    with pytest.raises(ValueError, match="missing columns"):
        build_us_item_decomposition(pd.DataFrame({"date": []}))
