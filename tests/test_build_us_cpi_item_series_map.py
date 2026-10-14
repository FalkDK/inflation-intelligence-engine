import pandas as pd
import pytest

from inflation_engine.data.build_us_cpi_item_series_map import (
    build_us_cpi_item_series_map,
    normalize_label,
    parse_cu_item,
    parse_cu_series,
    select_preferred_series,
    slugify_label,
)


CU_ITEM_TEXT = (
    "item_code\titem_name\tdisplay_level\tselectable\tsort_sequence\n"
    "SEHA\tRent of primary residence\t3\tT\t100\n"
    "SETB01\tGasoline (all types)\t4\tT\t200\n"
    "SEAA02\tMen's underwear, nightwear and accessories\t6\tT\t300\n"
    "SEFR02\tCandy and chewing gum\t6\tT\t400\n"
)

CU_SERIES_TEXT = (
    "series_id\tarea_code\titem_code\tseasonal\tperiodicity_code\tbase_code\t"
    "base_period\tseries_title\tfootnote_codes\tbegin_year\tbegin_period\t"
    "end_year\tend_period\n"
    "CUSR0000SEHA\t0000\tSEHA\tS\tR\tS\t1982-84=100\tt\t\t1953\tM01\t2026\tM08\n"
    "CUUR0000SEHA\t0000\tSEHA\tU\tR\tS\t1982-84=100\tt\t\t1953\tM01\t2026\tM08\n"
    "CUSR0000SETB01\t0000\tSETB01\tS\tR\tS\t1982-84=100\tt\t\t1978\tM01\t2026\tM08\n"
    "CUUR0000SEFR02\t0000\tSEFR02\tU\tR\tS\tDECEMBER 1997=100\tt\t\t1997\tM12\t2026\tM08\n"
    "CUUS0000SEFR02\t0000\tSEFR02\tU\tS\tS\tDECEMBER 1997=100\tt\t\t2018\tS01\t2026\tS01\n"
    "CUUR0000SEHAX\t0100\tSEHA\tU\tR\tS\t1982-84=100\tt\t\t1953\tM01\t2015\tM12\n"
)


def test_parse_cu_item_returns_code_and_name():
    frame = parse_cu_item(CU_ITEM_TEXT)
    assert dict(zip(frame["item_code"], frame["item_name"]))["SEHA"] == (
        "Rent of primary residence"
    )


def test_parse_cu_series_strips_whitespace():
    frame = parse_cu_series(CU_SERIES_TEXT)
    assert frame.loc[0, "series_id"] == "CUSR0000SEHA"
    assert frame.loc[0, "area_code"] == "0000"


def test_select_preferred_series_prefers_seasonally_adjusted():
    cu_series = parse_cu_series(CU_SERIES_TEXT)
    result = select_preferred_series(cu_series, min_end_year=2025)
    assert result["SEHA"] == "CUSR0000SEHA"


def test_select_preferred_series_falls_back_to_not_seasonally_adjusted():
    cu_series = parse_cu_series(CU_SERIES_TEXT)
    result = select_preferred_series(cu_series, min_end_year=2025)
    # SEFR02 has no monthly seasonally-adjusted series at area 0000 -- only
    # a not-seasonally-adjusted monthly one and an unrelated semiannual one.
    assert result["SEFR02"] == "CUUR0000SEFR02"


def test_select_preferred_series_excludes_stale_and_other_area_series():
    cu_series = parse_cu_series(CU_SERIES_TEXT)
    result = select_preferred_series(cu_series, min_end_year=2025)
    # The fixture's SEHA row at area 0100 (a different region) stopped
    # publishing in 2015 -- neither disqualifies the real area-0000 match.
    assert result["SEHA"] == "CUSR0000SEHA"
    assert "CUUR0000SEHAX" not in result.values()


def test_normalize_label_handles_curly_apostrophe_and_oxford_comma():
    assert normalize_label(
        "Men’s underwear, nightwear, and accessories"
    ) == "Men's underwear, nightwear and accessories"


def test_slugify_label_produces_snake_case_identifier():
    assert (
        slugify_label("Owners' equivalent rent of primary residence")
        == "owners_equivalent_rent_of_primary_residence"
    )
    assert slugify_label("Gasoline (all types)") == "gasoline_all_types"


def test_build_us_cpi_item_series_map_joins_tree_to_series_ids():
    tree = pd.DataFrame(
        {
            "level": [3, 4, 6],
            "label": [
                "Rent of primary residence",
                "Gasoline (all types)",
                "Men's underwear, nightwear, and accessories",
            ],
            "weight_pct": [7.8, 2.9, 0.13],
        }
    )
    cu_item = parse_cu_item(CU_ITEM_TEXT)
    cu_series = parse_cu_series(CU_SERIES_TEXT)

    result = build_us_cpi_item_series_map(tree, cu_item, cu_series, min_end_year=2025)

    assert list(result.columns) == ["level", "label", "weight_pct", "series_id"]
    assert set(result["series_id"]) == {"CUSR0000SEHA", "CUSR0000SETB01"}
    # The underwear item has no BLS-published series at all in this fixture
    # and must be dropped rather than raising, matching the frontier's own
    # cascade-to-nearest-ancestor behavior for unmatched labels.
    assert "Men's underwear, nightwear, and accessories" not in set(result["label"])


def test_build_us_cpi_item_series_map_raises_when_nothing_matches():
    tree = pd.DataFrame(
        {"level": [1], "label": ["Not a real BLS item"], "weight_pct": [1.0]}
    )
    cu_item = parse_cu_item(CU_ITEM_TEXT)
    cu_series = parse_cu_series(CU_SERIES_TEXT)
    with pytest.raises(ValueError, match="No BLS weight-tree labels matched"):
        build_us_cpi_item_series_map(tree, cu_item, cu_series, min_end_year=2025)
