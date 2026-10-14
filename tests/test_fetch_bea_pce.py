import pandas as pd
import pytest

from inflation_engine.data.fetch_bea_pce import (
    build_pce_detail,
    build_pce_panel,
    build_pce_structure,
    parse_nipa_monthly,
    parse_series_register,
)
from inflation_engine.utils.schema import validate_cpi_schema


# A miniature of BEA's register: PCE (1) = Goods (2) + Services (3), where
# Services = Hospitals (4) + NPISH (5), and NPISH = Gross output (6) less
# receipts (7), whose child (8) nets hospitals (DHOSRC, also on line 4)
# back out. Line 9 is an addendum with no in-table parent.
REGISTER = """%SeriesCode,SeriesLabel,MetricName,CalculationType,DefaultScale,TableId:LineNo,SeriesCodeParents
DPCERC,"Personal consumption expenditures","Current Dollars","Level",-6,T20405:1|U20405:1,A191RC
DGDSRC,"Goods","Current Dollars","Level",-6,U20405:2,DPCERC
DSERRC,"Services","Current Dollars","Level",-6,U20405:3,DPCERC
DHOSRC,"Hospitals (27)","Current Dollars","Level",-6,U20405:4|U20405:8,DSERRC|DNPSRC
DNPIRC,"Final consumption expenditures of NPISHs","Current Dollars","Level",-6,U20405:5,DPCERC|DSERRC
DNPERC,"Gross output of nonprofit institutions","Current Dollars","Level",-6,U20405:6,DNPIRC
DNPSRC,"Less: Receipts from sales of goods and services by nonprofit institutions","Current Dollars","Level",-6,U20405:7,DNPIRC
DPCCRC,"PCE excluding food and energy","Current Dollars","Level",-6,U20405:9,NONE
DPCERG,"Personal consumption expenditures","Fisher Price Index","Level",0,U20404:1,NONE
DGDSRG,"Goods","Fisher Price Index","Level",0,U20404:2,NONE
DSERRG,"Services","Fisher Price Index","Level",0,U20404:3,NONE
DHOSRG,"Hospitals (27)","Fisher Price Index","Level",0,U20404:4|U20404:8,NONE
DNPIRG,"Final consumption expenditures of NPISHs","Fisher Price Index","Level",0,U20404:5,NONE
DNPERG,"Gross output of nonprofit institutions","Fisher Price Index","Level",0,U20404:6,NONE
DNPSRG,"Less: Receipts","Fisher Price Index","Level",0,U20404:7,NONE
DPCCRG,"PCE excluding food and energy","Fisher Price Index","Level",0,U20404:9,NONE
"""


def _structure() -> pd.DataFrame:
    return build_pce_structure(parse_series_register(REGISTER)).set_index("table_line")


def test_structure_keys_by_line_and_resolves_closest_preceding_parent():
    structure = _structure()
    # Hospitals sits on two lines; each resolves to its own nearest parent.
    assert structure.at[4, "parent_line"] == 3
    assert structure.at[8, "parent_line"] == 7
    assert structure.at[7, "parent_line"] == 5
    # NPISH lists both PCE and Services; Services is the closer one.
    assert structure.at[5, "parent_line"] == 3
    assert structure.at[6, "parent_line"] == 5


def test_structure_flags_subtractive_lines_and_addenda():
    structure = _structure()
    assert structure.at[7, "aggregation_sign"] == -1
    assert structure.at[6, "aggregation_sign"] == 1
    assert bool(structure.at[8, "additive_tree"])
    assert not bool(structure.at[9, "additive_tree"])
    assert pd.isna(structure.at[9, "parent_line"])
    assert structure.at[9, "hierarchy_level"] == -1


def test_structure_levels_and_clean_labels():
    structure = _structure()
    assert structure.at[1, "hierarchy_level"] == 0
    assert structure.at[3, "hierarchy_level"] == 1
    assert structure.at[6, "hierarchy_level"] == 3
    assert structure.at[4, "label"] == "Hospitals"
    assert structure.at[4, "price_series_id"] == "DHOSRG"


def test_parse_nipa_monthly_strips_thousands_separators():
    text = '%SeriesCode,Period,Value\nDPCERG,2026M07,"131.172"\nDPCERC,2026M07,"22,117,708"\nOTHER,2026M07,"1"\n'
    frame = parse_nipa_monthly(text, {"DPCERG", "DPCERC"})
    assert set(frame["series_id"]) == {"DPCERG", "DPCERC"}
    assert frame.loc[frame["series_id"] == "DPCERC", "value"].item() == 22117708.0
    assert frame["date"].eq(pd.Timestamp("2026-07-01")).all()


def test_panel_uses_separate_geography_and_validates():
    structure = build_pce_structure(parse_series_register(REGISTER))
    dates = pd.date_range("2025-01-01", periods=2, freq="MS")
    observations = pd.DataFrame(
        [
            {"series_id": series_id, "date": date, "value": 100.0 + index}
            for series_id in ["DPCERG", "DPCCRG", "DPCERC", "DPCCRC"]
            for index, date in enumerate(dates)
        ]
    )
    detail = build_pce_detail(structure, observations)
    panel = build_pce_panel(
        detail,
        {
            "DPCERG": {"series_type": "headline", "component": "all_items"},
            "DPCCRG": {"series_type": "core", "component": "all_items_less_food_and_energy"},
        },
    )
    validate_cpi_schema(panel)
    assert set(panel["country_code"]) == {"US_PCE"}
    assert set(panel["series_role"]) == {"aggregate"}
    assert set(panel["index_base"]) == {"2017=100"}
    assert len(panel) == 4


def test_panel_raises_when_series_missing():
    structure = build_pce_structure(parse_series_register(REGISTER))
    observations = pd.DataFrame(
        {"series_id": ["DPCERC"], "date": [pd.Timestamp("2025-01-01")], "value": [1.0]}
    )
    detail = build_pce_detail(structure, observations)
    with pytest.raises(ValueError, match="DPCERG"):
        build_pce_panel(
            detail, {"DPCERG": {"series_type": "headline", "component": "all_items"}}
        )
