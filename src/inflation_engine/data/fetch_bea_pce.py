"""US PCE price indexes and nominal spending from BEA's NIPA bulk files.

BEA publishes every NIPA series as two keyless flat files -- a series
register (code, label, table/line placement, parent codes) and one long
monthly data file -- so no API key is needed. Two underlying-detail tables
carry the full product breakdown of personal consumption expenditures:

- Table 2.4.4U (``U20404``): chain-type Fisher price index per line
  (2017=100).
- Table 2.4.5U (``U20405``): nominal spending per line, in millions of
  current dollars at seasonally adjusted annual rates.

The two tables share line numbers, which is how price and spending are
paired. The additive product tree is rebuilt from the register's parent
codes on 2.4.5U (nominal levels add up; Fisher price indexes do not). Two
quirks of BEA's tree matter:

- Some series sit on more than one line -- e.g. nonprofit hospitals'
  services to households appear both under household health care and under
  NPISH final consumption's "Less: Receipts from sales..." branch, where
  they're subtracted back out. Nodes are therefore keyed by table line,
  never by series code, and each line's parent is the closest preceding
  line among the series' listed parents.
- Lines labelled "Less: ..." enter their parent with a negative sign.

Everything after the tree ("Control group", "PCE excluding food and
energy", the market-based aggregates, ...) is a published addendum that
overlaps the tree; those lines are kept, flagged ``additive_tree=False``.

This module writes two files and deliberately never touches the FRED/BLS
US CPI data:

- ``bea_us_pce.parquet``: every line of both tables, long format.
- ``bea_us_pce_cpi.parquet``: a small set of headline/core PCE aggregates
  (plus a derived "supercore") in the common CPI panel schema, under the
  separate ``US_PCE`` geography code so it can never collide with ``US``
  CPI rows.
"""

import argparse
import io
import re
import time
from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from inflation_engine.utils.config import COUNTRIES, series_role
from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.manifest import update_ingestion_manifest
from inflation_engine.utils.paths import BEA_US_PCE_CPI_PATH, BEA_US_PCE_PATH
from inflation_engine.utils.pce_tree import PceTree, chained_fisher_index
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS, validate_cpi_schema


BEA_SERIES_REGISTER_URL = "https://apps.bea.gov/national/Release/TXT/SeriesRegister.txt"
BEA_NIPA_MONTHLY_URL = "https://apps.bea.gov/national/Release/TXT/NipaDataM.txt"
BEA_DATASET_ID = "bea_us_pce"
PCE_COUNTRY_CODE = "US_PCE"

PRICE_TABLE_ID = "U20404"
NOMINAL_TABLE_ID = "U20405"
ROOT_LINE = 1

PCE_DETAIL_COLUMNS = [
    "date",
    "table_line",
    "label",
    "parent_line",
    "hierarchy_level",
    "aggregation_sign",
    "additive_tree",
    "price_series_id",
    "nominal_series_id",
    "price_index",
    "nominal_spending_musd",
]

# Price-index series (Table 2.4.4U) published into the CPI panel under
# US_PCE. Everything else stays in the detail file only.
US_PCE_PANEL_SERIES = {
    "DPCERG": {"series_type": "headline", "component": "all_items"},
    "DGDSRG": {"series_type": "headline", "component": "goods"},
    "DDURRG": {"series_type": "headline", "component": "durable_goods"},
    "DNDGRG": {"series_type": "headline", "component": "nondurable_goods"},
    "DSERRG": {"series_type": "headline", "component": "services"},
    "DFXARG": {
        "series_type": "headline",
        "component": "food_and_beverages_off_premises",
    },
    "DNRGRG": {"series_type": "headline", "component": "energy_goods_and_services"},
    "DPFERG": {"series_type": "headline", "component": "food_and_energy"},
    "DPCMRG": {"series_type": "headline", "component": "market_based"},
    "DPCCRG": {"series_type": "core", "component": "all_items_less_food_and_energy"},
    "IA000062": {"series_type": "core", "component": "goods_less_food_and_energy"},
    "IA000063": {"series_type": "core", "component": "services_less_energy"},
    "DPCXRG": {
        "series_type": "core",
        "component": "market_based_less_food_and_energy",
    },
}

# "Supercore" -- services excluding energy and housing -- isn't published by
# BEA, so it's rebuilt as a monthly-chained Fisher index over the services
# frontier minus the housing and electricity-and-gas subtrees (the same
# construction reproduces BEA's own published "PCE services excluding
# energy" to ~0.01pp YoY on average since 2000).
SUPERCORE_COMPONENT = "services_less_energy_and_housing"
SUPERCORE_SERIES_ID = "DERIVED:SERVICES_EX_ENERGY_HOUSING"
SUPERCORE_SCOPE_NOMINAL_SERIES = "DSERRC"
SUPERCORE_EXCLUDED_NOMINAL_SERIES = ("DHSGRC", "DELGRC")

_FOOTNOTE_SUFFIX = re.compile(r"\s*\((?:parts? of )?\d+(?:(?:,| and) ?\d+)*\)\s*$")


def _clean_label(label: str) -> str:
    """Drop BEA's trailing cross-table line references, e.g. "Electricity
    (27)" -> "Electricity", "Furniture and furnishings (parts of 31 and 32)"
    -> "Furniture and furnishings"."""
    return _FOOTNOTE_SUFFIX.sub("", label.strip())


def download_text(
    url: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 300.0,
    max_attempts: int = 3,
) -> tuple[str, str | None]:
    """Download one BEA flat file; returns its text and Last-Modified header."""
    client = session or requests.Session()
    last_error: requests.RequestException | None = None
    for attempt in range(max_attempts):
        try:
            response = client.get(url, timeout=timeout)
            response.raise_for_status()
            if response.text.lstrip().startswith("<"):
                raise ValueError(f"BEA returned an HTML page instead of data for {url}")
            return response.text, response.headers.get("Last-Modified")
        except requests.RequestException as error:
            last_error = error
            if attempt < max_attempts - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"BEA download failed for {url}") from last_error


def parse_series_register(text: str) -> pd.DataFrame:
    """One row per (series, table, line) placement from SeriesRegister.txt."""
    register = pd.read_csv(io.StringIO(text), dtype=str)
    register.columns = [column.lstrip("%") for column in register.columns]
    required = {"SeriesCode", "SeriesLabel", "TableId:LineNo", "SeriesCodeParents"}
    missing = required - set(register.columns)
    if missing:
        raise ValueError(f"BEA series register is missing columns: {sorted(missing)}")
    placements = register.assign(
        placement=register["TableId:LineNo"].str.split("|")
    ).explode("placement")
    placements[["table_id", "table_line"]] = placements["placement"].str.split(
        ":", expand=True
    )
    placements["table_line"] = placements["table_line"].astype(int)
    return placements.loc[
        :, ["SeriesCode", "SeriesLabel", "SeriesCodeParents", "table_id", "table_line"]
    ].reset_index(drop=True)


def build_pce_structure(placements: pd.DataFrame) -> pd.DataFrame:
    """The PCE product tree: one row per Table 2.4.5U line, paired with the
    Table 2.4.4U price series on the same line."""
    nominal = placements.loc[placements["table_id"] == NOMINAL_TABLE_ID]
    price = placements.loc[placements["table_id"] == PRICE_TABLE_ID]
    if nominal.empty or price.empty:
        raise ValueError("BEA series register has no PCE underlying-detail tables")
    if nominal["table_line"].duplicated().any() or price["table_line"].duplicated().any():
        raise ValueError("BEA PCE tables contain duplicate line numbers")

    structure = (
        nominal.rename(
            columns={
                "SeriesCode": "nominal_series_id",
                "SeriesLabel": "raw_label",
                "SeriesCodeParents": "parent_codes",
            }
        )
        .merge(
            price.loc[:, ["table_line", "SeriesCode"]].rename(
                columns={"SeriesCode": "price_series_id"}
            ),
            on="table_line",
            how="left",
            validate="one_to_one",
        )
        .sort_values("table_line")
        .reset_index(drop=True)
    )
    if structure["table_line"].iloc[0] != ROOT_LINE:
        raise ValueError("BEA PCE table does not start at line 1")

    lines_by_code: dict[str, list[int]] = {}
    for code, line in zip(structure["nominal_series_id"], structure["table_line"]):
        lines_by_code.setdefault(code, []).append(line)

    parent_lines: list[float] = []
    for line, codes in zip(structure["table_line"], structure["parent_codes"].fillna("")):
        candidates = [
            candidate
            for code in codes.split("|")
            for candidate in lines_by_code.get(code, [])
            if candidate < line
        ]
        parent_lines.append(float(max(candidates)) if candidates else np.nan)
    structure["parent_line"] = parent_lines

    # The additive tree is the contiguous block from line 1 up to the first
    # later line with no in-table parent; everything after is an addendum.
    orphan_lines = structure.loc[
        (structure["table_line"] > ROOT_LINE) & structure["parent_line"].isna(),
        "table_line",
    ]
    tree_end = orphan_lines.min() if not orphan_lines.empty else np.inf
    structure["additive_tree"] = structure["table_line"] < tree_end
    structure.loc[~structure["additive_tree"], "parent_line"] = np.nan

    structure["aggregation_sign"] = np.where(
        structure["raw_label"].str.strip().str.startswith("Less:"), -1, 1
    )
    structure["label"] = structure["raw_label"].map(_clean_label)

    level_by_line: dict[int, int] = {}
    levels = []
    for line, parent, in_tree in zip(
        structure["table_line"], structure["parent_line"], structure["additive_tree"]
    ):
        if not in_tree:
            levels.append(-1)
            continue
        level = 0 if np.isnan(parent) else level_by_line[int(parent)] + 1
        level_by_line[line] = level
        levels.append(level)
    structure["hierarchy_level"] = levels
    structure["parent_line"] = structure["parent_line"].astype("Int64")
    return structure.loc[
        :,
        [
            "table_line",
            "label",
            "parent_line",
            "hierarchy_level",
            "aggregation_sign",
            "additive_tree",
            "price_series_id",
            "nominal_series_id",
        ],
    ]


def parse_nipa_monthly(text: str, series_ids: set[str]) -> pd.DataFrame:
    """Long (series_id, date, value) rows for the requested series only."""
    frame = pd.read_csv(io.StringIO(text), dtype=str)
    frame.columns = ["series_id", "period", "value"]
    frame = frame.loc[frame["series_id"].isin(series_ids)].copy()
    frame["date"] = pd.to_datetime(frame["period"], format="%YM%m")
    frame["value"] = pd.to_numeric(frame["value"].str.replace(",", ""), errors="coerce")
    return frame.dropna(subset=["value"]).loc[:, ["series_id", "date", "value"]]


def build_pce_detail(structure: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """Every PCE line x month with its price index and nominal spending."""
    by_series = observations.pivot(index="date", columns="series_id", values="value")
    dates = by_series.index
    frames = []
    for row in structure.itertuples(index=False):
        price = (
            by_series[row.price_series_id]
            if isinstance(row.price_series_id, str) and row.price_series_id in by_series
            else pd.Series(np.nan, index=dates)
        )
        nominal = (
            by_series[row.nominal_series_id]
            if row.nominal_series_id in by_series
            else pd.Series(np.nan, index=dates)
        )
        frame = pd.DataFrame(
            {"date": dates, "price_index": price.to_numpy(), "nominal_spending_musd": nominal.to_numpy()}
        )
        frame = frame.dropna(subset=["price_index", "nominal_spending_musd"], how="all")
        for column in PCE_DETAIL_COLUMNS:
            if column not in frame:
                frame[column] = getattr(row, column)
        frames.append(frame)
    detail = pd.concat(frames, ignore_index=True)
    detail["price_series_id"] = detail["price_series_id"].fillna("")
    return (
        detail.loc[:, PCE_DETAIL_COLUMNS]
        .sort_values(["table_line", "date"], kind="stable")
        .reset_index(drop=True)
    )


def build_pce_panel(
    detail: pd.DataFrame,
    series: Mapping[str, Mapping[str, str]] = US_PCE_PANEL_SERIES,
) -> pd.DataFrame:
    """Headline/core PCE aggregates in the common CPI panel schema."""
    country = COUNTRIES[PCE_COUNTRY_CODE]
    frames = []
    for series_id, metadata in series.items():
        selected = detail.loc[
            (detail["price_series_id"] == series_id) & detail["price_index"].notna(),
            ["date", "price_index"],
        ].drop_duplicates("date")
        if selected.empty:
            raise ValueError(f"BEA PCE data has no observations for {series_id}")
        frame = selected.rename(columns={"price_index": "value"})
        frame["country"] = country["name"]
        frame["country_code"] = PCE_COUNTRY_CODE
        frame["region"] = country["region"]
        frame["series_type"] = metadata["series_type"]
        frame["series_role"] = series_role(
            PCE_COUNTRY_CODE, metadata["series_type"], metadata["component"]
        )
        frame["component"] = metadata["component"]
        frame["source"] = "BEA"
        frame["source_dataset"] = "NIPA Table 2.4.4U"
        frame["source_series_id"] = series_id
        frame["frequency"] = "monthly"
        frame["unit"] = "index"
        frame["index_base"] = "2017=100"
        frame["seasonal_adjustment"] = "seasonally_adjusted"
        frame["observation_status"] = "not_provided"
        frames.append(frame)
    panel = (
        pd.concat(frames, ignore_index=True)
        .loc[:, CPI_SCHEMA_COLUMNS]
        .sort_values(["date", "series_type", "component"], kind="stable")
        .reset_index(drop=True)
    )
    validate_cpi_schema(panel)
    return panel


def build_pce_supercore(detail: pd.DataFrame) -> pd.DataFrame:
    """The derived supercore index in the common CPI panel schema."""
    tree_rows = detail.loc[detail["additive_tree"]]
    tree = PceTree(tree_rows)
    lines = list(tree.parent)
    prices = tree_rows.pivot(index="date", columns="table_line", values="price_index")
    nominal = tree_rows.pivot(
        index="date", columns="table_line", values="nominal_spending_musd"
    )
    calendar = pd.date_range(prices.index.min(), prices.index.max(), freq="MS")
    prices = prices.reindex(index=calendar, columns=lines)
    nominal = nominal.reindex(index=calendar, columns=lines)
    (services,) = tree.lines_for_nominal_series([SUPERCORE_SCOPE_NOMINAL_SERIES])
    excluded = tree.lines_for_nominal_series(SUPERCORE_EXCLUDED_NOMINAL_SERIES)
    if len(excluded) != len(SUPERCORE_EXCLUDED_NOMINAL_SERIES):
        raise ValueError("PCE tree is missing the housing or electricity-and-gas line")
    index = chained_fisher_index(
        tree, prices, nominal, start=services, excluded_lines=excluded
    )
    country = COUNTRIES[PCE_COUNTRY_CODE]
    frame = pd.DataFrame({"date": index.index, "value": index.to_numpy()})
    frame["country"] = country["name"]
    frame["country_code"] = PCE_COUNTRY_CODE
    frame["region"] = country["region"]
    frame["series_type"] = "core"
    frame["series_role"] = series_role(PCE_COUNTRY_CODE, "core", SUPERCORE_COMPONENT)
    frame["component"] = SUPERCORE_COMPONENT
    frame["source"] = "BEA"
    frame["source_dataset"] = "Derived chained Fisher from NIPA Tables 2.4.4U/2.4.5U"
    frame["source_series_id"] = SUPERCORE_SERIES_ID
    frame["frequency"] = "monthly"
    frame["unit"] = "index"
    frame["index_base"] = "2017=100"
    frame["seasonal_adjustment"] = "seasonally_adjusted"
    frame["observation_status"] = "not_provided"
    frame = frame.loc[:, CPI_SCHEMA_COLUMNS]
    validate_cpi_schema(frame)
    return frame


def fetch_us_pce(
    *, session: requests.Session | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download BEA's register and monthly NIPA file; return (detail, panel)."""
    register_text, _ = download_text(BEA_SERIES_REGISTER_URL, session=session)
    data_text, last_modified = download_text(BEA_NIPA_MONTHLY_URL, session=session)
    structure = build_pce_structure(parse_series_register(register_text))
    series_ids = set(structure["nominal_series_id"]) | set(
        structure["price_series_id"].dropna()
    )
    detail = build_pce_detail(structure, parse_nipa_monthly(data_text, series_ids))
    panel = (
        pd.concat([build_pce_panel(detail), build_pce_supercore(detail)], ignore_index=True)
        .sort_values(["date", "series_type", "component"], kind="stable")
        .reset_index(drop=True)
    )
    validate_cpi_schema(panel)
    if last_modified:
        panel.attrs["source_updated_at"] = parsedate_to_datetime(last_modified).isoformat()
    return detail, panel


def persist_us_pce(
    detail: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    detail_path: str | Path = BEA_US_PCE_PATH,
    panel_path: str | Path = BEA_US_PCE_CPI_PATH,
) -> Path:
    atomic_write_parquet(detail, detail_path, columns=PCE_DETAIL_COLUMNS)
    validate_cpi_schema(panel)
    destination = atomic_write_parquet(panel, panel_path, columns=CPI_SCHEMA_COLUMNS)
    update_ingestion_manifest(
        BEA_DATASET_ID,
        panel,
        destination,
        source_updated_at=panel.attrs.get("source_updated_at"),
    )
    return destination


def run(
    detail_path: str | Path = BEA_US_PCE_PATH,
    panel_path: str | Path = BEA_US_PCE_CPI_PATH,
) -> Path:
    detail, panel = fetch_us_pce()
    return persist_us_pce(detail, panel, detail_path=detail_path, panel_path=panel_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch US PCE price indexes and nominal spending from BEA."
    )
    parser.add_argument("--detail-output", type=Path, default=BEA_US_PCE_PATH)
    parser.add_argument("--panel-output", type=Path, default=BEA_US_PCE_CPI_PATH)
    args = parser.parse_args()
    destination = run(args.detail_output, args.panel_output)
    print(f"Saved US PCE detail to {args.detail_output} and panel series to {destination}")


if __name__ == "__main__":
    main()
