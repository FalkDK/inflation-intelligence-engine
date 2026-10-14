"""Validated item-level contributions for Nordic CPI panels.

Norway is enabled from Statistics Norway's hierarchy-safe component contract.
Sweden is intentionally not approximated until its matching annual item-weight
table is ingested; five-bucket Eurostat weights are not item weights.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.paths import (
    SSB_NORWAY_COMPONENT_CONTRACT_PATH,
    NORDIC_ITEM_CONTRIBUTIONS_PATH,
    SCB_SWEDEN_CPI_WEIGHTS_PATH,
)


CONTRIBUTION_COLUMNS = [
    "date", "country", "country_code", "scope", "source_component_code",
    "description", "canonical_macro_bucket", "weight_current_pct",
    "weight_prior_pct", "current_index_value", "prior_index_value",
    "contribution_yoy_pp", "computed_scope_yoy", "official_scope_yoy",
    "official_reconciliation_yoy_pp", "is_reconciliation", "validation_status",
]


def _code(value: object) -> str:
    value = str(value).strip()
    if ":" in value:
        value = value.split(":", 1)[1]
    return value.split("_")[0]


def build_norway_item_contributions(
    raw_cpi: pd.DataFrame,
    contract_path: str | Path = SSB_NORWAY_COMPONENT_CONTRACT_PATH,
) -> pd.DataFrame:
    """Build headline/core Norway contributions using the official contract."""
    source = raw_cpi.loc[
        (raw_cpi["country_code"] == "NO")
        & (raw_cpi["series_role"] == "component")
    ].copy()
    source["date"] = pd.to_datetime(source["date"])
    source["source_component_code"] = source["source_series_id"].map(_code)
    source["index_value"] = pd.to_numeric(source["value"], errors="coerce")
    source = source.dropna(subset=["date", "index_value"])
    contract = pd.read_parquet(contract_path).copy()
    contract["source_component_code"] = contract["source_component_code"].map(_code)
    contract = contract.rename(columns={
        "source_component": "description",
        "annual_effective_weight_pct": "weight_pct",
    })
    available = set(source["source_component_code"])
    contract["match_code"] = contract["source_component_code"]
    contract.loc[~contract["match_code"].isin(available), "match_code"] = contract.loc[
        ~contract["match_code"].isin(available), "rate_source_component_code"
    ].map(_code)
    contract = contract.loc[contract["match_code"].isin(available)].copy()
    if contract.empty:
        raise ValueError("Norwegian contract has no matching component indexes")
    if not np.isclose(contract["weight_pct"].sum(), 100.0, atol=0.1):
        raise ValueError("Norwegian matched contract weights do not sum to 100")
    rows = []
    for date in sorted(source["date"].loc[source["date"].dt.year >= 2026].unique()):
        date = pd.Timestamp(date)
        prior_date = date - pd.DateOffset(years=1)
        current = source.loc[source["date"] == date, ["source_component_code", "index_value"]].rename(columns={"index_value": "current_index_value"})
        prior = source.loc[source["date"] == prior_date, ["source_component_code", "index_value"]].rename(columns={"index_value": "prior_index_value"})
        items = contract[["source_component_code", "match_code", "description", "canonical_macro_bucket", "weight_pct"]].rename(columns={"weight_pct": "weight_current_pct"})
        items = items.merge(items[["source_component_code", "weight_current_pct"]].rename(columns={"weight_current_pct": "weight_prior_pct"}), on="source_component_code")
        items = items.merge(current.rename(columns={"source_component_code": "match_code"}), on="match_code", how="left").merge(prior.rename(columns={"source_component_code": "match_code"}), on="match_code", how="left").dropna(subset=["current_index_value", "prior_index_value"])
        if items.empty:
            continue
        official = raw_cpi.loc[
            (raw_cpi["country_code"] == "NO")
            & (raw_cpi["source_series_id"].astype(str).isin(["KpiIndMnd", "KPI-JAE:KPIJustIndMnd"]))
            & (pd.to_datetime(raw_cpi["date"]).isin([date, prior_date]))
        ].copy()
        official["date"] = pd.to_datetime(official["date"])
        official["value"] = pd.to_numeric(official["value"], errors="coerce")
        for scope, excluded in (("headline", set()), ("core", {"food", "energy"})):
            scoped = items.loc[~items["canonical_macro_bucket"].isin(excluded)].copy()
            current_weight = scoped["weight_current_pct"].sum()
            prior_weight = scoped["weight_prior_pct"].sum()
            current_agg = (scoped["weight_current_pct"] * scoped["current_index_value"]).sum() / current_weight
            prior_agg = (scoped["weight_prior_pct"] * scoped["prior_index_value"]).sum() / prior_weight
            scoped["contribution_yoy_pp"] = ((scoped["weight_current_pct"] * scoped["current_index_value"] / current_weight) - (scoped["weight_prior_pct"] * scoped["prior_index_value"] / prior_weight)) / prior_agg * 100.0
            code = "KpiIndMnd" if scope == "headline" else "KPI-JAE:KPIJustIndMnd"
            published = official.loc[official["source_series_id"].eq(code)].set_index("date")["value"]
            if date not in published.index or prior_date not in published.index:
                continue
            official_yoy = (published.loc[date] / published.loc[prior_date] - 1.0) * 100.0
            computed_yoy = (current_agg / prior_agg - 1.0) * 100.0
            scoped["date"] = date; scoped["country"] = "Norway"; scoped["country_code"] = "NO"; scoped["scope"] = scope
            status = "validated_contract" if abs(official_yoy - computed_yoy) <= 0.10 else "review_mapping"
            scoped["computed_scope_yoy"] = computed_yoy; scoped["official_scope_yoy"] = official_yoy; scoped["official_reconciliation_yoy_pp"] = np.nan; scoped["is_reconciliation"] = False; scoped["validation_status"] = status
            rows.extend(scoped.loc[:, CONTRIBUTION_COLUMNS].to_dict("records"))
            rows.append({"date": date, "country": "Norway", "country_code": "NO", "scope": scope, "source_component_code": "__official_reconciliation__", "description": "Official aggregate minus reconstructed contract contributions", "canonical_macro_bucket": "reconciliation", "weight_current_pct": 0.0, "weight_prior_pct": 0.0, "current_index_value": np.nan, "prior_index_value": np.nan, "contribution_yoy_pp": official_yoy - computed_yoy, "computed_scope_yoy": computed_yoy, "official_scope_yoy": official_yoy, "official_reconciliation_yoy_pp": official_yoy - computed_yoy, "is_reconciliation": True, "validation_status": status})
    return pd.DataFrame(rows, columns=CONTRIBUTION_COLUMNS).sort_values(["date", "scope", "is_reconciliation", "source_component_code"]).reset_index(drop=True)


def save_norway_item_contributions(frame: pd.DataFrame, output_path: str | Path = NORDIC_ITEM_CONTRIBUTIONS_PATH) -> Path:
    return atomic_write_parquet(frame, output_path, columns=CONTRIBUTION_COLUMNS)


def build_sweden_item_contributions(
    raw_cpi: pd.DataFrame,
    weights_path: str | Path = SCB_SWEDEN_CPI_WEIGHTS_PATH,
) -> pd.DataFrame:
    """Build Sweden CPIF contributions from SCB's published item weights."""
    source = raw_cpi.loc[(raw_cpi["country_code"] == "SE") & (raw_cpi["series_role"] == "component")].copy()
    source["date"] = pd.to_datetime(source["date"])
    source["source_component_code"] = source["source_series_id"].astype(str).str.split(":").str[-1].str.replace(".", "", regex=False)
    source["index_value"] = pd.to_numeric(source["value"], errors="coerce")
    source = source.dropna(subset=["date", "index_value"])
    weights = pd.read_parquet(weights_path).copy()
    weights["coicop_code"] = weights["coicop_code"].astype(str).str.replace(".", "", regex=False)
    weights["weight_pct"] = pd.to_numeric(weights["weight_pct"], errors="coerce") / 10.0
    weights = weights.loc[weights["coicop_code"].ne("00")].copy()
    rows = []
    for date in sorted(source.loc[source["date"].dt.year >= 2026, "date"].unique()):
        date = pd.Timestamp(date); prior_date = date - pd.DateOffset(years=1)
        if not ((source["date"] == date).any() and (source["date"] == prior_date).any()):
            continue
        available = set(source["source_component_code"])
        items = weights.loc[weights["date"].isin([pd.Timestamp(f"{date.year}-01-01"), pd.Timestamp(f"{prior_date.year}-01-01")])].pivot_table(index="coicop_code", columns="date", values="weight_pct", aggfunc="first").reset_index()
        items.columns.name = None
        items = items.rename(columns={pd.Timestamp(f"{date.year}-01-01"): "weight_current_pct", pd.Timestamp(f"{prior_date.year}-01-01"): "weight_prior_pct"})
        items = items.loc[items["coicop_code"].isin(available)].copy()
        frontier = items.loc[~items["coicop_code"].map(lambda c: any(x.startswith(c) and len(x) > len(c) for x in set(items["coicop_code"])))].copy()
        current = source.loc[source["date"] == date, ["source_component_code", "index_value"]].rename(columns={"index_value": "current_index_value"})
        prior = source.loc[source["date"] == prior_date, ["source_component_code", "index_value"]].rename(columns={"index_value": "prior_index_value"})
        frontier = frontier.merge(current, left_on="coicop_code", right_on="source_component_code", how="inner").merge(prior, on="source_component_code", how="inner")
        for scope, excluded in (("headline", set()), ("core", {"045", "0722"})):
            scoped = frontier.loc[~frontier["coicop_code"].str.startswith(tuple(excluded))].copy() if excluded else frontier.copy()
            scoped["weight_current_pct"] = scoped["weight_current_pct"].fillna(0.0); scoped["weight_prior_pct"] = scoped["weight_prior_pct"].fillna(0.0)
            cw, pw = scoped["weight_current_pct"].sum(), scoped["weight_prior_pct"].sum()
            if cw < 80 or pw < 80: continue
            ca = (scoped.weight_current_pct * scoped.current_index_value).sum() / cw; pa = (scoped.weight_prior_pct * scoped.prior_index_value).sum() / pw
            scoped["contribution_yoy_pp"] = ((scoped.weight_current_pct * scoped.current_index_value / cw) - (scoped.weight_prior_pct * scoped.prior_index_value / pw)) / pa * 100
            code = "000007ZN" if scope == "headline" else "000007ZW"
            official = raw_cpi.loc[(raw_cpi.country_code == "SE") & (raw_cpi.source_series_id == code)].copy(); official["date"] = pd.to_datetime(official.date); official["value"] = pd.to_numeric(official.value, errors="coerce"); pub = official.set_index("date").value
            if date not in pub.index or prior_date not in pub.index: continue
            oy = (pub.loc[date] / pub.loc[prior_date] - 1) * 100; cy = (ca / pa - 1) * 100
            status = "validated_scb_weights" if abs(oy - cy) <= 0.10 else "review_mapping"
            scoped["date"] = date; scoped["country"] = "Sweden"; scoped["country_code"] = "SE"; scoped["scope"] = scope; scoped["description"] = scoped["coicop_code"]; scoped["canonical_macro_bucket"] = "item"; scoped["computed_scope_yoy"] = cy; scoped["official_scope_yoy"] = oy; scoped["official_reconciliation_yoy_pp"] = np.nan; scoped["is_reconciliation"] = False; scoped["validation_status"] = status
            rows.extend(scoped.loc[:, CONTRIBUTION_COLUMNS].to_dict("records"))
            rows.append({"date": date, "country": "Sweden", "country_code": "SE", "scope": scope, "source_component_code": "__official_reconciliation__", "description": "Official aggregate minus reconstructed SCB-weighted contributions", "canonical_macro_bucket": "reconciliation", "weight_current_pct": 0.0, "weight_prior_pct": 0.0, "current_index_value": np.nan, "prior_index_value": np.nan, "contribution_yoy_pp": oy - cy, "computed_scope_yoy": cy, "official_scope_yoy": oy, "official_reconciliation_yoy_pp": oy - cy, "is_reconciliation": True, "validation_status": status})
    return pd.DataFrame(rows, columns=CONTRIBUTION_COLUMNS).sort_values(["date", "scope", "is_reconciliation", "source_component_code"]).reset_index(drop=True)
