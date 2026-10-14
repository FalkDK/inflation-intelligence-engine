"""Focused five-bucket inflation decomposition for regime research."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from inflation_engine.data.fetch_bls_weights import (
    BUCKETS,
    validate_weights,
)
from inflation_engine.data.fetch_ssb_weights import CONTRACT_COLUMNS
from inflation_engine.utils.io import atomic_write_parquet
from inflation_engine.utils.paths import (
    BLS_US_CPI_WEIGHTS_PATH,
    CPI_PANEL_PATH,
    EUROSTAT_HICP_WEIGHTS_PATH,
    INFLATION_BUCKET_DECOMPOSITION_PATH,
    INFLATION_DECOMPOSITION_SUMMARY_PATH,
    ONS_UK_CPI_WEIGHTS_PATH,
    SSB_NORWAY_COMPONENT_CONTRACT_PATH,
    SSB_NORWAY_CPI_WEIGHTS_PATH,
)


DIRECT_BUCKET_COMPONENTS = {
    "food": "food",
    "energy": "energy",
    "shelter": "shelter",
    "core_goods": "commodities_less_food_and_energy_commodities",
}
RESIDUAL_BUCKET = "core_services_ex_housing"
HICP_COMPONENTS = {
    "food": "food_including_alcohol_tobacco",
    "energy": "energy",
    "shelter": "actual_rentals_for_housing",
    "core_goods": "non_energy_industrial_goods",
    "services": "services",
}
PRESSURE_GROUPS = {
    "food": "supply_sensitive",
    "energy": "supply_sensitive",
    "shelter": "cyclical",
    "core_goods": "supply_sensitive",
    "core_services_ex_housing": "cyclical",
}
BUCKET_OUTPUT_COLUMNS = [
    "date",
    "country",
    "country_code",
    "bucket",
    "pressure_group",
    "weight_pct",
    "inflation_yoy",
    "inflation_yoy_change_3m",
    "estimated_contribution_yoy_pp",
    "absolute_contribution_share",
    "above_2pct",
    "above_3pct",
    "above_4pct",
    "accelerating",
    "weight_reference_date",
    "weight_basis_year",
    "contribution_method",
]
SUMMARY_OUTPUT_COLUMNS = [
    "date",
    "country",
    "country_code",
    "headline_yoy",
    "estimated_contribution_sum_pp",
    "weighted_breadth_accelerating_pct",
    "weighted_share_above_2pct",
    "weighted_share_above_3pct",
    "weighted_share_above_4pct",
    "core_services_ex_housing_yoy",
    "core_goods_yoy",
    "services_goods_pressure",
    "cyclical_inflation_yoy",
    "supply_sensitive_inflation_yoy",
    "cyclical_supply_pressure",
    "contribution_concentration_hhi",
    "largest_absolute_contribution_share",
    "dominant_bucket",
    "feature_complete",
]


def _require_cpi_columns(frame: pd.DataFrame) -> None:
    required = [
        "date",
        "country",
        "country_code",
        "series_role",
        "component",
        "inflation_yoy",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Processed CPI panel is missing columns: {missing}")


def _applicable_weights(weights: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    applicable = weights.loc[
        (weights["country_code"] == "US")
        & (weights["effective_start"] <= date)
        & (weights["effective_end"] >= date)
    ]
    if applicable.empty:
        return applicable
    if len(applicable) != len(BUCKETS):
        raise ValueError(f"Incomplete CPI weights for {date:%Y-%m}")
    return applicable


def _calendar_change(frame: pd.DataFrame, periods: int = 3) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    group_columns = (
        ["country_code", "bucket"] if "country_code" in frame else ["bucket"]
    )
    for _, positions in frame.groupby(group_columns, sort=False).groups.items():
        bucket = frame.loc[positions].sort_values("date", kind="stable")
        calendar = pd.date_range(bucket["date"].min(), bucket["date"].max(), freq="MS")
        values = bucket.set_index("date")["inflation_yoy"].reindex(calendar)
        change = values - values.shift(periods)
        result.loc[bucket.index] = change.reindex(bucket["date"]).to_numpy()
    return result


def build_bucket_decomposition(
    cpi: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Build exhaustive US CPI bucket rates and estimated YoY contributions."""
    _require_cpi_columns(cpi)
    validate_weights(weights)
    us = cpi.loc[cpi["country_code"] == "US"].copy()
    if us.empty:
        raise ValueError("Processed CPI panel contains no US observations")

    required_components = {"all_items", *DIRECT_BUCKET_COMPONENTS.values()}
    selected = us.loc[us["component"].isin(required_components)]
    duplicate = selected.duplicated(["date", "component"], keep=False)
    if duplicate.any():
        raise ValueError("US CPI decomposition components contain duplicate dates")
    rates = selected.pivot(index="date", columns="component", values="inflation_yoy")
    country_name = str(us["country"].iloc[0])

    rows = []
    for date, date_rates in rates.sort_index().iterrows():
        date_weights = _applicable_weights(weights, date)
        if date_weights.empty:
            continue
        if not required_components.issubset(date_rates.dropna().index):
            continue
        weight_by_bucket = date_weights.set_index("bucket")["weight_pct"]
        direct_contributions = {
            bucket: weight_by_bucket[bucket] * date_rates[component] / 100.0
            for bucket, component in DIRECT_BUCKET_COMPONENTS.items()
        }
        headline_yoy = float(date_rates["all_items"])
        residual_contribution = headline_yoy - sum(direct_contributions.values())
        residual_rate = (
            residual_contribution
            / float(weight_by_bucket[RESIDUAL_BUCKET])
            * 100.0
        )
        bucket_rates = {
            bucket: float(date_rates[component])
            for bucket, component in DIRECT_BUCKET_COMPONENTS.items()
        }
        bucket_rates[RESIDUAL_BUCKET] = residual_rate
        contributions = {
            **direct_contributions,
            RESIDUAL_BUCKET: residual_contribution,
        }

        for bucket in BUCKETS:
            weight_row = date_weights.loc[date_weights["bucket"] == bucket].iloc[0]
            rows.append(
                {
                    "date": date,
                    "country": country_name,
                    "country_code": "US",
                    "bucket": bucket,
                    "pressure_group": PRESSURE_GROUPS[bucket],
                    "weight_pct": float(weight_row["weight_pct"]),
                    "inflation_yoy": bucket_rates[bucket],
                    "estimated_contribution_yoy_pp": contributions[bucket],
                    "weight_reference_date": weight_row["reference_date"],
                    "weight_basis_year": int(weight_row["weight_basis_year"]),
                    "contribution_method": (
                        "headline_accounting_residual"
                        if bucket == RESIDUAL_BUCKET
                        else "fixed_weight_rate_approximation"
                    ),
                }
            )

    if not rows:
        raise ValueError("No CPI dates have both applicable weights and complete rates")
    result = pd.DataFrame(rows)
    result["inflation_yoy_change_3m"] = _calendar_change(result)
    result["above_2pct"] = result["inflation_yoy"] > 2.0
    result["above_3pct"] = result["inflation_yoy"] > 3.0
    result["above_4pct"] = result["inflation_yoy"] > 4.0
    result["accelerating"] = result["inflation_yoy_change_3m"] > 0.0

    absolute = result["estimated_contribution_yoy_pp"].abs()
    totals = absolute.groupby(result["date"]).transform("sum")
    result["absolute_contribution_share"] = absolute.div(totals).where(totals > 0)

    result = result.sort_values(
        ["date", "bucket"],
        kind="stable",
    ).reset_index(drop=True)
    numeric = result[
        [
            "weight_pct",
            "inflation_yoy",
            "estimated_contribution_yoy_pp",
        ]
    ].stack()
    if not np.isfinite(numeric).all():
        raise ValueError("Inflation decomposition values must be finite")
    return result.loc[:, BUCKET_OUTPUT_COLUMNS]


def build_norway_bucket_decomposition(
    cpi: pd.DataFrame,
    weights: pd.DataFrame,
    contract: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate one non-overlapping SSB frontier into the five macro buckets."""
    _require_cpi_columns(cpi)
    validate_weights(weights)
    missing = [column for column in CONTRACT_COLUMNS if column not in contract]
    if missing:
        raise ValueError(f"Norway component contract is missing columns: {missing}")
    if contract["source_component_code"].duplicated().any():
        raise ValueError("Norway component contract contains duplicate source nodes")

    norway = cpi.loc[cpi["country_code"].eq("NO")].copy()
    if norway.empty:
        raise ValueError("Processed CPI panel contains no Norway observations")
    headline = (
        norway.loc[norway["component"].eq("all_items"), ["date", "inflation_yoy"]]
        .drop_duplicates("date")
        .set_index("date")["inflation_yoy"]
    )
    components = norway.loc[
        norway["source_series_id"].str.startswith("14700:", na=False),
        ["date", "source_series_id", "inflation_yoy"],
    ].copy()
    components["source_component_code"] = components["source_series_id"].str.removeprefix(
        "14700:"
    )
    rates = components.pivot(
        index="date",
        columns="source_component_code",
        values="inflation_yoy",
    )
    required_codes = set(contract["rate_source_component_code"])
    country_name = str(norway["country"].iloc[0])
    rows = []

    for date, date_rates in rates.sort_index().iterrows():
        date_contract = contract.loc[
            (contract["effective_start"] <= date)
            & (contract["effective_end"] >= date)
        ]
        date_weights = weights.loc[
            (weights["country_code"].eq("NO"))
            & (weights["effective_start"] <= date)
            & (weights["effective_end"] >= date)
        ]
        if date_contract.empty or date_weights.empty or date not in headline.index:
            continue
        if not required_codes.issubset(date_rates.dropna().index):
            continue
        nodes = date_contract.copy()
        nodes["inflation_yoy"] = nodes["rate_source_component_code"].map(date_rates)
        nodes["estimated_contribution_yoy_pp"] = (
            nodes["annual_effective_weight_pct"] * nodes["inflation_yoy"] / 100.0
        )
        grouped = nodes.groupby("canonical_macro_bucket").agg(
            weight_pct=("annual_effective_weight_pct", "sum"),
            estimated_contribution_yoy_pp=("estimated_contribution_yoy_pp", "sum"),
        )
        if set(grouped.index) != set(BUCKETS):
            raise ValueError(f"Norway decomposition buckets are incomplete for {date}")

        # Fixed-weight YoY contributions are approximate. Reconcile the small
        # chain-index residual to the broad residual services bucket.
        headline_yoy = float(headline.loc[date])
        reconciliation = headline_yoy - float(
            grouped["estimated_contribution_yoy_pp"].sum()
        )
        grouped.loc[
            RESIDUAL_BUCKET, "estimated_contribution_yoy_pp"
        ] += reconciliation
        grouped["inflation_yoy"] = (
            grouped["estimated_contribution_yoy_pp"] / grouped["weight_pct"] * 100.0
        )

        for bucket, values in grouped.iterrows():
            weight_row = date_weights.loc[date_weights["bucket"].eq(bucket)].iloc[0]
            rows.append(
                {
                    "date": date,
                    "country": country_name,
                    "country_code": "NO",
                    "bucket": bucket,
                    "pressure_group": PRESSURE_GROUPS[bucket],
                    "weight_pct": float(values["weight_pct"]),
                    "inflation_yoy": float(values["inflation_yoy"]),
                    "estimated_contribution_yoy_pp": float(
                        values["estimated_contribution_yoy_pp"]
                    ),
                    "weight_reference_date": weight_row["reference_date"],
                    "weight_basis_year": int(weight_row["weight_basis_year"]),
                    "contribution_method": (
                        "official_frontier_plus_headline_reconciliation"
                        if bucket == RESIDUAL_BUCKET
                        else "official_frontier_fixed_weight_approximation"
                    ),
                }
            )

    if not rows:
        raise ValueError("No Norway CPI dates have complete frontier rates and weights")
    result = pd.DataFrame(rows)
    result["inflation_yoy_change_3m"] = _calendar_change(result)
    result["above_2pct"] = result["inflation_yoy"] > 2.0
    result["above_3pct"] = result["inflation_yoy"] > 3.0
    result["above_4pct"] = result["inflation_yoy"] > 4.0
    result["accelerating"] = result["inflation_yoy_change_3m"] > 0.0
    absolute = result["estimated_contribution_yoy_pp"].abs()
    totals = absolute.groupby([result["country_code"], result["date"]]).transform(
        "sum"
    )
    result["absolute_contribution_share"] = absolute.div(totals).where(totals > 0)
    return result.sort_values(
        ["country_code", "date", "bucket"], kind="stable"
    ).loc[:, BUCKET_OUTPUT_COLUMNS].reset_index(drop=True)


def build_hicp_bucket_decomposition(
    cpi: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Build comparable five-bucket decompositions from HICP special aggregates."""
    _require_cpi_columns(cpi)
    validate_weights(weights)
    rows = []
    for country_code in sorted(weights["country_code"].unique()):
        country = cpi.loc[cpi["country_code"].eq(country_code)].copy()
        if country.empty:
            continue
        headline_component = (
            "hicp_all_items" if country_code in {"DK", "SE"} else "all_items"
        )
        required_components = {headline_component, *HICP_COMPONENTS.values()}
        selected = country.loc[
            country["component"].isin(required_components),
            ["date", "component", "inflation_yoy"],
        ]
        duplicate = selected.duplicated(["date", "component"], keep=False)
        if duplicate.any():
            raise ValueError(
                f"HICP decomposition components contain duplicate dates for "
                f"{country_code}"
            )
        rates = selected.pivot(
            index="date", columns="component", values="inflation_yoy"
        )
        country_name = str(country["country"].iloc[0])

        for date, date_rates in rates.sort_index().iterrows():
            date_weights = weights.loc[
                weights["country_code"].eq(country_code)
                & (weights["effective_start"] <= date)
                & (weights["effective_end"] >= date)
            ]
            if len(date_weights) != len(BUCKETS):
                continue
            if not required_components.issubset(date_rates.dropna().index):
                continue
            weight_by_bucket = date_weights.set_index("bucket")["weight_pct"]
            headline_yoy = float(date_rates[headline_component])
            direct_contributions = {
                bucket: float(weight_by_bucket[bucket])
                * float(date_rates[component])
                / 100.0
                for bucket, component in HICP_COMPONENTS.items()
                if bucket != "services"
            }
            services_contribution = (
                (
                    float(weight_by_bucket[RESIDUAL_BUCKET])
                    + float(weight_by_bucket["shelter"])
                )
                * float(date_rates[HICP_COMPONENTS["services"]])
                / 100.0
            )
            residual_contribution = (
                services_contribution - direct_contributions["shelter"]
            )
            contributions = {
                **direct_contributions,
                RESIDUAL_BUCKET: residual_contribution,
            }
            # HICP is chain-linked, so fixed annual weights times YoY rates are
            # an approximation. Put the small accounting residual into the
            # broad residual services bucket and retain exact headline closure.
            contributions[RESIDUAL_BUCKET] += headline_yoy - sum(
                contributions.values()
            )
            bucket_rates = {
                bucket: float(date_rates[component])
                for bucket, component in HICP_COMPONENTS.items()
                if bucket != "services"
            }
            bucket_rates[RESIDUAL_BUCKET] = (
                contributions[RESIDUAL_BUCKET]
                / float(weight_by_bucket[RESIDUAL_BUCKET])
                * 100.0
            )

            for bucket in BUCKETS:
                weight_row = date_weights.loc[
                    date_weights["bucket"].eq(bucket)
                ].iloc[0]
                rows.append(
                    {
                        "date": date,
                        "country": country_name,
                        "country_code": country_code,
                        "bucket": bucket,
                        "pressure_group": PRESSURE_GROUPS[bucket],
                        "weight_pct": float(weight_row["weight_pct"]),
                        "inflation_yoy": bucket_rates[bucket],
                        "estimated_contribution_yoy_pp": contributions[bucket],
                        "weight_reference_date": weight_row["reference_date"],
                        "weight_basis_year": int(weight_row["weight_basis_year"]),
                        "contribution_method": (
                            "special_aggregate_plus_headline_reconciliation"
                            if bucket == RESIDUAL_BUCKET
                            else "official_component_weight_rate_approximation"
                        ),
                    }
                )

    if not rows:
        raise ValueError("No HICP dates have complete rates and annual weights")
    result = pd.DataFrame(rows)
    result["inflation_yoy_change_3m"] = _calendar_change(result)
    result["above_2pct"] = result["inflation_yoy"] > 2.0
    result["above_3pct"] = result["inflation_yoy"] > 3.0
    result["above_4pct"] = result["inflation_yoy"] > 4.0
    result["accelerating"] = result["inflation_yoy_change_3m"] > 0.0
    absolute = result["estimated_contribution_yoy_pp"].abs()
    totals = absolute.groupby(
        [result["country_code"], result["date"]]
    ).transform("sum")
    result["absolute_contribution_share"] = absolute.div(totals).where(totals > 0)
    return result.sort_values(
        ["country_code", "date", "bucket"], kind="stable"
    ).loc[:, BUCKET_OUTPUT_COLUMNS].reset_index(drop=True)


def _weighted_rate(group: pd.DataFrame, pressure_group: str) -> float:
    selected = group.loc[group["pressure_group"] == pressure_group]
    return float(
        np.average(selected["inflation_yoy"], weights=selected["weight_pct"])
    )


def build_decomposition_summary(buckets: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in BUCKET_OUTPUT_COLUMNS if column not in buckets]
    if missing:
        raise ValueError(f"Bucket decomposition is missing columns: {missing}")

    rows = []
    for (_, date), group in buckets.groupby(["country_code", "date"], sort=True):
        by_bucket = group.set_index("bucket")
        total_weight = float(group["weight_pct"].sum())
        absolute_shares = group["absolute_contribution_share"].dropna()
        cyclical = _weighted_rate(group, "cyclical")
        supply = _weighted_rate(group, "supply_sensitive")
        services = float(by_bucket.loc[RESIDUAL_BUCKET, "inflation_yoy"])
        goods = float(by_bucket.loc["core_goods", "inflation_yoy"])
        rows.append(
            {
                "date": date,
                "country": group["country"].iloc[0],
                "country_code": group["country_code"].iloc[0],
                "headline_yoy": float(
                    group["estimated_contribution_yoy_pp"].sum()
                ),
                "estimated_contribution_sum_pp": float(
                    group["estimated_contribution_yoy_pp"].sum()
                ),
                "weighted_breadth_accelerating_pct": (
                    float(
                        group.loc[group["accelerating"], "weight_pct"].sum()
                        / total_weight
                        * 100.0
                    )
                    if group["inflation_yoy_change_3m"].notna().all()
                    else np.nan
                ),
                "weighted_share_above_2pct": float(
                    group.loc[group["above_2pct"], "weight_pct"].sum()
                    / total_weight
                    * 100.0
                ),
                "weighted_share_above_3pct": float(
                    group.loc[group["above_3pct"], "weight_pct"].sum()
                    / total_weight
                    * 100.0
                ),
                "weighted_share_above_4pct": float(
                    group.loc[group["above_4pct"], "weight_pct"].sum()
                    / total_weight
                    * 100.0
                ),
                "core_services_ex_housing_yoy": services,
                "core_goods_yoy": goods,
                "services_goods_pressure": services - goods,
                "cyclical_inflation_yoy": cyclical,
                "supply_sensitive_inflation_yoy": supply,
                "cyclical_supply_pressure": cyclical - supply,
                "contribution_concentration_hhi": (
                    float((absolute_shares**2).sum())
                    if not absolute_shares.empty
                    else np.nan
                ),
                "largest_absolute_contribution_share": (
                    float(absolute_shares.max())
                    if not absolute_shares.empty
                    else np.nan
                ),
                "dominant_bucket": str(
                    group.loc[
                        group["estimated_contribution_yoy_pp"].abs().idxmax(),
                        "bucket",
                    ]
                ),
                "feature_complete": bool(
                    group["inflation_yoy"].notna().all()
                    and group["inflation_yoy_change_3m"].notna().all()
                ),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_OUTPUT_COLUMNS)


def load_inputs(
    cpi_path: str | Path = CPI_PANEL_PATH,
    weights_path: str | Path = BLS_US_CPI_WEIGHTS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cpi_source = Path(cpi_path)
    weights_source = Path(weights_path)
    if not cpi_source.is_file():
        raise FileNotFoundError(f"Processed CPI panel does not exist: {cpi_source}")
    if not weights_source.is_file():
        raise FileNotFoundError(f"CPI weights do not exist: {weights_source}")
    return pd.read_parquet(cpi_source), pd.read_parquet(weights_source)


def save_decomposition(
    buckets: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    bucket_output: str | Path = INFLATION_BUCKET_DECOMPOSITION_PATH,
    summary_output: str | Path = INFLATION_DECOMPOSITION_SUMMARY_PATH,
) -> tuple[Path, Path]:
    bucket_path = atomic_write_parquet(
        buckets,
        bucket_output,
        columns=BUCKET_OUTPUT_COLUMNS,
    )
    summary_path = atomic_write_parquet(
        summary,
        summary_output,
        columns=SUMMARY_OUTPUT_COLUMNS,
    )
    return bucket_path, summary_path


def run(
    cpi_path: str | Path = CPI_PANEL_PATH,
    weights_path: str | Path = BLS_US_CPI_WEIGHTS_PATH,
    *,
    norway_weights_path: str | Path = SSB_NORWAY_CPI_WEIGHTS_PATH,
    norway_contract_path: str | Path = SSB_NORWAY_COMPONENT_CONTRACT_PATH,
    hicp_weights_path: str | Path = EUROSTAT_HICP_WEIGHTS_PATH,
    uk_weights_path: str | Path = ONS_UK_CPI_WEIGHTS_PATH,
    bucket_output: str | Path = INFLATION_BUCKET_DECOMPOSITION_PATH,
    summary_output: str | Path = INFLATION_DECOMPOSITION_SUMMARY_PATH,
) -> tuple[Path, Path]:
    cpi, weights = load_inputs(cpi_path, weights_path)
    bucket_frames = [build_bucket_decomposition(cpi, weights)]
    norway_weights_source = Path(norway_weights_path)
    norway_contract_source = Path(norway_contract_path)
    if norway_weights_source.is_file() and norway_contract_source.is_file():
        bucket_frames.append(
            build_norway_bucket_decomposition(
                cpi,
                pd.read_parquet(norway_weights_source),
                pd.read_parquet(norway_contract_source),
            )
        )
    comparable_weight_sources = [
        Path(path)
        for path in (hicp_weights_path, uk_weights_path)
        if Path(path).is_file()
    ]
    if comparable_weight_sources:
        bucket_frames.append(
            build_hicp_bucket_decomposition(
                cpi,
                pd.concat(
                    [
                        pd.read_parquet(source)
                        for source in comparable_weight_sources
                    ],
                    ignore_index=True,
                ),
            )
        )
    buckets = pd.concat(bucket_frames, ignore_index=True)
    summary = build_decomposition_summary(buckets)
    return save_decomposition(
        buckets,
        summary,
        bucket_output=bucket_output,
        summary_output=summary_output,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build focused inflation decompositions across model countries."
    )
    parser.add_argument("--cpi", type=Path, default=CPI_PANEL_PATH)
    parser.add_argument("--weights", type=Path, default=BLS_US_CPI_WEIGHTS_PATH)
    parser.add_argument(
        "--bucket-output",
        type=Path,
        default=INFLATION_BUCKET_DECOMPOSITION_PATH,
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=INFLATION_DECOMPOSITION_SUMMARY_PATH,
    )
    args = parser.parse_args()
    bucket_path, summary_path = run(
        args.cpi,
        args.weights,
        bucket_output=args.bucket_output,
        summary_output=args.summary_output,
    )
    print(f"Saved bucket decomposition to {bucket_path}")
    print(f"Saved decomposition summary to {summary_path}")


if __name__ == "__main__":
    main()
