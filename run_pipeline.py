import argparse
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inflation_engine.data.combine_sources import (  # noqa: E402
    RAW_CPI_SOURCES,
    combine_cpi_sources,
    save_combined_panel,
)
from inflation_engine.data.fetch_bea_pce import run as fetch_bea_pce  # noqa: E402
from inflation_engine.data.fetch_eurostat import run as fetch_eurostat  # noqa: E402
from inflation_engine.data.fetch_eurostat_items import (  # noqa: E402
    run as fetch_eurostat_items,
)
from inflation_engine.data.fetch_fred import run as fetch_fred  # noqa: E402
from inflation_engine.data.fetch_ons import run as fetch_ons  # noqa: E402
from inflation_engine.data.fetch_release_calendar import (  # noqa: E402
    run as fetch_release_calendar,
)
from inflation_engine.data.fetch_ssb import run as fetch_ssb  # noqa: E402
from inflation_engine.data.fetch_statbank_dk import (  # noqa: E402
    run as fetch_statbank_dk,
)
from inflation_engine.data.fetch_sweden import run as fetch_sweden  # noqa: E402
from inflation_engine.features.inflation_metrics import (  # noqa: E402
    SEASONAL_ADJUSTMENT_OPTIONS,
    calculate_inflation_metrics,
    save_cpi_panel,
)
from inflation_engine.features.danish_item_decomposition import (  # noqa: E402
    build_danish_aggregation_validation,
    build_danish_item_contributions,
    build_danish_item_decomposition,
    save_danish_aggregation_validation,
    save_danish_item_contributions,
    save_danish_item_decomposition,
)
from inflation_engine.features.eurozone_item_decomposition import (  # noqa: E402
    build_eurozone_item_decomposition,
    save_eurozone_item_decomposition,
)
from inflation_engine.features.nordic_item_contributions import (  # noqa: E402
    build_norway_item_contributions,
    build_sweden_item_contributions,
    save_norway_item_contributions,
)
from inflation_engine.features.norway_item_decomposition import (  # noqa: E402
    build_norway_item_decomposition,
    save_norway_item_decomposition,
)
from inflation_engine.features.uk_item_decomposition import (  # noqa: E402
    build_uk_item_decomposition,
    save_uk_item_decomposition,
)
from inflation_engine.features.us_item_decomposition import (  # noqa: E402
    build_us_item_decomposition,
    save_us_item_decomposition,
)
from inflation_engine.features.us_pce_item_decomposition import (  # noqa: E402
    build_us_pce_item_decomposition,
    save_us_pce_item_decomposition,
)
from inflation_engine.features.breadth import (  # noqa: E402
    build_bucket_breadth,
    build_item_breadth,
    save_breadth,
)
from inflation_engine.features.availability import (  # noqa: E402
    load_release_calendar,
)
from inflation_engine.features.regime_features import (  # noqa: E402
    build_regime_features,
    save_regime_features,
)
from inflation_engine.reports.charts import generate_charts  # noqa: E402
from inflation_engine.reports.generate_report import (  # noqa: E402
    generate_report,
)
from inflation_engine.utils.paths import (  # noqa: E402
    BEA_US_PCE_PATH,
    BLS_US_CPI_WEIGHTS_PATH,
    CPI_3M3M_CHART_PATH,
    CPI_PANEL_PATH,
    CPI_PANEL_RAW_COMBINED_PATH,
    CPI_YOY_CHART_PATH,
    EUROSTAT_HICP_WEIGHTS_PATH,
    INFLATION_REGIME_FEATURES_PATH,
    ONS_UK_CPI_WEIGHTS_PATH,
    SSB_NORWAY_COMPONENT_CONTRACT_PATH,
    SSB_NORWAY_CPI_WEIGHTS_PATH,
    DANISH_ITEM_DECOMPOSITION_PATH,
    DANISH_ITEM_CONTRIBUTIONS_PATH,
    DANISH_CPI_WEIGHTS_2026_PATH,
    EUROSTAT_HICP_ITEM_WEIGHTS_PATH,
    EUROZONE_ITEM_DECOMPOSITION_PATH,
    NORDIC_ITEM_CONTRIBUTIONS_PATH,
    NORWAY_ITEM_DECOMPOSITION_PATH,
    SSB_NORWAY_FULL_WEIGHTS_PATH,
    UK_ITEM_DECOMPOSITION_PATH,
    US_ITEM_DECOMPOSITION_PATH,
    US_PCE_ITEM_DECOMPOSITION_PATH,
    LATEST_INFLATION_REPORT_PATH,
)


RawFetcher = Callable[[], Path]


def raw_fetchers(
    *,
    full_refresh_incremental_sources: bool = False,
) -> dict[str, RawFetcher]:
    return {
        "fred_us_cpi": fetch_fred,
        "ons_uk_cpi": fetch_ons,
        "eurostat_hicp": fetch_eurostat,
        "eurostat_hicp_items": fetch_eurostat_items,
        "statbank_dk_cpi": lambda: fetch_statbank_dk(
            full_refresh=full_refresh_incremental_sources,
        ),
        "scb_sweden_cpi": lambda: fetch_sweden(
            full_refresh=full_refresh_incremental_sources,
        ),
        "ssb_norway_cpi": lambda: fetch_ssb(
            full_refresh=full_refresh_incremental_sources,
        ),
        "bea_us_pce": fetch_bea_pce,
        "release_calendar": fetch_release_calendar,
    }


def fetch_raw_sources(
    fetchers: Mapping[str, RawFetcher] | None = None,
    *,
    full_refresh_incremental_sources: bool = False,
) -> dict[str, Path]:
    selected_fetchers = fetchers or raw_fetchers(
        full_refresh_incremental_sources=full_refresh_incremental_sources,
    )
    outputs = {}
    for name, fetcher in selected_fetchers.items():
        print(f"Fetching {name}...")
        outputs[name] = fetcher()
    return outputs


def run_pipeline(
    *,
    source_paths: Mapping[str, str | Path] = RAW_CPI_SOURCES,
    fetch_raw: bool = False,
    fetchers: Mapping[str, RawFetcher] | None = None,
    full_refresh_incremental_sources: bool = False,
    combined_output: str | Path = CPI_PANEL_RAW_COMBINED_PATH,
    panel_output: str | Path = CPI_PANEL_PATH,
    regime_features_output: str | Path = INFLATION_REGIME_FEATURES_PATH,
    yoy_chart_output: str | Path = CPI_YOY_CHART_PATH,
    momentum_chart_output: str | Path = CPI_3M3M_CHART_PATH,
    report_output: str | Path = LATEST_INFLATION_REPORT_PATH,
    seasonal_adjustment: str = "source",
    x13_path: str | Path | None = None,
    chart_years: int = 5,
) -> dict[str, Path]:
    outputs = {}
    if fetch_raw:
        outputs.update(
            {
                f"raw_{name}": path
                for name, path in fetch_raw_sources(
                    fetchers,
                    full_refresh_incremental_sources=(
                        full_refresh_incremental_sources
                    ),
                ).items()
            }
        )

    combined = combine_cpi_sources(source_paths)
    combined_path = save_combined_panel(combined, combined_output)

    metrics = calculate_inflation_metrics(
        combined,
        seasonal_adjustment=seasonal_adjustment,
        x13_path=x13_path,
    )
    panel_path = save_cpi_panel(metrics, panel_output)
    # Additive diagnostic layer: this does not feed the canonical bucket
    # decomposition or regime features. It is only applicable when Denmark is
    # present in the requested source panel (important for focused test runs).
    danish_item_path = None
    danish_validation_path = None
    danish_contributions_path = None
    norway_contributions_path = None
    sweden_contributions_path = None
    nordic_contribution_frames = []
    if "DK" in set(combined["country_code"].dropna().unique()):
        danish_item_path = save_danish_item_decomposition(
            build_danish_item_decomposition(
                combined,
                DANISH_CPI_WEIGHTS_2026_PATH,
            ),
            DANISH_ITEM_DECOMPOSITION_PATH,
        )
        danish_validation_path = save_danish_aggregation_validation(
            build_danish_aggregation_validation(combined),
        )
        danish_contributions_path = save_danish_item_contributions(
            build_danish_item_contributions(combined),
            DANISH_ITEM_CONTRIBUTIONS_PATH,
        )
    norway_item_path = None
    norway_item_frame = None
    if "NO" in set(combined["country_code"].dropna().unique()):
        nordic_contribution_frames.append(build_norway_item_contributions(combined))
        norway_contributions_path = NORDIC_ITEM_CONTRIBUTIONS_PATH
        # Additive, more granular alternative to the contract above (which is
        # built for five-bucket classification, not maximum item detail);
        # needs the full SSB weight table fetched separately (see
        # fetch_ssb_weights.py's run()), so skip gracefully if absent.
        if SSB_NORWAY_FULL_WEIGHTS_PATH.is_file():
            norway_full_weights = pd.read_parquet(SSB_NORWAY_FULL_WEIGHTS_PATH)
            norway_item_frame = build_norway_item_decomposition(
                combined, norway_full_weights
            )
            norway_item_path = save_norway_item_decomposition(
                norway_item_frame, NORWAY_ITEM_DECOMPOSITION_PATH
            )
    if "SE" in set(combined["country_code"].dropna().unique()):
        nordic_contribution_frames.append(build_sweden_item_contributions(combined))
        sweden_contributions_path = NORDIC_ITEM_CONTRIBUTIONS_PATH
    if nordic_contribution_frames:
        save_norway_item_contributions(pd.concat(nordic_contribution_frames, ignore_index=True))
    uk_item_path = None
    uk_item_frame = None
    if "GB" in set(combined["country_code"].dropna().unique()):
        uk_item_frame = build_uk_item_decomposition(combined)
        uk_item_path = save_uk_item_decomposition(
            uk_item_frame,
            UK_ITEM_DECOMPOSITION_PATH,
        )
    eurozone_item_path = None
    eurozone_item_frame = None
    # Additive alternative to decomposition.py's five coarse harmonised
    # buckets, covering the four euro-area members with a detailed COICOP
    # item map fetched separately (see fetch_eurostat_item_weights.py's
    # run()); skip gracefully if that prerequisite data is missing.
    if EUROSTAT_HICP_ITEM_WEIGHTS_PATH.is_file():
        eurozone_item_weights = pd.read_parquet(EUROSTAT_HICP_ITEM_WEIGHTS_PATH)
        if set(eurozone_item_weights["country_code"]) & set(
            combined["country_code"].dropna().unique()
        ):
            eurozone_item_frame = build_eurozone_item_decomposition(
                combined, eurozone_item_weights
            )
            eurozone_item_path = save_eurozone_item_decomposition(
                eurozone_item_frame, EUROZONE_ITEM_DECOMPOSITION_PATH
            )
    us_item_frame = build_us_item_decomposition(combined)
    us_item_path = None
    # A minimal/focused source panel (see test_pipeline.py) only carries the
    # headline/core aggregates, so none of the detailed FRED item series this
    # depends on are present; skip saving rather than overwrite the real
    # output with an empty result (matching the DK/GB/NO/SE presence guards
    # above, which never trigger this because those tests omit those
    # countries entirely).
    if not us_item_frame.loc[~us_item_frame["reconciliation_row"]].empty:
        us_item_path = save_us_item_decomposition(
            us_item_frame, US_ITEM_DECOMPOSITION_PATH
        )
    # Separate from US CPI: reads only BEA's PCE detail file (written by
    # fetch_bea_pce.py alongside the US_PCE panel series), so skip
    # gracefully when a focused source panel carries no US_PCE rows.
    us_pce_item_path = None
    us_pce_item_frame = None
    us_pce_detail = None
    if "US_PCE" in set(combined["country_code"].dropna().unique()) and (
        BEA_US_PCE_PATH.is_file()
    ):
        us_pce_detail = pd.read_parquet(BEA_US_PCE_PATH)
        us_pce_item_frame = build_us_pce_item_decomposition(us_pce_detail)
        us_pce_item_path = save_us_pce_item_decomposition(
            us_pce_item_frame,
            US_PCE_ITEM_DECOMPOSITION_PATH,
        )
    # Breadth needs the same weight files as decomposition.py's five-bucket
    # build, which are fetched separately (see docs/decomposition.md) and are
    # not guaranteed to exist for a minimal/focused source panel; skip rather
    # than fail the whole pipeline when that prerequisite data is missing.
    breadth_bucket_path = None
    breadth_item_path = None
    try:
        bls_weights = pd.read_parquet(BLS_US_CPI_WEIGHTS_PATH)
        comparable_sources = [
            path
            for path in (EUROSTAT_HICP_WEIGHTS_PATH, ONS_UK_CPI_WEIGHTS_PATH)
            if path.is_file()
        ]
        no_weights_path, no_contract_path = (
            SSB_NORWAY_CPI_WEIGHTS_PATH,
            SSB_NORWAY_COMPONENT_CONTRACT_PATH,
        )
        breadth_buckets = build_bucket_breadth(
            metrics,
            bls_weights=bls_weights,
            no_weights=(
                pd.read_parquet(no_weights_path) if no_weights_path.is_file() else None
            ),
            no_contract=(
                pd.read_parquet(no_contract_path) if no_contract_path.is_file() else None
            ),
            comparable_weights=(
                pd.concat(
                    [pd.read_parquet(source) for source in comparable_sources],
                    ignore_index=True,
                )
                if comparable_sources else None
            ),
        )
        breadth_items = build_item_breadth(
            us_item_decomposition=us_item_frame,
            uk_item_decomposition=uk_item_frame,
            danish_item_decomposition=(
                pd.read_parquet(danish_item_path) if danish_item_path else None
            ),
            norway_item_decomposition=norway_item_frame,
            nordic_item_contributions=(
                pd.concat(nordic_contribution_frames, ignore_index=True)
                if nordic_contribution_frames else None
            ),
            eurozone_item_decomposition=eurozone_item_frame,
            us_pce_item_decomposition=us_pce_item_frame,
            us_pce_detail=us_pce_detail,
        )
        breadth_bucket_path, breadth_item_path = save_breadth(
            breadth_buckets, breadth_items
        )
    except (ValueError, FileNotFoundError):
        pass
    regime_features = build_regime_features(
        metrics,
        load_release_calendar(),
    )
    regime_features_path = save_regime_features(
        regime_features,
        regime_features_output,
    )
    yoy_path, momentum_path = generate_charts(
        metrics,
        yoy_path=yoy_chart_output,
        momentum_path=momentum_chart_output,
        years=chart_years,
    )
    report_path = generate_report(metrics, report_output)

    outputs.update(
        {
            "combined_panel": combined_path,
            "processed_panel": panel_path,
            "regime_features": regime_features_path,
            "yoy_chart": yoy_path,
            "momentum_chart": momentum_path,
            "report": report_path,
        }
    )
    if danish_item_path is not None:
        outputs["danish_item_decomposition"] = danish_item_path
    if danish_validation_path is not None:
        outputs["danish_item_aggregation_validation"] = danish_validation_path
    if danish_contributions_path is not None:
        outputs["danish_item_contributions"] = danish_contributions_path
    if norway_contributions_path is not None:
        outputs["norway_item_contributions"] = norway_contributions_path
    if norway_item_path is not None:
        outputs["norway_item_decomposition"] = norway_item_path
    if sweden_contributions_path is not None:
        outputs["sweden_item_contributions"] = sweden_contributions_path
    if uk_item_path is not None:
        outputs["uk_item_decomposition"] = uk_item_path
    if eurozone_item_path is not None:
        outputs["eurozone_item_decomposition"] = eurozone_item_path
    if us_item_path is not None:
        outputs["us_item_decomposition"] = us_item_path
    if us_pce_item_path is not None:
        outputs["us_pce_item_decomposition"] = us_pce_item_path
    if breadth_bucket_path is not None:
        outputs["breadth_buckets"] = breadth_bucket_path
    if breadth_item_path is not None:
        outputs["breadth_items"] = breadth_item_path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine raw CPI sources, calculate metrics, and generate charts "
            "and the latest inflation report."
        )
    )
    parser.add_argument(
        "--seasonal-adjustment",
        choices=SEASONAL_ADJUSTMENT_OPTIONS,
        default="source",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch raw source data before rebuilding derived outputs.",
    )
    parser.add_argument(
        "--full-refresh-incremental-sources",
        action="store_true",
        help=(
            "With --fetch, rebuild full Denmark, Sweden, and Norway histories "
            "instead of appending new periods."
        ),
    )
    parser.add_argument("--x13-path", type=Path)
    parser.add_argument("--chart-years", type=int, default=5)
    args = parser.parse_args()

    outputs = run_pipeline(
        fetch_raw=args.fetch,
        full_refresh_incremental_sources=args.full_refresh_incremental_sources,
        seasonal_adjustment=args.seasonal_adjustment,
        x13_path=args.x13_path,
        chart_years=args.chart_years,
    )
    print("Inflation pipeline completed")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
