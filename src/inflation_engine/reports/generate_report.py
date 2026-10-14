import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd

from inflation_engine.reports.charts import (
    load_processed_panel,
    select_primary_series,
)
from inflation_engine.utils.config import ALTERNATIVE_MEASURE_CODES
from inflation_engine.utils.paths import (
    CPI_PANEL_PATH,
    LATEST_INFLATION_REPORT_PATH,
)


def _format_percent(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.1f}%"


def _format_points(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.1f} pp"


def _format_signal(value: str) -> str:
    return str(value).replace("_", " ").title()


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def latest_primary_observations(frame: pd.DataFrame) -> pd.DataFrame:
    primary = select_primary_series(frame)
    return (
        primary.sort_values("date", kind="stable")
        .groupby(["country_code", "series_type"], as_index=False)
        .tail(1)
        .sort_values(["country", "series_type"], kind="stable")
        .reset_index(drop=True)
    )


def _latest_observation_table(latest: pd.DataFrame) -> str:
    rows = []
    for country, country_frame in latest.groupby("country", sort=True):
        by_type = country_frame.set_index("series_type")
        headline = by_type.loc["headline"]
        core = by_type.loc["core"]
        rows.append(
            [
                country,
                headline["date"].strftime("%Y-%m"),
                _format_percent(headline["inflation_yoy"]),
                core["date"].strftime("%Y-%m"),
                _format_percent(core["inflation_yoy"]),
                _format_points(headline["inflation_yoy_change_3m"]),
                _format_points(core["inflation_yoy_change_3m"]),
            ]
        )
    return _markdown_table(
        [
            "Country",
            "Headline month",
            "Headline YoY",
            "Core month",
            "Core YoY",
            "Headline 3m YoY change",
            "Core 3m YoY change",
        ],
        rows,
    )


def _ranking_table(
    latest: pd.DataFrame,
    *,
    metric: str,
) -> str:
    headline = latest.loc[
        (latest["series_type"] == "headline")
        & ~latest["country_code"].isin(ALTERNATIVE_MEASURE_CODES)
    ].sort_values(
        metric,
        ascending=False,
        na_position="last",
    )
    formatter = _format_points if metric == "inflation_yoy_change_3m" else _format_percent
    rows = [
        [
            str(rank),
            row.country,
            row.date.strftime("%Y-%m"),
            formatter(getattr(row, metric)),
        ]
        for rank, row in enumerate(headline.itertuples(index=False), start=1)
    ]
    value_header = "Change" if metric == "inflation_yoy_change_3m" else "Rate"
    return _markdown_table(["Rank", "Country", "Observation", value_header], rows)


def _momentum_table(latest: pd.DataFrame) -> str:
    headline = latest.loc[latest["series_type"] == "headline"].sort_values(
        "country"
    )
    rows = [
        [
            row.country,
            row.date.strftime("%Y-%m"),
            _format_points(row.inflation_yoy_change_3m),
            _format_signal(row.inflation_momentum_signal),
        ]
        for row in headline.itertuples(index=False)
    ]
    return _markdown_table(
        ["Country", "Observation", "3m YoY change", "Signal"],
        rows,
    )


def _coverage_table(frame: pd.DataFrame) -> str:
    primary = select_primary_series(frame)
    coverage = (
        primary.groupby(["country", "series_type"])
        .agg(start=("date", "min"), end=("date", "max"), observations=("date", "size"))
        .reset_index()
    )
    rows = [
        [
            row.country,
            row.series_type.title(),
            row.start.strftime("%Y-%m"),
            row.end.strftime("%Y-%m"),
            f"{row.observations:,}",
        ]
        for row in coverage.sort_values(["country", "series_type"]).itertuples(
            index=False
        )
    ]
    return _markdown_table(
        ["Country", "Series", "Start", "End", "Observations"],
        rows,
    )


def build_report(frame: pd.DataFrame) -> str:
    latest = latest_primary_observations(frame)
    latest_date = latest["date"].max().strftime("%B %Y")
    adjustment_methods = sorted(
        set(frame["analysis_seasonal_adjustment"].dropna())
    )
    adjustment_text = ", ".join(
        method.replace("_", " ") for method in adjustment_methods
    )

    sections = [
        "# Inflation Intelligence Report",
        "",
        f"Latest available observations through **{latest_date}**. "
        "Release months vary by country and series and are shown in the tables "
        "and chart endpoint labels.",
        "",
        "![CPI YoY by country](../charts/cpi_yoy_by_country.png)",
        "",
        "![Inflation direction](../charts/cpi_inflation_direction.png)",
        "",
        "## Latest Observations",
        "",
        _latest_observation_table(latest),
        "",
        "## Country Ranking by YoY Inflation",
        "",
        _ranking_table(latest, metric="inflation_yoy"),
        "",
        "## Country Ranking by Three-Month Change in YoY Inflation",
        "",
        _ranking_table(latest, metric="inflation_yoy_change_3m"),
        "",
        "## Momentum Signal",
        "",
        _momentum_table(latest),
        "",
        "The signal is accelerating when YoY inflation has risen by more than "
        "0.25 percentage points over three months, slowing when it has fallen "
        "by more than 0.25 points, and stable otherwise.",
        "",
        "## Data Coverage",
        "",
        _coverage_table(frame),
        "",
        "## Methodology Notes",
        "",
        "- YoY inflation is the 12-month percent change in the index.",
        "- The primary momentum measure is the three-month change in YoY "
        "inflation, which is comparable across adjusted and non-adjusted data.",
        "- Rates are calculated from `analysis_value`; source `value` data is "
        "preserved unchanged.",
        f"- Analysis seasonal-adjustment statuses present: {adjustment_text}.",
        "- MoM and annualized short-horizon metrics are not used in the "
        "cross-country rankings when a series is not seasonally adjusted.",
        "- Headline and core definitions differ across statistical agencies; "
        "cross-country comparisons are therefore indicative rather than exact.",
        "",
    ]
    return "\n".join(sections)


def save_report(
    report: str,
    output_path: str | Path = LATEST_INFLATION_REPORT_PATH,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(report)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def generate_report(
    frame: pd.DataFrame,
    output_path: str | Path = LATEST_INFLATION_REPORT_PATH,
) -> Path:
    return save_report(build_report(frame), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the latest Markdown inflation report."
    )
    parser.add_argument("--input", type=Path, default=CPI_PANEL_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=LATEST_INFLATION_REPORT_PATH,
    )
    args = parser.parse_args()

    frame = load_processed_panel(args.input)
    destination = generate_report(frame, args.output)
    print(f"Saved inflation report to {destination}")


if __name__ == "__main__":
    main()
