import argparse
import math
import os
import tempfile
from pathlib import Path

MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "inflation-engine-matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from inflation_engine.utils.config import primary_series_keys
from inflation_engine.utils.paths import (
    CPI_3M3M_CHART_PATH,
    CPI_PANEL_PATH,
    CPI_YOY_CHART_PATH,
)


PRIMARY_KEY = ["country_code", "series_type", "component"]
REPORT_METRIC_COLUMNS = [
    "inflation_yoy",
    "inflation_yoy_change_3m",
    "inflation_3m3m_ann",
    "inflation_3m3m_yoy_spread",
    "inflation_momentum_signal",
]
SERIES_COLORS = {
    "headline": "#24557A",
    "core": "#D17A22",
}
SERIES_LABELS = {
    "headline": "Headline",
    "core": "Core",
}
LATEST_LABEL_OFFSETS = {
    "headline": 8,
    "core": -12,
}


def load_processed_panel(path: str | Path = CPI_PANEL_PATH) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Processed CPI panel does not exist: {source}")

    frame = pd.read_parquet(source)
    required_columns = [
        "date",
        "country",
        *PRIMARY_KEY,
        "series_role",
        "analysis_seasonal_adjustment",
        *REPORT_METRIC_COLUMNS,
    ]
    missing_columns = [column for column in required_columns if column not in frame]
    if missing_columns:
        raise ValueError(f"Processed CPI panel is missing columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Processed CPI panel cannot be empty")
    if not pd.api.types.is_datetime64_any_dtype(frame["date"]):
        raise ValueError("Processed CPI dates must use a datetime dtype")
    return frame


def select_primary_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the designated headline and core series for each geography."""
    required_columns = [*PRIMARY_KEY, "date", "country"]
    missing_columns = [column for column in required_columns if column not in frame]
    if missing_columns:
        raise ValueError(f"CPI panel is missing columns: {missing_columns}")

    primary_keys = pd.DataFrame(primary_series_keys(), columns=PRIMARY_KEY)
    selected = frame.merge(primary_keys, on=PRIMARY_KEY, how="inner")
    if selected.empty:
        raise ValueError("CPI panel contains no designated primary series")
    if not selected["series_role"].eq("aggregate").all():
        raise ValueError("Designated primary series must have aggregate role")
    return selected.sort_values(
        ["country_code", "series_type", "date"],
        kind="stable",
    ).reset_index(drop=True)


def _save_figure(figure: plt.Figure, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        figure.savefig(
            temporary_path,
            format="png",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
        os.replace(temporary_path, destination)
    finally:
        plt.close(figure)
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def _plot_country_history(
    frame: pd.DataFrame,
    *,
    metric: str,
    title: str,
    subtitle: str,
    output_path: str | Path,
    years: int = 5,
) -> Path:
    if metric not in frame:
        raise ValueError(f"CPI panel does not contain metric: {metric}")
    if years < 1:
        raise ValueError("Chart history must include at least one year")

    selected = select_primary_series(frame)
    latest_date = selected["date"].max()
    cutoff = latest_date - pd.DateOffset(years=years)
    selected = selected.loc[selected["date"] >= cutoff]
    countries = (
        selected[["country_code", "country"]]
        .drop_duplicates()
        .sort_values("country")
    )

    columns = 3
    rows = math.ceil(len(countries) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, 3.7 * rows),
        sharex=True,
        constrained_layout=True,
    )
    figure.get_layout_engine().set(rect=(0, 0, 1, 0.94))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for axis, country_row in zip(axes_flat, countries.itertuples(index=False)):
        country_frame = selected.loc[
            selected["country_code"] == country_row.country_code
        ]
        for series_type in ("headline", "core"):
            series = country_frame.loc[
                country_frame["series_type"] == series_type
            ]
            if series.empty:
                continue
            axis.plot(
                series["date"],
                series[metric],
                color=SERIES_COLORS[series_type],
                label=SERIES_LABELS[series_type],
                linewidth=2.0,
            )
            latest = series.dropna(subset=[metric]).iloc[-1]
            axis.scatter(
                latest["date"],
                latest[metric],
                color=SERIES_COLORS[series_type],
                edgecolor="white",
                linewidth=0.8,
                s=30,
                zorder=4,
            )
            axis.annotate(
                f"{latest[metric]:.1f}%\n{latest['date']:%b %Y}",
                xy=(latest["date"], latest[metric]),
                xytext=(-5, LATEST_LABEL_OFFSETS[series_type]),
                textcoords="offset points",
                color=SERIES_COLORS[series_type],
                fontsize=7.5,
                fontweight="bold",
                ha="right",
                va="bottom" if series_type == "headline" else "top",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.8,
                },
                clip_on=True,
                zorder=5,
            )
        axis.axhline(0, color="#777777", linewidth=0.8, alpha=0.7)
        axis.margins(y=0.18)
        axis.set_title(country_row.country, loc="left", fontsize=11, weight="bold")
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.tick_params(axis="x", rotation=30)

    for axis in axes_flat[len(countries) :]:
        axis.set_visible(False)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=SERIES_COLORS[series_type],
            linewidth=2.4,
            label=SERIES_LABELS[series_type],
        )
        for series_type in ("headline", "core")
    ]
    figure.legend(
        handles=handles,
        loc="outside upper right",
        frameon=False,
        ncols=2,
    )
    figure.suptitle(
        title,
        fontsize=18,
        weight="bold",
        x=0.02,
        y=0.995,
        ha="left",
    )
    figure.text(0.02, 0.972, subtitle, fontsize=10, color="#555555")
    unit_label = (
        "Percentage points"
        if metric in {"inflation_yoy_change_3m", "inflation_3m3m_yoy_spread"}
        else "Percent"
    )
    figure.supylabel(unit_label, fontsize=11)
    return _save_figure(figure, Path(output_path))


def plot_yoy_by_country(
    frame: pd.DataFrame,
    output_path: str | Path = CPI_YOY_CHART_PATH,
    *,
    years: int = 5,
) -> Path:
    return _plot_country_history(
        frame,
        metric="inflation_yoy",
        title="CPI Inflation by Country",
        subtitle=(
            f"Headline and core year-over-year inflation, {years}-year history"
        ),
        output_path=output_path,
        years=years,
    )


def plot_3m3m_momentum(
    frame: pd.DataFrame,
    output_path: str | Path = CPI_3M3M_CHART_PATH,
    *,
    years: int = 5,
) -> Path:
    return _plot_country_history(
        frame,
        metric="inflation_yoy_change_3m",
        title="Inflation Direction",
        subtitle=(
            "Three-month change in year-over-year inflation, percentage "
            f"points, {years}-year history"
        ),
        output_path=output_path,
        years=years,
    )


def generate_charts(
    frame: pd.DataFrame,
    *,
    yoy_path: str | Path = CPI_YOY_CHART_PATH,
    momentum_path: str | Path = CPI_3M3M_CHART_PATH,
    years: int = 5,
) -> tuple[Path, Path]:
    return (
        plot_yoy_by_country(frame, yoy_path, years=years),
        plot_3m3m_momentum(frame, momentum_path, years=years),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate inflation charts from the processed CPI panel."
    )
    parser.add_argument("--input", type=Path, default=CPI_PANEL_PATH)
    parser.add_argument("--yoy-output", type=Path, default=CPI_YOY_CHART_PATH)
    parser.add_argument(
        "--momentum-output",
        type=Path,
        default=CPI_3M3M_CHART_PATH,
    )
    parser.add_argument("--years", type=int, default=5)
    args = parser.parse_args()

    frame = load_processed_panel(args.input)
    destinations = generate_charts(
        frame,
        yoy_path=args.yoy_output,
        momentum_path=args.momentum_output,
        years=args.years,
    )
    for destination in destinations:
        print(f"Saved chart to {destination}")


if __name__ == "__main__":
    main()
