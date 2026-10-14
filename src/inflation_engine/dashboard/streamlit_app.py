import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from inflation_engine.dashboard.data import (  # noqa: E402
    DEFAULT_COUNTRY_METRICS,
    DEFAULT_HEATMAP_METRICS,
    METRIC_COLUMNS,
    METRIC_LABELS,
    available_countries,
    country_latest_metrics,
    country_metric_history,
    country_ranking,
    heatmap_values,
    latest_primary_observations,
    load_dashboard_panel,
)


SERIES_LABELS = {
    "headline": "Headline",
    "core": "Core",
}
SERIES_COLORS = {
    "Headline": "#1F6F8B",
    "Core": "#C7612B",
}
SIGNAL_COLORS = {
    "accelerating": "#B42318",
    "stable": "#475467",
    "slowing": "#027A48",
    "not_available": "#667085",
}


def _format_percent(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.1f}%"


def _format_points(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.1f} pp"


def _format_signal(value: str) -> str:
    return str(value).replace("_", " ").title()


@st.cache_data(show_spinner=False)
def _load_data() -> pd.DataFrame:
    return load_dashboard_panel()


def _metric_options(metrics: list[str]) -> dict[str, str]:
    return {METRIC_LABELS[metric]: metric for metric in metrics}


def _line_chart(history: pd.DataFrame, metric: str) -> alt.Chart:
    chart_data = history.loc[history["metric"] == metric].copy()
    return (
        alt.Chart(chart_data)
        .mark_line(point=alt.OverlayMarkDef(size=35), strokeWidth=2.5)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y(
                "value:Q",
                title="Percent" if metric != "inflation_3m3m_yoy_spread" else "pp",
            ),
            color=alt.Color(
                "series_label:N",
                title=None,
                scale=alt.Scale(
                    domain=list(SERIES_COLORS),
                    range=list(SERIES_COLORS.values()),
                ),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%b %Y"),
                alt.Tooltip("series_label:N", title="Series"),
                alt.Tooltip("metric_label:N", title="Metric"),
                alt.Tooltip("value:Q", title="Value", format=".2f"),
            ],
        )
        .properties(height=260)
        .interactive()
    )


def _heatmap_chart(heatmap: pd.DataFrame) -> alt.Chart:
    base = (
        alt.Chart(heatmap)
        .encode(
            x=alt.X("metric_label:N", title=None, sort=None),
            y=alt.Y(
                "country:N",
                title=None,
                sort=list(heatmap["country"].cat.categories),
            ),
            tooltip=[
                alt.Tooltip("country:N", title="Country"),
                alt.Tooltip("date:T", title="Observation", format="%b %Y"),
                alt.Tooltip("metric_label:N", title="Metric"),
                alt.Tooltip("value:Q", title="Value", format=".2f"),
            ],
        )
    )
    cells = base.mark_rect().encode(
        color=alt.Color(
            "value:Q",
            title="Percent / pp",
            scale=alt.Scale(scheme="redyellowgreen", reverse=True),
        )
    )
    labels = base.mark_text(fontSize=12).encode(
        text=alt.Text("value:Q", format=".1f"),
        color=alt.condition(
            "abs(datum.value) > 5",
            alt.value("white"),
            alt.value("#1D2939"),
        ),
    )
    return (cells + labels).properties(
        height=max(320, 32 * heatmap["country"].nunique())
    )


def _ranking_table(ranking: pd.DataFrame, metric: str) -> pd.DataFrame:
    table = ranking[
        [
            "rank",
            "country",
            "region",
            "date",
            metric,
            "inflation_momentum_signal",
        ]
    ].copy()
    table["date"] = table["date"].dt.strftime("%Y-%m")
    table[metric] = table[metric].map(_format_percent)
    table["inflation_momentum_signal"] = table["inflation_momentum_signal"].map(
        _format_signal
    )
    return table.rename(
        columns={
            "rank": "Rank",
            "country": "Country",
            "region": "Region",
            "date": "Observation",
            metric: METRIC_LABELS[metric],
            "inflation_momentum_signal": "Signal",
        }
    )


def _latest_table(latest: pd.DataFrame) -> pd.DataFrame:
    table = latest[
        [
            "country",
            "series_type",
            "date",
            "inflation_mom",
            "inflation_yoy",
            "inflation_yoy_change_3m",
            "inflation_3m_ann",
            "inflation_3m3m_ann",
            "inflation_6m_ann",
            "inflation_6m6m_ann",
            "inflation_3m3m_yoy_spread",
            "inflation_momentum_signal",
        ]
    ].copy()
    table["series_type"] = table["series_type"].map(SERIES_LABELS)
    table["date"] = table["date"].dt.strftime("%Y-%m")
    for metric in METRIC_COLUMNS:
        points_metrics = {
            "inflation_yoy_change_3m",
            "inflation_3m3m_yoy_spread",
        }
        table[metric] = table[metric].map(
            _format_points if metric in points_metrics else _format_percent
        )
    table["inflation_momentum_signal"] = table["inflation_momentum_signal"].map(
        _format_signal
    )
    return table.rename(
        columns={
            "country": "Country",
            "series_type": "Series",
            "date": "Observation",
            "inflation_mom": "MoM",
            "inflation_yoy": "YoY",
            "inflation_yoy_change_3m": "3m change in YoY",
            "inflation_3m_ann": "3m Ann.",
            "inflation_3m3m_ann": "3m/3m Ann.",
            "inflation_6m_ann": "6m Ann.",
            "inflation_6m6m_ann": "6m/6m Ann.",
            "inflation_3m3m_yoy_spread": "3m/3m - YoY",
            "inflation_momentum_signal": "Signal",
        }
    )


def _render_country_metrics(latest: pd.DataFrame) -> None:
    columns = st.columns(len(latest))
    for column, row in zip(columns, latest.itertuples(index=False)):
        signal = _format_signal(row.inflation_momentum_signal)
        signal_color = SIGNAL_COLORS.get(row.inflation_momentum_signal, "#475467")
        with column:
            st.subheader(SERIES_LABELS[row.series_type])
            st.caption(f"{row.date:%B %Y}")
            st.metric("YoY", _format_percent(row.inflation_yoy))
            st.metric(
                "3m change in YoY",
                _format_points(row.inflation_yoy_change_3m),
            )
            st.metric("MoM", _format_percent(row.inflation_mom))
            st.metric("3m Ann.", _format_percent(row.inflation_3m_ann))
            st.metric("3m/3m Ann.", _format_percent(row.inflation_3m3m_ann))
            st.metric("6m/6m Ann.", _format_percent(row.inflation_6m6m_ann))
            st.markdown(
                f"<span style='color:{signal_color};font-weight:600'>{signal}</span>",
                unsafe_allow_html=True,
            )


def main() -> None:
    st.set_page_config(
        page_title="Inflation Intelligence Dashboard",
        page_icon="%",
        layout="wide",
    )

    st.title("Inflation Intelligence Dashboard")
    try:
        frame = _load_data()
    except Exception as exc:
        st.error(f"Could not load processed CPI panel: {exc}")
        st.stop()

    countries = available_countries(frame)
    country_labels = {
        f"{row.country} ({row.country_code})": row.country_code
        for row in countries.itertuples(index=False)
    }
    default_country = (
        list(country_labels.values()).index("US")
        if "US" in country_labels.values()
        else 0
    )

    with st.sidebar:
        selected_country_label = st.selectbox(
            "Country",
            options=list(country_labels),
            index=default_country,
        )
        selected_country_code = country_labels[selected_country_label]
        years = st.slider("History", min_value=1, max_value=10, value=5, step=1)
        selected_series_labels = st.multiselect(
            "Series",
            options=list(SERIES_LABELS.values()),
            default=list(SERIES_LABELS.values()),
        )
        selected_series = [
            key for key, label in SERIES_LABELS.items() if label in selected_series_labels
        ]
        metric_choices = _metric_options(METRIC_COLUMNS)
        selected_metric_labels = st.multiselect(
            "Country charts",
            options=list(metric_choices),
            default=[METRIC_LABELS[metric] for metric in DEFAULT_COUNTRY_METRICS],
        )
        selected_metrics = [metric_choices[label] for label in selected_metric_labels]

    latest = latest_primary_observations(frame)
    latest_date = latest["date"].max()
    st.caption(
        f"Latest available observations through {latest_date:%B %Y}. "
        "Release months vary by country and series."
    )

    country_tab, cross_country_tab = st.tabs(["Country", "Cross-country"])
    with country_tab:
        country_latest = country_latest_metrics(frame, selected_country_code)
        st.header(selected_country_label)
        _render_country_metrics(country_latest)

        if selected_metrics and selected_series:
            history = country_metric_history(
                frame,
                selected_country_code,
                metrics=selected_metrics,
                series_types=selected_series,
                years=years,
            )
            for first, second in zip(selected_metrics[::2], selected_metrics[1::2]):
                left, right = st.columns(2)
                with left:
                    st.subheader(METRIC_LABELS[first])
                    st.altair_chart(_line_chart(history, first), width="stretch")
                with right:
                    st.subheader(METRIC_LABELS[second])
                    st.altair_chart(_line_chart(history, second), width="stretch")
            if len(selected_metrics) % 2:
                metric = selected_metrics[-1]
                st.subheader(METRIC_LABELS[metric])
                st.altair_chart(_line_chart(history, metric), width="stretch")
        else:
            st.info("Select at least one series and metric to show country charts.")

    with cross_country_tab:
        st.header("Latest Country Ranking")
        ranking_metric_labels = _metric_options(METRIC_COLUMNS)
        ranking_metric_label = st.selectbox(
            "Ranking metric",
            options=list(ranking_metric_labels),
            index=list(ranking_metric_labels).index("YoY"),
        )
        ranking_series_label = st.radio(
            "Ranking series",
            options=list(SERIES_LABELS.values()),
            horizontal=True,
        )
        ranking_metric = ranking_metric_labels[ranking_metric_label]
        ranking_series = [
            key for key, label in SERIES_LABELS.items() if label == ranking_series_label
        ][0]
        ranking = country_ranking(
            frame,
            metric=ranking_metric,
            series_type=ranking_series,
        )
        st.dataframe(
            _ranking_table(ranking, ranking_metric),
            hide_index=True,
            width="stretch",
        )

        st.header("Inflation Heatmap")
        heatmap_series_label = st.radio(
            "Heatmap series",
            options=list(SERIES_LABELS.values()),
            horizontal=True,
            key="heatmap_series",
        )
        heatmap_metric_labels = st.multiselect(
            "Heatmap metrics",
            options=list(ranking_metric_labels),
            default=[METRIC_LABELS[metric] for metric in DEFAULT_HEATMAP_METRICS],
        )
        heatmap_series = [
            key for key, label in SERIES_LABELS.items() if label == heatmap_series_label
        ][0]
        heatmap_metrics = [ranking_metric_labels[label] for label in heatmap_metric_labels]
        if heatmap_metrics:
            heatmap = heatmap_values(
                frame,
                metrics=heatmap_metrics,
                series_type=heatmap_series,
            )
            st.altair_chart(_heatmap_chart(heatmap), width="stretch")
        else:
            st.info("Select at least one metric to show the heatmap.")

        with st.expander("Latest observations table"):
            st.dataframe(
                _latest_table(latest),
                hide_index=True,
                width="stretch",
            )


if __name__ == "__main__":
    main()
