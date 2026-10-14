"""Shared calendar-safe 12-month percent change helper.

Row-based ``.pct_change(12)`` silently compares against 11 months prior
instead of 12 for every date after a series has any missing month (e.g. the
Oct 2025 BLS release gap from that year's federal government shutdown).
``calendar_yoy`` reindexes each group to its own contiguous monthly calendar
first, so a gap produces a missing rate at the affected date rather than a
shifted one.
"""

import pandas as pd


def calendar_yoy(
    frame: pd.DataFrame,
    *,
    group_column: str,
    date_column: str = "date",
    value_column: str = "index_value",
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float, index=frame.index)
    parts = []
    for _, group in frame.groupby(group_column, sort=False):
        monthly_index = pd.date_range(
            group[date_column].min(), group[date_column].max(), freq="MS"
        )
        values = group.set_index(date_column)[value_column].reindex(monthly_index)
        rate = values.pct_change(periods=12, fill_method=None) * 100.0
        parts.append(group[date_column].map(rate))
    return pd.concat(parts).reindex(frame.index)
