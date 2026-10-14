import pandas as pd
import pytest

from inflation_engine.features.calendar_yoy import calendar_yoy


def test_calendar_yoy_does_not_shift_across_a_missing_month():
    """A series with the Oct-2025-shaped gap (BLS's 2025 shutdown-delayed
    release) must have its 12-month change measured against the calendar
    month a year prior, not against whatever row happens to sit 12 rows
    back -- row-based pct_change(12) silently drifts by one month for
    every date after a gap like this."""
    dates = pd.date_range("2025-01-01", "2026-08-01", freq="MS")
    dates = dates[dates != pd.Timestamp("2025-10-01")]
    # A value of exactly (year * 100 + month) makes the correct calendar
    # ratio easy to hand-verify and very different from the row-shifted one.
    values = [d.year * 100 + d.month for d in dates]
    frame = pd.DataFrame({"date": dates, "index_value": values, "series_id": "X"})

    result = calendar_yoy(frame, group_column="series_id")

    aug_2026 = frame.index[frame["date"] == "2026-08-01"][0]
    aug_2025_value = 2025 * 100 + 8
    aug_2026_value = 2026 * 100 + 8
    expected = (aug_2026_value / aug_2025_value - 1.0) * 100.0
    assert result[aug_2026] == pytest.approx(expected)

    # Prove the fixture actually exercises the gap: the naive row-based
    # pct_change(12) this replaces would have compared against September
    # (11 calendar months back, one row short) and landed on a different
    # number -- if it didn't, this test wouldn't be testing anything.
    wrong_row_based = frame["index_value"].pct_change(periods=12) * 100.0
    assert wrong_row_based[aug_2026] != pytest.approx(expected)

    # The month right after the gap has no true prior-year value 12
    # calendar months back within this short fixture, so it must be NaN --
    # never silently filled from whatever row happens to be 12 rows earlier.
    nov_2025 = frame.index[frame["date"] == "2025-11-01"][0]
    assert pd.isna(result[nov_2025])


def test_calendar_yoy_handles_an_empty_frame():
    empty = pd.DataFrame({"date": [], "index_value": [], "series_id": []})
    result = calendar_yoy(empty, group_column="series_id")
    assert result.empty
