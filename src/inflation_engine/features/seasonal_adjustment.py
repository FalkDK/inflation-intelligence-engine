import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd


X13_BINARY_NAMES = (
    "x13as",
    "x13as_ascii",
    "x13as_html",
    "x13as.exe",
    "x13as_ascii.exe",
    "x13as_html.exe",
)
MINIMUM_X13_OBSERVATIONS = 60
X13_METHOD = "x13_arima_seats"


def _find_binary_in_directory(directory: Path) -> Path | None:
    for binary_name in X13_BINARY_NAMES:
        candidate = directory / binary_name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def resolve_x13_path(x13_path: str | Path | None = None) -> Path:
    """Find an executable X-13ARIMA-SEATS binary."""
    if x13_path is not None:
        candidate = Path(x13_path).expanduser()
        if candidate.is_dir():
            executable = _find_binary_in_directory(candidate)
        else:
            executable = candidate
        if executable is not None and executable.is_file():
            if os.access(executable, os.X_OK):
                return executable.resolve()
            raise FileNotFoundError(
                f"X-13ARIMA-SEATS binary is not executable: {executable}"
            )
        raise FileNotFoundError(
            f"X-13ARIMA-SEATS binary was not found at: {candidate}"
        )

    for environment_variable in ("X13PATH", "X12PATH"):
        configured_path = os.getenv(environment_variable)
        if configured_path:
            return resolve_x13_path(configured_path)

    for binary_name in X13_BINARY_NAMES:
        executable = shutil.which(binary_name)
        if executable:
            return Path(executable).resolve()

    raise FileNotFoundError(
        "X-13ARIMA-SEATS executable not found. Provide --x13-path, set "
        "X13PATH, or add x13as to PATH."
    )


def _validate_monthly_series(values: pd.Series) -> pd.Series:
    if not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError("X-13 input must use a DatetimeIndex")
    if values.empty:
        raise ValueError("X-13 input cannot be empty")
    if values.isna().any():
        raise ValueError("X-13 input cannot contain missing values")
    if (values <= 0).any():
        raise ValueError("X-13 input values must be greater than zero")
    if len(values) < MINIMUM_X13_OBSERVATIONS:
        raise ValueError(
            "X-13 seasonal adjustment requires at least "
            f"{MINIMUM_X13_OBSERVATIONS} monthly observations"
        )

    ordered = values.sort_index()
    expected_index = pd.date_range(
        ordered.index.min(),
        ordered.index.max(),
        freq="MS",
    )
    if not ordered.index.equals(expected_index):
        raise ValueError(
            "X-13 seasonal adjustment requires contiguous month-start data"
        )
    return ordered.astype(float)


def build_x13_seats_spec(values: pd.Series) -> str:
    """Build an automatic regARIMA and SEATS specification."""
    values = _validate_monthly_series(values)
    start = values.index[0]
    data = "\n".join(f"{value:.12g}" for value in values)
    return (
        "series {\n"
        f'  title = "{values.name or "CPI series"}"\n'
        f"  start = {start.year}.{start.month}\n"
        "  period = 12\n"
        "  precision = 12\n"
        "  data = (\n"
        f"{data}\n"
        "  )\n"
        "}\n"
        "transform { function = auto }\n"
        "automdl { }\n"
        "outlier { }\n"
        "forecast { maxlead = 12 }\n"
        "seats { save = s11 }\n"
    )


def _read_seats_adjustment(path: Path, dates: pd.DatetimeIndex) -> pd.Series:
    if not path.is_file():
        raise RuntimeError(
            f"X-13ARIMA-SEATS did not create expected output: {path.name}"
        )

    table = pd.read_csv(
        path,
        skiprows=2,
        header=None,
        sep="\t",
        engine="python",
    )
    if table.shape[1] < 2 or len(table) != len(dates):
        raise RuntimeError(
            "X-13ARIMA-SEATS returned an unexpected seasonal-adjustment table"
        )

    adjusted = pd.to_numeric(table.iloc[:, -1], errors="coerce")
    if adjusted.isna().any() or (adjusted <= 0).any():
        raise RuntimeError(
            "X-13ARIMA-SEATS returned invalid seasonally adjusted values"
        )
    return pd.Series(
        adjusted.to_numpy(),
        index=dates,
        name="analysis_value",
    )


def seasonal_adjust_x13(
    values: pd.Series,
    *,
    x13_path: str | Path | None = None,
    _runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> pd.Series:
    """Seasonally adjust a monthly index with X-13ARIMA-SEATS using SEATS."""
    values = _validate_monthly_series(values)
    executable = resolve_x13_path(x13_path)

    with tempfile.TemporaryDirectory(prefix="inflation-x13-") as temporary:
        base_path = Path(temporary) / "series"
        specification_path = base_path.with_suffix(".spc")
        specification_path.write_text(
            build_x13_seats_spec(values),
            encoding="ascii",
        )
        process = _runner(
            [str(executable), str(base_path)],
            cwd=temporary,
            capture_output=True,
            text=True,
            check=False,
        )
        error_path = base_path.with_suffix(".err")
        error_output = (
            error_path.read_text(encoding="utf-8", errors="replace")
            if error_path.is_file()
            else ""
        )
        if process.returncode != 0 or "ERROR" in error_output.upper():
            details = error_output.strip() or process.stdout.strip()
            if not details:
                details = f"exit status {process.returncode}"
            raise RuntimeError(f"X-13ARIMA-SEATS failed: {details}")

        return _read_seats_adjustment(
            base_path.with_suffix(".s11"),
            values.index,
        )
