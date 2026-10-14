import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def atomic_write_parquet(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    columns: Sequence[str] | None = None,
) -> Path:
    """Write a Parquet file atomically without exposing partial output."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = frame if columns is None else frame.loc[:, list(columns)]

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        output.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return destination
