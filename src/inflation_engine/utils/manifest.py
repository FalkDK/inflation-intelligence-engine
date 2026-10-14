import fcntl
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from inflation_engine.utils.paths import INGESTION_MANIFEST_PATH, PROJECT_ROOT
from inflation_engine.utils.schema import validate_cpi_schema


MANIFEST_SCHEMA_VERSION = 1


def _utc_timestamp(value: str | datetime | None) -> str | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def update_ingestion_manifest(
    dataset_id: str,
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    source_updated_at: str | datetime | None = None,
    retrieved_at: str | datetime | None = None,
    manifest_path: str | Path = INGESTION_MANIFEST_PATH,
) -> Path:
    """Atomically add or replace one dataset entry in the ingestion manifest."""
    validate_cpi_schema(frame)
    destination = Path(output_path)
    if not destination.is_file():
        raise FileNotFoundError(f"Ingestion output does not exist: {destination}")

    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest.with_name(f".{manifest.name}.lock")

    entry = {
        "retrieved_at": _utc_timestamp(
            retrieved_at or datetime.now(timezone.utc)
        ),
        "source_updated_at": _utc_timestamp(source_updated_at),
        "output_file": _display_path(destination),
        "sha256": _sha256(destination),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "first_observation_date": frame["date"].min().date().isoformat(),
        "latest_observation_date": frame["date"].max().date().isoformat(),
        "countries": sorted(frame["country_code"].unique().tolist()),
        "sources": sorted(frame["source"].unique().tolist()),
        "source_datasets": sorted(frame["source_dataset"].unique().tolist()),
        "source_series_count": int(frame["source_series_id"].nunique()),
        "index_bases": sorted(frame["index_base"].unique().tolist()),
        "seasonal_adjustments": sorted(
            frame["seasonal_adjustment"].unique().tolist()
        ),
        "observation_status_counts": {
            status: int(count)
            for status, count in sorted(
                frame["observation_status"].value_counts().items()
            )
        },
    }

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if manifest.exists():
            try:
                document = json.loads(manifest.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid ingestion manifest: {manifest}") from error
        else:
            document = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "updated_at": None,
                "datasets": {},
            }

        if document.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError("Unsupported ingestion manifest schema version")
        datasets = document.get("datasets")
        if not isinstance(datasets, dict):
            raise ValueError("Ingestion manifest datasets must be an object")

        datasets[dataset_id] = entry
        document["updated_at"] = entry["retrieved_at"]

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=manifest.parent,
                prefix=f".{manifest.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                json.dump(document, temporary_file, indent=2, sort_keys=True)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_path = Path(temporary_file.name)
            os.replace(temporary_path, manifest)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    return manifest
