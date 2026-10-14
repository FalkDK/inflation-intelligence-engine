"""Release-aware local automation. Only Denmark is enabled initially.

The daily watcher sleeps only on release/retry days; ordinary checks exit.
All release comparisons use UTC. A local timezone defines the watcher day.
"""

import argparse
from contextlib import contextmanager
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd

from inflation_engine.data.fetch_statbank_dk import (
    fetch_danish_cpi,
    persist_danish_cpi,
)
from inflation_engine.features.availability import load_release_calendar
from inflation_engine.utils.paths import (
    CPI_PANEL_PATH, PROJECT_ROOT, STATBANK_DK_CPI_PATH,
)
from inflation_engine.utils.schema import validate_cpi_schema


RUNTIME_DIR = PROJECT_ROOT / "outputs" / "automation"
LOG = logging.getLogger("inflation_automation")


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as file:
        json.dump(state, file, indent=2)
        temporary = Path(file.name)
    os.replace(temporary, path)


def has_observation(frame: pd.DataFrame, month: pd.Timestamp) -> bool:
    rows = frame.loc[
        frame["country_code"].eq("DK") & frame["series_role"].eq("aggregate")
        & pd.to_datetime(frame["date"]).eq(month)
    ]
    return {"headline", "core"}.issubset(rows["series_type"])


def next_action(calendar, raw, panel, state, now, *, timezone="Europe/London",
                delay_minutes=15):
    """Return (UTC wake time, observation month), or None. No network or writes."""
    now = utc(now)
    dk = calendar.loc[calendar["country_code"].eq("DK")].copy()
    if "publication_datetime_utc" not in dk:
        raise ValueError("Calendar has no release times; refresh it first")
    dk = dk.dropna(subset=["publication_datetime_utc"])
    if dk.empty:
        raise ValueError("Calendar contains no timed Danish releases")
    dk["fetch_at"] = dk["publication_datetime_utc"] + pd.Timedelta(minutes=delay_minutes)
    due = dk.loc[dk["fetch_at"] <= now].sort_values("fetch_at")
    if not due.empty:
        # Older calendar history is not a backlog of releases to refetch.
        latest = due.iloc[-1]
        month = latest["observation_date"]
        if (state.get("build_pending") or not has_observation(raw, month)
                or not has_observation(panel, month)):
            retry = state.get("next_attempt_utc")
            return (max(now, utc(retry)) if retry else now, month)
    end = (now.tz_convert(timezone).normalize() + pd.DateOffset(days=1)).tz_convert("UTC")
    upcoming = dk.loc[(dk["fetch_at"] > now) & (dk["fetch_at"] < end)].sort_values("fetch_at")
    for row in upcoming.itertuples():
        if not has_observation(raw, row.observation_date) or not has_observation(panel, row.observation_date):
            return row.fetch_at, row.observation_date
    return None


def rebuild_with_rollback(root: Path = PROJECT_ROOT) -> None:
    """Restore all previous pipeline artifacts if the rebuild fails.

    The raw fetch is retained so a subsequent run can retry just the rebuild.
    This is a local pipeline, with no public hosting or Git publishing.
    """
    folders = [root / "data/interim", root / "data/processed",
               root / "outputs/charts", root / "outputs/reports"]
    existing = [p for folder in folders if folder.exists()
                for p in folder.rglob("*") if p.is_file()]
    with tempfile.TemporaryDirectory(prefix="inflation-build-") as temp:
        backup = Path(temp)
        for path in existing:
            target = backup / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(root / "tmp/matplotlib")
        try:
            result = subprocess.run(
                [sys.executable, "-u", str(root / "run_pipeline.py")],
                cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=1800,
            )
            if result.returncode:
                raise RuntimeError(f"Pipeline rebuild exited {result.returncode}: {result.stdout[-8000:]}")
            LOG.info("Pipeline rebuild completed")
        except Exception:
            for folder in folders:
                if folder.exists():
                    for path in folder.rglob("*"):
                        if path.is_file() and path not in existing:
                            path.unlink()
            for path in existing:
                shutil.copy2(backup / path.relative_to(root), path)
            raise


def refresh_denmark(month: pd.Timestamp, state: dict, state_path: Path,
                    raw_path=STATBANK_DK_CPI_PATH, panel_path=CPI_PANEL_PATH,
                    fetcher=fetch_danish_cpi, persist=persist_danish_cpi,
                    rebuild=rebuild_with_rollback) -> bool:
    raw_path, panel_path = Path(raw_path), Path(panel_path)
    raw = pd.read_parquet(raw_path) if raw_path.exists() else pd.DataFrame()
    if raw.empty or not has_observation(raw, month):
        LOG.info("Fetching Denmark for expected month %s", month.date())
        fetched = fetcher(existing_frame=None if raw.empty else raw)
        validate_cpi_schema(fetched)
        if not has_observation(fetched, month):
            LOG.info("Denmark %s not available yet", month.date())
            return False
        if not raw.empty:
            keys = ["date", "country_code", "series_type", "component"]
            if not pd.MultiIndex.from_frame(raw[keys]).isin(pd.MultiIndex.from_frame(fetched[keys])).all():
                raise ValueError("Danish refresh dropped existing observations")
        # Persist pending intent before touching raw data, so interrupted builds retry.
        state["build_pending"] = True
        write_state(state_path, state)
        persist(fetched, raw_path)
    state["build_pending"] = True
    write_state(state_path, state)
    rebuild()
    panel = pd.read_parquet(panel_path)
    if not has_observation(panel, month):
        raise ValueError("Rebuilt panel is missing expected Danish headline/core")
    state["build_pending"] = False
    state["last_success_utc"] = utc_now().isoformat()
    state["last_observation"] = str(month.date())
    state.pop("next_attempt_utc", None)
    state.pop("last_error", None)
    write_state(state_path, state)
    return True


def check(*, watch=False, dry_run=False, timezone="Europe/London") -> None:
    state_path = RUNTIME_DIR / "state.json"
    with exclusive_lock(RUNTIME_DIR / "daily.lock") as acquired:
        if not acquired:
            LOG.info("Another daily checker is running; exiting")
            return
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        end = (utc_now().tz_convert(timezone).normalize() + pd.DateOffset(days=1)).tz_convert("UTC")
        while True:
            calendar = load_release_calendar()
            raw = pd.read_parquet(STATBANK_DK_CPI_PATH)
            panel = pd.read_parquet(CPI_PANEL_PATH)
            now = utc_now()
            if now >= end:
                LOG.info("Watcher day ended; any remaining release retries resume tomorrow")
                return
            action = next_action(calendar, raw, panel, state, now, timezone=timezone)
            if action is None:
                LOG.info("Denmark up to date; no release fetch needed today")
                return
            wake, month = action
            LOG.info("Next action: Denmark %s at %s", month.date(), wake.tz_convert(timezone))
            if dry_run:
                return
            if wake > now:
                if not watch or wake >= end:
                    return
                # No polling or network requests during this release-time wait.
                time.sleep((wake - now).total_seconds())
                continue
            state["last_attempt_utc"] = now.isoformat()
            state["next_attempt_utc"] = (now + pd.Timedelta(hours=1)).isoformat()
            write_state(state_path, state)
            try:
                if refresh_denmark(month, state, state_path):
                    return
            except Exception as error:
                state["last_error"] = str(error)
                LOG.exception("Denmark refresh/build failed; previous outputs retained")
            # Retry no sooner than an hour after completion, even for slow fetches.
            state["next_attempt_utc"] = (utc_now() + pd.Timedelta(hours=1)).isoformat()
            write_state(state_path, state)
            if not watch:
                return


def refresh_calendar() -> None:
    from inflation_engine.data.fetch_release_calendar import run

    with exclusive_lock(RUNTIME_DIR / "calendar.lock") as acquired:
        if acquired:
            path = run(recent_months=2)
            LOG.info("Release calendar refreshed: %s", path)
        else:
            LOG.info("Calendar refresh already running; exiting")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "calendar"])
    parser.add_argument("--watch", action="store_true", help="Wait for today's releases and hourly retries.")
    parser.add_argument("--dry-run", action="store_true", help="Plan locally without fetching or rebuilding.")
    parser.add_argument("--timezone", default="Europe/London")
    args = parser.parse_args()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.StreamHandler(), RotatingFileHandler(
            RUNTIME_DIR / "automation.log", maxBytes=2_000_000, backupCount=3)]:
        handler.setFormatter(formatter)
        LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    try:
        if args.command == "calendar":
            if args.dry_run:
                LOG.info("Dry run: would refresh calendar (recent two months + upcoming six)")
            else:
                refresh_calendar()
        else:
            check(watch=args.watch, dry_run=args.dry_run, timezone=args.timezone)
    except Exception:
        LOG.exception("Automation stopped")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
