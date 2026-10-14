"""Install only the managed inflation cron block; preserve unrelated jobs."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BEGIN = "# BEGIN inflation-release-automation"
END = "# END inflation-release-automation"


def updated_crontab(existing: str) -> str:
    lines = existing.splitlines()
    if lines.count(BEGIN) != lines.count(END) or lines.count(BEGIN) > 1:
        raise ValueError("Malformed inflation cron block; refusing to edit")
    if BEGIN in lines:
        start, end = lines.index(BEGIN), lines.index(END)
        if end < start:
            raise ValueError("Malformed inflation cron block; refusing to edit")
        lines = lines[:start] + lines[end + 1:]
    wrapper = shlex.quote(str(ROOT / "scripts/inflation_automation.sh"))
    log = shlex.quote(str(ROOT / "outputs/automation/cron.log"))
    interpreter = "INFLATION_PYTHON=" + shlex.quote(sys.executable)
    block = [BEGIN,
             "# Mac system timezone: Europe/London. Initial scope: Denmark.",
             f"0 9 * * 0 {interpreter} /bin/sh {wrapper} calendar >> {log} 2>&1",
             f"0 7 * * * {interpreter} /bin/sh {wrapper} check --watch >> {log} 2>&1",
             END]
    return "\n".join(lines + block) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="Install; otherwise print proposed crontab.")
    args = parser.parse_args()
    current = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if current.returncode and "no crontab for" not in current.stderr.lower():
        raise RuntimeError(current.stderr)
    existing = current.stdout
    proposed = updated_crontab(existing)
    if not args.install:
        print(proposed, end="")
        return
    folder = ROOT / "outputs/automation"
    folder.mkdir(parents=True, exist_ok=True)
    backup = folder / ("crontab-before-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".txt")
    backup.write_text(existing)
    target = folder / "crontab-installed.txt"
    target.write_text(proposed)
    # macOS crontab truncates long input filenames; stdin avoids that limit.
    subprocess.run(["crontab", "-"], input=proposed, text=True, check=True)
    installed = subprocess.run(["crontab", "-l"], check=True, capture_output=True, text=True).stdout
    if installed != proposed:
        raise RuntimeError("Installed crontab did not match the proposed configuration")
    print(f"Installed Denmark daily check (07:00) and weekly calendar refresh (Sunday 09:00).\nBackup: {backup}")


if __name__ == "__main__":
    main()
