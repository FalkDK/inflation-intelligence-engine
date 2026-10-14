# Release-aware automation

The initial local deployment enables **Denmark only**. Other countries still
refresh through `run_pipeline.py --fetch` when run manually. No files are
published to GitHub or a hosted website by these jobs.

Two cron jobs, interpreted in the Mac's system timezone (currently
Europe/London), drive the automation:

| Job | Schedule | Behaviour |
| --- | --- | --- |
| Calendar | Sunday 09:00 | Refresh two recent months and six upcoming months, retaining older history. |
| Daily check | Every day 07:00 | Read local calendar and Danish headline/core coverage. Exit on ordinary days. |

On release days the daily process sleeps until the UTC release timestamp plus
15 minutes, then fetches Denmark. If the API has not published the expected
month, it retries hourly. Retries stop at local midnight and resume on the next
daily run. An overdue release is checked too, so a missed daily run is recovered
at the next run. Cron does not wake a sleeping Mac or replay a missed job as soon
as it wakes. An always-on host is needed for reliable publication-time fetches.

The first fetch must include both official headline and core observations for
the expected month and preserve existing raw history. Only then is raw data
saved and the pipeline rebuilt without fetching other countries. An explicit
pending-build state lets a failed rebuild retry without downloading again.
Pipeline artifacts are backed up before rebuilding and restored on errors;
the new raw data is retained. Individual files are atomically saved, but the
whole multi-file output directory is not an atomic publication unit. Avoid
running a manual pipeline concurrently with the automated rebuild.

The daily checker and weekly refresh each have their own process lock. The
checker reloads the calendar after a wait, so a revised release time can delay
the fetch. Calendar refreshes retain their previous file on failure. Unknown
release times are never treated as midnight. This first version triggers on
new observation months, not revisions to already-complete observations.

Run manually from the repository root:

```sh
# Local plan only: no network fetch or pipeline rebuild.
/bin/sh scripts/inflation_automation.sh check --dry-run

# One due attempt, without waiting for a future release.
/bin/sh scripts/inflation_automation.sh check

# Same behaviour as the daily cron job, including waits and retries.
/bin/sh scripts/inflation_automation.sh check --watch

# Weekly calendar refresh.
/bin/sh scripts/inflation_automation.sh calendar
```

The wrapper uses `INFLATION_PYTHON`, `.venv/bin/python`, or `python3` in that
order. The cron installer pins the interpreter it is run with, so run it using
the environment where the project's dependencies are installed. It resolves the
repository directory itself and sets `PYTHONPATH` and a writable Matplotlib
cache so cron does not depend on an interactive shell.

Local state and rotating logs are in `outputs/automation/` (Git ignored):

- `automation.log`: checks, planned times, attempts, successes and errors.
- `state.json`: retry timing, pending rebuild and last successful release.
- `cron.log`: shell/startup output from scheduled jobs.

Existing cron entries are preserved. The inflation entries are grouped between
`# BEGIN inflation-release-automation` and `# END inflation-release-automation`.
Use `crontab -l` to inspect them; remove only that block to disable scheduling.

To preview or install the two entries on your own Mac:

```sh
python scripts/install_inflation_cron.py
python scripts/install_inflation_cron.py --install
```
