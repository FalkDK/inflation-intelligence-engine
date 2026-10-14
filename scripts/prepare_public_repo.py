"""Create an audited standalone source snapshot, without secrets or Git history."""

import argparse
from pathlib import Path
import shutil

from audit_secrets import ROOT, SKIP_DIRS, audit_files, env_secrets


TOP_FILES = [".env.example", ".gitignore", "README.md", "SECURITY.md", "LICENSE",
             "pyproject.toml", "run_pipeline.py", "inflation-intelligence-engine.md"]
FOLDERS = ["src", "tests", "scripts", "docs", ".github", "data/reference"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/public-github")
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        raise SystemExit("Output already exists; choose a new empty destination.")
    files = [ROOT / name for name in TOP_FILES if (ROOT / name).is_file()]
    for name in FOLDERS:
        folder = ROOT / name
        if not folder.exists():
            continue
        files.extend(path for path in folder.rglob("*") if path.is_file()
                     and not any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts)
                     and path.name != ".DS_Store")
    # Audit before writing anything. Local raw data/logs are not part of this export.
    from audit_secrets import findings

    secrets = env_secrets(ROOT / ".env")
    issues = []
    for path in files:
        if (path.is_symlink() or path.suffix in {".pem", ".key"}
                or (path.name.startswith(".env") and path.name != ".env.example")):
            issues.append((path.relative_to(ROOT), "unexpected environment file or symlink"))
        issues.extend((path.relative_to(ROOT), reason) for reason in findings(path.read_bytes(), secrets))
    if issues:
        for path, reason in issues:
            print(f"FAIL: {path}: {reason}")
        raise SystemExit("Publication audit failed; no snapshot written.")
    destination.mkdir(parents=True)
    for path in files:
        target = destination / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    problems, count = audit_files(destination, secrets)
    if problems:
        raise SystemExit("Export failed its final audit; do not publish it.")
    print(f"Prepared {count} audited files in {destination}")
    print("No .env, raw/processed data, logs, Git backups or prior Git history copied.")
    print("Run tests here before creating a GitHub remote. Nothing was published.")


if __name__ == "__main__":
    main()
