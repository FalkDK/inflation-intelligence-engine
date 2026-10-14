"""Check publication files and optionally project Git history without printing secrets.

This is a focused guard, not a replacement for GitHub secret scanning. Known
local .env values are checked literally; common token formats are also checked.
"""

import argparse
import io
from pathlib import Path
import re
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".git.standalone-backup", ".venv", "venv", "__pycache__",
             ".pytest_cache", ".ruff_cache", ".claude", ".codex", ".agents", ".aws", "tmp"}
PATTERNS = [
    re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb'''(?i)(?:FRED_API_KEY|BLS_API_KEY|api_key|registrationkey|access_token|password)\s*[:=]\s*["']([A-Za-z0-9_./+=-]{24,})["']'''),
]


def env_secrets(path: Path):
    if not path.exists():
        return []
    from dotenv import dotenv_values

    return [value.encode() for key, value in dotenv_values(path).items()
            if value and len(value) >= 8 and re.search("KEY|TOKEN|SECRET|PASSWORD", key, re.I)
            and not value.startswith(("your_", "replace_"))]


def findings(data: bytes, secrets):
    messages = []
    if any(secret in data for secret in secrets):
        messages.append("matches a local .env secret")
    if any(pattern.search(data) for pattern in PATTERNS):
        messages.append("possible embedded credential")
    if data.startswith(b"PK"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for item in archive.infolist():
                    if item.file_size < 20_000_000:
                        content = archive.read(item)
                        if any(secret in content for secret in secrets) or any(p.search(content) for p in PATTERNS):
                            messages.append("possible credential in archive member")
                            break
        except zipfile.BadZipFile:
            pass
    return sorted(set(messages))


def audit_files(root: Path, secrets):
    issues, count = [], 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if path.name == ".env":
            continue
        if path.name.startswith(".env.") and path.name != ".env.example":
            issues.append((str(relative), "unexpected environment file outside .env"))
            continue
        count += 1
        issues.extend((str(relative), reason) for reason in findings(path.read_bytes(), secrets))
    return issues, count


def audit_history(root: Path, secrets):
    repo = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root,
                          capture_output=True, text=True, check=True).stdout.strip()
    relative = root.resolve().relative_to(Path(repo).resolve()).as_posix()
    objects = subprocess.run(["git", "rev-list", "--objects", "--all", "--", relative],
                             cwd=repo, capture_output=True, text=True, check=True).stdout
    issues, count = [], 0
    for line in objects.splitlines():
        oid, _, name = line.partition(" ")
        kind = subprocess.run(["git", "cat-file", "-t", oid], cwd=repo,
                              capture_output=True, text=True, check=True).stdout.strip()
        if kind != "blob":
            continue
        count += 1
        data = subprocess.run(["git", "cat-file", "blob", oid], cwd=repo,
                              capture_output=True, check=True).stdout
        if Path(name).name == ".env" or (Path(name).name.startswith(".env.") and Path(name).name != ".env.example"):
            issues.append((name, "environment file exists in Git history"))
        issues.extend((f"{name} (blob {oid[:10]})", reason) for reason in findings(data, secrets))
    staged = subprocess.run(["git", "ls-files", "--stage", "--", relative], cwd=repo,
                            capture_output=True, text=True, check=True).stdout
    for line in staged.splitlines():
        metadata, name = line.split("\t", 1)
        oid = metadata.split()[1]
        if Path(name).name == ".env" or (Path(name).name.startswith(".env.") and Path(name).name != ".env.example"):
            issues.append((name, "environment file exists in Git index"))
        data = subprocess.run(["git", "cat-file", "blob", oid], cwd=repo,
                              capture_output=True, check=True).stdout
        issues.extend((f"{name} (Git index)", reason) for reason in findings(data, secrets))
    return issues, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    secrets = env_secrets(args.env_file)
    issues, count = audit_files(args.root, secrets)
    print(f"Scanned {count} files; checked {len(secrets)} local credential values (values never printed).")
    if args.history:
        historic, blobs = audit_history(args.root, secrets)
        issues.extend(historic)
        print(f"Scanned {blobs} project history blobs and the Git index.")
    for path, reason in issues:
        print(f"FAIL: {path}: {reason}")
    print("PASS: no credential matches found." if not issues else f"FAIL: {len(issues)} findings.")
    raise SystemExit(bool(issues))


if __name__ == "__main__":
    main()
