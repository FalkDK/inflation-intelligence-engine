# Credentials and security

Store local FRED and BLS API keys only in the repository-root `.env`, copied
from `.env.example`. Never put real credentials in source code, notebooks,
command examples, cron entries, screenshots, or issue reports. The template
contains placeholders only. Environment variables supplied by a CI secret
store are also supported; CI's unit tests do not require provider keys.

`.env`, environment variants, private key files, runtime logs, virtual
environments and Git backups are excluded from publication. The public export
uses an explicit allowlist and does not copy the working Git history.

Before publication, run:

```sh
python scripts/audit_secrets.py --history
```

The audit checks known local `.env` values, common credential patterns,
workbook archive contents, the Git index and project history. It reports paths
only, never the matching values. Pattern checks cannot guarantee that every
unknown secret is detected. Enable GitHub secret scanning and push protection
where available, and review the files before pushing.

If a real key is ever committed or shared, revoke and replace it at the
provider; deleting the visible file alone does not remove it from history.
Do not put suspected credentials into a public issue. Use GitHub's private
vulnerability reporting when enabled on the repository.
