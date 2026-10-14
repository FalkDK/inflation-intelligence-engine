import io
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.audit_secrets import audit_files, findings
from scripts.install_inflation_cron import updated_crontab


class PublicationGuardsTests(unittest.TestCase):
    def test_known_secret_detected_without_returning_value(self):
        secret = b"local-only-value"
        result = findings(b"prefix " + secret, [secret])
        self.assertEqual(result, ["matches a local .env secret"])
        self.assertNotIn(secret.decode(), str(result))

    def test_workbook_archive_members_are_scanned(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("xl/sharedStrings.xml", "hidden-local-value")
        self.assertIn("possible credential in archive member",
                      findings(buffer.getvalue(), [b"hidden-local-value"]))

    def test_env_allowed_locally_but_variant_flagged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".env").write_text("local-only-value")
            (root / ".env.production").write_text("another-value")
            (root / "README.md").write_text("local-only-value")
            issues, _ = audit_files(root, [b"local-only-value"])
        self.assertEqual({path for path, _ in issues}, {".env.production", "README.md"})

    def test_cron_install_preserves_other_jobs_and_is_idempotent(self):
        existing = "0 12 * * 1-5 /some/existing/job\n"
        proposed = updated_crontab(existing)
        self.assertTrue(proposed.startswith(existing))
        self.assertEqual(updated_crontab(proposed), proposed)
        self.assertEqual(proposed.count("# BEGIN inflation-release-automation"), 1)

    def test_cron_install_rejects_unbalanced_managed_block(self):
        with self.assertRaises(ValueError):
            updated_crontab("# BEGIN inflation-release-automation\n")


if __name__ == "__main__":
    unittest.main()
