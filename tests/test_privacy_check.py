import unittest
from tools.check_privacy import forbidden_path, scan_content


class PrivacyChecks(unittest.TestCase):
    def test_runtime_artifacts(self):
        for path in ["x/__pycache__/a.pyc", "x.log", ".env", ".env.local", "runs.sqlite3", "automation/output/example.png"]:
            self.assertTrue(forbidden_path(path), path)
        for path in [".env.example", "README.md", "backend/loader.py"]:
            self.assertFalse(forbidden_path(path), path)

    def test_token_is_detected_without_literal_secret_fixture(self):
        self.assertIn("github-token", scan_content(b"ghp_" + b"A" * 36))
        self.assertIn("google-key", scan_content(b"AIza" + b"B" * 35))

    def test_private_key_header(self):
        self.assertIn("private-key", scan_content(b"-----BEGIN " + b"PRIVATE KEY-----"))

    def test_email_policy(self):
        self.assertIn("non-example-email", scan_content(b"person@" + b"mail.invalid"))
        self.assertEqual([], scan_content(b"person@example.com 123+person@users.noreply.github.com"))

    def test_local_home_path(self):
        self.assertIn("personal-home-path", scan_content(b"C:" + b"/Users/" + b"person/private"))


if __name__ == "__main__":
    unittest.main()
