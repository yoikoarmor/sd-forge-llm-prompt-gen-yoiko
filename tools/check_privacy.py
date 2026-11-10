"""Fail closed on common tracked secrets and local artifacts. Never print values."""
import argparse
import re
import subprocess
import sys
from pathlib import PurePosixPath


PATTERNS = {
    "private-key": rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "github-token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})",
    "api-token": rb"\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{25,})",
    "google-key": rb"AIza[0-9A-Za-z_-]{30,}",
    "aws-key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "personal-home-path": rb"(?:[A-Za-z]:[\\/](?:Users|home)[\\/][^\\/\s]+|/home/[A-Za-z0-9_-]+)",
}
EMAIL = re.compile(rb"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
ALLOWED_EMAIL_DOMAINS = {b"example.com", b"example.org", b"example.net", b"users.noreply.github.com", b"github.com"}


def scan_content(data):
    findings = [name for name, pattern in PATTERNS.items() if re.search(pattern, data)]
    if any(m.group(1).lower() not in ALLOWED_EMAIL_DOMAINS for m in EMAIL.finditer(data)):
        findings.append("non-example-email")
    return findings


def forbidden_path(path):
    p = PurePosixPath(path)
    return ("__pycache__" in p.parts or p.suffix.lower() in
            {".pyc", ".pyo", ".log", ".pem", ".key", ".p12", ".pfx", ".db", ".sqlite", ".sqlite3"}
            or (p.name.startswith(".env") and p.name != ".env.example")
            or path.startswith(("automation/output/", "automation/models/")))


def git(*args):
    return subprocess.check_output(["git", *args])


def introduced_commits(base, head):
    if not base:
        return [head]
    if set(base) == {"0"}:
        return git("rev-list", head).decode().splitlines()
    try:
        resolved = subprocess.check_output(
            ["git", "rev-parse", "--verify", base + "^{commit}"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except subprocess.CalledProcessError:
        # History rewrites can make the event's previous SHA unavailable.
        # Inspect MORE history, never skip the check or silently pass.
        print("Base commit unavailable; scanning all history reachable from head.")
        return git("rev-list", head).decode().splitlines()
    return git("rev-list", resolved + ".." + head).decode().splitlines()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    head = git("rev-parse", "--verify", args.head + "^{commit}").decode().strip()
    commits = introduced_commits(args.base, head)
    failures = []
    seen = set()
    for commit in set(commits + [head]):
        for entry in git("ls-tree", "-rz", commit).split(b"\0"):
            if not entry:
                continue
            meta, raw_path = entry.split(b"\t", 1)
            _, kind, sha = meta.split()
            if kind != b"blob":
                continue
            path = raw_path.decode("utf-8", "replace")
            if forbidden_path(path):
                failures.append((path, "tracked-local-artifact"))
            pair = (sha, path)
            if pair in seen:
                continue
            seen.add(pair)
            data = git("cat-file", "blob", sha.decode())
            # Decode UTF-16 text before checking; also scan raw bytes, including binaries.
            if data.startswith((b"\xff\xfe", b"\xfe\xff")):
                data = data.decode("utf-16", "replace").encode("utf-8")
            failures.extend((path, finding) for finding in scan_content(data))
    for commit in commits:
        metadata = git("show", "-s", "--format=%ae%n%ce%n%B", commit)
        failures.extend(("commit:" + commit[:12], finding) for finding in scan_content(metadata))
    for path, finding in sorted(set(failures)):
        print(f"FAIL {finding}: {path}")
    if failures:
        print("Privacy check failed. Values are intentionally redacted.")
        return 1
    print(f"Privacy check passed: {len(seen)} file versions; {len(commits)} new commits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
