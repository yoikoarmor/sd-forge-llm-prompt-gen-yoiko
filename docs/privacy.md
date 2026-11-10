# Publication privacy checks

- Do not commit runtime logs, bytecode caches, local databases, credentials, or experiment output.
- Use a GitHub noreply author and committer email. Set these locally with `git config user.email` before committing.
- `.gitignore` does not untrack files already in Git. Inspect `git diff --cached` and `git ls-files` before publication.
- Run `python tools/check_privacy.py` against the intended commit, and the tests in `tests/test_privacy_check.py`.
- The Privacy check workflow inspects tracked blobs, common secret formats, home paths, non-example emails, and new commit metadata. For branch pushes with a zero or unavailable base SHA it scans all history reachable from the new head.
- Passing this heuristic check is not proof that every secret or identifying detail is absent. Review documents, images, archives, logs, and licensing manually before publication.
- CI runs after upload and is not a substitute for a pre-push check. A failed check must block release; repository administrators may additionally configure protected-branch requirements.
- Deleting files in a new commit does not remove earlier versions, forks, clones, or cached copies. History sanitization and hosting-provider cache handling are separate procedures.
