"""Create the GitHub repository and push, without the GitHub CLI.

The build plan called for `gh`, which on macOS normally arrives via Homebrew.
Homebrew was not installed on this machine and installing it prompts for a sudo
password, so the same two operations -- create a repo, push to it -- are done
directly against the GitHub REST API and plain `git` instead. No extra tooling,
no password prompt, identical result.

**Token handling.** ``GITHUB_TOKEN`` is read from ``.env`` into the environment
and passed to git through an inline credential helper that echoes it from that
environment variable. It is therefore never written into ``.git/config``, never
embedded in the remote URL, and never printed. The remote stays a clean
``https://github.com/<user>/<repo>.git`` that is safe to share.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

API = "https://api.github.com"
# Reads the token from the environment at push time; keeps it out of any file.
CREDENTIAL_HELPER = (
    '!f() { echo "username=x-access-token"; echo "password=$GITHUB_TOKEN"; }; f'
)


def _token() -> str:
    from dotenv import load_dotenv

    load_dotenv(config.PROJECT_ROOT / ".env")
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN is not set. Create .env from .env.example and fill it in."
        )
    return token


def _api(path: str, token: str, payload: dict | None = None) -> dict:
    """Call the GitHub API, returning the decoded JSON body."""
    req = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "candlestick-pattern-recognition-system",
            "Content-Type": "application/json",
        },
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def whoami(token: str) -> str:
    """Return the authenticated GitHub login."""
    return _api("/user", token)["login"]


def ensure_repo(token: str, name: str = config.GITHUB_REPO) -> str:
    """Create the public repo if absent; return its clone URL either way.

    Raises:
        RuntimeError: on any API failure other than "already exists".
    """
    user = whoami(token)
    try:
        _api("/user/repos", token, {
            "name": name,
            "private": False,
            "description": (
                "YOLO candlestick pattern detector on rule-based TA-Lib labels, "
                "with an honest test of whether detected patterns improve "
                "next-day return prediction."
            ),
            "has_issues": True,
            "has_wiki": False,
        })
        logger.info("created github.com/%s/%s", user, name)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        if exc.code == 422 and "already exists" in body:
            logger.info("repo github.com/%s/%s already exists", user, name)
        else:
            raise RuntimeError(f"GitHub API {exc.code}: {body}") from exc
    return f"https://github.com/{user}/{name}.git"


def _git(*args: str, cwd: Path | None = None, token: str | None = None) -> str:
    """Run a git command, injecting the credential helper when a token is given."""
    env = dict(os.environ)
    cmd = ["git"]
    if token:
        env["GITHUB_TOKEN"] = token
        cmd += ["-c", f"credential.helper={CREDENTIAL_HELPER}"]
    cmd += list(args)
    proc = subprocess.run(
        cmd, cwd=str(cwd or config.PROJECT_ROOT),
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        # Defensive: never let a token appear in an error message.
        err = proc.stderr.replace(token, "***") if token else proc.stderr
        raise RuntimeError(f"git {' '.join(args)} failed:\n{err}")
    return proc.stdout.strip()


def push(token: str, remote_url: str, branch: str = "main") -> None:
    """Point `origin` at the repo and push the current branch."""
    existing = ""
    try:
        existing = _git("remote", "get-url", "origin")
    except RuntimeError:
        pass
    if existing != remote_url:
        _git("remote", "remove", "origin") if existing else None
        _git("remote", "add", "origin", remote_url)
    _git("branch", "-M", branch)
    _git("push", "-u", "origin", branch, token=token)
    logger.info("pushed to %s (%s)", remote_url, branch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", default=None, help="commit before pushing")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    token = _token()
    url = ensure_repo(token)
    if args.message:
        _git("add", "-A")
        try:
            _git("commit", "-m", args.message)
        except RuntimeError as exc:
            if "nothing to commit" not in str(exc):
                raise
    if not args.no_push:
        push(token, url)
    print(url.removesuffix(".git"))


if __name__ == "__main__":
    main()
