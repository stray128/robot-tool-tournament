"""scripts/leak_check.sh: finds planted private strings without printing them."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "leak_check.sh"
SECRET_WORD = "zebrafinch-internal"  # a made-up private pattern for this test only
SECRET_ID = "projectabc123xyz"
API_KEY = "v9.QqSyntheticKeyForTestsOnly0000"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, root / "scripts" / "leak_check.sh")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text(".env\nignored/\n")
    (root / "clean.txt").write_text("nothing to see here\n")
    (root / ".env").write_text(f"NEBIUS_API_KEY={API_KEY}\n")
    config = tmp_path / "config"
    config.mkdir()
    (config / "patterns.txt").write_text(f"# comment\n{SECRET_WORD}\n\nab\n")
    (config / "nebius.env").write_text(f"NEBIUS_PROJECT_ID={SECRET_ID}\nNEBIUS_REGION=eu-north1\n")
    env = {
        **os.environ,
        "RTT_LEAK_PATTERNS": str(config / "patterns.txt"),
        "RTT_NEBIUS_ENV": str(config / "nebius.env"),
    }
    return root, env


def run(root: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(root / "scripts" / "leak_check.sh"), *args],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_clean_tree_passes(repo) -> None:
    root, env = repo
    res = run(root, env)
    assert res.returncode == 0, res.stderr
    assert "clean" in res.stdout
    assert "eu-north1" not in res.stdout  # public settings are not patterns


@pytest.mark.parametrize(
    ("text", "source"),
    [
        (f"see {SECRET_WORD.upper()} docs", "leak-patterns.txt"),
        (f"parent {SECRET_ID}", "nebius.env:NEBIUS_PROJECT_ID"),
        (f"key={API_KEY}", ".env:NEBIUS_API_KEY"),
        ("token v9.QqOtherRotatedKey", ".env:NEBIUS_API_KEY prefix"),
    ],
)
def test_planted_strings_are_reported_without_their_text(repo, text: str, source: str) -> None:
    root, env = repo
    (root / "notes.md").write_text(f"line one\n{text}\n")
    res = run(root, env)
    assert res.returncode == 1
    assert f"notes.md:2  [{source}" in res.stderr
    for secret in (SECRET_WORD, SECRET_ID, API_KEY, "v9.Qq"):
        assert secret.lower() not in (res.stdout + res.stderr).lower()


def test_ignored_files_and_dotenv_are_not_scanned(repo) -> None:
    root, env = repo
    (root / "ignored").mkdir()
    (root / "ignored" / "x.txt").write_text(SECRET_WORD)
    assert run(root, env).returncode == 0


def test_staged_and_history_modes(repo) -> None:
    root, env = repo
    (root / "leak.txt").write_text(SECRET_ID)
    subprocess.run(["git", "add", "leak.txt"], cwd=root, check=True)
    assert run(root, env, "--staged").returncode == 1
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "x"],
        cwd=root,
        check=True,
    )
    (root / "leak.txt").unlink()
    assert run(root, env).returncode == 0  # the work tree is clean again
    hist = run(root, env, "--history")
    assert hist.returncode == 1 and "leak.txt:1" in hist.stderr


def test_missing_pattern_file_is_a_setup_error(repo, tmp_path: Path) -> None:
    root, env = repo
    env = {**env, "RTT_LEAK_PATTERNS": str(tmp_path / "missing.txt")}
    assert run(root, env).returncode == 2
