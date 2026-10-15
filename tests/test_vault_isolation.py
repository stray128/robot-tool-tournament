"""Held-out vault isolation: the seed is readable only by explicit path, never during a search.

Checks the vault's commit-reveal API, the audit-hook guard (files, directory listings,
worker threads), that :func:`run_search` runs inside the guard end to end, and that no
search module refers to the vault's reading API or to held-out paths.
"""

from __future__ import annotations

import ast
import os
import stat
import sys
import threading
from pathlib import Path

import pytest

import rtt.search
from rtt.search.fakes import FakeEvaluator
from rtt.search.runner import run_search
from rtt.store.vault import HeldoutVault, VaultAccessError, VaultGuard, guard_active, seed_sha256

SEED = "a1" * 32


@pytest.fixture
def vault(tmp_path: Path) -> HeldoutVault:
    return HeldoutVault.create(tmp_path / "secrets" / "heldout_seed.txt", SEED)


def test_vault_needs_an_explicit_path_and_never_overwrites(tmp_path: Path, vault: HeldoutVault) -> None:
    with pytest.raises(ValueError):
        HeldoutVault("")
    with pytest.raises(TypeError):
        HeldoutVault()  # type: ignore[call-arg]
    assert stat.S_IMODE(vault.path.stat().st_mode) == 0o600
    assert vault.read_seed() == SEED
    assert vault.sha256() == seed_sha256(SEED)
    assert vault.verify(seed_sha256(SEED).upper()) == SEED
    with pytest.raises(ValueError, match="commitment"):
        vault.verify("0" * 64)
    with pytest.raises(FileExistsError):
        HeldoutVault.create(vault.path, SEED)
    fresh = HeldoutVault.create(tmp_path / "other" / "heldout_seed.txt")
    assert len(fresh.read_seed()) == 64 and fresh.read_seed() != SEED
    with pytest.raises(ValueError, match="64 lowercase hex"):
        HeldoutVault.create(tmp_path / "bad.txt", "not-a-seed")


def test_guard_blocks_files_listings_and_threads(tmp_path: Path, vault: HeldoutVault) -> None:
    other = tmp_path / "public.txt"
    other.write_text("ok")
    stray = tmp_path / "elsewhere" / "heldout_seed.backup"
    stray.parent.mkdir()
    stray.write_text(SEED)
    assert not guard_active()
    with VaultGuard([vault.path.parent]) as guard:
        assert guard_active() and guard.protects(vault.path)
        with pytest.raises(VaultAccessError):
            vault.read_seed()
        with pytest.raises(VaultAccessError):
            open(vault.path).close()
        with pytest.raises(VaultAccessError):
            os.listdir(vault.path.parent)
        with pytest.raises(VaultAccessError):
            stray.read_text()  # any file named heldout_seed*, wherever it is
        with pytest.raises(VaultAccessError):
            os.open(os.path.relpath(vault.path), os.O_RDONLY)
        errors: list[BaseException] = []

        def reader() -> None:
            try:
                vault.path.read_bytes()
            except BaseException as exc:  # noqa: BLE001 - the test inspects it
                errors.append(exc)

        thread = threading.Thread(target=reader)
        thread.start()
        thread.join()
        assert len(errors) == 1 and isinstance(errors[0], VaultAccessError)
        assert other.read_text() == "ok"
    assert not guard_active()
    assert vault.read_seed() == SEED


def test_search_runs_inside_the_guard(tmp_path: Path, vault: HeldoutVault) -> None:
    opened: list[str] = []

    def spy(event: str, args: tuple) -> None:
        if event == "open" and isinstance(args[0], str | bytes | os.PathLike):
            opened.append(os.fsdecode(args[0]))

    recording = [True]

    def hook(event: str, args: tuple) -> None:
        if recording[0]:
            spy(event, args)

    sys.addaudithook(hook)
    try:
        run_search(
            "random", 1, 16, [3], FakeEvaluator(0), tmp_path / "out", protected_paths=[vault.path.parent]
        )
    finally:
        recording[0] = False
    secrets_dir = str(vault.path.parent.resolve())
    assert opened, "the spy should have seen the search write its outputs"
    assert not [p for p in opened if os.path.realpath(p).startswith(secrets_dir)]

    class Snooping(FakeEvaluator):
        def evaluate(self, designs):  # type: ignore[no-untyped-def]
            vault.path.read_text()  # what a careless evaluator might try
            return super().evaluate(designs)

    with pytest.raises(VaultAccessError):
        run_search("random", 1, 16, [3], Snooping(0), tmp_path / "snoop", protected_paths=[vault.path.parent])
    assert not guard_active()


def _search_sources() -> list[Path]:
    root = Path(rtt.search.__file__).parent
    return sorted(root.rglob("*.py"))


def test_search_code_never_touches_the_vault_reader() -> None:
    forbidden_names = {"HeldoutVault", "read_seed", "read_secret_seed", "heldout_suite", "seed_commitment"}
    allowed_vault_imports = {"VaultGuard"}
    for path in _search_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("vault"):
                names = {alias.name for alias in node.names}
                assert names <= allowed_vault_imports, (path.name, names)
            if isinstance(node, ast.Import):
                assert not any(alias.name.endswith("vault") for alias in node.names), path.name
            if isinstance(node, ast.Name | ast.Attribute):
                name = node.id if isinstance(node, ast.Name) else node.attr
                assert name not in forbidden_names, (path.name, name)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "heldout_seed" not in node.value and "secrets/" not in node.value, path.name


def test_held_out_prompt_words_are_only_used_by_the_guard() -> None:
    for path in _search_sources():
        text = path.read_text(encoding="utf-8")
        assert "/secret/" not in text and "heldout_v1" not in text, path.name


def test_vault_refuses_unignored_paths_inside_the_repo() -> None:
    repo = Path(rtt.search.__file__).resolve().parents[3]
    if not (repo / ".git").exists():
        pytest.skip("needs the git checkout")
    tracked_dir = repo / "vault-probe-not-ignored"
    with pytest.raises(PermissionError, match="not git-ignored"):
        HeldoutVault.create(tracked_dir / "heldout_seed.txt", SEED)
    assert not tracked_dir.exists()
    ignored_dir = repo / "runs" / "vault-probe-ignored"  # runs/ is git-ignored
    try:
        vault = HeldoutVault.create(ignored_dir / "heldout_seed.txt", SEED)
        assert vault.read_seed() == SEED
    finally:
        (ignored_dir / "heldout_seed.txt").unlink(missing_ok=True)
        if ignored_dir.exists():
            ignored_dir.rmdir()
