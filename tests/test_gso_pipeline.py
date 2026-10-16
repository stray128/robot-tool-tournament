"""Real-object held-out pipeline (rtt.pipeline.heldout_real): select, write, read, score.

Offline: a synthetic mirror stands in for the scanned-objects repository (see
tests/test_gso.py), and the suite file is a copy of the repository's with the mirror's
source and convex-hull decomposition.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rtt.design.schema import load_entrants
from rtt.pipeline import heldout_real
from rtt.search.select import FrozenDesign
from rtt.sim import gso
from rtt.sim.m1 import DEFAULT_ENTRANTS
from rtt.store.vault import HeldoutVault, seed_sha256
from test_gso import _write_mirror

ENTRANTS = {d.name: d for d in load_entrants(DEFAULT_ENTRANTS)}


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("real")
    mirror = base / "mirror"
    source = _write_mirror(mirror)
    raw = yaml.safe_load(gso.DEFAULT_CONFIG.read_text())
    raw["name"] = "heldout_real_test"
    raw["source"] = {
        "repo": source.repo,
        "commit": source.commit,
        "models_tree": source.models_tree,
        "n_models": source.n_models,
        "mesh_file": "model.obj",
    }
    raw["draw"].update(n_objects=2, n_dev=1, seeds_per_condition=1, exclude_names=[])
    raw["rules"]["decomposition"]["method"] = "hull"
    raw["denylist"] = {"tokens": ["car"], "name_sha256": []}
    # the dev draw does not depend on exclude_names, so draw it first and pin it
    cache = gso.GsoCache(source, root=base / "cache", fetch=gso.LocalMirror(mirror))
    parsed = gso.gso_config_from_dict(raw)
    dev = gso.select_objects(parsed, gso.dev_seed_material(parsed.draw.dev_seed), mode="dev", cache=cache)
    raw["draw"]["exclude_names"] = [o.name for o in dev.objects]
    unpinned = base / "suite-unpinned.yaml"
    unpinned.write_text(yaml.safe_dump({**raw, "draw": {**raw["draw"], "exclude_names": []}}))
    config = base / "suite.yaml"
    config.write_text(yaml.safe_dump(raw))
    secret = base / "secret" / "heldout_seed.txt"
    HeldoutVault.create(secret)
    return {
        "base": base,
        "mirror": mirror,
        "config": config,
        "unpinned": unpinned,
        "cache": base / "cache",
        "secret": secret,
    }


def _select(setup: dict[str, Path], out: Path, mode: str, **kw: object) -> gso.Selection:
    return heldout_real.select_real(
        out,
        mode=mode,  # type: ignore[arg-type]
        config=setup["config"],
        cache_root=setup["cache"],
        fetch=gso.LocalMirror(setup["mirror"]),
        workers=1,
        **kw,  # type: ignore[arg-type]
    )


def test_dev_selection_writes_a_self_contained_pack(setup: dict[str, Path]) -> None:
    out = setup["base"] / "dev"
    sel = _select(setup, out, "dev")
    assert sel.mode == "dev" and len(sel.objects) == 1
    doc = json.loads((out / "selection.json").read_text())
    assert doc["digest"] == sel.digest() and doc["seed"]["kind"] == "public"
    assert (out / "objects" / f"{sel.objects[0].name}.npz").is_file()
    assert sel.objects[0].name in (out / "ATTRIBUTION.md").read_text()
    back, objects = heldout_real.read_selection(out)
    assert back.digest() == sel.digest() and objects[0].geometry_sha256 == sel.objects[0].geometry_sha256
    with pytest.raises(FileExistsError):
        _select(setup, out, "dev")


def test_scored_selection_needs_the_committed_secret(setup: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="seed file"):
        _select(setup, setup["base"] / "s0", "scored")
    with pytest.raises(ValueError, match="commitment"):
        _select(setup, setup["base"] / "s1", "scored", seed_file=setup["secret"], commitment="0" * 64)
    secret = setup["secret"].read_text().strip()
    out = setup["base"] / "scored"
    sel = _select(setup, out, "scored", seed_file=setup["secret"], commitment=seed_sha256(secret))
    dev = _select(setup, setup["base"] / "dev-for-scored", "dev")
    assert sel.seed_info["dev_excluded"] == [o.name for o in dev.objects]
    assert sel.seed_info["dev_draw"] == [o.name for o in dev.objects]
    assert not {o.name for o in dev.objects} & {o.name for o in sel.objects}
    assert json.loads((out / "selection.json").read_text())["code"]["git_sha"]
    assert sel.seed_info["commitment_verified"] and len(sel.objects) == 2
    assert sel.digest() != dev.digest()
    for path in out.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes(), path


def test_run_scores_entrants_on_the_selection(setup: dict[str, Path]) -> None:
    sel_dir = setup["base"] / "dev-run"
    _select(setup, sel_dir, "dev")
    out = setup["base"] / "run"
    report = heldout_real.run_real(
        sel_dir, out, entrants=[ENTRANTS["straight"]], entrant_settings="all", workers=2
    )
    assert report["schema"] == heldout_real.HELDOUT_REAL_SCHEMA and report["exploratory"]
    assert [d["key"] for d in report["designs"]] == [f"entrant-straight-k{k}" for k in range(3)]
    assert report["episodes"] == 3 * 1 * 1  # settings x objects x seeds
    for d in report["designs"]:
        assert 0.0 <= d["success"] <= 1.0 and d["best_training_setting"] in (0, 1, 2)
        assert set(d["per_object"]) == {report["objects"][0]["object"]}
    assert report["attribution"][0]["license"] == "CC-BY-4.0"
    assert report["suite"]["training_widest_mm"] == 50.0
    assert sum(report["suite"]["objects_by_width"].values()) == 1
    d0 = report["designs"][0]
    assert d0["open_gap_mm"] == 84.0 and set(d0["success_by_width"]) == {"within_training", "wider"}
    assert report["objects"][0]["width_band"] in ("within_training", "wider")
    assert (out / "episodes.parquet").is_file() and (out / "heldout_real.json").is_file()
    with pytest.raises(FileExistsError):
        heldout_real.run_real(sel_dir, out, entrants=[ENTRANTS["straight"]])


def _frozen(design_name: str) -> FrozenDesign:
    d = ENTRANTS[design_name]
    return FrozenDesign(
        arm="C",
        run_index=0,
        run_seed=0,
        label="test",
        design_id=d.design_id(),
        design=d,
        controller_setting=0,
        grasp_height_frac=None,
        training_success=0.5,
        training_hold_fraction=0.5,
        mass_g=10.0,
        origin="test",
        content_sha256="0" * 64,
        frozen_at="2026-10-15T00:00:00Z",
    )


def test_scored_draw_excludes_every_development_object(setup: dict[str, Path], tmp_path: Path) -> None:
    secret = setup["secret"].read_text().strip()
    commitment = seed_sha256(secret)
    # a dev draw that is not pinned in draw.exclude_names stops the scored draw
    with pytest.raises(ValueError, match="exclude_names"):
        heldout_real.select_real(
            tmp_path / "unpinned",
            mode="scored",
            config=setup["unpinned"],
            cache_root=setup["cache"],
            fetch=gso.LocalMirror(setup["mirror"]),
            seed_file=setup["secret"],
            commitment=commitment,
            workers=1,
        )
    assert not (tmp_path / "unpinned" / "selection.json").exists()
    # an object simulated in an earlier dev draw is excluded too, with its own reason
    raw = yaml.safe_load(setup["config"].read_text())
    dev_name = raw["draw"]["exclude_names"][0]
    earlier = next(n for n in ("Small_Box", "Pill_Bottle", "Soap_Bar", "Spice_Jar") if n != dev_name)
    raw["draw"]["exclude_names"] = [dev_name, earlier]
    config = tmp_path / "suite.yaml"
    config.write_text(yaml.safe_dump(raw))
    sel = heldout_real.select_real(
        tmp_path / "scored",
        mode="scored",
        config=config,
        cache_root=setup["cache"],
        fetch=gso.LocalMirror(setup["mirror"]),
        seed_file=setup["secret"],
        commitment=commitment,
        workers=1,
    )
    assert {o.name for o in sel.objects}.isdisjoint({dev_name, earlier}) and len(sel.objects) == 2
    details = {e.name: e.detail for e in sel.exclusions if e.reason == "dev_subset"}
    assert details == {
        dev_name: "drawn for the dev subset",
        earlier: "simulated during development (draw.exclude_names)",
    }
    assert sel.seed_info["dev_excluded"] == sorted([dev_name, earlier])
    # a typo in the list is refused, in dev mode as well
    raw["draw"]["exclude_names"] = [dev_name, "No_Such_Object"]
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="not in the catalogue"):
        heldout_real.select_real(
            tmp_path / "typo",
            mode="dev",
            config=config,
            cache_root=setup["cache"],
            fetch=gso.LocalMirror(setup["mirror"]),
            workers=1,
        )


def test_scored_draw_is_refused_inside_a_search_guard(setup: dict[str, Path], tmp_path: Path) -> None:
    from rtt.store.vault import VaultAccessError, VaultGuard

    with VaultGuard(), pytest.raises(VaultAccessError):
        _select(setup, tmp_path / "guarded", "scored", seed_file=setup["secret"])
    assert not (tmp_path / "guarded" / "selection.json").exists()


def test_scored_report_counts_only_with_a_verified_commitment(setup: dict[str, Path]) -> None:
    secret = setup["secret"].read_text().strip()
    frozen = [_frozen("straight")]
    unverified = setup["base"] / "scored-unverified"
    _select(setup, unverified, "scored", seed_file=setup["secret"], n=1)
    report = heldout_real.run_real(unverified, setup["base"] / "run-unverified", frozen, workers=1)
    assert report["selection"]["mode"] == "scored" and report["exploratory"]
    verified = setup["base"] / "scored-verified"
    _select(setup, verified, "scored", seed_file=setup["secret"], commitment=seed_sha256(secret), n=1)
    report = heldout_real.run_real(verified, setup["base"] / "run-verified", frozen, workers=1)
    assert not report["exploratory"] and report["arms"]["C"]["runs"] == [report["designs"][0]["success"]]


def test_run_refuses_a_tampered_pack(setup: dict[str, Path], tmp_path: Path) -> None:
    sel_dir = setup["base"] / "dev-tamper"
    sel = _select(setup, sel_dir, "dev")
    npz = sel_dir / "objects" / f"{sel.objects[0].name}.npz"
    other = next(p for p in (setup["cache"]).rglob("*.npz") if p.name != npz.name)
    npz.write_bytes(other.read_bytes())
    with pytest.raises(ValueError, match="geometry"):
        heldout_real.run_real(sel_dir, tmp_path / "out", entrants=[ENTRANTS["straight"]])


def test_command_line_select(setup: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    out = setup["base"] / "cli"
    code = heldout_real.main(
        [
            "select",
            "--mode",
            "dev",
            "--out",
            str(out),
            "--config",
            str(setup["config"]),
            "--cache",
            str(setup["cache"]),
            "--mirror",
            str(setup["mirror"]),
            "--workers",
            "1",
        ]
    )
    assert code == 0 and (out / "selection.json").is_file()
    assert "dev selection: 1 objects" in capsys.readouterr().out


def test_width_band_splits_at_the_widest_training_part() -> None:
    assert heldout_real.width_band(0.050, 0.050) == "within_training"
    assert heldout_real.width_band(0.0501, 0.050) == "wider"


def test_scored_cli_reads_the_commitment_file_not_a_typed_value(
    setup: dict[str, Path], tmp_path: Path, capsys
) -> None:
    """`select --mode scored` without --commitment uses the committed file; a missing file refuses."""
    missing = tmp_path / "none.sha256"
    argv = ["select", "--mode", "scored", "--out", str(tmp_path / "a"), "--config", str(setup["config"])]
    argv += [
        "--seed-file",
        str(setup["secret"]),
        "--cache",
        str(setup["cache"]),
        "--mirror",
        str(setup["mirror"]),
    ]
    assert heldout_real.main([*argv, "--commitment-file", str(missing)]) == 2
    assert "needs a commitment" in capsys.readouterr().err
    wrong = tmp_path / "wrong.sha256"
    wrong.write_text("0" * 64 + "  secrets/heldout_seed.txt\n")
    with pytest.raises(ValueError, match="commitment"):
        heldout_real.main([*argv, "--commitment-file", str(wrong)])
    assert heldout_real.read_commitment(wrong) == "0" * 64
    assert heldout_real.read_commitment(missing) is None
