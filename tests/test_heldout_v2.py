"""The v2 held-out runners (rtt.heldout_v2): H1, H2, the secondary family, the command lines.

Offline and small: a throwaway secret (never the committed one), four hand-written frozen
picks in two fake search outputs per arm (seeds 1001 and 1002), 1-2 procedural conditions
at one jitter seed, and three objects of the synthetic scan mirror. Physics runs for real
(task_v2) except where a test only checks plumbing.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

import rtt
from heldout_v2_support import COMMITMENT, PICKS, SECRET, fake_simulate, mirror_config, write_search
from rtt.cli import main as rtt_main
from rtt.heldout_v2 import common, procedural, real, secondary
from rtt.sim.gso import LocalMirror
from rtt.store.vault import HeldoutVault, VaultAccessError, VaultGuard

N_ENTRANTS = 6  # 4 round-0 entrants + 2 references (configs/entrants_v2.yaml)


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("heldout_v2")


@pytest.fixture(scope="module")
def searches(base: Path) -> dict[str, Path]:
    root = base / "searches"
    root.mkdir()
    return {arm: write_search(root, arm) for arm in ("random", "nemotron")}


@pytest.fixture(scope="module")
def frozen(searches: dict[str, Path]) -> list[Any]:
    return common.collect_frozen_v2(searches.values())


@pytest.fixture(scope="module")
def scans(base: Path) -> dict[str, Path]:
    return mirror_config(base / "scans")


@pytest.fixture(scope="module")
def h1(base: Path, frozen: list[Any], scans: dict[str, Path]) -> dict[str, Any]:
    out = base / "h1"
    report = procedural.run_heldout_v2(
        scans["secret"], out, frozen, commitment=COMMITMENT, n_conditions=2, seeds_per_condition=1, workers=2
    )
    return {"out": out, "report": report}


@pytest.fixture(scope="module")
def selection(base: Path, scans: dict[str, Path]) -> Path:
    out = base / "selection"
    real.select_real_v2(
        out,
        mode="scored",
        seed_file=scans["secret"],
        commitment=COMMITMENT,
        config=scans["config"],
        workers=1,
        cache_root=scans["cache"],
        fetch=LocalMirror(scans["mirror"]),
    )
    return out


@pytest.fixture(scope="module")
def h2(base: Path, frozen: list[Any], selection: Path) -> dict[str, Any]:
    out = base / "h2"
    return {"out": out, "report": real.run_real_v2(selection, out, frozen, workers=2)}


def _files_without_secret(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            assert SECRET.encode() not in path.read_bytes(), path


# --------------------------------------------------------------------------- frozen picks


def test_collect_frozen_reads_every_layout_and_refuses_bad_sets(
    searches: dict[str, Path], frozen: list[Any], tmp_path: Path
) -> None:
    assert [(f.arm, f.run_index, f.run_seed) for f in frozen] == sorted((a, i, s) for a, i, s, _ in PICKS)
    cloud = write_search(tmp_path, "random", layout="cloud")
    assert common.frozen_digest(common.collect_frozen_v2([cloud])) == common.frozen_digest(
        [f for f in frozen if f.arm == "random"]
    )
    one = searches["random"] / "frozen" / "random-r00.json"
    assert len(common.collect_frozen_v2([one])) == 1
    with pytest.raises(ValueError, match="share an arm and run index"):
        common.collect_frozen_v2([searches["random"], one])
    tampered = tmp_path / "tampered.json"
    data = json.loads(one.read_text())
    data["training_success"] = 1.0
    tampered.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="sha256"):
        common.collect_frozen_v2([tampered])
    with pytest.raises(FileNotFoundError):
        common.collect_frozen_v2([tmp_path / "missing"])
    # the digest ignores order but not content
    assert common.frozen_digest(frozen[::-1]) == common.frozen_digest(frozen)
    assert common.frozen_digest(frozen[:3]) != common.frozen_digest(frozen)


def test_entries_are_the_picks_then_the_six_entrants(frozen: list[Any]) -> None:
    entries = common.held_out_entries(frozen, common.load_entrants())
    roles = [e.role for e in entries]
    assert roles == ["frozen"] * 4 + ["round0_entrant"] * 4 + ["reference_entrant"] * 2
    assert [e.key for e in entries[4:]] == [
        "entrant-stock",
        "entrant-flat",
        "entrant-imprint_configurator",
        "entrant-imprint_swept",
        "entrant-ref_wrap",
        "entrant-ref_hook",
    ]
    assert all(e.frozen_sha256 for e in entries[:4]) and not any(e.frozen_sha256 for e in entries[4:])


# --------------------------------------------------------------------------- H1


def test_h1_scores_every_pick_and_entrant_under_task_v2(h1: dict[str, Any]) -> None:
    report, out = h1["report"], h1["out"]
    assert report["schema"] == "rtt.heldout/2" and report["endpoint"] == "H1"
    assert report["commitment_verified"] and report["commitment"] == COMMITMENT
    assert report["task"]["name"] == "task_v2" and report["generator_version"] == "rtt-freeform-gen/2"
    assert report["suite"]["n_conditions"] == 2 and report["suite"]["episodes_per_design"] == 2
    assert report["suite"]["domain"] == "rtt-heldout-procedural-v2"
    # a development-size run is never an H1 result
    assert report["exploratory"] and any("n_conditions" in r for r in report["exploratory_reasons"])
    assert len(report["designs"]) == 4 + N_ENTRANTS and report["episodes"] == (4 + N_ENTRANTS) * 2
    assert set(report["arms"]) == {"random", "nemotron"}
    for a in report["arms"].values():
        assert [r["run_seed"] for r in a["runs"]] == [1001, 1002] and a["n"] == 2
        assert a["mean"] == pytest.approx(sum(r["success"] for r in a["runs"]) / 2)
    episodes = pd.read_parquet(out / procedural.EPISODES)
    assert len(episodes) == report["episodes"] and set(episodes["task"]) == {"task_v2"}
    for d in report["designs"]:
        rows = episodes[episodes["design_id"].isin({d["design_id"]})]
        if rows.empty:  # a named-recipe entrant's records carry its recipe's id
            continue
        assert d["success"] == pytest.approx(rows["success"].mean())
        assert d["lift_success"] >= d["success"]
        assert set(d["by_size_mode"]["success"]) <= {"near", "down", "up", "unseen"}
    assert report["heldout_code"] == common.code_listing()
    suite = json.loads((out / procedural.SUITE).read_text())
    assert suite["digest"] == report["suite"]["digest"] and len(suite["conditions"]) == 2
    _files_without_secret(out)
    with pytest.raises(FileExistsError):
        procedural.run_heldout_v2(h1["out"].parent / "secret-unused", out, [], commitment=COMMITMENT)


def test_h1_is_deterministic_across_worker_counts(
    h1: dict[str, Any], frozen: list[Any], scans: dict[str, Path], tmp_path: Path
) -> None:
    """Serial builds and in-process episodes give the records of the parallel run."""
    report = procedural.run_heldout_v2(
        scans["secret"],
        tmp_path / "serial",
        frozen,
        commitment=COMMITMENT,
        entrants=[],
        n_conditions=2,
        seeds_per_condition=1,
        workers=1,
    )
    first = {d["key"]: d for d in h1["report"]["designs"]}
    for d in report["designs"]:
        for key in ("design_id", "success", "lift_success", "failure_modes", "by_family", "by_size_mode"):
            assert d[key] == first[d["key"]][key], (d["key"], key)
    assert report["suite"]["digest"] == h1["report"]["suite"]["digest"]
    drop = ["wall_time_s"]
    a = pd.read_parquet(tmp_path / "serial" / procedural.EPISODES).drop(columns=drop)
    b = pd.read_parquet(h1["out"] / procedural.EPISODES).drop(columns=drop)
    b = b[b["design_id"].isin(set(a["design_id"]))].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_h1_full_size_with_the_commitment_is_confirmatory(
    frozen: list[Any], scans: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plumbing only (no physics): the frozen suite at 100 x 2 is the H1 suite."""
    monkeypatch.setattr(common, "simulate", fake_simulate)
    report = procedural.run_heldout_v2(scans["secret"], tmp_path / "full", frozen, commitment=COMMITMENT)
    assert not report["exploratory"] and report["exploratory_reasons"] == []
    assert report["suite"]["n_conditions"] == 100 and report["suite"]["seeds_per_condition"] == 2
    assert report["suite"]["episodes_per_design"] == 200
    assert report["suite"]["summary"]["families"].keys() >= {"pipe_elbow", "spur_gear"}
    # rule 4: a design with no jaws for a part fails every episode there
    assert all(d["success"] == 0.0 and d["invalid_build_episodes"] == 200 for d in report["designs"])
    unchecked = procedural.run_heldout_v2(scans["secret"], tmp_path / "unchecked", frozen, commitment=None)
    assert unchecked["exploratory"] and "commitment not checked" in unchecked["exploratory_reasons"]


def test_procedural_suite_is_the_same_under_either_task() -> None:
    """The suite is drawn with the module's default task (frozen); its only task input, the grip's
    mass cap, is equal in task_v1 and task_v2, so the draw cannot depend on which one."""
    from rtt.parts.heldout import heldout_procedural_suite
    from rtt.sim.task import default_task, default_task_v2

    a = heldout_procedural_suite(SECRET, n=12)
    assert heldout_procedural_suite(SECRET, n=12, task=default_task_v2()).digest() == a.digest()
    assert heldout_procedural_suite(SECRET, n=12, task=default_task()).digest() == a.digest()


def test_h1_reads_the_secret_only_through_the_vault(
    frozen: list[Any], scans: dict[str, Path], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="commitment"):
        procedural.run_heldout_v2(scans["secret"], tmp_path / "wrong", frozen, commitment="0" * 64)
    with VaultGuard([scans["secret"].parent]), pytest.raises(VaultAccessError):
        procedural.run_heldout_v2(scans["secret"], tmp_path / "guarded", frozen, commitment=COMMITMENT)
    assert not (tmp_path / "guarded" / procedural.REPORT).exists()
    assert not (tmp_path / "wrong" / procedural.REPORT).exists()


def test_a_search_cannot_reach_the_held_out_runners(scans: dict[str, Path], tmp_path: Path) -> None:
    """A search runs inside the vault guard, so an evaluator that tries to score held-out
    parts (or to draw the scans) fails, and nothing is written."""
    from rtt.search.fakes import FakeEvaluator
    from rtt.search.runner import run_search

    class Snooping(FakeEvaluator):
        def evaluate(self, designs):  # type: ignore[no-untyped-def]
            procedural.run_heldout_v2(scans["secret"], tmp_path / "snoop-h1", [], commitment=COMMITMENT)
            return super().evaluate(designs)

    with pytest.raises(VaultAccessError):
        run_search(
            "random", 1, 16, [3], Snooping(0), tmp_path / "s", protected_paths=[scans["secret"].parent]
        )
    assert not (tmp_path / "snoop-h1").exists() or not (tmp_path / "snoop-h1" / procedural.REPORT).exists()


def test_no_frozen_or_search_module_imports_the_runners() -> None:
    """Search, simulator, parts, generator and analysis code (frozen in PREREG A5) never
    import rtt.heldout_v2 or the cloud entry: the dependency only runs the other way."""
    src = Path(rtt.__file__).parent
    for package in ("search", "sim", "parts", "design", "analysis", "pipeline", "store"):
        for path in (src / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                for name in names:
                    assert not name.startswith(("rtt.heldout_v2", "rtt.cloud")), (path, name)


# --------------------------------------------------------------------------- H2


def test_scored_selection_records_hull_hashes(selection: Path, scans: dict[str, Path]) -> None:
    from rtt.pipeline.heldout_real import read_selection

    sel, objects = read_selection(selection)
    assert sel.mode == "scored" and sel.seed_info["commitment_verified"] and len(sel.objects) == 3
    hulls = json.loads((selection / real.HULLS).read_text())
    assert hulls == real.hull_hashes(selection)
    assert [o["name"] for o in hulls["objects"]] == [o.name for o in sel.objects]
    assert [o["geometry_sha256"] for o in hulls["objects"]] == [o.geometry_sha256 for o in objects]
    listing = "".join(f"{o['geometry_sha256']}  {o['name']}\n" for o in hulls["objects"])
    assert hulls["hull_listing_sha256"] == hashlib.sha256(listing.encode()).hexdigest()
    dev = set(yaml.safe_load(scans["config"].read_text())["draw"]["exclude_names"])
    assert dev and dev.isdisjoint(o.name for o in sel.objects)
    _files_without_secret(selection)


def test_width_bands_split_at_50_mm() -> None:
    assert real.width_band(0.050) == "width_le_50mm"
    assert real.width_band(0.05 + 1e-12) == "width_le_50mm"  # floating-point noise only
    assert real.width_band(0.050001) == "width_gt_50mm"  # a micrometre over the edge
    assert real.width_band(0.0501) == "width_gt_50mm"
    with pytest.raises(ValueError, match="fixed"):
        real.width_band(0.04, edge_mm=44.3)


def test_h2_scores_the_draw_by_band(h2: dict[str, Any], selection: Path) -> None:
    report, out = h2["report"], h2["out"]
    assert report["schema"] == "rtt.heldout_real/2" and report["endpoint"] == "H2"
    assert report["task"]["name"] == "task_v2"
    assert report["selection"]["screen_task"]["name"] == "task_v1"
    assert report["selection"]["seed"]["commitment_verified"]
    # the test config (a mirror, 3 objects) is not the frozen draw: exploratory
    assert report["exploratory"] and any("config differs" in r for r in report["exploratory_reasons"])
    assert report["suite"]["band_edge_mm"] == 50.0 and report["suite"]["training_widest_mm"] == 44.25
    bands = report["suite"]["objects_by_band"]
    assert set(bands) == set(real.BANDS) and sum(bands.values()) == 3
    for o in report["objects"]:
        assert o["band"] == real.width_band(o["width_mm"] / 1000.0)
    assert report["episodes"] == (4 + N_ENTRANTS) * 3
    for d in report["designs"]:
        assert list(d["by_band"]["success"]) == list(real.BANDS)
        n = {b: c for b, c in d["by_band"]["episodes"].items() if c}
        rates = d["by_band"]["success"]
        whole = sum(rates[b] * c for b, c in n.items()) / sum(n.values())
        assert whole == pytest.approx(d["success"])
    for a in report["arms"].values():
        assert all(set(r["by_band"]["success"]) == set(real.BANDS) for r in a["runs"])
    assert len(report["attribution"]) == 3 and report["attribution"][0]["license"] == "CC-BY-4.0"
    assert len(pd.read_parquet(out / real.EPISODES)) == report["episodes"]
    _files_without_secret(out)


def test_h2_refuses_a_selection_screened_under_another_task(scans: dict[str, Path], tmp_path: Path) -> None:
    from rtt.pipeline.heldout_real import select_real
    from rtt.sim.task import default_task_v2

    sel = tmp_path / "v2-screen"
    select_real(
        sel,
        mode="dev",
        config=scans["config"],
        workers=1,
        cache_root=tmp_path / "cache",
        fetch=LocalMirror(scans["mirror"]),
        task=default_task_v2(),
    )
    with pytest.raises(ValueError, match="screened"):
        real.run_real_v2(sel, tmp_path / "out", entrants=common.load_entrants()[:1])


def test_scored_scan_draw_needs_the_vault(scans: dict[str, Path], tmp_path: Path) -> None:
    kw: dict[str, Any] = {
        "mode": "scored",
        "config": scans["config"],
        "workers": 1,
        "cache_root": scans["cache"],
        "fetch": LocalMirror(scans["mirror"]),
    }
    with pytest.raises(ValueError, match="commitment"):
        real.select_real_v2(tmp_path / "a", seed_file=scans["secret"], commitment="0" * 64, **kw)
    with VaultGuard([scans["secret"].parent]), pytest.raises(VaultAccessError):
        real.select_real_v2(tmp_path / "b", seed_file=scans["secret"], commitment=COMMITMENT, **kw)
    assert not (tmp_path / "b" / "selection.json").exists()


# --------------------------------------------------------------------------- secondary family


def test_secondary_family_s2_h1_h2_holm(
    searches: dict[str, Path], h1: dict[str, Any], h2: dict[str, Any]
) -> None:
    kw = {"heldout": h1["out"], "heldout_real": h2["out"]}
    # the small held-out reports are exploratory, so the confirmatory analysis refuses them
    with pytest.raises(ValueError, match="exploratory"):
        secondary.analyze([searches["random"]], [searches["nemotron"]], **kw)
    result = secondary.analyze([searches["random"]], [searches["nemotron"]], confirmatory=False, **kw)
    assert result["schema"] == "rtt.secondary_v2/1" and result["family"] == ["H1", "H2", "S2"]
    assert "S3" in result["not_run"]
    p = {k: result["tests"][k]["p"] for k in result["family"]}
    order = sorted(p, key=p.get)
    expect, running = {}, 0.0
    for i, k in enumerate(order):  # Holm step-down by hand
        running = max(running, min(1.0, (3 - i) * p[k]))
        expect[k] = running
    for k in p:
        t = result["tests"][k]
        assert t["p_holm"] == pytest.approx(expect[k]) and t["p_holm"] >= t["p"]
        assert t["rejected"] == (t["p_holm"] <= 0.05)
        assert t["n_a"] == t["n_c"] == 2 and "exact permutation over C(4,2)" in t["method"]
    h1_runs = {r["run_seed"]: r["success"] for r in h1["report"]["arms"]["random"]["runs"]}
    assert result["tests"]["H1"]["values_a"] == [h1_runs[1001], h1_runs[1002]]
    from rtt.analysis.s1 import best_so_far_area
    from rtt.analysis.tau import load_runs_v2

    areas = [
        best_so_far_area(r.scores) for r in sorted(load_runs_v2(searches["random"]), key=lambda r: r.run_seed)
    ]
    assert result["tests"]["S2"]["values_a"] == areas
    bands = result["exploratory"]["H2_width_bands"]
    assert set(bands) >= {"width_le_50mm", "width_gt_50mm", "note", "band_edge_mm"}
    assert {"S2_lift_only", "H1_lift_only", "H2_lift_only"} <= set(result["exploratory"])
    # deterministic
    again = secondary.analyze([searches["random"]], [searches["nemotron"]], confirmatory=False, **kw)
    assert {k: v["p"] for k, v in again["tests"].items()} == {k: v["p"] for k, v in result["tests"].items()}


def test_secondary_refuses_a_report_from_other_frozen_picks(
    searches: dict[str, Path], h1: dict[str, Any], tmp_path: Path
) -> None:
    report = json.loads((h1["out"] / procedural.REPORT).read_text())
    report["arms"]["random"]["runs"][0]["frozen_sha256"] = "0" * 64
    bad = tmp_path / procedural.REPORT
    bad.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="another frozen record"):
        secondary.analyze([searches["random"]], [searches["nemotron"]], heldout=bad, confirmatory=False)
    report["arms"]["random"]["runs"] = report["arms"]["random"]["runs"][:1]
    bad.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="seeds"):
        secondary.analyze([searches["random"]], [searches["nemotron"]], heldout=bad, confirmatory=False)
    only_s2 = secondary.analyze([searches["random"]], [searches["nemotron"]], confirmatory=False)
    assert only_s2["family"] == ["S2"] and only_s2["tests"]["S2"]["p_holm"] == only_s2["tests"]["S2"]["p"]


def test_secondary_applies_the_confirmatory_checks(searches: dict[str, Path], tmp_path: Path) -> None:
    seeds = tmp_path / "seeds.yaml"
    seeds.write_text("seeds: [2001, 2002]\n")
    with pytest.raises(ValueError, match="confirmatory seeds"):
        secondary.analyze([searches["random"]], [searches["nemotron"]], seeds_file=seeds)
    with pytest.raises(ValueError, match="nemotron arm holds runs labelled"):
        secondary.analyze([searches["random"]], [searches["random"]])
    ok = secondary.analyze([searches["random"]], [searches["nemotron"]])  # S2 alone passes the checks
    assert ok["confirmatory"] and ok["family"] == ["S2"]


# --------------------------------------------------------------------------- command lines


def test_command_lines_end_to_end(
    searches: dict[str, Path], selection: Path, scans: dict[str, Path], tmp_path: Path, capsys: Any
) -> None:
    frozen_args = [str(searches["random"]), str(searches["nemotron"])]
    h1_out = tmp_path / "h1"
    code = rtt_main(
        ["heldout", "--space", "freeform-v2", "--seed-file", str(scans["secret"]), "--commitment-file",
         str(scans["commitment_file"]), "--frozen", *frozen_args, "--out", str(h1_out), "--n", "1",
         "--seeds-per-condition", "1", "--no-entrants", "--workers", "2"]
    )  # fmt: skip
    assert code == 0 and "commitment verified" in capsys.readouterr().out
    h1_report = json.loads((h1_out / procedural.REPORT).read_text())
    assert h1_report["episodes"] == 4 and h1_report["commitment"] == COMMITMENT
    # a typed-in wrong commitment is refused; a missing commitment file needs --allow-uncommitted
    assert rtt_main(
        ["heldout", "--space", "v2", "--seed-file", str(scans["secret"]), "--commitment", "0" * 64,
         "--frozen", *frozen_args, "--out", str(tmp_path / "x"), "--n", "1", "--no-entrants"]
    ) == 2  # fmt: skip
    assert rtt_main(
        ["heldout", "--space", "v2", "--seed-file", str(scans["secret"]), "--commitment-file",
         str(tmp_path / "none"), "--frozen", *frozen_args, "--out", str(tmp_path / "y")]
    ) == 2  # fmt: skip
    capsys.readouterr()
    # H2 in one command: the scored draw, then the scores
    h2_out = tmp_path / "h2"
    code = rtt_main(
        ["heldout-real", "run", "--space", "freeform-v2", "--seed-file", str(scans["secret"]),
         "--commitment-file", str(scans["commitment_file"]), "--config", str(scans["config"]),
         "--cache", str(scans["cache"]), "--mirror", str(scans["mirror"]), "--select-workers", "1",
         "--frozen", *frozen_args, "--no-entrants", "--out", str(h2_out), "--workers", "2"]
    )  # fmt: skip
    assert code == 0, capsys.readouterr()
    h2_report = json.loads((h2_out / real.REPORT).read_text())
    assert h2_report["selection"]["mode"] == "scored" and (h2_out / "selection" / real.HULLS).is_file()
    # the same draw as the fixture's (a pure function of the secret and the rule)
    assert (
        h2_report["selection"]["digest"] == json.loads((selection / "selection.json").read_text())["digest"]
    )
    out = tmp_path / "secondary"
    code = rtt_main(
        ["analyze-secondary", "--random", str(searches["random"]), "--nemotron", str(searches["nemotron"]),
         "--heldout", str(h1_out), "--heldout-real", str(h2_out), "--exploratory", "--out", str(out)]
    )  # fmt: skip
    text = capsys.readouterr().out
    assert code == 0 and "H1:" in text and "S2:" in text
    result = json.loads((out / secondary.OUT).read_text())
    assert result["family"] == ["H1", "H2", "S2"] and all(
        math.isfinite(t["p"]) for t in result["tests"].values()
    )
    _files_without_secret(tmp_path)


def test_heldout_real_space_routing_keeps_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    from rtt import cli

    seen: list[tuple[str, list[str]]] = []
    monkeypatch.setattr("rtt.pipeline.heldout_real.main", lambda argv: seen.append(("v1", list(argv))) or 0)
    monkeypatch.setattr("rtt.heldout_v2.real.main", lambda argv: seen.append(("v2", list(argv))) or 0)
    assert cli.main(["heldout-real", "select", "--mode", "dev", "--out", "x"]) == 0
    assert cli.main(["heldout-real", "select", "--space", "box-v1", "--mode", "dev", "--out", "y"]) == 0
    assert cli.main(["heldout-real", "run", "--space=freeform-v2", "--out", "z"]) == 0
    assert seen == [
        ("v1", ["select", "--mode", "dev", "--out", "x"]),  # v1 unchanged: no --space reaches it
        ("v1", ["select", "--mode", "dev", "--out", "y"]),
        ("v2", ["run", "--space=freeform-v2", "--out", "z"]),
    ]
    assert cli.main(["heldout-real", "run", "--space", "box-v3", "--out", "z"]) == 2


def test_vault_create_and_read_round_trip(tmp_path: Path) -> None:
    """The fixtures' secrets go through the vault like the real one (64 hex digits, mode 600)."""
    vault = HeldoutVault.create(tmp_path / "s" / "heldout_seed.txt", SECRET)
    assert vault.verify(COMMITMENT) == SECRET
