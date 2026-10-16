"""Real-object held-out set (rtt.sim.gso): catalogue, category rule, screening, draw, suite.

Everything runs offline against a synthetic mirror of the scanned-objects repository
(``LocalMirror``), with convex-hull decomposition so the tests stay fast. One test reads
the real catalogue from GitHub; it runs only with ``RTT_GSO_LIVE=1``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Self

import numpy as np
import pytest
import trimesh

from rtt.design.schema import load_entrants
from rtt.sim import gso
from rtt.sim import objects as objs
from rtt.sim.evaluate import evaluate_designs
from rtt.sim.m1 import DEFAULT_ENTRANTS
from rtt.sim.mesh import read_mesh_object, save_mesh_object
from rtt.sim.task import load_task

TASK = load_task()
ENTRANTS = {d.name: d for d in load_entrants(DEFAULT_ENTRANTS)}
HASHED_NAME = "Zorbium_Widget"  # stands in for an object the config lists only by name hash


def _box(x: float, y: float, z: float) -> trimesh.Trimesh:
    return trimesh.creation.box((x, y, z))


def _objects() -> dict[str, trimesh.Trimesh | bytes]:
    return {
        "Small_Box": _box(0.040, 0.060, 0.050),
        "Pill_Bottle": trimesh.creation.cylinder(radius=0.022, height=0.080, sections=32),
        "Soap_Bar": _box(0.090, 0.055, 0.030),
        "Spice_Jar": trimesh.creation.cylinder(radius=0.018, height=0.060, sections=24),
        "Big_Cube": _box(0.100, 0.100, 0.100),
        "Flat_Card": _box(0.060, 0.040, 0.003),
        "Long_Stick": _box(0.300, 0.020, 0.020),
        "Heavy_Block": _box(0.200, 0.070, 0.070),
        "Toy_Car_Red": _box(0.050, 0.030, 0.030),
        "Zorbium_Widget": _box(0.050, 0.030, 0.030),
        "Broken_Scan": b"this is not a mesh\n",
    }


def _write_mirror(root: Path, extra_listed: tuple[str, ...] = ("Missing_File",)) -> gso.GsoSource:
    """A mirror with the synthetic objects; ``extra_listed`` names are listed but have no file."""
    entries = []
    for name, mesh in _objects().items():
        path = root / "models" / name / "model.obj"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = mesh if isinstance(mesh, bytes) else mesh.export(file_type="obj").encode()
        path.write_bytes(data)
        entries.append(
            {"path": name, "mode": "040000", "type": "tree", "sha": hashlib.sha1(data).hexdigest()}
        )
    for name in extra_listed:
        entries.append({"path": name, "mode": "040000", "type": "tree", "sha": "0" * 40})
    tree = gso.git_tree_sha1(entries)
    (root / "models_tree.json").write_text(json.dumps({"sha": tree, "tree": entries, "truncated": False}))
    return gso.GsoSource(repo="test/objects", commit="a" * 40, models_tree=tree, n_models=len(entries))


def _config(source: gso.GsoSource, **draw: int) -> gso.GsoConfig:
    rules = gso.ScreenRules(decomposition=gso.DecompositionRules(method="hull"))
    return gso.GsoConfig(
        name="heldout_real_test",
        source=source,
        draw=gso.DrawConfig(**{"n_objects": 3, "n_dev": 1, "dev_seed": 7, **draw}),
        placement=gso.PlacementRanges(),
        rules=rules,
        denylist=gso.Denylist(
            tokens=frozenset({"car"}),
            name_sha256=frozenset({hashlib.sha256(HASHED_NAME.encode()).hexdigest()}),
        ),
    )


@pytest.fixture(scope="module")
def mirror(tmp_path_factory: pytest.TempPathFactory) -> tuple[gso.GsoConfig, gso.GsoCache]:
    root = tmp_path_factory.mktemp("mirror")
    source = _write_mirror(root)
    cache = gso.GsoCache(source, root=tmp_path_factory.mktemp("cache"), fetch=gso.LocalMirror(root))
    return _config(source), cache


@pytest.fixture(scope="module")
def screened(mirror: tuple[gso.GsoConfig, gso.GsoCache]) -> dict[str, gso.CandidateResult]:
    cfg, cache = mirror
    return {
        n: gso.evaluate_candidate(n, cache, cfg.rules, TASK)
        for n in _objects()
        if cfg.denylist.match(n) is None
    }


# --------------------------------------------------------------------------- the repository's config


def test_repository_config_matches_the_task_and_the_plan() -> None:
    cfg = gso.load_gso_config()
    assert cfg.name == "heldout_real_v1"
    assert len(cfg.source.commit) == 40 and len(cfg.source.models_tree) == 40
    assert cfg.source.n_models == 1030
    assert 50 <= cfg.draw.n_objects <= 100 and cfg.draw.n_dev == 10 and cfg.draw.exclude_dev
    # 84 mm opening minus the synthetic held-out suite's 5 mm margin
    assert cfg.rules.max_grip_width_mm == pytest.approx(2000.0 * TASK.gripper.open_half_gap - 5.0)
    # mu_min F / (g * 1.5) with mu_min = 0.3 * 0.9 and F = 50 N
    assert cfg.rules.mass_cap_kg(TASK) == pytest.approx(0.27 * 50.0 / (9.81 * 1.5))
    assert cfg.rules.object_density(TASK) == TASK.physics.object_density
    assert cfg.rules.rest.duration_s == 2.0
    assert all(len(h) == 64 for h in cfg.denylist.name_sha256)
    # every hashed entry is a full catalogue name, so the hashes can not encode words
    assert len(cfg.denylist.name_sha256) == 6
    for name in ("CITY_TAXI_POLICE_CAR", "SCHOOL_BUS", "FIRE_ENGINE", "Melissa_Doug_Chunky_Puzzle_Vehicles"):
        assert cfg.denylist.match(name) is not None, name
    for name in ("ACE_Coffee_Mug_Kristen_16_oz_cup", "Crayola_Crayons_24_count", "Cartridge_Box"):
        assert cfg.denylist.match(name) is None, name
    assert cfg.digest() == gso.load_gso_config().digest()
    # the objects simulated in development: both dev draws (10 + 6 not in the current one)
    names = cfg.draw.exclude_names
    assert len(names) == 16 and list(names) == sorted(names)
    assert "Crayola_Crayons_24_count" in names and all(cfg.denylist.match(n) is None for n in names)


def test_draw_config_validates_excluded_names() -> None:
    with pytest.raises(ValueError):
        gso.DrawConfig(exclude_names=("../escape",))
    with pytest.raises(ValueError, match="duplicates"):
        gso.DrawConfig(exclude_names=("A_B", "A_B"))
    raw = gso.load_gso_config().describe()
    raw["draw"]["exclude_names"] = "Lutein"
    with pytest.raises(TypeError, match="exclude_names"):
        gso.gso_config_from_dict(raw)


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    raw = gso.load_gso_config().describe()
    raw["rules"]["rest"]["max_wobble"] = 1.0
    with pytest.raises(ValueError, match="rules.rest"):
        gso.gso_config_from_dict(raw)
    raw = gso.load_gso_config().describe()
    raw["extra"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        gso.gso_config_from_dict(raw)


# --------------------------------------------------------------------------- source, catalogue, names


def test_urls_are_pinned_to_the_commit_and_quoted() -> None:
    src = gso.load_gso_config().source
    url = src.raw_url("Pokémon_Yellow")
    assert url.startswith(
        f"https://raw.githubusercontent.com/{src.repo}/{src.commit}/models/Pok%C3%A9mon_Yellow/"
    )
    assert url.endswith("/model.obj")
    assert src.commit in src.page_url("Small_Box")
    for bad in ("../etc", "a/b", "", ".hidden"):
        with pytest.raises(ValueError):
            src.raw_url(bad)


def test_catalogue_is_verified_against_the_pinned_tree(tmp_path: Path) -> None:
    root = tmp_path / "mirror"
    source = _write_mirror(root)
    cache = gso.GsoCache(source, root=tmp_path / "cache", fetch=gso.LocalMirror(root))
    names = cache.catalogue()
    assert list(names) == sorted(names) and len(names) == source.n_models
    assert (cache.dir / "catalogue.json").is_file()

    wrong = replace(source, models_tree="b" * 40)
    with pytest.raises(ValueError, match="does not hash"):
        gso.GsoCache(wrong, root=tmp_path / "cache2", fetch=gso.LocalMirror(root)).catalogue()
    short = replace(source, n_models=source.n_models + 1)
    with pytest.raises(ValueError, match="expected"):
        gso.GsoCache(short, root=tmp_path / "cache3", fetch=gso.LocalMirror(root)).catalogue()


def test_git_tree_hash_matches_git() -> None:
    # `git mktree` of these two subtree entries (both the empty tree) gives this id
    empty = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
    entries = [
        {"path": "b", "mode": "040000", "type": "tree", "sha": empty},
        {"path": "a", "mode": "040000", "type": "tree", "sha": empty},
    ]
    body = b"".join(b"40000 " + n + b"\0" + bytes.fromhex(empty) for n in (b"a", b"b"))
    assert gso.git_tree_sha1(entries) == hashlib.sha1(b"tree %d\0" % len(body) + body).hexdigest()


def test_mirror_refuses_paths_outside_it(tmp_path: Path) -> None:
    mirror = gso.LocalMirror(tmp_path)
    with pytest.raises(ValueError):
        mirror("https://raw.githubusercontent.com/o/r/" + "a" * 40 + "/../../secret.txt")
    with pytest.raises(ValueError):
        mirror("https://example.com/models/x/model.obj")


class _Response:
    def __init__(self, final_url: str, body: bytes) -> None:
        self.final_url, self.body = final_url, body

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def geturl(self) -> str:
        return self.final_url

    def read(self, n: int = -1) -> bytes:
        return self.body[:n] if n >= 0 else self.body


def test_http_fetcher_refuses_plain_http_and_oversize(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = gso.HttpFetcher(retries=0, max_bytes=8)
    with pytest.raises(ValueError, match="non-HTTPS URL"):
        fetch("http://example.com/x")
    monkeypatch.setattr(gso.urllib.request, "urlopen", lambda *a, **k: _Response("http://evil/x", b"ok"))
    with pytest.raises(ValueError, match="redirected"):
        fetch("https://raw.githubusercontent.com/o/r/c/models/x/model.obj")
    monkeypatch.setattr(gso.urllib.request, "urlopen", lambda *a, **k: _Response("https://x/y", b"0" * 9))
    with pytest.raises(ValueError, match="larger"):
        fetch("https://raw.githubusercontent.com/o/r/c/models/x/model.obj")
    monkeypatch.setattr(gso.urllib.request, "urlopen", lambda *a, **k: _Response("https://x/y", b"0" * 8))
    assert fetch("https://raw.githubusercontent.com/o/r/c/models/x/model.obj") == b"0" * 8


def test_category_rule_on_tokens_and_name_hashes() -> None:
    deny = gso.Denylist(
        tokens=frozenset({"car", "truck"}),
        name_sha256=frozenset({hashlib.sha256(HASHED_NAME.encode()).hexdigest()}),
    )
    assert gso.name_tokens("CITY_TAXI-Car 2") == ("city", "taxi", "car", "2")
    assert deny.match("Toy_CAR_Red") == ("token", "car")
    assert deny.match(HASHED_NAME) == ("name_hash", None)
    assert deny.match("Zorbium_Widget_2") is None  # whole names, not tokens or prefixes
    assert deny.match("Cartridge_Box") is None  # tokens, not substrings
    with pytest.raises(ValueError):
        gso.Denylist(tokens=frozenset({"two words"}))
    with pytest.raises(ValueError):
        gso.Denylist(name_sha256=frozenset({"abc"}))


def test_config_refuses_word_hashes() -> None:
    # hashes of single words are reversed with a word list; only whole names may be hashed
    raw = gso.load_gso_config().describe()
    raw["denylist"]["token_sha256"] = [hashlib.sha256(b"widget").hexdigest()]
    with pytest.raises(ValueError, match="unknown keys"):
        gso.gso_config_from_dict(raw)


def test_draw_order_is_a_seeded_permutation() -> None:
    names = [f"obj_{i:02d}" for i in range(20)]
    a = gso.draw_order(names, gso.dev_seed_material(1))
    assert sorted(a) == names
    assert a == gso.draw_order(list(reversed(names)), gso.dev_seed_material(1))  # input order irrelevant
    assert a != gso.draw_order(names, gso.dev_seed_material(2))
    secret = "ab" * 32
    assert a != gso.draw_order(names, gso.secret_seed_material(secret))
    with pytest.raises(ValueError):
        gso.secret_seed_material("not-a-secret")


# --------------------------------------------------------------------------- screening


def test_pose_fit_uses_the_narrowest_horizontal_extent() -> None:
    mesh = gso.geometry_mesh(_box(0.040, 0.060, 0.050).export(file_type="obj").encode())
    fits = gso.pose_fits(mesh)
    assert fits[0].probability >= fits[-1].probability
    top = fits[0]  # rests on its largest face (60 x 50): width 50, length 60, height 40
    assert (top.width, top.length, top.height) == pytest.approx((0.050, 0.060, 0.040), abs=1e-6)
    rotated = trimesh.creation.box((0.040, 0.060, 0.050))
    rotated.apply_transform(trimesh.transformations.rotation_matrix(0.6, (0, 0, 1)))
    turned = gso.pose_fits(gso.geometry_mesh(rotated.export(file_type="obj").encode()))[0]
    assert turned.width == pytest.approx(0.050, abs=1e-6)


def test_choose_pose_takes_the_most_probable_fitting_pose() -> None:
    rules = gso.ScreenRules(min_pose_probability=0.0)
    fits = [
        gso.PoseFit(0, 0.6, 0.120, 0.150, 0.020),  # too wide
        gso.PoseFit(1, 0.3, 0.020, 0.150, 0.120),  # fits
        gso.PoseFit(2, 0.1, 0.020, 0.120, 0.150),  # fits, less probable
    ]
    fit, reason, _ = gso.choose_pose(fits, rules)
    assert fit is not None and fit.index == 1 and reason is None
    strict = gso.ScreenRules(min_pose_probability=0.5)
    fit, reason, detail = gso.choose_pose(fits, strict)
    assert fit is None and reason == "too_wide" and "pose 0" in detail


def test_mass_cap_follows_the_grip_force() -> None:
    rules = gso.ScreenRules(mass_safety_factor=1.0)
    stronger = TASK.with_overrides(gripper={"grip_force": 100.0})
    assert rules.mass_cap_kg(stronger) == pytest.approx(2.0 * rules.mass_cap_kg(TASK))
    assert gso.ScreenRules().mass_cap_kg(TASK) == pytest.approx(rules.mass_cap_kg(TASK) / 1.5)


def test_screening_reasons(screened: dict[str, gso.CandidateResult]) -> None:
    expect = {
        "Small_Box": None,
        "Pill_Bottle": None,
        "Soap_Bar": None,
        "Spice_Jar": None,
        "Big_Cube": "too_wide",
        "Flat_Card": "too_flat",
        "Long_Stick": "too_long",
        "Heavy_Block": "too_heavy",
        "Broken_Scan": "mesh_invalid",
    }
    for name, reason in expect.items():
        res = screened[name]
        assert res.reason == reason, (name, res.detail)
        assert res.accepted is (reason is None)
    box = screened["Small_Box"]
    assert box.file_sha256 and box.geometry_sha256 and box.n_parts == 1 and box.watertight
    assert box.size_m is not None and box.size_m[1] == pytest.approx(0.050, abs=1e-6)
    assert box.mass_kg == pytest.approx(0.040 * 0.060 * 0.050 * TASK.physics.object_density)
    assert box.rest is not None and box.rest["shift_mm"] < 0.5 and box.rest["tilt_deg"] < 0.5
    heavy = screened["Heavy_Block"]
    assert heavy.mass_kg == pytest.approx(0.2 * 0.07 * 0.07 * 1200.0)


def test_missing_file_is_a_fetch_failure(mirror: tuple[gso.GsoConfig, gso.GsoCache]) -> None:
    cfg, cache = mirror
    res = gso.evaluate_candidate("Missing_File", cache, cfg.rules, TASK)
    assert not res.accepted and res.reason == "fetch_failed"


class _FlakyFetcher:
    def __call__(self, url: str) -> bytes:
        raise ConnectionError("network down")


def test_network_errors_abort_instead_of_excluding(tmp_path: Path) -> None:
    root = tmp_path / "mirror"
    source = _write_mirror(root)
    cfg = _config(source)
    cache = gso.GsoCache(source, root=tmp_path / "cache", fetch=gso.LocalMirror(root))
    cache.catalogue()
    flaky = gso.GsoCache(source, root=tmp_path / "cache", fetch=_FlakyFetcher())
    with pytest.raises(ConnectionError):
        gso.evaluate_candidate("Small_Box", flaky, cfg.rules, TASK)
    assert not (flaky.screen_dir(cfg.rules.key(TASK)) / "Small_Box.json").exists()


def test_screening_results_are_cached(mirror: tuple[gso.GsoConfig, gso.GsoCache], screened: dict) -> None:
    cfg, cache = mirror
    key = cfg.rules.key(TASK)
    assert (cache.screen_dir(key) / "Small_Box.json").is_file()
    assert (cache.objects_dir(key) / "Small_Box.npz").is_file()
    again = gso.evaluate_candidate("Small_Box", cache, cfg.rules, TASK)
    assert again.stable_dict() == screened["Small_Box"].stable_dict()
    assert key != gso.ScreenRules().key(TASK)  # the decomposition method is part of the key


def test_rest_test_fails_a_part_balanced_on_a_point() -> None:
    cone = trimesh.creation.cone(radius=0.03, height=0.06, sections=24)
    # apex down and tilted 10 deg, so the centre of mass is not above the contact point
    cone.apply_transform(trimesh.transformations.rotation_matrix(math.pi + 0.17, (1, 0, 0)))
    lo, hi = cone.bounds
    cone.apply_translation(-(lo + hi) / 2)
    inertia = cone.moment_inertia  # trimesh's default density is 1, so this is per unit density
    obj = objs.MeshObject(
        name="balanced_cone",
        parts=(np.asarray(cone.vertices),),
        faces=(np.asarray(cone.faces, dtype=np.int32),),
        volume=float(cone.volume),
        com=tuple(float(v) for v in cone.center_mass),  # type: ignore[arg-type]
        unit_inertia=(
            inertia[0, 0],
            inertia[1, 1],
            inertia[2, 2],
            inertia[0, 1],
            inertia[0, 2],
            inertia[1, 2],
        ),
    )
    check = gso.rest_test(obj, TASK)
    assert not check.passed and check.reason == "rest_unstable"
    assert check.tilt_deg > 5.0
    upright = gso.rest_test(objs.ObjectSpec("box", (0.04, 0.05, 0.06)), TASK)
    assert upright.passed and upright.shift_mm < 0.5 and not upright.gripper_contact


# --------------------------------------------------------------------------- selection


def test_selection_walks_the_draw_order(mirror: tuple[gso.GsoConfig, gso.GsoCache]) -> None:
    cfg, cache = mirror
    material = gso.dev_seed_material(3)
    seen: list[int] = []
    sel = gso.select_objects(
        cfg, material, mode="dev", n=2, cache=cache, task=TASK, progress=lambda i, _r: seen.append(i)
    )
    assert len(sel.objects) == 2 and all(o.accepted for o in sel.objects)
    assert seen == list(range(sel.examined))
    order = gso.draw_order([n for n in cache.catalogue() if cfg.denylist.match(n) is None], material)
    accepted_positions = [order.index(o.name) for o in sel.objects]
    assert accepted_positions == sorted(accepted_positions) and accepted_positions[-1] == sel.examined - 1
    examined = {e.name for e in sel.exclusions if e.position >= 0}
    assert examined | {o.name for o in sel.objects} == set(order[: sel.examined])

    category = [e for e in sel.exclusions if e.reason == "category"]
    assert len(category) == 2 and all(e.position == -1 for e in category)
    hashed = next(e for e in category if e.detail == "name listed by hash")
    assert hashed.name is None and hashed.name_sha256 == hashlib.sha256(HASHED_NAME.encode()).hexdigest()
    assert sel.pool_size == sel.catalogue_size - 2
    assert len(sel.conditions) == 2 and [c["object"] for c in sel.conditions] == [o.name for o in sel.objects]

    again = gso.select_objects(cfg, material, mode="dev", n=2, cache=cache, task=TASK)
    assert again.digest() == sel.digest()
    more = gso.select_objects(cfg, material, mode="dev", n=3, cache=cache, task=TASK)
    assert [o.name for o in more.objects[:2]] == [o.name for o in sel.objects]
    assert list(more.conditions[:2]) == list(sel.conditions)  # placements are per object


def test_selection_runs_out_and_excludes_names(mirror: tuple[gso.GsoConfig, gso.GsoCache]) -> None:
    cfg, cache = mirror
    sel = gso.select_objects(cfg, gso.dev_seed_material(4), mode="scored", n=50, cache=cache, task=TASK)
    assert {o.name for o in sel.objects} == {"Small_Box", "Pill_Bottle", "Soap_Bar", "Spice_Jar"}
    assert sel.examined == sel.pool_size and sel.requested == 50
    reasons = sel.reason_counts()
    assert reasons["category"] == 2 and reasons["fetch_failed"] == 1 and reasons["mesh_invalid"] == 1
    without = gso.select_objects(
        cfg, gso.dev_seed_material(4), mode="scored", n=50, cache=cache, task=TASK, exclude=["Small_Box"]
    )
    assert "Small_Box" not in {o.name for o in without.objects}
    dev = next(e for e in without.exclusions if e.reason == "dev_subset")
    assert dev.name == "Small_Box" and dev.position == -1
    with pytest.raises(ValueError, match="not in the catalogue"):
        gso.select_objects(cfg, gso.dev_seed_material(4), mode="scored", cache=cache, exclude=["Smal_Box"])


def test_parallel_selection_matches_serial(tmp_path: Path) -> None:
    root = tmp_path / "mirror"
    source = _write_mirror(root)
    cfg = _config(source)
    serial = gso.select_objects(
        cfg,
        gso.dev_seed_material(5),
        mode="dev",
        n=3,
        cache=gso.GsoCache(source, root=tmp_path / "c1", fetch=gso.LocalMirror(root)),
        task=TASK,
    )
    parallel = gso.select_objects(
        cfg,
        gso.dev_seed_material(5),
        mode="dev",
        n=3,
        cache=gso.GsoCache(source, root=tmp_path / "c2", fetch=gso.LocalMirror(root)),
        task=TASK,
        workers=2,
    )
    assert parallel.digest() == serial.digest()
    assert parallel.examined == serial.examined


def test_selection_json_round_trip(mirror: tuple[gso.GsoConfig, gso.GsoCache]) -> None:
    cfg, cache = mirror
    sel = gso.select_objects(cfg, gso.dev_seed_material(3), mode="dev", n=2, cache=cache, task=TASK)
    doc = json.loads(json.dumps(sel.to_json()))
    back = gso.Selection.from_json(doc)
    assert back.digest() == sel.digest() == doc["digest"]
    assert doc["objects"][0]["attribution"]["license"] == "CC-BY-4.0"
    del doc["digest"]
    with pytest.raises(ValueError, match="digest"):
        gso.Selection.from_json(doc)  # an unverifiable selection is refused
    doc["digest"] = sel.digest()
    doc["conditions"][0]["x"] += 0.001
    with pytest.raises(ValueError, match="digest"):
        gso.Selection.from_json(doc)
    # a self-consistent digest over conditions that do not follow the objects is refused too
    swapped = replace(sel, conditions=tuple(reversed(sel.conditions)))
    with pytest.raises(ValueError, match="draw order"):
        gso.Selection.from_json(json.loads(json.dumps(swapped.to_json())))


def test_objects_load_and_are_verified(mirror: tuple[gso.GsoConfig, gso.GsoCache], tmp_path: Path) -> None:
    cfg, cache = mirror
    sel = gso.select_objects(cfg, gso.dev_seed_material(3), mode="dev", n=2, cache=cache, task=TASK)
    loaded = gso.load_selection_objects(sel, cache=cache)
    assert [o.name for o in loaded] == [o.name for o in sel.objects]
    assert [o.geometry_sha256 for o in loaded] == [o.geometry_sha256 for o in sel.objects]

    pack = tmp_path / "pack"
    pack.mkdir()
    src = cache.objects_dir(sel.rules_key)
    first, second = (o.name for o in sel.objects)
    (pack / f"{first}.npz").write_bytes((src / f"{second}.npz").read_bytes())  # wrong geometry
    (pack / f"{second}.npz").write_bytes((src / f"{second}.npz").read_bytes())
    with pytest.raises(ValueError, match="geometry"):
        gso.load_selection_objects(sel, pack)

    # same geometry under another object's name
    renamed = tmp_path / "renamed"
    renamed.mkdir()
    for o in sel.objects:
        obj = read_mesh_object(src / f"{o.name}.npz")
        save_mesh_object(
            replace(obj, name="Other_Name") if o.name == first else obj, renamed / f"{o.name}.npz"
        )
    with pytest.raises(ValueError, match="different object"):
        gso.load_selection_objects(sel, renamed)

    # a zip bomb: tiny on disk, far larger once decompressed
    bomb = tmp_path / "bomb"
    bomb.mkdir()
    for o in sel.objects:
        (bomb / f"{o.name}.npz").write_bytes((src / f"{o.name}.npz").read_bytes())
    with zipfile.ZipFile(bomb / f"{first}.npz", "a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("padding.npy", b"\0" * (gso.MAX_PACK_FILE_BYTES + 1))
    assert (bomb / f"{first}.npz").stat().st_size < gso.MAX_PACK_FILE_BYTES
    with pytest.raises(ValueError, match="expands"):
        gso.load_selection_objects(sel, bomb)


def test_selection_suite_evaluates_like_any_suite(mirror: tuple[gso.GsoConfig, gso.GsoCache]) -> None:
    cfg, cache = mirror
    sel = gso.select_objects(cfg, gso.dev_seed_material(3), mode="dev", n=2, cache=cache, task=TASK)
    suite = gso.selection_suite(sel, gso.load_selection_objects(sel, cache=cache))
    assert suite.name == "heldout_real_test-dev" and len(suite) == 2
    for cond, rec in zip(suite, sel.objects, strict=True):
        assert cond.family == "gso" and cond.size_label == rec.name and cond.condition_id.endswith(rec.name)
        p = cond.placement
        assert abs(p.x) <= 0.08 and abs(p.y) <= 0.08 and abs(p.err_x) <= 0.003 and abs(p.err_y) <= 0.003
        assert abs(p.err_yaw) <= math.radians(8.0) + 1e-9 and len(cond.seeds) == cfg.draw.seeds_per_condition
        assert cond.describe()["object"]["geometry_sha256"] == rec.geometry_sha256
    one = gso.selection_suite(sel, gso.load_selection_objects(sel, cache=cache), limit=1)
    records = evaluate_designs([ENTRANTS["straight"]], one, controller_settings=(1,), workers=1, task=TASK)
    assert len(records) == cfg.draw.seeds_per_condition
    assert {r.object_kind for r in records} == {"mesh"} and {r.family for r in records} == {"gso"}


# --------------------------------------------------------------------------- attribution


def test_attribution_names_the_licence_source_and_changes() -> None:
    src = gso.load_gso_config().source
    rec = gso.attribution("Small_Box", src)
    assert rec["license"] == "CC-BY-4.0" and rec["xml_license"] == "MIT"
    assert src.commit in rec["source_url"] and "2204.11918" in rec["citation"]
    assert "convex" in rec["changes"]
    text = gso.attribution_markdown(["Small_Box", "Pill_Bottle"], src)
    assert "CC BY 4.0" in text and "| Pill_Bottle |" in text and src.commit in text


@pytest.mark.skipif(os.environ.get("RTT_GSO_LIVE") != "1", reason="network test; set RTT_GSO_LIVE=1")
def test_live_catalogue_matches_the_pinned_tree(tmp_path: Path) -> None:
    cfg = gso.load_gso_config()
    names = gso.GsoCache(cfg.source, root=tmp_path).catalogue()
    assert len(names) == cfg.source.n_models
