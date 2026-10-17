"""Fixtures for the v2 held-out runner tests: fake frozen picks, search outputs, a scan mirror.

Nothing here touches Nebius, the network or the repository's held-out secret: every secret is
a throwaway string, every search output is written by hand (scores on the 1/48 grid, frozen
records hashed like :func:`rtt.search.select.freeze_top1` hashes them), and the scanned
objects come from the synthetic mirror of ``tests/test_gso.py``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rtt.heldout_v2 import common
from rtt.llm.cache import canonical_json
from rtt.search.select import FrozenDesign, write_frozen
from rtt.search.space import get_space
from rtt.sim import gso

G = 1.0 / 48
SECRET = "cd" * 32  # a throwaway held-out secret (64 hex digits), never the committed one
COMMITMENT = hashlib.sha256(SECRET.encode()).hexdigest()

#: Four distinct valid recipes (schema rules), one per (arm, run).
RECIPES: list[dict[str, Any]] = [
    {
        "pad": "conform",
        "tip": "blunt",
        "grip_height_frac": 0.53,
        "pad_height_mm": 25.2,
        "finger_width_mm": 35.8,
        "pad_depth_mm": 10.1,
        "wall_draft_deg": 67.0,
        "clearance_mm": 0.4,
    },
    {
        "pad": "v_funnel",
        "tip": "chamfer",
        "grip_height_frac": 0.5,
        "pad_height_mm": 20.0,
        "finger_width_mm": 30.0,
        "pad_depth_mm": 5.0,
        "wall_draft_deg": 40.0,
        "tip_size_mm": 2.0,
    },
    {
        "pad": "conform",
        "tip": "hook",
        "grip_height_frac": 0.4,
        "pad_height_mm": 20.0,
        "finger_width_mm": 30.0,
        "pad_depth_mm": 6.0,
        "wall_draft_deg": 30.0,
        "clearance_mm": 0.5,
        "tip_size_mm": 4.0,
    },
    {"pad": "flat", "tip": "blunt", "grip_height_frac": 0.6, "pad_height_mm": 15.0, "finger_width_mm": 25.0},
]
#: (arm, run index, seed, recipe index) of the fake confirmatory runs: R = 2 per arm.
PICKS = [("random", 0, 1001, 0), ("random", 1, 1002, 3), ("nemotron", 0, 1001, 1), ("nemotron", 1, 1002, 2)]


def make_frozen(arm: str, run_index: int, run_seed: int, params: dict[str, Any]) -> FrozenDesign:
    space = get_space("freeform-v2")
    design = space.recipe(**params)
    frozen = FrozenDesign(
        arm=arm,
        run_index=run_index,
        run_seed=run_seed,
        label="d12",
        design_id=space.design_id(design),
        design=design,
        controller_setting=0,
        grasp_height_frac=float(params["grip_height_frac"]),
        training_success=30 * G,
        training_hold_fraction=1.0,
        mass_g=20.0,
        origin=arm,
        content_sha256="",
        frozen_at="2026-10-17T00:00:00.000Z",
    )
    digest = hashlib.sha256(canonical_json(frozen.content()).encode("utf-8")).hexdigest()
    return dataclasses.replace(frozen, content_sha256=digest)


def write_search(root: Path, arm: str, *, layout: str = "local") -> Path:
    """A finished ``freeform-v2`` search output of ``arm`` with its two runs and frozen picks.

    ``layout="local"`` is an ``rtt search`` directory; ``"cloud"`` a fetched job directory
    (``runs/cloud/<run_id>``: job manifest on top, the search under ``search/``), which
    :func:`rtt.cloud.runs.verify_run` accepts.
    """
    out = root / f"search-{arm}"
    search = out / "search" if layout == "cloud" else out
    per_run = []
    for a, idx, seed, recipe in PICKS:
        if a != arm:
            continue
        run_id = f"{arm}-r{idx:02d}"
        rng = np.random.default_rng(seed + (7 if arm == "nemotron" else 0))
        steps = rng.integers(8, 40, 40)
        lines = [
            json.dumps(
                {
                    "index": j + 1,
                    "score": {"success": float(s) * G, "lift_success": float(min(48, s + 6)) * G},
                    "proposal": {"origin": "entrant" if j < 4 else "sobol" if j < 8 else arm},
                }
            )
            for j, s in enumerate(steps)
        ]
        hist = search / "runs" / run_id / "history.jsonl"
        hist.parent.mkdir(parents=True)
        hist.write_text("\n".join(lines) + "\n")
        write_frozen(make_frozen(arm, idx, seed, RECIPES[recipe]), search / "frozen")
        per_run.append({"run_id": run_id, "run_seed": seed, "fallback_random": 0, "llm": {}})
    manifest = {
        "kind": "search",
        "arm": arm,
        "status": "complete",
        "runs": len(per_run),
        "budget": 40,
        "round_sizes": [8, 8, 8, 8, 8],
        "design_space": "freeform-v2",
        "evaluator": {"episodes_per_design": 48, "task_hash": "task-a", "suite_hash": "suite-a"},
        "per_run": per_run,
    }
    (search / "manifest.json").write_text(json.dumps(manifest))
    if layout == "cloud":
        files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
        size = sum((out / f).stat().st_size for f in files)
        job = {
            "kind": "search",
            "exit_status": 0,
            "inputs": {"design_space": "freeform-v2", "arm": arm},
            "files": [*files, "manifest.json"],
            "publish": {"bytes": size},
        }
        (out / "manifest.json").write_text(json.dumps(job))
        (out / "job.json").write_text(json.dumps({"region": "uk-south2"}))
    return out


def mirror_config(base: Path) -> dict[str, Path]:
    """The v2 scan config rewritten for the synthetic mirror (convex hulls, 3 objects, 1 seed),
    with the dev draw pinned in ``exclude_names``, a cache, and the throwaway secret."""
    from rtt.store.vault import HeldoutVault
    from test_gso import _write_mirror

    mirror = base / "mirror"
    source = _write_mirror(mirror)
    raw = yaml.safe_load(gso.DEFAULT_CONFIG_V2.read_text())
    raw["source"] = {
        "repo": source.repo,
        "commit": source.commit,
        "models_tree": source.models_tree,
        "n_models": source.n_models,
        "mesh_file": "model.obj",
    }
    raw["draw"].update(n_objects=3, n_dev=1, seeds_per_condition=1, exclude_names=[])
    raw["rules"]["decomposition"] = {"method": "hull"}
    raw["denylist"] = {"tokens": ["car"], "name_sha256": []}
    cache = gso.GsoCache(source, root=base / "cache", fetch=gso.LocalMirror(mirror))
    parsed = gso.gso_config_from_dict(raw)
    dev = gso.select_objects(parsed, gso.dev_seed_material(parsed.draw.dev_seed), mode="dev", cache=cache)
    raw["draw"]["exclude_names"] = [o.name for o in dev.objects]
    config = base / "heldout_real_test.yaml"
    config.write_text(yaml.safe_dump(raw))
    secret = base / "secret" / "heldout_seed.txt"
    if not secret.exists():
        HeldoutVault.create(secret, SECRET)
    commitment_file = base / "commitment.sha256"
    commitment_file.write_text(f"{COMMITMENT}  secrets/heldout_seed.txt\n")
    return {
        "mirror": mirror,
        "config": config,
        "cache": base / "cache",
        "secret": secret,
        "commitment_file": commitment_file,
    }


def fake_simulate(entries: Any, suite: Any, task: Any, *, executor: Any = "cpu", workers: Any = None) -> Any:
    """:func:`rtt.heldout_v2.common.simulate` without physics: every episode an invalid build."""
    from rtt.sim.evaluate import invalid_record_v2

    by: dict[str, list[Any]] = {}
    for e in entries:
        if e.design_id not in by:
            by[e.design_id] = [
                invalid_record_v2(e.design_id, c, suite.name, int(s), "fake", task)
                for c in suite
                for s in c.seeds
            ]
    records = [r for chunk in by.values() for r in chunk]
    return common.Simulation(records, by, 0.0, 0.0, 0, 0)
