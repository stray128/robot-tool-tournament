"""``rtt export-web``: publish run directories into the web app's static data.

The site (``web/``) reads only ``web/public/data/`` (contract: ``web/lib/types.ts``,
checked by ``web/lib/validate.ts`` at build time). This module turns run directories
into entries of that contract:

* a **slice** directory (``slice.json``, :mod:`rtt.pipeline.slice`) becomes one bout
  (``kind: "training"``) whose lanes are its recorded replays, the designs it
  proposed (with STL/GLB print files and the provenance of the model call that made
  each one) and one job entry;
* a **search** directory (``manifest.json`` of kind ``search``) contributes each run's
  frozen top-1 design and one job entry;
* a **held-out** directory (``heldout.json``, :mod:`rtt.pipeline.heldout`) contributes
  one job entry;
* a ``certificates/<design_id>.json`` file (``rtt certify --out``) next to any of them
  fills that design's robustness certificate.

Where the physics ran comes from the run itself: a run fetched from a Nebius job
carries ``job.json`` (job ID, platform, preset, region, wall time, estimated cost) and
is labelled as a Nebius job; anything else is a local run and says so.

Entries are owned by a ``source`` tag, ``export:<run name>``: re-exporting a run
replaces exactly its own entries and keeps everyone else's (the demo importer uses
``demo-m1``). A design that another source already lists is not duplicated.
Everything is computed first and written last, so a failing export changes nothing.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from rtt.design.schema import GEOMETRY_PARAMS, FingerDesign, design_json_schema

__all__ = ["DEFAULT_DATA_DIR", "ExportResult", "export_web", "task_summary"]

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR: Final = REPO_ROOT / "web" / "public" / "data"
ENGINE_PREFIX: Final = "MuJoCo"
LOCAL_HOST: Final = "Developer laptop"
ORIGINS: Final = {
    "entrant": "manual",
    "sobol": "random",
    "random": "random",
    "fallback_random": "random",
    "nemotron": "nemotron",
    "zeroshot": "zeroshot",
}
LANE_KINDS: Final = {"box": "box", "cylinder": "cylinder", "rod": "capsule"}


@dataclass
class ExportResult:
    """What one :func:`export_web` call wrote."""

    data_dir: Path
    bouts: list[str] = field(default_factory=list)
    designs: list[str] = field(default_factory=list)
    jobs: list[str] = field(default_factory=list)
    replays: int = 0
    skipped: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"export-web: {len(self.bouts)} bouts ({', '.join(self.bouts) or '-'}), {len(self.designs)} designs, "
            f"{len(self.jobs)} jobs, {self.replays} replays -> {self.data_dir}"
            + (f"; skipped: {', '.join(self.skipped)}" if self.skipped else "")
        )


# --------------------------------------------------------------------------- helpers


def _r(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _slug(text: str) -> str:
    out = "".join(c if c.isalnum() or c in "._-" else "-" for c in text.lower()).strip("-.")
    return out or "run"


def _iso_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def task_summary(task: Any) -> dict[str, Any]:
    """The bout-level task constants the viewer needs (``TaskSummary`` in ``web/lib/types.ts``)."""
    c, s, g = task.controller, task.score, task.gripper
    t_descend = c.settle_time
    t_close = t_descend + c.descend_time
    t_lift = t_close + c.close_time + c.squeeze_time
    t_hold = t_lift + c.lift_time
    return {
        "name": task.name,
        "phases": [
            {"name": "settle", "start_s": 0, "end_s": _r(t_descend)},
            {"name": "descend", "start_s": _r(t_descend), "end_s": _r(t_close)},
            {"name": "close", "start_s": _r(t_close), "end_s": _r(t_lift)},
            {"name": "lift", "start_s": _r(t_lift), "end_s": _r(t_hold)},
            {"name": "hold", "start_s": _r(t_hold), "end_s": None},
        ],
        "success_lift_m": s.success_lift,
        "success_hold_s": s.success_hold,
        "record_tail_s": s.record_tail,
        "max_time_s": s.max_time,
        "grip_force_n": g.grip_force,
        "max_opening_mm": _r(2000.0 * g.open_half_gap, 3),
    }


def _object_summary(kind: str, size_m: Sequence[float]) -> dict[str, Any]:
    """Object extents in mm: box x y z; cylinder and capsule diameter, length."""
    if kind == "box":
        size = [1000.0 * v for v in size_m]
    else:
        size = [2000.0 * size_m[0], 1000.0 * size_m[1]]
    return {"kind": LANE_KINDS[kind], "size_mm": [_r(v, 3) for v in size]}


def _params_schema() -> dict[str, Any]:
    props = design_json_schema()["properties"]
    out: dict[str, Any] = {}
    for key in GEOMETRY_PARAMS:
        p = props[key]
        entry: dict[str, Any] = {"description": p.get("description", "")}
        if "enum" in p:
            entry["options"] = list(p["enum"])
        else:
            entry["min"], entry["max"] = p.get("minimum"), p.get("maximum")
        out[key] = entry
    return out


def _read_index(path: Path, schema: str, key: str) -> dict[str, Any]:
    if not path.is_file():
        return {"schema": schema, "version": 1, key: []}
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != schema:
        raise ValueError(f"{path}: expected schema {schema}, found {doc.get('schema')}")
    return doc


def _merge(
    existing: Sequence[Mapping[str, Any]], fresh: Sequence[Mapping[str, Any]], sources: set[str]
) -> list:
    return [x for x in existing if x.get("source") not in sources] + list(fresh)


def _write_json(path: Path, doc: Any) -> None:
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- compute provenance


def _cloud_job(run_dir: Path) -> dict[str, Any] | None:
    """``job.json`` written by ``rtt job`` / ``python -m rtt.cloud run`` next to a fetched run."""
    for candidate in (run_dir / "job.json", run_dir.parent / "job.json"):
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def _job_cost(job: Mapping[str, Any]) -> float:
    attempts = job.get("attempts") or []
    return _r(sum(float(a.get("est_usd") or 0.0) for a in attempts), 4)


def _compute(
    run_dir: Path, executor: str, wall_s: float, sim_steps: int | None, note: str, image_digest: str | None
) -> dict[str, Any]:
    import mujoco

    engine = f"{ENGINE_PREFIX} {mujoco.__version__}"
    job = _cloud_job(run_dir)
    if job is not None and job.get("job_id") and job.get("platform"):
        return {
            "kind": "nebius",
            "job_id": str(job["job_id"]),
            "platform": str(job["platform"]),
            "preset": str(job.get("preset") or "-"),
            "region": str(job.get("region") or "eu-north1"),
            "executor": executor,
            "engine": engine,
            "wall_s": _r(job.get("wall_s") or wall_s, 3),
            "cost_usd": _job_cost(job),
            "image_digest": image_digest,
        }
    return {
        "kind": "local",
        "executor": executor,
        "engine": engine,
        "host": LOCAL_HOST,
        "wall_s": _r(wall_s, 3),
        "sim_steps": sim_steps,
        "note": note,
    }


def _job_entry(
    run_dir: Path,
    source: str,
    name: str,
    purpose: str,
    executor: str,
    started_at: str,
    ended_at: str,
    wall_s: float,
    episodes: int | None,
    sim_steps: int | None,
    bouts: Sequence[str],
    status: str = "completed",
) -> dict[str, Any]:
    job = _cloud_job(run_dir)
    if job is not None and job.get("job_id") and job.get("platform"):
        return {
            "id": str(job["job_id"]),
            "source": source,
            "name": name,
            "purpose": purpose,
            "kind": "nebius",
            "executor": executor,
            "platform": job.get("platform"),
            "preset": job.get("preset"),
            "region": job.get("region") or "eu-north1",
            "status": _job_status(job.get("state")),
            "started_at": job.get("started_at") or job.get("created_at") or started_at,
            "ended_at": job.get("finished_at") or ended_at,
            "wall_s": _r(job.get("wall_s") or wall_s, 3),
            "cost_usd": _job_cost(job),
            "episodes": episodes,
            "sim_steps": sim_steps,
            "bouts": list(bouts),
        }
    return {
        "id": f"local-{_run_name(run_dir)}",
        "source": source,
        "name": name,
        "purpose": purpose,
        "kind": "local",
        "executor": executor,
        "platform": None,
        "preset": None,
        "region": None,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_s": _r(wall_s, 3),
        "cost_usd": None,
        "episodes": episodes,
        "sim_steps": sim_steps,
        "bouts": list(bouts),
    }


def _job_status(state: Any) -> str:
    s = str(state or "").upper()
    if s in ("COMPLETED", "SUCCEEDED"):
        return "completed"
    if s in ("CANCELLED", "CANCELLING"):
        return "cancelled"
    if s in ("FAILED", "ERROR"):
        return "failed"
    return "running" if s else "completed"


# --------------------------------------------------------------------------- designs


def _design_files(design: FingerDesign, gripper: Any, target: Path) -> tuple[dict[str, str], dict[str, Any]]:
    from rtt.design.export import export_glb, export_stl, to_trimesh
    from rtt.design.generate import generate_finger, mass_and_inertia

    geom = generate_finger(design)
    mesh = to_trimesh(geom)
    target.mkdir(parents=True, exist_ok=True)
    files = {
        "stl_left": export_stl(geom, target / "finger-left.stl").name,
        "stl_right": export_stl(geom, target / "finger-right.stl", mirror=True).name,
        "glb": export_glb(geom, target / "gripper.glb", gripper).name,
    }
    facts = {
        "mass_printed_g": _r(1000.0 * abs(float(mesh.volume)) * geom.density_kg_m3, 2),
        "mass_sim_g": _r(1000.0 * mass_and_inertia(geom).mass_kg, 2),
        "bbox_mm": [_r(1000.0 * float(e), 2) for e in mesh.extents],
        "watertight": bool(mesh.is_watertight),
        "n_boxes": len(geom.boxes),
    }
    return files, facts


def _certificate(run_dir: Path, design_id: str) -> dict[str, Any] | None:
    """The design's robustness certificate (``rtt certify --out``) as web checks, if one exists.

    Placement tests are grouped by error magnitude (0, 1, 2, 3 mm: each non-zero group holds
    +-x and +-y), shakes and pushes by category.
    """
    path = run_dir / "certificates" / f"{design_id}.json"
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    certs = doc.get("certificates") or [doc]
    groups: dict[str, list[bool]] = {}
    conditions: dict[str, str] = {}
    for cert in certs:
        for test in cert.get("tests", []):
            if test["category"] == "placement":
                err = abs(float(test["params"].get("error_mm") or 0.0))
                key = f"Placement error {err:g} mm"
                conditions[key] = (
                    "Object at the aimed pose"
                    if err == 0
                    else f"Object offset by {err:g} mm from the aimed pose, along x and along y"
                )
            elif test["category"] == "shake":
                key = "Shake during hold"
                conditions[key] = "Palm shakes along x, y and z while holding the object"
            else:
                key = "Lateral push"
                conditions[key] = "Sideways push at the object's top, along x and along y"
            groups.setdefault(key, []).append(bool(test["success"]))
    checks = [
        {
            "name": key,
            "condition": conditions[key],
            "success_rate": _r(sum(results) / len(results), 4),
            "episodes": len(results),
        }
        for key, results in groups.items()
    ]
    return {"status": "issued", "issued_at": doc.get("issued_at"), "checks": checks}


def _provenance(
    entry: Mapping[str, Any], model: Mapping[str, Any] | None, provider: str | None, seed: int
) -> dict:
    origin = entry.get("origin")
    if origin == "entrant":
        return {"kind": "manual", "note": "Hand-designed baseline entrant"}
    if origin in ("nemotron", "zeroshot") and entry.get("llm_call") and model:
        call = entry["llm_call"]
        return {
            "kind": "llm",
            "model": str(model.get("model_id")),
            "provider": str(provider or "nebius-tokenfactory"),
            "tokens_in": int(call.get("prompt_tokens") or 0),
            "tokens_out": int(call.get("completion_tokens") or 0),
            "latency_s": _r(call.get("latency_s") or 0.0, 3),
            "cost_usd": _r(call.get("cost_usd") or 0.0, 6),
            "round": entry.get("round"),
        }
    return {"kind": "random", "sampler": "scrambled Sobol (validator-filtered)", "seed": int(seed)}


def _rationale(entry: Mapping[str, Any]) -> str | None:
    if entry.get("origin") == "fallback_random":
        return f"Random fallback: the model's proposal for this slot was rejected ({entry.get('fallback_reason')})."
    return entry.get("hypothesis") or None


# --------------------------------------------------------------------------- slice


@dataclass
class _Staged:
    bouts: list[dict[str, Any]] = field(default_factory=list)
    designs: list[dict[str, Any]] = field(default_factory=list)
    jobs: list[dict[str, Any]] = field(default_factory=list)
    copies: list[tuple[Path, str]] = field(default_factory=list)  # (source file, path under data dir)
    design_dirs: list[tuple[Path, str]] = field(default_factory=list)  # (staged dir, design id)


def _slice_entries(
    run_dir: Path, summary: Mapping[str, Any], known_designs: set[str], staging: Path, source: str
) -> _Staged:
    from rtt.design.validate import validate
    from rtt.sim.conditions import training_suite
    from rtt.sim.replay import read_replay
    from rtt.sim.task import default_task, task_digest

    task = default_task()
    if summary["task"]["digest"] != task_digest(task):
        raise ValueError(
            f"{run_dir}: recorded under task {summary['task']['digest']}, the current task is "
            f"{task_digest(task)}; re-run the slice (replays must match the task the site prints)"
        )
    gripper = training_suite().gripper_task(task)
    staged = _Staged()
    bout_id = _slug(run_dir.name if run_dir.name.startswith("slice") else f"slice-{run_dir.name}")
    model = summary.get("model") or {}
    provider = summary.get("provider")
    seed = int(summary["config"]["seed"])
    entries = {d["design_id"]: d for d in summary["designs"]}
    cond = summary["condition"]
    lanes = []
    total_steps = 0
    total_wall = 0.0
    for lane in summary["lanes"]:
        entry = entries[lane["design_id"]]
        stem = lane["stem"]
        rel = f"replays/{bout_id}/{stem}"
        replay = read_replay(run_dir / lane["replay"])
        if replay.n_frames != lane["n_frames"]:
            raise ValueError(
                f"{run_dir}: replay {stem} has {replay.n_frames} frames, slice.json says {lane['n_frames']}"
            )
        staged.copies += [
            (run_dir / f"{lane['replay']}.bin", f"{rel}.bin"),
            (run_dir / f"{lane['replay']}.json", f"{rel}.json"),
        ]
        origin = entry["origin"]
        if origin == "entrant":
            label = f"Baseline: {entry['name']} finger"
        elif origin == "fallback_random":
            label = f"Random fallback ({(entry.get('slot') or '').replace('_', ' ')})"
        else:
            label = f"Nemotron {(entry.get('slot') or entry.get('operator') or 'design').replace('_', ' ')}"
        total_steps += int(lane["n_steps"])
        total_wall += float(lane["wall_time_s"])
        lanes.append(
            {
                "id": _slug(stem),
                "replay": rel,
                "label": label,
                "condition": cond["description"],
                "entrant": entry.get("name") or entry["label"],
                "design_id": lane["design_id"],
                "object": _object_summary(cond["kind"], cond["size_m"]),
                "controller_setting": int(lane["controller_setting"]),
                "grasp_height_frac": lane["grasp_height_frac"],
                "seed": int(lane["seed"]),
                "n_frames": int(lane["n_frames"]),
                "duration_s": _r((lane["n_frames"] - 1) / task.replay.fps, 4),
                "outcome": {
                    "success": bool(lane["success"]),
                    "failure_mode": lane["failure_mode"],
                    "max_lift_mm": _r(1000.0 * lane["max_lift_m"], 1),
                    "hold_s": _r(lane["hold_time_s"], 3),
                    "flags": list(lane["flags"]),
                    "source": "scorer",
                    "wall_s": _r(lane["wall_time_s"], 3),
                    "sim_steps": int(lane["n_steps"]),
                },
                "design_provenance": _provenance(entry, model, provider, seed),
            }
        )
    usage = summary.get("llm") or {}
    proposed = [d for d in summary["designs"] if d["round"] >= 1]
    n_model = sum(1 for d in proposed if d["origin"] in ("nemotron", "zeroshot"))
    n_fallback = len(proposed) - n_model
    n_calls = int(usage.get("calls") or 0)
    image = (summary.get("compute") or {}).get("job", {}).get("image_digest")
    note = (
        "Local run on a developer laptop, not a Nebius job. Designs proposed live by Nemotron on "
        "Nebius Token Factory."
    )
    kept = f"{n_model} of {len(proposed)}" if n_fallback else f"all {len(proposed)}"
    if n_fallback == 1:
        fallback = (
            " The remaining slot failed the validator or the diversity check twice and was filled at "
            "random, as the protocol requires."
        )
    elif n_fallback > 1:
        fallback = (
            f" The other {n_fallback} slots failed the validator or the diversity check twice and were "
            "filled at random, as the protocol requires."
        )
    else:
        fallback = ""
    staged.bouts.append(
        {
            "id": bout_id,
            "source": source,
            "kind": "training",
            "title": f"Nemotron's first {len(proposed)} proposals against the stock finger",
            "summary": (
                f"Nemotron 3 Super read the results of {summary['config']['round0_size']} starting designs on "
                f"{summary['suite']['n_conditions']} training grasps and proposed {len(proposed)} new fingers; "
                f"{kept} passed the checks ({n_calls} model call{'s' if n_calls != 1 else ''}, "
                f"{int(usage.get('prompt_tokens') or 0):,} prompt and "
                f"{int(usage.get('completion_tokens') or 0):,} completion tokens, "
                f"${float(usage.get('cost_usd') or 0):.4f}).{fallback} "
                f"Each lane is a design's own best grasp height on the training condition where their outcomes "
                f"differ most: {cond['description']}."
            ),
            "featured": False,
            "recorded_at": summary.get("finished_at"),
            "task": task_summary(task),
            "compute": _compute(
                run_dir, summary["compute"]["executor"], total_wall, total_steps, note, image
            ),
            "lanes": lanes,
        }
    )
    lane_designs = {lane["design_id"] for lane in summary["lanes"]}
    for design_id in sorted(
        lane_designs, key=lambda d: [x["design_id"] for x in summary["designs"]].index(d)
    ):
        if design_id in known_designs:
            continue
        entry = entries[design_id]
        design = FingerDesign.model_validate({**entry["params"], "name": entry.get("name")})
        check = validate(design, gripper)
        target = staging / "designs" / design_id
        files, facts = _design_files(design, gripper, target)
        staged.design_dirs.append((target, design_id))
        name = entry.get("name") or f"{ORIGINS.get(entry['origin'], 'design')} {entry['label']}"
        staged.designs.append(
            {
                "id": design_id,
                "source": source,
                "name": name,
                "origin": ORIGINS.get(entry["origin"], "random"),
                "rationale": _rationale(entry),
                "params": entry["params"],
                "valid": check.ok,
                "errors": list(check.errors),
                "warnings": list(check.warnings),
                **facts,
                "files": {k: f"designs/{design_id}/{v}" for k, v in files.items()},
                "provenance": _provenance(entry, model, provider, seed),
                "bouts": [bout_id],
                "certificate": _certificate(run_dir, design_id),
            }
        )
        known_designs.add(design_id)
    sim = summary.get("sim") or {}
    staged.jobs.append(
        _job_entry(
            run_dir,
            source,
            "Vertical slice",
            f"One Nemotron round end to end: {summary['config']['round0_size']} starting designs, "
            f"{len(proposed)} proposed designs ({n_model} kept from the model), {int(sim.get('episodes_simulated') or 0)} training episodes, "
            f"{len(lanes)} recorded replays.",
            summary["compute"]["executor"],
            summary["started_at"],
            summary["finished_at"],
            float(summary["wall_s"]),
            int(sim.get("episodes_simulated") or 0) + len(lanes),
            None,
            [bout_id],
        )
    )
    return staged


# --------------------------------------------------------------------------- search / held-out


def _search_entries(
    run_dir: Path, manifest: Mapping[str, Any], known_designs: set[str], staging: Path, source: str
) -> _Staged:
    from rtt.design.validate import validate
    from rtt.search.select import read_frozen
    from rtt.sim.conditions import training_suite
    from rtt.sim.task import default_task

    staged = _Staged()
    gripper = training_suite().gripper_task(default_task())
    for path in sorted((run_dir / "frozen").glob("*.json")):
        frozen = read_frozen(path)
        if frozen.design_id in known_designs:
            continue
        check = validate(frozen.design, gripper)
        target = staging / "designs" / frozen.design_id
        files, facts = _design_files(frozen.design, gripper, target)
        staged.design_dirs.append((target, frozen.design_id))
        origin = ORIGINS.get(frozen.origin, "random")
        staged.designs.append(
            {
                "id": frozen.design_id,
                "source": source,
                "name": f"{manifest['arm']} run {frozen.run_index} top-1",
                "origin": origin,
                "rationale": (
                    f"Frozen top-1 of {manifest['arm']} run {frozen.run_index} (seed {frozen.run_seed}): "
                    f"{100 * frozen.training_success:.0f}% training success at grasp height "
                    f"{frozen.grasp_height_frac}."
                ),
                "params": frozen.design.geometry_params(),
                "valid": check.ok,
                "errors": list(check.errors),
                "warnings": list(check.warnings),
                **facts,
                "files": {k: f"designs/{frozen.design_id}/{v}" for k, v in files.items()},
                "provenance": (
                    {"kind": "manual", "note": "Hand-designed baseline entrant"}
                    if frozen.origin == "entrant"
                    else {
                        "kind": "random",
                        "sampler": "scrambled Sobol (validator-filtered)",
                        "seed": frozen.run_seed,
                    }
                    if origin == "random"
                    else {
                        "kind": "llm",
                        "model": str((manifest.get("model") or {}).get("model_id")),
                        "provider": str(manifest.get("provider") or "nebius-tokenfactory"),
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "latency_s": 0.0,
                        "cost_usd": 0.0,
                        "round": None,
                    }
                ),
                "bouts": [],
                "certificate": _certificate(run_dir, frozen.design_id),
            }
        )
        known_designs.add(frozen.design_id)
    sim = manifest.get("sim") or {}
    staged.jobs.append(
        _job_entry(
            run_dir,
            source,
            f"Search: {manifest['arm']} arm",
            f"{manifest['runs']} run{'s' if manifest['runs'] != 1 else ''} x {manifest['budget']} designs on the training suite"
            + (
                f"; {int((manifest.get('llm') or {}).get('calls') or 0)} model calls"
                if manifest.get("model")
                else ""
            ),
            str(manifest.get("executor") or "cpu"),
            manifest["started_at"],
            manifest.get("finished_at") or manifest["started_at"],
            float(manifest.get("wall_s") or 0.0),
            int(sim.get("episodes_simulated") or 0),
            None,
            [],
            status="completed" if manifest.get("status") == "complete" else "failed",
        )
    )
    return staged


def _heldout_entries(run_dir: Path, report: Mapping[str, Any], source: str) -> _Staged:
    staged = _Staged()
    staged.jobs.append(
        _job_entry(
            run_dir,
            source,
            "Held-out evaluation",
            f"{len(report.get('designs', []))} frozen designs x {report.get('episodes_per_design')} held-out episodes",
            str(report.get("executor") or "cpu"),
            report["started_at"],
            report["finished_at"],
            float(report.get("wall_s") or 0.0),
            int(report.get("episodes") or 0),
            None,
            [],
        )
    )
    return staged


# --------------------------------------------------------------------------- entry point


def export_web(
    run_dirs: Iterable[str | Path],
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    featured: str | None = None,
) -> ExportResult:
    """Export ``run_dirs`` into ``data_dir`` (default ``web/public/data``).

    ``featured`` names a bout id to mark as the landing page's featured bout (other
    bouts this export writes are not featured; other sources' flags are untouched).
    """
    data = Path(data_dir)
    data.mkdir(parents=True, exist_ok=True)
    files = {name: data / f"{name}.json" for name in ("bouts", "designs", "jobs")}
    bouts_doc = _read_index(files["bouts"], "rtt-web-bouts", "bouts")
    designs_doc = _read_index(files["designs"], "rtt-web-designs", "designs")
    jobs_doc = _read_index(files["jobs"], "rtt-web-jobs", "jobs")
    result = ExportResult(data_dir=data)

    with tempfile.TemporaryDirectory(prefix="rtt-export-") as tmp:
        staging = Path(tmp)
        sources: set[str] = set()
        combined = _Staged()
        for raw in run_dirs:
            run_dir = Path(raw).resolve()
            source = f"export:{_run_name(run_dir)}"
            sources.add(source)
            known = {d["id"] for d in designs_doc["designs"] if d.get("source") not in sources}
            known |= {d["id"] for d in combined.designs}
            if (run_dir / "slice.json").is_file():
                summary = json.loads((run_dir / "slice.json").read_text(encoding="utf-8"))
                staged = _slice_entries(run_dir, summary, known, staging, source)
            elif (run_dir / "heldout.json").is_file():
                report = json.loads((run_dir / "heldout.json").read_text(encoding="utf-8"))
                staged = _heldout_entries(run_dir, report, source)
            elif (run_dir / "manifest.json").is_file() and _manifest_kind(run_dir) == "search":
                manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                staged = _search_entries(run_dir, manifest, known, staging, source)
            else:
                result.skipped.append(str(raw))
                continue
            combined.bouts += staged.bouts
            combined.designs += staged.designs
            combined.jobs += staged.jobs
            combined.copies += staged.copies
            combined.design_dirs += staged.design_dirs

        if featured is not None:
            for bout in combined.bouts:
                bout["featured"] = bout["id"] == featured
            if featured in {b["id"] for b in combined.bouts}:
                for bout in bouts_doc["bouts"]:
                    if bout.get("source") not in sources:
                        bout["featured"] = False

        # publish: replays and print files first, then the indexes that point at them
        stale = {b["id"] for b in combined.bouts}
        stale |= {b["id"] for b in bouts_doc["bouts"] if b.get("source") in sources}
        for bout_id in stale:  # this export's earlier replays, including bouts it no longer writes
            shutil.rmtree(data / "replays" / bout_id, ignore_errors=True)
        for src, rel in combined.copies:
            dst = data / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        for staged_dir, design_id in combined.design_dirs:
            dst = data / "designs" / design_id
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(staged_dir, dst)
        _write_json(
            files["bouts"],
            {
                "schema": "rtt-web-bouts",
                "version": 1,
                "bouts": _merge(bouts_doc["bouts"], combined.bouts, sources),
            },
        )
        _write_json(
            files["designs"],
            {
                "schema": "rtt-web-designs",
                "version": 1,
                "params": designs_doc.get("params") or _params_schema(),
                "designs": _merge(designs_doc["designs"], combined.designs, sources),
            },
        )
        _write_json(
            files["jobs"],
            {
                "schema": "rtt-web-jobs",
                "version": 1,
                "jobs": _merge(jobs_doc["jobs"], combined.jobs, sources),
            },
        )
    result.bouts = [b["id"] for b in combined.bouts]
    result.designs = [d["id"] for d in combined.designs]
    result.jobs = [j["id"] for j in combined.jobs]
    result.replays = len(combined.copies) // 2
    return result


def _run_name(run_dir: Path) -> str:
    """A run's export name: its directory, prefixed by the parent for generic names.

    A fetched Nebius run keeps its search under ``runs/cloud/<run_id>/search``, so two
    such runs would otherwise both be called ``search``.
    """
    if run_dir.name in ("search", "heldout", "slice", "out"):
        return _slug(f"{run_dir.parent.name}-{run_dir.name}")
    return _slug(run_dir.name)


def _manifest_kind(run_dir: Path) -> str | None:
    try:
        return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")).get("kind")
    except (OSError, json.JSONDecodeError):
        return None
