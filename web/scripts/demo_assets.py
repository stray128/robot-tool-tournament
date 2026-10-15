"""Export the manual entrants' print files and re-run the G1 gate for the web demo.

Called by ``web/scripts/import-demo.mjs`` with the repository's virtualenv
Python. It uses the same library code as ``rtt export-geom`` and ``rtt m1``, so
the files the site offers for download are exactly what the simulator collides
with, and the gate numbers come from the real scorer (not from the 30 fps
replay frames). With ``--replays`` the gate also records every case's replay
(``<case>.bin`` + ``<case>.json``, as ``rtt m1`` writes them) in the same run,
so the replays the site plays and the numbers it prints beside them come from
one simulation under the current task config.

Writes, for every entrant in the entrants YAML, under ``--out``::

    <design_id>/finger-left.stl    left finger, millimetres
    <design_id>/finger-right.stl   right (mirrored) finger, millimetres
    <design_id>/gripper.glb        palm plus both fingers, fully open, metres, Y-up

and prints one JSON document on stdout (see ``main``). Nothing else is printed
on stdout; library warnings go to stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _schema_params() -> dict[str, Any]:
    """Bounds, enum options and descriptions of the 10 geometry parameters."""
    from rtt.design.schema import GEOMETRY_PARAMS, design_json_schema

    props = design_json_schema()["properties"]
    out: dict[str, Any] = {}
    for key in GEOMETRY_PARAMS:
        p = props[key]
        entry: dict[str, Any] = {"description": p.get("description", "")}
        if "enum" in p:
            entry["options"] = list(p["enum"])
        else:
            entry["min"] = p.get("minimum")
            entry["max"] = p.get("maximum")
        out[key] = entry
    return out


def _export_entrants(entrants_path: Path, task_path: Path | None, out_dir: Path) -> list[dict[str, Any]]:
    from rtt.design.export import export_glb, export_stl, to_trimesh
    from rtt.design.generate import generate_finger, mass_and_inertia
    from rtt.design.schema import load_entrants
    from rtt.design.validate import validate
    from rtt.sim.task import gripper_task_for, load_task

    gripper = gripper_task_for(load_task(task_path))
    designs = []
    for design in load_entrants(entrants_path):
        design_id = design.design_id()
        check = validate(design, gripper)
        geom = generate_finger(design)
        mesh = to_trimesh(geom)
        target = out_dir / design_id
        files = {}
        if check.ok:
            files = {
                "stl_left": export_stl(geom, target / "finger-left.stl").name,
                "stl_right": export_stl(geom, target / "finger-right.stl", mirror=True).name,
                "glb": export_glb(geom, target / "gripper.glb", gripper).name,
            }
        designs.append(
            {
                "id": design_id,
                "name": design.name,
                "rationale": design.rationale,
                "params": design.geometry_params(),
                "valid": check.ok,
                "errors": list(check.errors),
                "warnings": list(check.warnings),
                "mass_printed_g": round(1000.0 * abs(float(mesh.volume)) * geom.density_kg_m3, 2),
                "mass_sim_g": round(1000.0 * mass_and_inertia(geom).mass_kg, 2),
                "bbox_mm": [round(1000.0 * float(e), 2) for e in mesh.extents],
                "watertight": bool(mesh.is_watertight),
                "n_boxes": len(geom.boxes),
                "files": files,
            }
        )
    return designs


def _run_gate(
    entrants_path: Path, task_path: Path | None, seed: int, replays_dir: Path | None
) -> dict[str, Any]:
    """Run the G1 gate; with ``replays_dir``, record each case's replay there (no video)."""
    from rtt.sim.m1 import run_gate
    from rtt.sim.task import load_task

    started = time.time()
    report = run_gate(
        replays_dir, seed=seed, video=False, task=load_task(task_path), entrants_path=entrants_path
    )
    ended = time.time()
    doc = report.to_dict()
    for case in doc["cases"].values():
        case.pop("files", None)  # local paths; the importer knows where it asked for the replays
    return {"started_at": _iso(started), "ended_at": _iso(ended), "recorded": replays_dir is not None, **doc}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entrants", type=Path, required=True, help="entrants YAML")
    parser.add_argument("--task", type=Path, default=None, help="task YAML (default: configs/task_v1.yaml)")
    parser.add_argument("--out", type=Path, required=True, help="directory for <design_id>/ print files")
    parser.add_argument("--seed", type=int, default=0, help="G1 gate seed (the demo replays use 0)")
    parser.add_argument("--no-gate", action="store_true", help="skip re-running the G1 gate")
    parser.add_argument(
        "--replays",
        type=Path,
        default=None,
        help="record the gate's <case>.bin/.json replays into this directory",
    )
    args = parser.parse_args(argv)
    if args.no_gate and args.replays is not None:
        parser.error("--replays needs the gate; drop --no-gate")
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

    import mujoco

    doc: dict[str, Any] = {
        "versions": {"rtt": version("robot-tool-tournament"), "mujoco": mujoco.__version__},
        "schema": _schema_params(),
        "designs": _export_entrants(args.entrants, args.task, args.out),
        "gate": None if args.no_gate else _run_gate(args.entrants, args.task, args.seed, args.replays),
    }
    json.dump(doc, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
