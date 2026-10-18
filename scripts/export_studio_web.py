#!/usr/bin/env python
"""Regenerate every Studio page's data from tournament report files (product mode, track S7).

::

    python scripts/export_studio_web.py --reports DIR [--sessions DIR] [--heroes h1,h2,h3]
        [--data-dir web/public/data] [--assets-dir web/public/assets] [--workers 4]
        [--no-replays] [--no-packs] [--no-blocker] [--check]

``DIR`` holds one folder per hero case (``h1_tote``, ``h2_coupling``, ``h3_cabinet``) with the
``report.json`` that :func:`rtt.studio.search.run_task_tournament` wrote (``report.to_dict()``)
and, optionally, the ``determinism.json`` of the run. Everything the Studio pages show is
derived from these files, the recorded Task Engineer plans and (with ``--sessions``) the recorded
Grasp Engineer sessions, so the orchestrator reruns this script after any refresh of a report:

* ``web/public/data/studio/index.json`` (``rtt-web-studio``) and ``<hero>.json``
  (``rtt-web-studio-hero``): the use case, the frozen plan's steps, the candidates screened and
  pruned with their reasons, the success matrix, the finalists against the best single
  off-the-shelf baseline, the recommendation with per-step fidelity tiers and caveats (or, with
  none, the closest setups and the physical blocker), the assumed constants and provenance;
* ``studio/replays/<hero>/<lane>.{json,bin}``: replay v2 lanes of the recommended (or closest)
  setup and the best single baseline on one condition, re-simulated by
  :func:`rtt.studio.replay.record_lane`, which must reproduce the report's ``traj_sha`` bit for bit;
  their GLB assets go to ``web/public/assets`` (content addressed, merged into its manifest,
  meshopt-compressed by the web build's compressor);
* ``studio/<hero>/build_pack.zip``: the build pack, re-exported from the report with
  :func:`rtt.studio.buildpack.part_jaw.export_part_jaw_pack` (deterministic zip), at most
  :data:`PACK_CAP_BYTES`;
* ``studio/sessions/<hero>.json``: the recorded session, scrubbed (held-out words, brands,
  local paths; the chat transcript dropped);
* ``no_winner.blocker`` of a task nothing completes: :func:`rtt.studio.replay.blocker_probe`
  re-runs the step every finalist stops at for every candidate screened on it and reports the
  peak of its blocking quantity per tool family.

Every string written is checked for brand tokens (the contract's list plus the library's), URLs
appear only in fields named ``url``, and every document carries the product-mode label. Nothing
is written when a check fails. ``--check`` rebuilds everything in a scratch folder and compares
it with the published files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from rtt.effectors.lanes import FAMILY_TITLES
from rtt.effectors.types import PRODUCT_MODE_LABEL
from rtt.library.brands import find_library_brands
from rtt.studio.engineer.gate import RECORDED_GATE
from rtt.studio.types import (
    CONTRACT_VERSION,
    QUANTITIES,
    TemplateKind,
    get_template,
)

INDEX_SCHEMA: Final = "rtt-web-studio"
PAGE_SCHEMA: Final = "rtt-web-studio-hero"
WEB_VERSION: Final = 1
#: Hero route slug -> recorded case id (the frozen S4e plans).
HEROES: Final[tuple[tuple[str, str], ...]] = (("h1", "h1_tote"), ("h2", "h2_coupling"), ("h3", "h3_cabinet"))
#: The largest build pack zip the static site serves.
PACK_CAP_BYTES: Final = 2_000_000
DEFAULT_DATA_DIR: Final = REPO / "web" / "public" / "data"
DEFAULT_ASSETS_DIR: Final = REPO / "web" / "public" / "assets"
KIND_TITLES: Final[Mapping[str, str]] = {
    "single": "Single tool",
    "multi_head": "Two-tool head",
    "tool_change": "Tool change",
}
PRIMITIVE_TITLES: Final[Mapping[str, str]] = {
    "grasp": "Grasp",
    "suction": "Suction",
    "wrap": "Wrap",
    "hook_pull": "Hook pull",
    "twist": "Twist",
    "insert": "Insert",
    "push": "Push",
}
FAILURE_TEXT: Final[Mapping[str, str]] = {
    "criterion_not_met": "criterion not met",
    "constraint_exceeded": "force or torque limit exceeded",
    "slip": "slip",
    "jam": "jam",
    "dropped": "dropped",
    "physics_invalid": "simulator reset",
    "physics_unreliable": "penetration over 1 mm",
    "invalid_build": "cannot be built",
    "not_simulated": "not simulated yet",
    "not_reached": "not reached",
}
_URL = re.compile(r"https?://", re.IGNORECASE)
_LOCAL = re.compile(r"(/home/|/Users/|/tmp/|/root/|[A-Za-z]:\\)")


class ExportError(RuntimeError):
    """A check failed; nothing was written."""


# --------------------------------------------------------------------------- small helpers


def _r(v: Any, nd: int = 4) -> Any:
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        return round(v, nd)
    return v


def _rd(d: Mapping[str, Any] | None, nd: int = 4) -> dict[str, Any]:
    return {k: _r(v, nd) for k, v in (d or {}).items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_version() -> str:
    """The checkout's commit, 12 hex digits (``+dirty`` for a modified tree)."""
    from rtt.sim.replay import code_version as cv

    return cv(REPO)


def family_title(family: str) -> str:
    return FAMILY_TITLES.get(family, family.replace("_", " ").capitalize())


def criterion_text(quantity: str, at_least: float | None, at_most: float | None, obj_label: str = "") -> str:
    q = QUANTITIES.get(quantity)
    unit = q.unit if q is not None else ""
    name = {
        "angle_deg": "angle",
        "travel_mm": "travel",
        "depth_mm": "depth",
        "turns": "turns",
        "lift_mm": "lift",
        "hold_s": "hold",
        "in_target": "in the target",
        "tilt_deg": "tilt",
        "slip_deg": "slip",
        "slip_mm": "slip",
        "force_n": "force",
        "lateral_force_n": "lateral force",
        "torque_nm": "torque",
        "jam": "jam",
        "penetration_mm": "penetration",
    }.get(quantity, quantity.replace("_", " "))
    unit = {"turn": "turns", "flag": ""}.get(unit, unit)

    def num(v: float) -> str:
        return f"{v:g}"

    if quantity == "in_target":
        return "ends inside the target" if (at_least or 0) >= 1 else "may end outside the target"
    if quantity == "jam":
        return "no jam" if at_most is not None and at_most <= 0 else "jam allowed"
    u = f" {unit}" if unit and unit != "turns" else (" turns" if unit == "turns" else "")
    if at_least is not None and at_most is not None:
        return f"{name} {num(at_least)} to {num(at_most)}{u}"
    if at_least is not None:
        return f"{name} at least {num(at_least)}{u}"
    if at_most is not None:
        return f"{name} at most {num(at_most)}{u}"
    return name


def condition_text(cond: Mapping[str, Any]) -> str:
    offsets = cond.get("aim_offsets_mm") or {}
    seed = cond.get("seed", 0)
    if not offsets:
        return f"nominal placement, seed {seed}"
    worst = max(math.hypot(*map(float, v)) for v in offsets.values())
    return f"placement errors up to {worst:.1f} mm per object, seed {seed}"


# --------------------------------------------------------------------------- inputs


def load_case(case_id: str, recorded: Path = RECORDED_GATE) -> dict[str, Any]:
    """The recorded Task Engineer case (use-case text, title, setting, how the plan was reached)."""
    manifest = json.loads((recorded / "manifest.json").read_text(encoding="utf-8"))
    for row in manifest["cases"]:
        if row["case_id"] == case_id:
            return {**row, "model": manifest.get("model")}
    raise ExportError(f"case {case_id} is not in the recorded gate run")


def load_report(reports: Path, case_id: str) -> tuple[dict[str, Any], Path]:
    path = reports / case_id / "report.json"
    if not path.is_file():
        raise ExportError(f"no report for {case_id}: {path.name} missing in its folder")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("format") != "rtt-studio-tournament":
        raise ExportError(f"{case_id}: not a Studio tournament report")
    if report.get("label") != PRODUCT_MODE_LABEL:
        raise ExportError(f"{case_id}: the report is not labelled product mode")
    if not report.get("evidence") or report.get("controllers") != "default":
        raise ExportError(f"{case_id}: reference-controller results are not evidence")
    return report, path


# --------------------------------------------------------------------------- plan


def plan_doc(report: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    task = report["task"]
    labels = {o["name"]: o.get("label") or o["name"] for o in task["objects"]}
    objects = []
    for o in task["objects"]:
        tmpl = get_template(TemplateKind(o["kind"]))
        objects.append(
            {
                "name": o["name"],
                "kind": o["kind"],
                "kind_title": tmpl.display_name,
                "label": o.get("label") or o["name"],
                "params": dict(o.get("params") or {}),
                "pose": {k: _r(float(v), 2) for k, v in (o.get("pose") or {}).items()},
                "ref": o.get("ref"),
            }
        )
    fidelity = task.get("fidelity") or {}
    from rtt.studio.types import task_spec_from_tool_args

    keys = (
        "title",
        "description",
        "objects",
        "steps",
        "assumptions",
        "clarifications",
        "unsupported",
        "refusal",
    )
    spec = task_spec_from_tool_args({k: task[k] for k in keys if k in task}, source="recorded")
    if spec.task_id != report["task_id"]:
        raise ExportError(f"the report's task does not hash to its task_id ({spec.task_id[:12]})")
    steps = []
    for s in task["steps"]:
        success = []
        for c in s.get("success") or []:
            q = QUANTITIES.get(c["quantity"])
            success.append(
                {
                    "quantity": c["quantity"],
                    "unit": q.unit if q is not None else "",
                    "at_least": _r(c.get("at_least")),
                    "at_most": _r(c.get("at_most")),
                    "text": criterion_text(
                        c["quantity"], c.get("at_least"), c.get("at_most"), labels[s["object"]]
                    ),
                }
            )
        steps.append(
            {
                "id": s["id"],
                "primitive": s["primitive"],
                "primitive_title": PRIMITIVE_TITLES[s["primitive"]],
                "object": s["object"],
                "handle": s["handle"],
                "target": s.get("target"),
                "params": dict(s.get("params") or {}),
                "constraints": {k: _r(v) for k, v in (s.get("constraints") or {}).items()},
                "success": success,
                "after": list(s.get("after") or []),
                "hold_after": bool(s.get("hold_after")),
                "description": s.get("description") or "",
                "fidelity": fidelity.get(s["id"], spec.fidelity(s["id"]).value),
                "caveats": list(spec.caveats(s["id"])),
            }
        )
    return {
        "task_id": report["task_id"],
        "title": task["title"],
        "description": task["description"],
        "source": task.get("source", "recorded"),
        "result": case.get("result"),
        "calls": int(case.get("calls") or 0),
        "model": case.get("model"),
        "layout_moves": list(case.get("layout_moves") or []),
        "assumptions": list(task.get("assumptions") or []),
        "clarifications": [
            {"question": c.get("question", ""), "assumed": c.get("assumed_answer", c.get("assumed", ""))}
            for c in task.get("clarifications") or []
        ],
        "unsupported": [
            {"what": u.get("what", ""), "why": u.get("reason", ""), "nearest": u.get("nearest_primitive")}
            for u in task.get("unsupported") or []
        ],
        "objects": objects,
        "steps": steps,
    }


# --------------------------------------------------------------------------- screen and matrix


def screen_doc(report: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    sc = report["screen"]
    steps = [s["id"] for s in report["task"]["steps"]]
    pruned_all: dict[str, str] = {}
    pruned_steps: dict[str, list[dict[str, str]]] = {}
    for p in sc["pruned"]:
        if p.get("step_id") is None:
            pruned_all[p["effector_id"]] = p["reason"]
        else:
            pruned_steps.setdefault(p["effector_id"], []).append(
                {"step_id": p["step_id"], "reason": p["reason"]}
            )
    cells: dict[tuple[str, str], Mapping[str, Any]] = {
        (c["effector_id"], c["step_id"]): c for c in sc["cells"]
    }
    candidates = []
    for c in sc["candidates"]:
        eid = c["effector_id"]
        screened = any((eid, s) in cells for s in steps)
        candidates.append(
            {
                "effector_id": eid,
                "display_name": c["display_name"],
                "family": c["family"],
                "family_title": family_title(c["family"]),
                "source": c.get("source") or "preset",
                "off_the_shelf": bool(c.get("off_the_shelf")),
                "mechanism": c.get("mechanism") or "tool",
                "fidelity": _tier(c.get("fidelity")),
                "built_for": c.get("built_for"),
                "status": "screened" if screened else "pruned",
                "pruned_reason": pruned_all.get(eid),
                "pruned_steps": pruned_steps.get(eid, []),
            }
        )
    reasons = Counter()
    for p in sc["pruned"]:
        reasons[_reason_group(p["reason"])] += 1
    pruned_reasons = [
        {"reason": r, "count": n} for r, n in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    n_screened = sum(1 for c in candidates if c["status"] == "screened")
    screen = {
        "n_candidates": len(candidates),
        "n_screened": n_screened,
        "n_conditions": len(sc.get("conditions") or []),
        "runs": int(sc.get("runs") or 0),
        "wall_s": _r(float(sc.get("wall_s") or 0.0), 1),
        "candidates": candidates,
        "pruned_reasons": pruned_reasons,
    }
    rows = []
    by_id = {c["effector_id"]: c for c in candidates}
    for row in sc["matrix"]["rows"]:
        eid = row["effector_id"]
        cand = by_id.get(eid)
        if cand is None or cand["status"] != "screened":
            continue
        out_cells: dict[str, Any] = {}
        for s in steps:
            cell = cells.get((eid, s))
            raw = (row.get("cells") or {}).get(s) or {}
            if cell is not None:
                out_cells[s] = {
                    "state": "run",
                    "rate": _r(float(cell["rate"]), 4),
                    "n": int(cell["n"]),
                    "successes": int(cell["successes"]),
                    "primitive_used": cell.get("primitive_used") or cell.get("primitive"),
                    "failure_modes": dict(cell.get("failure_modes") or {}),
                    "context": cell.get("context") or "start",
                }
            else:
                reason = raw.get("pruned") or next(
                    (p["reason"] for p in pruned_steps.get(eid, []) if p["step_id"] == s), None
                )
                out_cells[s] = (
                    {"state": "pruned", "reason": reason}
                    if reason
                    else {
                        "state": "not_run",
                        "reason": "not screened on this step (an earlier step of its chain stopped it)",
                    }
                )
        rows.append(
            {
                "effector_id": eid,
                "display_name": cand["display_name"],
                "family_title": cand["family_title"],
                "fidelity": cand["fidelity"],
                "off_the_shelf": cand["off_the_shelf"],
                "cells": out_cells,
            }
        )

    def passing(r: Mapping[str, Any]) -> tuple[int, float, str]:
        run = [c for c in r["cells"].values() if c["state"] == "run"]
        return (-sum(1 for c in run if c["rate"] > 0), -sum(c["rate"] for c in run), r["display_name"])

    rows.sort(key=passing)
    return screen, {"steps": steps, "rows": rows}


#: Pruning reasons grouped for the summary: (a phrase of the reason, the group's title). The
#: first phrase found wins; a reason that matches none is its own group.
REASON_GROUPS: Final[tuple[tuple[str, str], ...]] = (
    ("closing axis; the gripper opens", "Opens narrower than the part"),
    ("too_narrow", "Its open digits stand too close for the part"),
    ("capability tags do not include", "Its capability tags do not include the step's primitive"),
    ("no hook or fingers", "A suction cup has no hook or fingers to engage a handle or lip"),
    ("cannot pick or hold a part", "It cannot pick or hold a part"),
    ("digit_collision", "Its digits collide when they close on the probe part"),
    ("cannot hold a member stiffly", "A soft arm cannot hold a member stiffly along the seat axis"),
    ("cup cannot seal", "No closed top face for a cup to seal on"),
    ("cup is wider than", "Its cup is wider than the face it must seal on"),
    ("the arm hangs only", "The part is wider than the soft arm reaches around"),
    ("no stiff face to push with", "A slack soft arm has no stiff face to push with"),
    ("the arm carries at most", "Heavier than the arm payload limit [assumed]"),
    ("shaped and sized for", "Part-shaped jaws serve only the object they are shaped for"),
    ("pre-shaping around", "Opens too little to pre-shape around the feature"),
    ("cannot be built", "Cannot be built for this object (the family's own checks)"),
    ("built against the block", "Cannot be built on the probe part"),
    ("amber licence", "An amber licence: never competes or ships"),
    ("frozen v2 episode", "Runs only inside the frozen v2 episode"),
    ("opens", "Opens narrower than the object"),
)


def _reason_group(reason: str) -> str:
    """The summary group of a pruning reason (:data:`REASON_GROUPS`)."""
    for phrase, title in REASON_GROUPS:
        if phrase in reason:
            return title
    return reason[:200]


# --------------------------------------------------------------------------- finalists


def _setup_tools(setup: Mapping[str, Any], off_ids: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "index": i,
            "effector_id": t["effector_id"],
            "display_name": t["display_name"],
            "family": t["family"],
            "family_title": family_title(t["family"]),
            "fidelity": _tier(t.get("fidelity")),
            "off_the_shelf": t["effector_id"] in off_ids,
        }
        for i, t in enumerate(setup["tools"])
    ]


def _tier(fidelity: str | None) -> str:
    """A family's fidelity as a Studio tier (the v1 ``core`` and ``rigid`` are both ``rigid``)."""
    return {"core": "rigid"}.get(fidelity or "rigid", fidelity or "rigid")


def _stops_at(row: Mapping[str, Any]) -> str | None:
    stops = row.get("stops_at") or {}
    if not stops:
        return None
    return max(stops.items(), key=lambda kv: (kv[1].get("runs", 0), kv[0]))[0]


def brief(row: Mapping[str, Any]) -> dict[str, Any]:
    n = int(row.get("n_runs") or 0)
    rate = float(row.get("completion_rate") or 0.0)
    completed = round(rate * n)
    return {
        "setup_id": row["setup_id"],
        "kind": row["kind"],
        "kind_title": KIND_TITLES[row["kind"]],
        "label": row["label"],
        "tools": [t["display_name"] if isinstance(t, Mapping) else str(t) for t in row.get("tools") or []],
        "completion_rate": _r(rate, 4),
        "completed_runs": completed,
        "n_runs": n,
        "fidelity": row.get("fidelity") or "rigid",
        "stops_at": _stops_at(row) if completed < n else None,
        "cycle_time_s": _r(row.get("cycle_time_s"), 2) if completed else None,
    }


def finalist_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rec = report["recommendation"]
    off_ids = {c["effector_id"] for c in report["screen"]["candidates"] if c.get("off_the_shelf")}
    setups = {f["setup"]["setup_id"]: f["setup"] for f in report["finalists"]}
    rec_id = (rec.get("recommendation") or {}).get("setup_id")
    best_id = report.get("best_single")
    rigid_id = ((rec.get("runner_up_rigid") or {}).get("setup") or {}).get("setup_id")
    closest = [c["setup_id"] for c in (rec.get("no_winner") or {}).get("closest") or []]
    rows = []
    for row in rec["ranking"]:
        sid = row["setup_id"]
        roles = []
        if sid == rec_id:
            roles.append("recommended")
        if sid == best_id:
            roles.append("best_single")
        if sid == rigid_id:
            roles.append("runner_up_rigid")
        if closest and sid == closest[0] and not rec_id:
            roles.append("closest")
        rows.append(
            {
                **brief(row),
                "roles": roles or ["finalist"],
                "eligible": bool(row.get("eligible")),
                "reason": row.get("reason"),
                "tool_details": _setup_tools(setups[sid], off_ids),
                "step_success": _rd(row.get("step_success")),
                "weakest_step": {
                    "step_id": (row.get("weakest_step") or {}).get("step_id"),
                    "success_rate": _r((row.get("weakest_step") or {}).get("success_rate")),
                },
                "complexity": {
                    "n_tools": int(row["complexity"]["n_tools"]),
                    "n_actuators": int(row["complexity"]["n_actuators"]),
                    "total_mass_kg": _r(float(row["complexity"]["total_mass_kg"]), 3),
                    "n_changes": int(row["complexity"]["n_changes"]),
                    "bracket": bool(row["complexity"]["bracket"]),
                },
                "payload_kg": _r(row.get("payload_kg"), 3),
                "failure_modes": dict(row.get("failure_modes") or {}),
                "stops_at_detail": {
                    k: {"runs": int(v.get("runs", 0)), "failure_modes": dict(v.get("failure_modes") or {})}
                    for k, v in (row.get("stops_at") or {}).items()
                },
            }
        )
    return rows


# --------------------------------------------------------------------------- recommendation


def _primitive_used(report: Mapping[str, Any], effector_id: str, step_id: str, fallback: str) -> str:
    for c in report["screen"]["cells"]:
        if c["effector_id"] == effector_id and c["step_id"] == step_id:
            return c.get("primitive_used") or fallback
    return fallback


def _family_caveat(family: str) -> str | None:
    from rtt.effectors.registry import get_family

    try:
        impl = get_family(family)
    except Exception:  # noqa: BLE001 - a family that does not load has no caveat to print
        return None
    text = str(getattr(impl, "caveat", "") or "").strip()
    return text or None


def recommendation_doc(report: Mapping[str, Any]) -> dict[str, Any] | None:
    rec = report["recommendation"]
    r = rec.get("recommendation")
    if not r:
        return None
    sid = r["setup_id"]
    finalist = next(f for f in report["finalists"] if f["setup"]["setup_id"] == sid)
    setup = finalist["setup"]
    runs = finalist["runs"]
    off_ids = {c["effector_id"] for c in report["screen"]["candidates"] if c.get("off_the_shelf")}
    tools = _setup_tools(setup, off_ids)
    for t in tools:
        t["caveat"] = _family_caveat(t["family"]) if t["fidelity"] == "approximation" else None
    by_step: dict[str, Mapping[str, Any]] = {}
    for run in runs:
        for s in run["steps"]:
            by_step.setdefault(s["step_id"], s)
    steps = []
    for s in report["task"]["steps"]:
        tool = int(setup["assignment"][s["id"]])
        outcome = by_step.get(s["id"]) or {}
        tier = outcome.get("fidelity") or (report["task"].get("fidelity") or {}).get(s["id"], "rigid")
        steps.append(
            {
                "step_id": s["id"],
                "primitive": s["primitive"],
                "primitive_used": _primitive_used(
                    report, setup["tools"][tool]["effector_id"], s["id"], s["primitive"]
                ),
                "tool": tool,
                "tool_label": setup["tools"][tool]["display_name"],
                "success_rate": _r(float((r.get("step_success") or {}).get(s["id"], 0.0)), 4),
                "fidelity": tier,
                "caveats": list(outcome.get("caveats") or []),
                "verify_on_hardware": tier == "approximation",
            }
        )
    geometry: dict[str, Any] | None = None
    if setup.get("mounts"):
        geometry = {
            "mounts": [_rd(m, 2) for m in setup["mounts"]],
            "clearance_mm": _r(((finalist.get("plan") or {}).get("geometry") or {}).get("clearance_mm"), 2),
            "present_s": _r(setup.get("present_s")),
        }
    elif setup.get("changer"):
        geometry = {"changer": _rd(setup["changer"], 3)}
    rigid = rec.get("runner_up_rigid") or {}
    return {
        **brief(r),
        "why": r.get("why"),
        "caveat": r.get("caveat"),
        "verify_on_hardware": any(s["verify_on_hardware"] for s in steps)
        or r.get("fidelity") == "approximation",
        "tools": tools,
        "geometry": geometry,
        "steps": steps,
        "runner_up_rigid": {
            "label": (rigid.get("setup") or {}).get("label"),
            "basis": rigid.get("basis") or "",
        },
        "versus_best_single": _rd(rec.get("versus_best_single"), 3)
        if rec.get("versus_best_single")
        else None,
    }


# --------------------------------------------------------------------------- sensitivity, constants


def sensitivity_doc(report: Mapping[str, Any]) -> dict[str, Any]:
    s = report.get("sensitivity") or {}
    combos = s.get("combos") or []
    return {
        "recommendation_changes": bool(s.get("recommendation_changes")),
        "order_changes": int(s.get("order_changes") or 0),
        "n_pairs": len(combos),
        "tool_change_s": sorted({float(c["tool_change_s"]) for c in combos}),
        "present_s": sorted({float(c["present_s"]) for c in combos}),
        "note": s.get("note") or "",
    }


def constants_doc(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "key": k,
            "value": _r(v["value"]),
            "unit": v.get("unit", ""),
            "note": v.get("note", ""),
            "tag": v.get("tag", "assumed"),
        }
        for k, v in (report.get("constants") or {}).items()
    ]


# --------------------------------------------------------------------------- headline


def headline(
    rec: Mapping[str, Any] | None,
    best: Mapping[str, Any] | None,
    closest: Mapping[str, Any] | None,
    steps: Mapping[str, str],
) -> str:
    def runs(b: Mapping[str, Any]) -> str:
        return f"{b['completed_runs']} of {b['n_runs']} runs"

    if rec is not None:
        kind = {
            "single": "A single tool",
            "multi_head": "A two-tool head",
            "tool_change": "A tool-change sequence",
        }[rec["kind"]]
        text = f"{kind} completes the task in {runs(rec)}"
        if best is not None and best["setup_id"] != rec["setup_id"]:
            text += f"; the best single off-the-shelf tool in {runs(best)}"
        return text + "."
    text = "No setup completes the task"
    if closest is not None and closest.get("stops_at"):
        text += f"; the closest setup stops at {steps.get(closest['stops_at'], closest['stops_at'])}"
    return text + "."


# --------------------------------------------------------------------------- checks


def iter_strings(value: Any, where: str = "$") -> Iterable[tuple[str, str, str]]:
    """(path, key, text) for every string in ``value``."""
    if isinstance(value, str):
        yield where, where.rsplit(".", 1)[-1], value
    elif isinstance(value, Mapping):
        for k, v in value.items():
            yield from iter_strings(v, f"{where}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from iter_strings(v, f"{where}[{i}]")


def check_doc(doc: Any, name: str) -> list[str]:
    """Brand tokens anywhere but in ``url`` fields; URLs outside ``url`` fields; local paths."""
    problems = []
    for where, key, text in iter_strings(doc):
        if key == "url":
            if not text.startswith("https://"):
                problems.append(f"{name} {where}: a link must be https")
            continue
        hits = find_library_brands(text)
        if hits:
            problems.append(f"{name} {where}: names a brand ({', '.join(sorted(set(hits)))})")
        if _URL.search(text):
            problems.append(f"{name} {where}: a URL outside a url field")
        if _LOCAL.search(text):
            problems.append(f"{name} {where}: a local path")
    return problems


# --------------------------------------------------------------------------- page


def hero_page(
    hero: str,
    case_id: str,
    report: Mapping[str, Any],
    report_path: Path,
    *,
    case: Mapping[str, Any] | None = None,
    determinism: Sequence[Mapping[str, Any]] = (),
    lanes: Sequence[Mapping[str, Any]] = (),
    build_pack: Mapping[str, Any] | None = None,
    build_pack_note: str | None = None,
    blocker: Mapping[str, Any] | None = None,
    session: Mapping[str, Any] | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """The ``rtt-web-studio-hero`` document of one hero (pure: no simulation, no files written)."""
    case = case or load_case(case_id)
    if case.get("task_id") and case["task_id"] != report["task_id"]:
        raise ExportError(
            f"{case_id}: the report ran task {report['task_id'][:12]}, the recorded plan is {case['task_id'][:12]}"
        )
    plan = plan_doc(report, case)
    screen, matrix = screen_doc(report)
    finalists = finalist_rows(report)
    rec = recommendation_doc(report)
    rows = {r["setup_id"]: r for r in finalists}
    best = rows.get(report.get("best_single") or "")
    nw = report["recommendation"].get("no_winner")
    no_winner = None
    closest = None
    if nw:
        closest_rows = [rows[c["setup_id"]] for c in nw.get("closest") or [] if c["setup_id"] in rows]
        closest = closest_rows[0] if closest_rows else None
        no_winner = {
            "reason": nw.get("reason") or "",
            "closest": closest_rows,
            "blocker": dict(blocker) if blocker else None,
        }
    labels = {o["name"]: o["label"] for o in plan["objects"]}
    titles = {
        s["id"]: f"step {i}, {s['primitive_title'].lower()} the {labels[s['object']].lower()}"
        for i, s in enumerate(plan["steps"], 1)
    }
    combos = report.get("combinations") or {}
    doc = {
        "schema": PAGE_SCHEMA,
        "version": WEB_VERSION,
        "label": PRODUCT_MODE_LABEL,
        "contract": CONTRACT_VERSION,
        "hero": hero,
        "case_id": case_id,
        "title": case.get("title") or plan["title"],
        "setting": case.get("setting") or "",
        "use_case": case.get("text") or plan["description"],
        "outcome": "recommended" if rec else "no_recommendation",
        "headline": headline(rec, best, closest, titles),
        "plan": plan,
        "screen": screen,
        "matrix": matrix,
        "combinations": {
            "counts": {k: int(v) for k, v in (combos.get("counts") or {}).items()},
            "rejected": [
                {"label": x.get("label", ""), "reason": x.get("reason", "")}
                for x in combos.get("rejected") or []
            ],
        },
        "finalists": finalists,
        "recommendation": rec,
        "best_single": best,
        "no_winner": no_winner,
        "sensitivity": sensitivity_doc(report),
        "constants": constants_doc(report),
        "fixed_caveats": list(report.get("fixed_caveats") or []),
        "lanes": [dict(lane) for lane in lanes],
        "build_pack": dict(build_pack) if build_pack else None,
        "build_pack_note": build_pack_note,
        "session": dict(session) if session else None,
        "provenance": {
            "code_version": version or code_version(),
            "report_sha256": _sha256(report_path),
            "task_id": report["task_id"],
            "controllers": report.get("controllers"),
            "evidence": report["recommendation"].get("evidence"),
            "conditions": [
                {
                    "condition_id": c["condition_id"],
                    "seed": int(c["seed"]),
                    "aim_offsets_mm": {
                        k: [_r(float(x), 3) for x in v] for k, v in (c.get("aim_offsets_mm") or {}).items()
                    },
                }
                for c in report["conditions"]
            ],
            "fairness_sha": _fairness_sha(report),
            "workers": report.get("workers"),
            "timings": _rd(report.get("timings"), 1),
            "determinism": [{"condition": d["condition"], "same": bool(d["same"])} for d in determinism],
        },
    }
    return doc


def _fairness_sha(report: Mapping[str, Any]) -> str | None:
    shas = {run.get("fairness_sha") for f in report["finalists"] for run in f["runs"]}
    shas.discard(None)
    shas.discard("unbuilt")
    return shas.pop() if len(shas) == 1 else None


def index_entry(page: Mapping[str, Any]) -> dict[str, Any]:
    def strip(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        keys = (
            "setup_id",
            "kind",
            "kind_title",
            "label",
            "tools",
            "completion_rate",
            "completed_runs",
            "n_runs",
            "fidelity",
            "stops_at",
            "cycle_time_s",
        )
        return {k: row[k] for k in keys}

    nw = page.get("no_winner") or {}
    return {
        "hero": page["hero"],
        "case_id": page["case_id"],
        "title": page["title"],
        "setting": page["setting"],
        "use_case": page["use_case"],
        "page": f"studio/{page['hero']}.json",
        "task_id": page["plan"]["task_id"],
        "n_steps": len(page["plan"]["steps"]),
        "n_candidates": page["screen"]["n_candidates"],
        "outcome": page["outcome"],
        "headline": page["headline"],
        "recommended": strip(page["recommendation"]),
        "best_single": strip(page["best_single"]),
        "closest": strip((nw.get("closest") or [None])[0]) if nw else None,
    }


def index_doc(pages: Sequence[Mapping[str, Any]], version: str) -> dict[str, Any]:
    return {
        "schema": INDEX_SCHEMA,
        "version": WEB_VERSION,
        "label": PRODUCT_MODE_LABEL,
        "contract": CONTRACT_VERSION,
        "code_version": version,
        "heroes": [index_entry(p) for p in pages],
    }


# --------------------------------------------------------------------------- build pack


def export_pack(
    report_path: Path, report: Mapping[str, Any], work: Path
) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    """Re-export the build pack of the recommendation (or, with none, the closest setup) into
    ``work`` with :func:`export_part_jaw_pack`; returns the zip, its page block and a note."""
    from rtt.studio.buildpack.part_jaw import export_part_jaw_pack

    rec = report["recommendation"]
    target = "recommended"
    sid = (rec.get("recommendation") or {}).get("setup_id")
    if sid is None:
        closest = (rec.get("no_winner") or {}).get("closest") or []
        if not closest:
            return None, None, "No setup was run on the full grid, so there is nothing to pack."
        sid, target = closest[0]["setup_id"], "closest"
    out = work / "build_pack"
    pack = export_part_jaw_pack(dict(report), out, setup_id=sid, overwrite=True)
    zip_path = Path(pack.zip_path)
    size = zip_path.stat().st_size
    if size > PACK_CAP_BYTES:
        return (
            None,
            None,
            (
                f"The build pack is {size / 1e6:.1f} MB, over the {PACK_CAP_BYTES / 1e6:.0f} MB this site serves; "
                "export it locally with the Studio's export_build_pack."
            ),
        )
    manifest = pack.manifest
    files = []
    for f in manifest.files:
        p = out / f.path
        files.append(
            {
                "path": f.path,
                "kind": str(getattr(f.kind, "value", f.kind)),
                "format": f.format,
                "license": f.license,
                "bytes": p.stat().st_size if p.is_file() else 0,
            }
        )
    links = [
        {"name": link.name, "url": link.url, "license": link.license, "reason": link.reason}
        for link in manifest.links
    ]
    label = next(f["setup"]["label"] for f in report["finalists"] if f["setup"]["setup_id"] == sid)
    block = {
        "zip": "",  # set by the caller (data-relative)
        "bytes": size,
        "sha256": _sha256(zip_path),
        "setup_id": sid,
        "label": label,
        "for": target,
        "files": files,
        "links": links,
        "caveats": list(manifest.caveats),
    }
    note = None
    if target == "closest":
        note = "No setup completes this task, so this is the pack of the closest setup, for inspection only."
    return zip_path, block, note


# --------------------------------------------------------------------------- sessions


def scrub_session(doc: Mapping[str, Any]) -> dict[str, Any]:
    """A recorded Grasp Engineer session as the site serves it: no chat transcript (it repeats the
    system prompt), every string scrubbed of held-out words and brands, local paths cut."""
    from rtt.agent.recorder import SESSION_FORMAT, redact_paths
    from rtt.studio.engineer.engineer import scrub_text

    if doc.get("format") != SESSION_FORMAT or not doc.get("events"):
        raise ExportError("not a recorded session with events")

    def clean(v: Any) -> Any:
        if isinstance(v, str):
            return scrub_text(redact_paths(v))
        if isinstance(v, Mapping):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v

    out = {k: clean(v) for k, v in doc.items() if k != "messages"}
    for event in out["events"]:
        if event.get("type") == "summary" and isinstance(event.get("text"), str):
            event["text"] = trim_capped(event["text"])
    if isinstance(out.get("summary"), str):
        out["summary"] = trim_capped(out["summary"])
    out["kind"] = "recorded"
    return out


#: The summarizer's length cap: a summary this long was cut by it, possibly mid-word.
SUMMARY_CAP_CHARS: Final = 400


def trim_capped(text: str) -> str:
    """A summary the recorder cut at :data:`SUMMARY_CAP_CHARS`, shown up to its last full sentence
    (an uncut summary is returned as it is)."""
    if len(text) < SUMMARY_CAP_CHARS - 1 or text.rstrip().endswith((".", "!", "?")):
        return text
    end = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
    return text[: end + 1] if end > 0 else text


def session_block(doc: Mapping[str, Any], rel: str) -> dict[str, Any]:
    finished = next((e for e in reversed(doc["events"]) if e.get("type") == "session.finished"), {})
    models = (
        doc.get("models")
        or next((e.get("models") for e in doc["events"] if e.get("type") == "session.started"), {})
        or {}
    )
    spend = finished.get("spend") or doc.get("spend") or {}
    return {
        "file": rel,
        "session_id": doc.get("session_id"),
        "status": finished.get("status") or doc.get("status"),
        "model": models.get("agent"),
        "turns": int(finished.get("turns") or 0),
        "tool_calls": int(finished.get("tool_calls") or 0),
        "spend_usd": _r(float(spend.get("spent_usd") or 0.0), 6),
        "answer": finished.get("answer") or doc.get("answer") or "",
    }


# --------------------------------------------------------------------------- driver


def _write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def export(
    reports: Path,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    assets_dir: Path = DEFAULT_ASSETS_DIR,
    sessions: Path | None = None,
    heroes: Sequence[str] = tuple(h for h, _ in HEROES),
    replays: bool = True,
    packs: bool = True,
    blocker: bool = True,
    workers: int = 4,
    compress: bool = True,
    log: Any = print,
) -> dict[str, Any]:
    """Build every hero's data in a scratch folder, check it, then publish (module docstring)."""
    from rtt.sim.replay import write_replay

    version = code_version()
    cases = dict(HEROES)
    unknown = [h for h in heroes if h not in cases]
    if unknown:
        raise ExportError(f"unknown heroes {unknown}; known: {sorted(cases)}")
    staged: dict[str, Path] = {}  # data-relative path -> staged file
    problems: list[str] = []
    served: dict[str, tuple[bytes, Mapping[str, Any]]] = {}
    with tempfile.TemporaryDirectory(prefix="studio-web-") as tmp_name:
        tmp = Path(tmp_name)
        store = None
        if replays:
            from rtt.studio.replay import staging_store

            store = staging_store(tmp / "raw_assets")
        # 1. per hero: lanes (re-simulated), the build pack, the blocker probe, the session
        work: list[dict[str, Any]] = []
        for hero in heroes:
            case_id = cases[hero]
            report, report_path = load_report(reports, case_id)
            det_path = report_path.with_name("determinism.json")
            determinism = json.loads(det_path.read_text(encoding="utf-8")) if det_path.is_file() else []
            case = load_case(case_id)
            log(f"{hero}: {case_id}, task {report['task_id'][:12]}")
            item: dict[str, Any] = {
                "hero": hero,
                "case_id": case_id,
                "case": case,
                "report": report,
                "report_path": report_path,
                "determinism": determinism,
                "lanes": [],
                "pack": None,
                "pack_note": None,
                "blocker": None,
                "session": None,
            }
            if replays:
                from rtt.studio.replay import export_lanes

                item["lanes"] = export_lanes(hero, report, store, code_version=version, log=log)
            if packs:
                zip_path, block, note = export_pack(report_path, report, tmp / "packs" / hero)
                item["pack_note"] = note
                if zip_path is not None and block is not None:
                    rel = f"studio/{hero}/build_pack.zip"
                    block["zip"] = rel
                    staged[rel] = zip_path
                    item["pack"] = block
                    log(
                        f"  build pack: {block['label'][:50]} ({block['for']}), {block['bytes'] / 1e3:.0f} kB"
                    )
            if blocker and not report["recommendation"].get("recommendation"):
                from rtt.studio.replay import blocker_probe

                item["blocker"] = blocker_probe(report, workers=workers, log=log)
            if sessions is not None:
                src = sessions / f"{hero}.json"
                if src.is_file():
                    doc = scrub_session(json.loads(src.read_text(encoding="utf-8")))
                    rel = f"studio/sessions/{hero}.json"
                    problems += check_doc(doc, rel)
                    out = tmp / "sessions" / f"{hero}.json"
                    _write_json(out, doc)
                    staged[rel] = out
                    item["session"] = session_block(doc, rel)
            work.append(item)
        # 2. the replays' GLBs: served once (compressed), every header pointed at them
        if store is not None:
            from rtt.studio.replay import serve_assets

            all_lanes = [lane for item in work for lane in item["lanes"]]
            served = serve_assets(store, all_lanes, compress=compress)
            for item in work:
                for lane in item["lanes"]:
                    stem = tmp / "replays" / item["hero"] / lane.stem
                    bin_path, json_path = write_replay(lane.replay, stem)
                    rel = f"studio/replays/{item['hero']}/{lane.stem}"
                    staged[f"{rel}.json"] = json_path
                    staged[f"{rel}.bin"] = bin_path
                    lane.doc["replay"] = rel
                    lane.doc["bytes"] = json_path.stat().st_size + bin_path.stat().st_size
                    lane.doc["asset_bytes"] = sum(len(served[a][0]) for a in lane.assets)
                    problems += check_doc(json.loads(json_path.read_text(encoding="utf-8")), f"{rel}.json")
        # 3. pages and the index
        pages = []
        for item in work:
            page = hero_page(
                item["hero"],
                item["case_id"],
                item["report"],
                item["report_path"],
                case=item["case"],
                determinism=item["determinism"],
                lanes=[lane.doc for lane in item["lanes"]],
                build_pack=item["pack"],
                build_pack_note=item["pack_note"],
                blocker=item["blocker"],
                session=item["session"],
                version=version,
            )
            problems += check_doc(page, f"studio/{item['hero']}.json")
            pages.append(page)
            out = tmp / "pages" / f"{item['hero']}.json"
            _write_json(out, page)
            staged[f"studio/{item['hero']}.json"] = out
        if set(heroes) != {h for h, _ in HEROES}:  # a partial export keeps the other heroes' pages
            for h, _ in HEROES:
                if h not in heroes and (data_dir / "studio" / f"{h}.json").is_file():
                    pages.append(json.loads((data_dir / "studio" / f"{h}.json").read_text(encoding="utf-8")))
            order = [h for h, _ in HEROES]
            pages.sort(key=lambda p: order.index(p["hero"]))
        index = index_doc(pages, version)
        problems += check_doc(index, "studio/index.json")
        _write_json(tmp / "pages" / "index.json", index)
        staged["studio/index.json"] = tmp / "pages" / "index.json"
        if problems:
            raise ExportError("refused to publish:\n  " + "\n  ".join(problems[:40]))
        # 4. publish: the heroes' own files are replaced, everything else is kept
        for hero in heroes:
            for sub in (data_dir / "studio" / "replays" / hero, data_dir / "studio" / hero):
                if sub.is_dir():
                    shutil.rmtree(sub)
            sess = data_dir / "studio" / "sessions" / f"{hero}.json"
            if sess.is_file() and f"studio/sessions/{hero}.json" not in staged:
                sess.unlink()
        for rel, path in sorted(staged.items()):
            dst = data_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dst)
        if served:
            from rtt.studio.replay import publish_assets

            publish_assets(served, assets_dir)
        sizes = {rel: (data_dir / rel).stat().st_size for rel in staged}
    total = sum(sizes.values())
    asset_total = sum(len(b) for b, _ in served.values())
    log(
        f"published {len(staged)} files ({total / 1e6:.2f} MB) under {data_dir.name}/studio and "
        f"{len(served)} GLB assets ({asset_total / 1e6:.2f} MB)"
    )
    return {"files": sizes, "total_bytes": total, "assets": sorted(served), "asset_bytes": asset_total}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports", type=Path, required=True, help="folder with <case_id>/report.json")
    parser.add_argument(
        "--sessions", type=Path, default=None, help="folder with recorded <hero>.json sessions"
    )
    parser.add_argument("--heroes", default=",".join(h for h, _ in HEROES))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-replays", action="store_true")
    parser.add_argument("--no-packs", action="store_true")
    parser.add_argument("--no-blocker", action="store_true")
    parser.add_argument("--no-compress", action="store_true", help="serve raw GLBs (tests)")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be 1 to 4 (the host is shared)")
    try:
        export(
            args.reports,
            data_dir=args.data_dir,
            assets_dir=args.assets_dir,
            sessions=args.sessions,
            heroes=[h.strip() for h in args.heroes.split(",") if h.strip()],
            replays=not args.no_replays,
            packs=not args.no_packs,
            blocker=not args.no_blocker,
            workers=args.workers,
            compress=not args.no_compress,
        )
    except ExportError as exc:
        print(f"export_studio_web: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
