"""The intake page's data, generated from the contract and the recorded runs (``export-web``).

:func:`write_web_data` writes three files into ``web/components/studio/intake/``:

* ``catalog.json``: :func:`rtt.studio.engineer.catalog.web_catalog` (templates, primitives,
  quantities, limits): what the step editor offers and validates against.
* ``recorded.json``: the recorded plans the static site plays (every gate case and the upload
  example), each with its attempts, problems, warnings and the final plan.
* ``parity.json``: edited versions of the recorded plans (the edits the step editor can make)
  with the exact problems the Python side reports, so the page's validator is tested against
  :func:`rtt.studio.engineer.engineer.judge_args` message for message.

A test regenerates all three and compares them with the committed copies.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from rtt.effectors.types import PRODUCT_MODE_LABEL
from rtt.studio.engineer.catalog import web_catalog
from rtt.studio.engineer.engineer import ENGINEER_MODEL, judge_args, normalise_nulls
from rtt.studio.engineer.examples import GATE_CASES, HERO_CASES, UPLOAD_CASE
from rtt.studio.engineer.gate import RECORDED_GATE
from rtt.studio.types import CONTRACT_VERSION

__all__ = ["RECORDED_UPLOAD", "parity_cases", "recorded_plans", "write_web_data"]

RECORDED_UPLOAD: Final = Path(__file__).with_name("recorded") / "upload-2026-10-18"
FILES: Final = ("catalog.json", "recorded.json", "parity.json")


def _dump(data: Any) -> str:
    return json.dumps(data, indent=1, ensure_ascii=False) + "\n"


def _example(row: dict[str, Any], *, hero: bool) -> dict[str, Any]:
    plan = row["plan"]
    return {
        "case_id": row["case_id"],
        "setting": row["setting"],
        "title": row["title"],
        "text": row["text"],
        "hero": hero,
        "expect": row["expect"],
        "status": plan["status"],
        "first_try": plan["first_try"],
        "repaired": plan["repaired"],
        "attempts": [
            {
                k: a[k]
                for k in (
                    "stage",
                    "outcome",
                    "problems",
                    "cost_usd",
                    "latency_s",
                    "prompt_tokens",
                    "completion_tokens",
                )
            }
            for a in plan["attempts"]
        ],
        "problems": plan["problems"],
        "warnings": plan["warnings"],
        "notes": plan["notes"],
        "uploads": plan["uploads"],
        "plan": plan["plan"],
        "cost_usd": plan["cost_usd"],
    }


def recorded_plans(recorded: Path | None = None) -> dict[str, Any]:
    """``recorded.json``: the recorded gate run's plans (heroes first) and the upload example."""
    root = recorded or RECORDED_GATE
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    heroes = {c.case_id for c in HERO_CASES}
    rows = {r["case_id"]: r for r in manifest["cases"]}
    examples = [_example(rows[c.case_id], hero=c.case_id in heroes) for c in GATE_CASES if c.case_id in rows]
    upload_path = RECORDED_UPLOAD / "plan.json"
    if upload_path.is_file():
        plan = json.loads(upload_path.read_text(encoding="utf-8"))
        row = {
            "case_id": UPLOAD_CASE.case_id,
            "setting": UPLOAD_CASE.setting,
            "title": UPLOAD_CASE.title,
            "text": UPLOAD_CASE.text,
            "expect": "plan",
            "plan": plan,
        }
        examples.append(_example(row, hero=False))
    return {
        "format": "rtt-studio-recorded-plans",
        "version": 1,
        "label": PRODUCT_MODE_LABEL,
        "contract": CONTRACT_VERSION,
        "model": ENGINEER_MODEL,
        "recorded_at": manifest["written_at"],
        "gate": {"passed": manifest["gate"]["passed"], "summary": manifest["summary"]},
        "examples": examples,
    }


# --------------------------------------------------------------------------- parity cases


def _args(example: dict[str, Any]) -> dict[str, Any]:
    plan = example["plan"]
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
    return copy.deepcopy({k: plan[k] for k in keys})


def _step(args: dict[str, Any], primitive: str) -> dict[str, Any]:
    return next(s for s in args["steps"] if s["primitive"] == primitive)


def _object(args: dict[str, Any], kind: str) -> dict[str, Any]:
    return next(o for o in args["objects"] if o["kind"] == kind)


def _edits() -> list[tuple[str, str, Callable[[dict[str, Any]], None]]]:
    """(name, base case id, edit) for every parity case; each edit is one the page can make."""

    def setp(kind: str, key: str, value: Any) -> Callable[[dict[str, Any]], None]:
        return lambda a: _object(a, kind)["params"].__setitem__(key, value)

    def move(i: int, j: int) -> Callable[[dict[str, Any]], None]:
        def f(a: dict[str, Any]) -> None:
            a["steps"].insert(j, a["steps"].pop(i))

        return f

    def chain(*fs: Callable[[dict[str, Any]], None]) -> Callable[[dict[str, Any]], None]:
        def f(a: dict[str, Any]) -> None:
            for g in fs:
                g(a)

        return f

    def crit(primitive: str, rows: list[dict[str, Any]]) -> Callable[[dict[str, Any]], None]:
        return lambda a: _step(a, primitive).__setitem__("success", rows)

    def cons(primitive: str, key: str, value: Any) -> Callable[[dict[str, Any]], None]:
        return lambda a: _step(a, primitive)["constraints"].__setitem__(key, value)

    def sparam(primitive: str, key: str, value: Any) -> Callable[[dict[str, Any]], None]:
        return lambda a: _step(a, primitive)["params"].__setitem__(key, value)

    def sfield(primitive: str, key: str, value: Any) -> Callable[[dict[str, Any]], None]:
        return lambda a: _step(a, primitive).__setitem__(key, value)

    def drop_object(kind: str) -> Callable[[dict[str, Any]], None]:
        return lambda a: a["objects"].remove(_object(a, kind))

    def drop_step(primitive: str) -> Callable[[dict[str, Any]], None]:
        return lambda a: a["steps"].remove(_step(a, primitive))

    def pose(kind: str, key: str, value: float) -> Callable[[dict[str, Any]], None]:
        return lambda a: _object(a, kind)["pose"].__setitem__(key, value)

    def dup_step(primitive: str) -> Callable[[dict[str, Any]], None]:
        return lambda a: a["steps"].append(copy.deepcopy(_step(a, primitive)))

    def many_steps(a: dict[str, Any]) -> None:
        base = a["steps"][-1]
        for i in range(13 - len(a["steps"])):
            extra = copy.deepcopy(base)
            extra["id"] = f"extra_{i}"
            a["steps"].append(extra)

    return [
        ("valid_h1", "h1_tote", lambda a: None),
        ("valid_h2", "h2_coupling", lambda a: None),
        ("valid_h3", "h3_cabinet", lambda a: None),
        ("valid_e3", "e3_isolation_valve", lambda a: None),
        ("refusal_u1", "u1_fold_shirt", lambda a: None),
        ("object_param_high", "h1_tote", setp("bin", "width_mm", 900)),
        ("object_param_low", "h3_cabinet", setp("port_plug", "clearance_mm", 0.01)),
        ("object_param_int", "h3_cabinet", setp("handwheel_valve", "spokes", 2.5)),
        (
            "object_two_bad_params",
            "h3_cabinet",
            chain(setp("quarter_turn_latch", "turn_deg", 10), setp("quarter_turn_latch", "torque_nm", 9)),
        ),
        ("pose_out_of_range", "h1_tote", pose("part", "x_mm", 400)),
        ("pose_and_param", "h1_tote", chain(pose("bin", "y_mm", -350), setp("bin", "wall_mm", 30))),
        ("step_param_high", "h1_tote", sparam("hook_pull", "speed_mm_s", 500)),
        (
            "criterion_bound",
            "h3_cabinet",
            crit("hook_pull", [{"quantity": "angle_deg", "at_least": 2000, "at_most": None}]),
        ),
        (
            "criterion_inverted",
            "h3_cabinet",
            crit("hook_pull", [{"quantity": "angle_deg", "at_least": 100, "at_most": 50}]),
        ),
        (
            "criterion_empty_bounds",
            "h3_cabinet",
            crit("hook_pull", [{"quantity": "angle_deg", "at_least": None, "at_most": None}]),
        ),
        (
            "criterion_unmeasured",
            "h1_tote",
            crit("hook_pull", [{"quantity": "turns", "at_least": 1, "at_most": None}]),
        ),
        (
            "criterion_no_goal",
            "h1_tote",
            crit("push", [{"quantity": "force_n", "at_least": None, "at_most": 20}]),
        ),
        (
            "criteria_five",
            "h1_tote",
            crit("push", [{"quantity": "angle_deg", "at_least": None, "at_most": 5}] * 5),
        ),
        ("criteria_none", "h1_tote", crit("push", [])),
        ("constraint_high", "h3_cabinet", cons("insert", "max_force_n", 900)),
        (
            "in_target_without_target",
            "h3_cabinet",
            crit("grasp", [{"quantity": "in_target", "at_least": 1, "at_most": None}]),
        ),
        ("target_not_bin", "h1_tote", sfield("grasp", "target", "tote_lid")),
        ("target_unknown", "h1_tote", sfield("grasp", "target", "nowhere")),
        ("target_and_hold", "h1_tote", sfield("grasp", "hold_after", True)),
        ("hold_on_push", "h1_tote", sfield("push", "hold_after", True)),
        ("handle_wrong", "h1_tote", sfield("push", "handle", "handle")),
        ("description_empty", "h1_tote", sfield("push", "description", "")),
        ("description_long", "h1_tote", sfield("push", "description", "x" * 201)),
        ("object_deleted", "h1_tote", drop_object("bin")),
        ("step_deleted_insert_unheld", "h2_coupling", drop_step("grasp")),
        ("steps_reordered", "h2_coupling", move(1, 0)),
        ("grasp_moved_first", "h3_cabinet", move(3, 0)),
        ("insert_moved_first", "h3_cabinet", move(4, 0)),
        ("after_later_step", "h1_tote", sfield("hook_pull", "after", ["close_lid"])),
        ("duplicate_step", "h1_tote", dup_step("push")),
        ("too_many_steps", "e3_isolation_valve", many_steps),
        (
            "unreachable_angle",
            "h1_tote",
            crit("hook_pull", [{"quantity": "angle_deg", "at_least": 150, "at_most": None}]),
        ),
        (
            "unreachable_turns",
            "h2_coupling",
            crit("twist", [{"quantity": "turns", "at_least": 9, "at_most": None}]),
        ),
        ("wrist_torque", "e3_isolation_valve", setp("handwheel_valve", "torque_nm", 18)),
        ("torque_cap_low", "e3_isolation_valve", cons("twist", "max_torque_nm", 1)),
        (
            "force_cap_low",
            "h3_cabinet",
            chain(setp("port_plug", "insert_force_n", 30), cons("insert", "max_force_n", 10)),
        ),
        (
            "lift_over_param",
            "h3_cabinet",
            crit("grasp", [{"quantity": "lift_mm", "at_least": 150, "at_most": None}]),
        ),
        ("label_long", "h1_tote", lambda a: _object(a, "bin").__setitem__("label", "L" * 61)),
        ("label_empty", "h1_tote", lambda a: _object(a, "bin").__setitem__("label", "")),
        ("title_empty", "h1_tote", lambda a: a.__setitem__("title", "")),
        ("last_step_holds", "h3_cabinet", chain(drop_step("insert"))),
    ]


def parity_cases(recorded: Path | None = None) -> dict[str, Any]:
    """``parity.json``: each edit's arguments and the problems and warnings Python reports."""
    examples = {e["case_id"]: e for e in recorded_plans(recorded)["examples"]}
    cases = []
    for name, base, edit in _edits():
        args = _args(examples[base])
        edit(args)
        args, _ = normalise_nulls(args)
        spec, problems, warnings = judge_args(args, source="manual")
        status = "invalid" if problems or spec is None else ("refused" if spec.refusal else "valid")
        cases.append(
            {
                "name": name,
                "base": base,
                "args": args,
                "status": status,
                "problems": problems,
                "warnings": warnings,
            }
        )
    return {"format": "rtt-studio-parity", "version": 1, "cases": cases}


def write_web_data(recorded: Path | None = None, out: Path | None = None) -> list[Path]:
    """Write :data:`FILES` into ``out`` (default ``web/components/studio/intake``)."""
    out = out or Path(__file__).resolve().parents[4] / "web" / "components" / "studio" / "intake"
    out.mkdir(parents=True, exist_ok=True)
    payloads = {
        "catalog.json": web_catalog(),
        "recorded.json": recorded_plans(recorded),
        "parity.json": parity_cases(recorded),
    }
    paths = []
    for name in FILES:
        path = out / name
        path.write_text(_dump(payloads[name]), encoding="utf-8")
        paths.append(path)
    return paths
