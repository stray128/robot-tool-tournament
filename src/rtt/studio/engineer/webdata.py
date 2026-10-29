"""The intake page's data, generated from the contract and the recorded runs (``export-web``).

:func:`write_web_data` writes three files into ``web/components/studio/intake/``:

* ``catalog.json``: :func:`rtt.studio.engineer.catalog.web_catalog` (templates, primitives,
  quantities, limits): what the step editor offers and validates against.
* ``recorded.json``: the recorded plans the static site plays, each with its attempts, problems,
  warnings, the final plan and the day it was recorded: Studio v2's six heroes (recorded on the
  current prompt; a hero whose recorded plan is not valid, h4_cnc, is listed but not offered as a
  hero example), then the rest of the v1 test set (the cases the v2 heroes do not replace, recorded
  on the v1 prompt), then the upload example.
* ``parity.json``: edited versions of the recorded plans (the edits the step editor can make)
  with the exact problems the Python side reports, so the page's validator is tested against
  :func:`rtt.studio.engineer.engineer.judge_args` message for message. The edits written for the
  v1 plans keep the v1 plans as their bases (``v1:<case>``); the job templates' edits use the
  Studio v2 heroes.

A test regenerates all three and compares them with the committed copies.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

from rtt.effectors.types import PRODUCT_MODE_LABEL
from rtt.studio.engineer.catalog import web_catalog
from rtt.studio.engineer.engineer import ENGINEER_MODEL, judge_args, normalise_nulls
from rtt.studio.engineer.examples import HERO_CASES, UPLOAD_CASE, V1_HERO_CASES
from rtt.studio.engineer.gate import RECORDED_GATE, RECORDED_GATE_V1
from rtt.studio.types import CONTRACT_VERSION

__all__ = ["RECORDED_UPLOAD", "parity_cases", "recorded_plans", "write_web_data"]

RECORDED_UPLOAD: Final = Path(__file__).with_name("recorded") / "upload-2026-10-18-s4e"
FILES: Final = ("catalog.json", "recorded.json", "parity.json")


def _dump(data: Any, *, compact: bool = False) -> str:
    if compact:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
    return json.dumps(data, indent=1, ensure_ascii=False) + "\n"


def _now(plan: dict[str, Any]) -> str:
    """The status today's rules (v2, what the page validates with) give a recorded plan."""
    doc = plan.get("plan")
    if not doc:
        return str(plan["status"])
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
    uploads = [SimpleNamespace(upload_id=u["upload_id"]) for u in plan.get("uploads") or []]  # ids only
    spec, problems, _ = judge_args({k: doc[k] for k in keys}, uploads=uploads, source="recorded")  # type: ignore[arg-type]
    return "invalid" if problems or spec is None else ("refused" if spec.refusal else "valid")


def _example(row: dict[str, Any], *, hero: bool, recorded_at: str, rules: str = "v2") -> dict[str, Any]:
    plan = row["plan"]
    return {
        "case_id": row["case_id"],
        "setting": row["setting"],
        "title": row["title"],
        "text": row["text"],
        "hero": hero,
        "expect": row["expect"],
        "recorded_at": recorded_at,
        "rules": rules,
        "status_now": _now(plan),
        "status": plan["status"],
        "first_try": plan["first_try"],
        "repaired": plan["repaired"],
        "layout_fixed": plan["layout_fixed"],
        "result": plan["result"],
        "layout_moves": plan["layout_moves"],
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


def _manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def recorded_plans(recorded: Path | None = None) -> dict[str, Any]:
    """``recorded.json`` (module docstring). With ``recorded``, that one run's plans in the order it
    ran (its heroes marked: the v1 heroes for the v1 run) and the upload example."""
    root = recorded or RECORDED_GATE
    manifest = _manifest(root)
    v1 = root.resolve() == RECORDED_GATE_V1.resolve()
    heroes = {c.case_id for c in (V1_HERO_CASES if v1 else HERO_CASES)}
    at, rules = manifest["written_at"], str(manifest.get("rules_version", "v1"))
    examples = [  # a hero with no valid recorded plan (h4_cnc) is not offered as a hero example
        _example(
            r, hero=r["case_id"] in heroes and r["plan"]["status"] == "valid", recorded_at=at, rules=rules
        )
        for r in manifest["cases"]
    ]
    if recorded is None:  # the v2 heroes first, then the rest of the v1 test set
        order = {c.case_id: i for i, c in enumerate(HERO_CASES)}
        examples.sort(key=lambda e: order.get(e["case_id"], len(order)))
        old = _manifest(RECORDED_GATE_V1)
        taken = {e["case_id"] for e in examples}
        examples += [
            _example(r, hero=False, recorded_at=old["written_at"], rules="v1")
            for r in old["cases"]
            if r["case_id"] not in taken
        ]
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
        examples.append(_example(row, hero=False, recorded_at=_upload_recorded_at(), rules="v1"))
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


def _upload_recorded_at() -> str:
    """When the upload example was recorded: its ledger's first entry (it has no manifest)."""
    from rtt.llm.ledger import read_ledger

    entries = read_ledger(RECORDED_UPLOAD / "ledger.jsonl")
    return str(entries[0].ts) if entries else ""


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

    def nth(primitive: str, n: int, key: str, value: Any) -> Callable[[dict[str, Any]], None]:
        return lambda a: [x for x in a["steps"] if x["primitive"] == primitive][n].__setitem__(key, value)

    def nth_param(primitive: str, n: int, key: str, value: Any) -> Callable[[dict[str, Any]], None]:
        return lambda a: [x for x in a["steps"] if x["primitive"] == primitive][n]["params"].__setitem__(
            key, value
        )

    def add_step(index: int, step: dict[str, Any]) -> Callable[[dict[str, Any]], None]:
        return lambda a: a["steps"].insert(index if index >= 0 else len(a["steps"]), copy.deepcopy(step))

    def add_step_on(index: int, step: dict[str, Any], kind: str) -> Callable[[dict[str, Any]], None]:
        """``step`` on the plan's object of ``kind`` (whatever the planner named it)."""

        def f(a: dict[str, Any]) -> None:
            a["steps"].insert(index, {**copy.deepcopy(step), "object": _object(a, kind)["name"]})

        return f

    def no_cons(primitive: str) -> Callable[[dict[str, Any]], None]:
        return lambda a: _step(a, primitive).__setitem__(
            "constraints",
            {k: None for k in ("max_force_n", "max_torque_nm", "precision_mm", "precision_deg")},
        )

    none_cons = {"max_force_n": None, "max_torque_nm": None, "precision_mm": None, "precision_deg": None}
    grasp_coupling = {
        "id": "pick_coupling",
        "primitive": "grasp",
        "object": "pipe_coupling",
        "handle": "member",
        "target": None,
        "description": "Pick the coupling up.",
        "params": {"lift_mm": 30, "hold_s": 1},
        "constraints": none_cons,
        "success": [{"quantity": "lift_mm", "at_least": 20, "at_most": None}],
        "after": [],
        "hold_after": False,
    }
    close_valve = {
        "id": "close_valve",
        "primitive": "twist",
        "object": "valve",
        "handle": "wheel",
        "target": None,
        "description": "Turn the handwheel back shut.",
        "params": {"direction": "cw", "speed_deg_s": 60},
        "constraints": none_cons,
        "success": [{"quantity": "angle_deg", "at_least": None, "at_most": 5}],
        "after": [],
        "hold_after": False,
    }
    grasp_plug = {
        "id": "take_plug",
        "primitive": "grasp",
        "object": "port",
        "handle": "plug",
        "target": None,
        "description": "Take the test connector.",
        "params": {"lift_mm": 60, "hold_s": 0.5},
        "constraints": none_cons,
        "success": [{"quantity": "lift_mm", "at_least": 40, "at_most": None}],
        "after": [],
        "hold_after": True,
    }
    angle = {"quantity": "angle_deg", "at_least": 70, "at_most": None}

    def near_bin(a: dict[str, Any]) -> None:
        tote, part = _object(a, "bin"), _object(a, "part")
        edge = tote["pose"]["x_mm"] + 0.5 * tote["params"]["length_mm"]
        part["pose"].update(
            {"x_mm": edge + 10 + 0.5 * part["params"]["size_x_mm"], "y_mm": tote["pose"]["y_mm"]}
        )

    def in_drawer_travel(a: dict[str, Any]) -> None:
        drawer, part = _object(a, "drawer"), _object(a, "part")
        front = drawer["pose"]["y_mm"] - 0.5 * drawer["params"]["depth_mm"]
        part["pose"].update(
            {"x_mm": drawer["pose"]["x_mm"], "y_mm": front - 0.5 * drawer["params"]["travel_mm"]}
        )

    def lid_over_bin(
        feature: str, over_x: float, over_y: float, turn: float = 0.0
    ) -> Callable[[dict[str, Any]], None]:
        """Put the lid on its bin with ``feature`` and an overall overhang of ``over_x`` x ``over_y``
        mm (CR-S2b-1), the bin turned ``turn`` degrees relative to the lid."""

        def f(a: dict[str, Any]) -> None:
            tote, lid = _object(a, "bin"), _object(a, "hinged_lid")
            tote["pose"].update({"x_mm": lid["pose"]["x_mm"], "y_mm": lid["pose"]["y_mm"]})
            tote["pose"]["yaw_deg"] = lid["pose"]["yaw_deg"] + turn
            tote["params"]["height_mm"] = 120
            lid["params"].update(
                {
                    "feature": feature,
                    "feature_mm": 15,
                    "width_mm": tote["params"]["length_mm"] + over_x,
                    "depth_mm": tote["params"]["width_mm"] + over_y,
                }
            )

        return f

    def shut_before_picks(a: dict[str, Any]) -> None:
        lid = _object(a, "hinged_lid")["name"]
        first_pick = next(
            i for i, x in enumerate(a["steps"]) if x["primitive"] in ("grasp", "suction", "wrap")
        )
        a["steps"].insert(
            first_pick,
            {
                "id": "shut_early",
                "primitive": "push",
                "object": lid,
                "handle": "lid",
                "target": None,
                "description": "Push the lid shut too early.",
                "params": {"speed_mm_s": 40},
                "constraints": dict(none_cons),
                "success": [{"quantity": "angle_deg", "at_least": None, "at_most": 5}],
                "after": [],
                "hold_after": False,
            },
        )

    v1 = [
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
        ("in_target_without_target", "h1_tote", sfield("grasp", "target", None)),
        ("target_not_bin", "h1_tote", sfield("grasp", "target", "tote_lid")),
        ("target_unknown", "h1_tote", sfield("grasp", "target", "nowhere")),
        ("target_and_hold", "h1_tote", sfield("grasp", "hold_after", True)),
        ("hold_on_push", "h1_tote", sfield("push", "hold_after", True)),
        ("handle_wrong", "h1_tote", sfield("push", "handle", "handle")),
        ("description_empty", "h1_tote", sfield("push", "description", "")),
        ("description_long", "h1_tote", sfield("push", "description", "x" * 201)),
        ("object_deleted", "h1_tote", drop_object("bin")),
        ("steps_reordered", "h1_tote", move(3, 0)),
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
        (
            "wrist_torque",
            "e3_isolation_valve",
            chain(setp("handwheel_valve", "torque_nm", 18), no_cons("twist")),
        ),
        ("torque_cap_low", "e3_isolation_valve", cons("twist", "max_torque_nm", 1)),
        (
            "force_cap_low",
            "h3_cabinet",
            chain(setp("port_plug", "insert_force_n", 30), cons("insert", "max_force_n", 10)),
        ),
        (
            "force_default_low",
            "h3_cabinet",
            chain(setp("port_plug", "insert_force_n", 40), cons("insert", "max_force_n", None)),
        ),
        (
            "lift_over_param",
            "h1_tote",
            crit("grasp", [{"quantity": "lift_mm", "at_least": 200, "at_most": None}]),
        ),
        ("label_long", "h1_tote", lambda a: _object(a, "bin").__setitem__("label", "L" * 61)),
        ("label_empty", "h1_tote", lambda a: _object(a, "bin").__setitem__("label", "")),
        ("title_empty", "h1_tote", lambda a: a.__setitem__("title", "")),
        ("last_step_holds", "h2_coupling", sfield("twist", "hold_after", True)),
        # the simulator's conventions (S2)
        ("twist_cw_latch", "h3_cabinet", nth_param("twist", 0, "direction", "cw")),
        ("twist_cw_valve", "h3_cabinet", nth_param("twist", 1, "direction", "cw")),
        ("twist_back_after_open", "h3_cabinet", add_step_on(3, close_valve, "handwheel_valve")),
        ("twist_ccw_thread", "h2_coupling", sparam("twist", "direction", "ccw")),
        ("grasp_on_thread", "h2_coupling", add_step_on(0, grasp_coupling, "threaded_joint")),
        ("twist_without_insert", "h2_coupling", drop_step("insert")),
        (
            "insert_too_deep",
            "h2_coupling",
            crit("insert", [{"quantity": "depth_mm", "at_least": 10, "at_most": None}]),
        ),
        ("insert_holds_thread", "h2_coupling", sfield("insert", "hold_after", True)),
        (
            "thread_torque_capped",
            "h2_coupling",
            chain(
                setp("threaded_joint", "engage_turns", 3),
                crit(
                    "twist",
                    [
                        {"quantity": "turns", "at_least": 3, "at_most": None},
                        {"quantity": "torque_nm", "at_least": None, "at_most": 6},
                    ],
                ),
                cons("twist", "max_torque_nm", 6),
            ),
        ),
        (
            "thread_torque_wrist",
            "h2_coupling",
            chain(
                setp("threaded_joint", "engage_turns", 3),
                setp("threaded_joint", "final_torque_nm", 18),
                no_cons("twist"),
                crit("twist", [{"quantity": "turns", "at_least": 3, "at_most": None}]),
            ),
        ),
        (
            "latch_torque_stop",
            "h3_cabinet",
            nth("twist", 0, "success", [angle, {"quantity": "torque_nm", "at_least": 0.5, "at_most": None}]),
        ),
        (
            "snug_goal",
            "h2_coupling",
            crit("twist", [{"quantity": "torque_nm", "at_least": 5, "at_most": None}]),
        ),
        (
            "push_drawer_default",
            "e1_drawer_kitting",
            chain(setp("drawer", "pull_force_n", 40), no_cons("push")),
        ),
        (
            "push_drawer_share",
            "e1_drawer_kitting",
            chain(setp("drawer", "pull_force_n", 20), cons("push", "max_force_n", 20)),
        ),
        ("plug_grasped_first", "h3_cabinet", add_step_on(3, grasp_plug, "port_plug")),
        (
            "lid_low_stop",
            "h1_tote",
            chain(
                setp("hinged_lid", "max_angle_deg", 80),
                setp("bin", "height_mm", 120),
                crit("hook_pull", [angle]),
            ),
        ),
        ("lid_on_a_200_mm_bin", "h1_tote", setp("bin", "height_mm", 200)),
        ("lid_and_bin_apart", "h1_tote", chain(pose("bin", "y_mm", -200), setp("bin", "height_mm", 200))),
        # a lip lid overhangs its bin by feature_mm on every side (S4d, CR-S2b-1)
        ("lid_lip_flush_with_its_bin", "h1_tote", lid_over_bin("lip", 0, 0)),
        ("lid_handle_flush_with_its_bin", "h1_tote", lid_over_bin("handle", 0, 0)),
        ("lid_lip_overhangs_its_bin", "h1_tote", lid_over_bin("lip", 30, 30)),
        ("lid_lip_short_in_depth", "h1_tote", lid_over_bin("lip", 30, 29)),
        ("lid_lip_on_a_turned_bin", "h1_tote", lid_over_bin("lip", 30, 30, 90)),
        # a pick into a lidded bin needs the lid opened past vertical first (S4e)
        (
            "lid_opened_to_30",
            "h1_tote",
            crit("hook_pull", [{"quantity": "angle_deg", "at_least": 30, "at_most": None}]),
        ),
        (
            "lid_opened_without_at_least",
            "h1_tote",
            crit("hook_pull", [{"quantity": "angle_deg", "at_least": None, "at_most": 110}]),
        ),
        (
            "lid_opened_to_95",
            "h1_tote",
            crit("hook_pull", [{"quantity": "angle_deg", "at_least": 95, "at_most": None}]),
        ),
        ("lid_never_opened", "h1_tote", drop_step("hook_pull")),
        ("lid_shut_before_the_picks", "h1_tote", shut_before_picks),
        # layout and windows (S4c)
        (
            "layout_part_in_its_bin",
            "h1_tote",
            lambda a: _object(a, "part")["pose"].update({"x_mm": 0.0, "y_mm": 0.0}),
        ),
        ("layout_part_near_a_bin", "h1_tote", near_bin),
        (
            "layout_latch_in_the_door_swing",
            "h3_cabinet",
            lambda a: _object(a, "quarter_turn_latch")["pose"].update({"x_mm": 0.0, "y_mm": -100.0}),
        ),
        ("layout_part_in_a_drawer_travel", "e1_drawer_kitting", in_drawer_travel),
        ("layout_yawed_door", "h3_cabinet", pose("hinged_lid", "yaw_deg", 180)),
        (
            "window_zero_depth",
            "h2_coupling",
            crit("insert", [{"quantity": "depth_mm", "at_least": 4, "at_most": 4}]),
        ),
        (
            "window_narrow_turns",
            "h2_coupling",
            chain(
                setp("threaded_joint", "engage_turns", 3),
                crit("twist", [{"quantity": "turns", "at_least": 2.9, "at_most": 3}]),
            ),
        ),
        (
            "window_narrow_angle",
            "h3_cabinet",
            nth("twist", 0, "success", [{"quantity": "angle_deg", "at_least": 88, "at_most": 90}]),
        ),
        (
            "window_wide_enough",
            "h3_cabinet",
            nth("twist", 0, "success", [{"quantity": "angle_deg", "at_least": 85, "at_most": 90}]),
        ),
        (
            "window_flag_exempt",
            "h1_tote",
            crit("grasp", [{"quantity": "in_target", "at_least": 1, "at_most": 1}]),
        ),
    ]

    # Studio v2's job templates, on the v2 heroes (found by kind and primitive, whatever the names)
    def q(quantity: str, lo: float | None = None, hi: float | None = None) -> dict[str, Any]:
        return {"quantity": quantity, "at_least": lo, "at_most": hi}

    def add_crate(a: dict[str, Any]) -> None:
        a["objects"].append(
            {
                "name": "crate",
                "kind": "bin",
                "label": "Crate",
                "params": {"length_mm": 100, "width_mm": 60, "height_mm": 120, "wall_mm": 4, "fixed": True},
                "pose": {"x_mm": 0.0, "y_mm": -270.0, "yaw_deg": 0.0},
                "ref": None,
            }
        )

    def place(kind: str, x: float, y: float) -> Callable[[dict[str, Any]], None]:
        return lambda a: _object(a, kind)["pose"].update({"x_mm": x, "y_mm": y})

    def pick_field(key: str, value: Any) -> Callable[[dict[str, Any]], None]:
        """``key`` of the plan's first pick, whichever pick primitive the model chose."""
        return lambda a: next(
            x for x in a["steps"] if x["primitive"] in ("grasp", "suction", "wrap")
        ).__setitem__(key, value)

    def say(line: str) -> Callable[[dict[str, Any]], None]:
        return lambda a: a["assumptions"].append(line)

    def fixture_inside(a: dict[str, Any]) -> None:
        """The fixture tray moved under the machine's opening, onto the finished part there (what stands
        inside a machine keeps its 20 mm from the rest)."""
        machine = _object(a, "sliding_door")
        load = [x for x in a["steps"] if x["primitive"] == "grasp"][-1]
        fixture = next(o for o in a["objects"] if o["name"] == load["target"])
        fixture["pose"].update({"x_mm": machine["pose"]["x_mm"] + 40.0, "y_mm": machine["pose"]["y_mm"]})

    v2 = [
        *[(f"valid_v2_{c.case_id}", c.case_id, lambda a: None) for c in HERO_CASES],
        ("push_from_the_top_is_dropped", "h1_tote", sparam("push", "approach", "top")),
        # a panel on its rails and its clamp
        ("panel_out_of_reach", "h5_solar", pose("panel_rail", "x_mm", 200)),
        (
            "panel_turned_out_of_reach",
            "h5_solar",
            chain(pose("panel_rail", "y_mm", -180), pose("panel_rail", "yaw_deg", 90)),
        ),
        ("panel_pick_on_its_clamp", "h5_solar", pick_field("handle", "member")),
        ("panel_pick_into_a_bin", "h5_solar", chain(add_crate, pick_field("target", "crate"))),
        (
            "panel_pick_without_a_lift",
            "h5_solar",
            pick_field("success", [q("gap_mm", hi=5), q("lift_mm", 0)]),
        ),
        ("panel_insert_short_of_its_thread", "h5_solar", crit("insert", [q("depth_mm", 0, 5)])),
        # a pick's lift leaves the tool's give under the part's weight (rule set v2-r2)
        (
            "panel_lift_inside_the_give",
            "h5_solar",
            chain(
                pick_field("params", {"lift_mm": 20, "hold_s": 1}),
                pick_field("success", [q("gap_mm", hi=5), q("lift_mm", 10)]),
            ),
        ),
        (
            "panel_lift_past_the_give",
            "h5_solar",
            chain(
                pick_field("params", {"lift_mm": 60, "hold_s": 1}),
                pick_field("success", [q("gap_mm", hi=5), q("lift_mm", 10)]),
            ),
        ),
        (
            "panel_hold_under_the_threshold",
            "h5_solar",
            chain(
                pick_field("params", {"lift_mm": 40, "hold_s": 2}),
                pick_field("success", [q("gap_mm", hi=5), q("hold_s", 1)]),
            ),
        ),
        (
            "tile_lift_inside_the_give",
            "h6_tile",
            pick_field("params", {"lift_mm": 10, "hold_s": 1}),
        ),
        ("panel_clamp_ccw", "h5_solar", sparam("twist", "direction", "ccw")),
        ("panel_twist_without_insert", "h5_solar", drop_step("insert")),
        ("panel_insert_holds", "h5_solar", sfield("insert", "hold_after", True)),
        ("panel_wrist_torque_goal", "h5_solar", crit("twist", [q("torque_nm", 6)])),
        ("panel_clamp_over_its_torque", "h5_solar", crit("twist", [q("clamp_torque_nm", 8)])),
        ("panel_clamp_torque_capped", "h5_solar", cons("twist", "max_torque_nm", 5)),
        ("panel_insert_too_deep", "h5_solar", crit("insert", [q("depth_mm", 6)])),
        # a tile on its bed
        (
            "tile_press_without_force",
            "h6_tile",
            crit("push", [q("height_err_mm", hi=1), q("tilt_deg", hi=0.5)]),
        ),
        ("tile_press_too_light", "h6_tile", crit("push", [q("force_n", 15), q("height_err_mm", hi=1)])),
        (
            "tile_press_too_hard",
            "h6_tile",
            chain(
                crit("push", [q("force_n", 58), q("height_err_mm", hi=1)]), cons("push", "max_force_n", 100)
            ),
        ),
        ("tile_press_capped", "h6_tile", cons("push", "max_force_n", 30)),
        # a call button on the wall
        ("button_from_the_top", "h7_hotel", sparam("push", "approach", "top")),
        ("button_judged_by_travel", "h7_hotel", crit("push", [q("travel_mm", 2)])),
        (
            "button_stiff",
            "h7_hotel",
            chain(setp("spring_button", "actuation_force_n", 25), cons("push", "max_force_n", 30)),
        ),
        ("button_out_of_reach", "h7_hotel", place("spring_button", 0.0, -250.0)),
        ("button_turned_away", "h7_hotel", pose("spring_button", "yaw_deg", 180)),
        ("button_on_a_desk_from_the_front", "h7_hotel", setp("spring_button", "mount", "desk")),
        # a machine's sliding door and what stands inside it
        ("door_slip_bound", "h4_cnc", crit("hook_pull", [q("travel_mm", 230), q("slip_mm", hi=15)])),
        ("door_opened_too_little", "h4_cnc", crit("hook_pull", [q("travel_mm", 100)])),
        ("door_never_opened", "h4_cnc", drop_step("hook_pull")),
        ("door_shut_before_the_picks", "h4_cnc", move(3, 1)),
        ("door_short_travel", "h4_cnc", setp("sliding_door", "travel_mm", 120)),
        ("door_out_of_reach", "h4_cnc", pose("sliding_door", "y_mm", -250)),
        ("door_heavy", "h4_cnc", setp("sliding_door", "pull_force_n", 30)),
        ("inside_part_pokes_out", "h4_cnc", pose("part", "x_mm", -130)),
        ("inside_contents_kept_apart", "h4_cnc", fixture_inside),
        # a hook pull's force limit over the door's pull, and the finger room inside (rule set v2-r3)
        ("door_pull_capped", "h4_cnc", cons("hook_pull", "max_force_n", 20)),
        ("door_pull_force_bound", "h4_cnc", crit("hook_pull", [q("travel_mm", 230), q("force_n", hi=20)])),
        ("picked_part_at_the_wall", "h4_cnc", pose("part", "x_mm", 100)),
        # the repairs name the field to change, and a claim binds what a clause names with the machine (v2-r4)
        ("door_shut_fast", "h4_cnc", sparam("push", "speed_mm_s", 60)),
        ("panel_start_gap_over_its_bound", "h5_solar", setp("panel_rail", "start_gap_mm", 10)),
        ("claim_inside_another_bin", "h4_cnc", say("The finished part is inside the out bin.")),
        ("claim_inside_the_machine", "h4_cnc", say("The out bin stands inside the machine.")),
    ]
    return [(name, f"v1:{base}", edit) for name, base, edit in v1] + v2


def parity_cases(recorded: Path | None = None) -> dict[str, Any]:
    """``parity.json``: each edit's arguments and the problems and warnings Python reports (the v1
    plans' edits on the v1 recording, the others on ``recorded``, by default the v2 one)."""
    examples = {e["case_id"]: e for e in recorded_plans(recorded or RECORDED_GATE)["examples"]}
    examples |= {f"v1:{e['case_id']}": e for e in recorded_plans(RECORDED_GATE_V1)["examples"]}
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
                "task_id": spec.task_id if spec is not None and status == "valid" else None,
                "fidelity": {st.id: spec.fidelity(st.id).value for st in spec.steps}
                if spec is not None
                else {},
                "caveats": {st.id: list(spec.caveats(st.id)) for st in spec.steps}
                if spec is not None
                else {},
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
        path.write_text(_dump(payloads[name], compact=name == "parity.json"), encoding="utf-8")
        paths.append(path)
    return paths
