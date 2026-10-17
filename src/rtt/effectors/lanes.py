"""Web lanes and pages of the effector tournament (product mode; ``docs/EFFECTORS.md`` section 6).

A **lane** is one recorded episode, replay version 2, drawn from the collision geoms the physics
saw (no GLB visuals for effector links this round). Every lane of a part plays the same
condition (:data:`~rtt.effectors.tournament.LANE_CONDITION`, the part 3 mm off centre, first
seed), on one clock:

* the episode is re-run with recording on and must reproduce its screen record bit for bit
  (``traj_sha``); a lane that disagrees with the score is never published;
* the recorder's version 1 header is lifted with :func:`rtt.sim.replay.upgrade_header`
  (``visuals = {}``, ``assets = {}``), ``meta.design_space = "effector-<family>"``,
  ``meta.design_id = effector_id``, ``meta.effector`` and ``provenance.product_mode = true``;
* overlays: ``offset_mm`` (from the reference), ``hold_s``, ``n_contacts`` and
  ``seat_short_mm`` (NaN for every family but the jaw);
* at most :data:`LANE_BUDGET_BYTES` as served (header, poses and overlays: there are no GLBs);
  an over-budget lane drops to 15 then 10 fps, else it is refused;
* no brand token anywhere in the header (:func:`rtt.effectors.types.find_brand_tokens`).

Lanes cover every finalist (two per group), the recommendation first, then the rigid
runner-up, then the others in ranking order. :func:`export_web` writes the lanes under
``web/public/data/replays/effectors-<part>/`` and the page data under
``web/public/data/effectors/`` (``index.json`` and ``<part>.json``), from a tournament's
output directory (:func:`rtt.effectors.tournament.write_outputs`).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from rtt.effectors.recommend import card_note
from rtt.effectors.tournament import GROUP_ORDER, LANE_CONDITION, spec_from_dict
from rtt.effectors.types import (
    CONTRACT_VERSION,
    FAMILY_FIDELITY,
    PRODUCT_MODE_LABEL,
    EffectorFamily,
    EffectorSpec,
    RegistrationReference,
    find_brand_tokens,
)

if TYPE_CHECKING:
    from rtt.parts.types import PartSpec
    from rtt.sim.replay import Replay

__all__ = [
    "FAMILY_CAVEATS",
    "FAMILY_TITLES",
    "GROUP_TITLES",
    "LANE_BUDGET_BYTES",
    "LaneError",
    "LanePlan",
    "export_web",
    "lane_payload",
    "lane_plan",
    "lift_lane",
    "record_lane",
]

#: The hero payload budget (``web/lib/assets.ts`` ``HERO_PAYLOAD_BUDGET_BYTES``).
LANE_BUDGET_BYTES: Final = 300_000
PAGE_SCHEMA: Final = "rtt-web-effector-part"
INDEX_SCHEMA: Final = "rtt-web-effectors"
WEB_VERSION: Final = 1
_STRIDES: Final = (1, 2, 3)

#: On-screen names of the families (generic: never a brand).
FAMILY_TITLES: Final[Mapping[str, str]] = {
    "parallel_jaw": "Parallel jaw",
    "catalog_hand": "Multi-finger hand",
    "catalog_gripper": "Industrial gripper",
    "soft_arm": "Soft arm",
    "suction": "Suction cup",
    "designed": "Designed morphology",
}
FAMILY_BLURBS: Final[Mapping[str, str]] = {
    "parallel_jaw": "The open-source parallel gripper of the core experiment, with its entrant fingers.",
    "catalog_hand": "Open multi-finger hand models, palm down, closed by scripted power and pinch synergies.",
    "catalog_gripper": "Open two-finger grippers and procedural two- and three-finger adaptive grippers.",
    "soft_arm": "An octopus-style tentacle: spring-jointed segments curled by a cable.",
    "suction": "A suction cup on a short bellows, its vacuum switched on only while the lip seals.",
    "designed": "Fingered morphologies proposed by Nemotron, with a random control from the same space.",
}
GROUP_TITLES: Final[Mapping[str, str]] = {
    "reference": "Reference",
    "catalog": "Catalog",
    "soft_arm": "Soft arm",
    "suction": "Suction",
    "designed": "Designed (Nemotron)",
    "random": "Random control",
}
#: The fidelity caveats of ``docs/EFFECTORS.md`` section 10, in plain words.
FAMILY_CAVEATS: Final[Mapping[str, str]] = {
    "parallel_jaw": (
        "The core experiment's own scene, checks and scorer. Registration is measured against the jaw's "
        "centre line."
    ),
    "catalog_hand": (
        "Simplified rigid models with no skin compliance, driven by scripted open-to-closed synergies: no "
        "tactile feedback, no regrasp. Actuator limits are the upstream model's. Rubber pads would grip "
        "better than the part's mu 0.3 used here, so results are conservative for padded hands."
    ),
    "catalog_gripper": (
        "Rigid models with the upstream actuator limits; the adaptive grippers are our own geometry. "
        "Scripted closing, no tactile feedback."
    ),
    "soft_arm": (
        "Approximation: a chain of rigid capsules with spring-damper joints, curled by a tension-limited "
        "cable. Not modelled: continuum mechanics, material non-linearity, suckers. Verify on hardware."
    ),
    "suction": (
        "Approximation: the seal is a geometric test gating an adhesion actuator capped at vacuum times lip "
        "area. Not modelled: air flow, leaks, porosity, vacuum build-up time, lip hysteresis. Treat as a "
        "screen, then verify on hardware."
    ),
    "designed": (
        "Rigid models generated from a recipe and driven by scripted synergies. The random control from the "
        "same space is shown beside the Nemotron designs; no superiority is claimed."
    ),
}
COMMON_CAVEAT: Final = (
    "Every family: a top-down 4-DoF wrist (no roll or pitch), open-loop timing, no perception (the placement "
    "error is unknown to the controller), soft contacts and a 1 ms timestep. The recommendation is a "
    "simulation ranking under these conditions, not a hardware guarantee. The jaw's registration reference "
    "is its centre line; every other family's is repeatability against its own nominal episode, and each "
    "family runs at its own declared actuation limits."
)
_FAILURES: Final = frozenset(
    {
        "missed_grasp",
        "slip_during_lift",
        "ejected_on_close",
        "unstable_hold",
        "finger_env_collision",
        "not_registered",
        "physics_invalid",
        "physics_unreliable",
        "invalid_build",
    }
)


class LaneError(RuntimeError):
    """A lane could not be published (disagrees with its score, too large, or names a brand)."""


@dataclass(frozen=True)
class LanePlan:
    """One lane to record: the candidate, its roles, and the episode."""

    effector_id: str
    roles: tuple[str, ...]
    condition_id: str
    seed: int

    @property
    def stem(self) -> str:
        return f"{self.roles[0].replace('_', '-')}-{self.effector_id[:12]}"


# --------------------------------------------------------------------------- recording


def _overlay_matrix(overlays: Mapping[str, np.ndarray] | None, n: int) -> tuple[np.ndarray | None, list[str]]:
    from rtt.sim.replay import OVERLAY_CHANNELS

    if not overlays:
        return None, []
    channels = [c for c in OVERLAY_CHANNELS if c in overlays]
    cols = [np.asarray(overlays[c], dtype=np.float64).reshape(-1)[:n] for c in channels]
    return np.stack(cols, axis=1).astype(np.float16), channels


def lift_lane(
    result: Any,
    part: PartSpec,
    *,
    run_id: str,
    code_version: str,
    condition_id: str,
    suite: str,
    episode: str,
    role: str,
    created_utc: str | None = None,
) -> Replay:
    """The version 2 lane of a recorded :class:`~rtt.effectors.types.EffectorEpisodeResult`."""
    from rtt.sim import replay as rp

    replay = result.replay
    if replay is None:
        raise LaneError("the episode was not recorded")
    spec: EffectorSpec = result.spec
    n = replay.n_frames
    ovl, channels = _overlay_matrix(result.overlays, n)
    v1 = {**replay.header, "n_frames": n}
    if ovl is not None:
        v1["overlay_channels"] = channels
    effector = {
        "family": spec.family.value,
        "model": spec.model,
        "display_name": spec.display_name,
        "primitive": spec.primitive,
        "origin": spec.origin,
        "fidelity": spec.fidelity.value,
        "group": spec.group.value,
        "label": PRODUCT_MODE_LABEL,
    }
    meta = {
        "design_space": f"effector-{spec.family.value}",
        "design_id": spec.effector_id,
        "geometry_id": None,
        "part_id": part.name,
        "effector": effector,
        "label": PRODUCT_MODE_LABEL,
        "role": role,
        "condition_id": condition_id,
        "reference": result.reference,
        "success": bool(result.success),
        "held": bool(result.held),
        "failure_mode": result.failure_mode,
    }
    prov = rp.provenance(
        run_id=run_id,
        code_version=code_version,
        product_mode=True,
        episode=episode,
        suite=suite,
        condition=condition_id,
        seed=int(result.seed),
        created_utc=created_utc,
    )
    header = rp.upgrade_header(v1, visuals={}, assets={}, meta=meta, provenance=prov)
    errs = rp.check_header(header, replay.frames, ovl)
    if errs:
        raise LaneError("lane header is invalid: " + "; ".join(errs))
    return rp.Replay(frames=replay.frames.copy(), header=header, overlays=ovl)


def _decimate(replay: Replay, stride: int) -> Replay:
    from rtt.sim import replay as rp

    if stride == 1:
        return replay
    header = dict(replay.header)
    fps = int(header["fps"])
    if fps % stride:
        raise LaneError(f"cannot decimate {fps} fps by {stride}")
    frames = replay.frames[::stride].copy()
    header.update(fps=fps // stride, n_frames=int(frames.shape[0]))
    header["meta"] = {**header["meta"], "decimated_from_fps": fps}
    ovl = replay.overlays[::stride].copy() if replay.overlays is not None else None
    return rp.Replay(frames=frames, header=header, overlays=ovl)


def lane_payload(replay: Replay) -> dict[str, int]:
    """Bytes a page downloads for one lane (compact header, poses, overlays; no GLBs)."""
    header = json.dumps({**replay.header, "bin": "x.bin"}, separators=(",", ":"), ensure_ascii=False)
    poses = int(replay.frames.shape[0]) * int(replay.frames.shape[1]) * 7 * 4
    ovl = int(replay.overlays.size) * 2 if replay.overlays is not None else 0
    head = len(header.encode()) + 1 + 32  # newline, and room for the real file names
    return {"header": head, "poses": poses, "overlays": ovl, "total": head + poses + ovl}


def fit_budget(replay: Replay, budget: int = LANE_BUDGET_BYTES) -> Replay:
    """``replay`` at the highest frame rate (30, 15 or 10 fps) that fits ``budget``."""
    for stride in _STRIDES:
        out = _decimate(replay, stride)
        if lane_payload(out)["total"] <= budget:
            return out
    raise LaneError(
        f"the lane is {lane_payload(_decimate(replay, _STRIDES[-1]))['total']:,} bytes at 10 fps, over {budget:,}"
    )


def record_lane(
    spec: EffectorSpec,
    part: PartSpec,
    placement: Any,
    seed: int,
    *,
    reference: RegistrationReference | None,
    condition_id: str,
    expect_traj_sha: str | None,
    task: Any = None,
) -> Any:
    """Re-run one screen episode with recording on; it must reproduce ``expect_traj_sha``."""
    from rtt.effectors.runner import run_effector_episode

    r = run_effector_episode(
        spec, part, placement, seed, task=task, record=True, reference=reference, condition_id=condition_id
    )
    if expect_traj_sha is not None and r.traj_sha != expect_traj_sha:
        raise LaneError(
            f"{spec.display_name}: the recorded episode ({condition_id}, seed {seed}) has traj_sha "
            f"{r.traj_sha[:12]}, its screen record {expect_traj_sha[:12]}; not publishing a lane that "
            "disagrees with the score"
        )
    return r


# --------------------------------------------------------------------------- plans


def lane_plan(recommendation: Mapping[str, Any], tournament: Mapping[str, Any]) -> list[LanePlan]:
    """Every finalist's lane: the recommendation first, then the rigid runner-up, then the
    other finalists in ranking order (each with every role it has)."""
    cond = next(
        (c for c in tournament["conditions"] if c["label"] == LANE_CONDITION), tournament["conditions"][0]
    )
    roles: dict[str, list[str]] = {}

    def add(eid: str | None, role: str) -> None:
        if eid:
            roles.setdefault(eid, []).append(role)

    winner = recommendation.get("winner")
    add(winner["effector"]["effector_id"] if winner else None, "recommended")
    ru = recommendation.get("runner_up_rigid")
    add(ru["effector"]["effector_id"] if ru and ru.get("effector") else None, "runner_up")
    for req, alt in (recommendation.get("alternatives") or {}).items():
        add(alt["effector_id"] if alt else None, f"recommended_{req}")
    for g in GROUP_ORDER:
        add(recommendation["by_group"].get(g.value), f"best_{g.value}")
    finalists = {e for g in tournament["groups"].values() for e in g["finalists"]}
    for row in recommendation["ranking"]:
        if row["effector_id"] in finalists:
            add(row["effector_id"], "finalist")
    return [
        LanePlan(eid, tuple(dict.fromkeys(rs)), cond["condition_id"], int(cond["seeds"][0]))
        for eid, rs in roles.items()
    ]


# --------------------------------------------------------------------------- page data


def _r(v: Any, digits: int = 4) -> Any:
    if v is None:
        return None
    f = float(v)
    return round(f, digits) if math.isfinite(f) else None


def _cert_summary(cert: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not cert:
        return None
    return {
        "reference": cert["reference"],
        "held_score": cert["held_score"],
        "registered_score": cert["registered_score"],
        "held_scores": cert["held_scores"],
        "registered_scores": cert["registered_scores"],
        "held_tolerance_mm": cert["held_tolerance_mm"],
        "registered_tolerance_mm": cert["registered_tolerance_mm"],
        "held": cert["held"],
        "registered": cert["registered"],
        "effector_mass_kg": cert.get("effector_mass_kg"),
        "tests": [
            {
                "name": t["name"],
                "category": t["category"],
                "held": t["held"],
                "registered": t["registered"],
                "failure_mode": t["failure_mode"],
                "registration_flags": t["registration_flags"],
                "offset_mm": t["offset_mm"],
            }
            for t in cert["tests"]
        ],
    }


def _effector_brief(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "effector_id": spec["effector_id"],
        "display_name": spec["display_name"],
        "family": spec["family"],
        "model": spec["model"],
        "primitive": spec["primitive"],
        "origin": spec["origin"],
        "group": spec["group"],
        "fidelity": spec["fidelity"],
        "hypothesis": (spec.get("recipe") or {}).get("hypothesis") or None,
        "note": card_note(spec["family"], spec["model"]),
    }


def _card(card: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not card or not card.get("effector"):
        return None if not card else {"effector": None, "basis": card.get("basis")}
    out = {
        "effector": _effector_brief(card["effector"]),
        "certificate": _cert_summary(card.get("certificate")),
        "screen": card["screen"],
        "why": card.get("why"),
        "score": card.get("score"),
        "tolerance_mm": card.get("tolerance_mm"),
        "n_actuators": card.get("n_actuators"),
        "mass_kg": card.get("mass_kg"),
        "actuation": card.get("actuation") or {},
        "caveat": card.get("caveat"),
    }
    if "basis" in card:
        out["basis"] = card["basis"]
    return out


def _ranking(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keep = (
        "effector_id",
        "display_name",
        "family",
        "group",
        "fidelity",
        "status",
        "certified",
        "eligible",
        "reason",
        "score",
        "tolerance_mm",
        "screen_rate",
        "n_actuators",
        "mass_kg",
        "actuation",
    )
    return [{**{k: row.get(k) for k in keep}, "note": _note_of(row)} for row in rows]


def _note_of(row: Mapping[str, Any]) -> str | None:
    """A row's card note: recomputed from its family and model, so older runs get it too."""
    model = row.get("model")
    return card_note(row["family"], model) if model else row.get("note")


def _requirement_block(block: Mapping[str, Any]) -> dict[str, Any]:
    nw = block.get("no_winner")
    return {
        "winner": _card(block.get("winner")),
        "runner_up_rigid": _card(block.get("runner_up_rigid")),
        "no_winner": None
        if not nw
        else {
            "reason": nw["reason"],
            "closest": [
                {
                    **{
                        k: c.get(k)
                        for k in (
                            "effector_id",
                            "display_name",
                            "family",
                            "fidelity",
                            "score",
                            "screen_rate",
                            "reason",
                        )
                    },
                    "failure_modes": c.get("failure_modes", {}),
                }
                for c in nw["closest"]
            ],
        },
        "ranking": _ranking(block["ranking"]),
    }


def _families(tournament: Mapping[str, Any], recommendation: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for c in tournament["candidates"]:
        by_family.setdefault(c["family"], []).append(c)
    order = {row["effector_id"]: i for i, row in enumerate(recommendation["ranking"])}
    out = []
    for fam in EffectorFamily:
        status = tournament["families"].get(fam.value, "not available")
        cands = by_family.get(fam.value, [])
        screened = [c for c in cands if c["status"] == "screened"]
        best = min(screened, key=lambda c: order.get(c["effector_id"], 10**6)) if screened else None
        out.append(
            {
                "family": fam.value,
                "title": FAMILY_TITLES[fam.value],
                "blurb": FAMILY_BLURBS[fam.value],
                "fidelity": FAMILY_FIDELITY[fam].value,
                "available": status == "available",
                "status": "available"
                if status == "available"
                else "not built yet"
                if "not available" in status
                else status,
                "candidates": len(cands),
                "screened": len(screened),
                "invalid": sum(c["status"] == "invalid_build" for c in cands),
                "best": None
                if best is None
                else {
                    "effector_id": best["effector_id"],
                    "display_name": best["display_name"],
                    "screen": best.get("screen"),
                    "certified": best["certified"],
                    "note": card_note(best["family"], best["model"]),
                },
                "caveat": FAMILY_CAVEATS[fam.value],
                "label": PRODUCT_MODE_LABEL,
            }
        )
    return out


def _candidate_rows(tournament: Mapping[str, Any]) -> list[dict[str, Any]]:
    keep = (
        "effector_id",
        "display_name",
        "family",
        "model",
        "primitive",
        "origin",
        "group",
        "fidelity",
        "source",
        "status",
        "reason",
        "n_actuators",
        "mass_kg",
        "actuation",
        "reach_mm",
        "max_opening_mm",
        "reference",
        "finalist",
        "certified",
        "screen",
    )
    return [
        {**{k: c.get(k) for k in keep}, "note": card_note(c["family"], c["model"])}
        for c in tournament["candidates"]
    ]


def _task_doc(task: Any) -> dict[str, Any]:
    from rtt.pipeline.export_web import task_summary

    t = task_summary(task)
    return {
        k: t[k] for k in ("name", "phases", "success_lift_m", "success_hold_s", "record_tail_s", "max_time_s")
    }


def _lane_doc(
    plan: LanePlan, r: Any, rel: str, replay: Replay, payload: Mapping[str, int], cond: Mapping[str, Any]
) -> dict[str, Any]:
    spec: EffectorSpec = r.spec
    mode = (
        r.failure_mode
        if r.failure_mode in _FAILURES
        else (None if r.failure_mode is None else "physics_invalid")
    )
    return {
        "id": plan.stem,
        "replay": rel,
        "roles": list(plan.roles),
        "effector": _effector_brief(spec.to_dict()),
        "condition_id": plan.condition_id,
        "condition": _describe_condition(cond),
        "seed": str(plan.seed),  # 63-bit seeds: text, so a browser shows them exactly
        "fps": int(replay.header["fps"]),
        "n_frames": int(replay.n_frames),
        "duration_s": round((replay.n_frames - 1) / float(replay.header["fps"]), 4),
        "bytes": int(payload["total"]),
        "reference": r.reference,
        "outcome": {
            "registered": bool(r.success),
            "held": bool(r.held),
            "engaged": bool(r.engaged),
            "failure_mode": mode,
            "registration_flags": list(r.registration_flags),
            "max_lift_mm": round(1000.0 * float(r.max_lift), 2),
            "hold_s": round(float(r.hold_time), 3),
            "penetration_mm": _r(r.penetration_mm, 3),
            "sim_steps": int(r.n_steps),
            "traj_sha": r.traj_sha,
        },
    }


def _describe_condition(cond: Mapping[str, Any]) -> str:
    p = cond["placement"]
    bits = []
    if abs(p["err_x_mm"]) > 1e-9:
        bits.append(f"{abs(p['err_x_mm']):g} mm off centre across")
    if abs(p["err_y_mm"]) > 1e-9:
        bits.append(f"{abs(p['err_y_mm']):g} mm off centre along the closing axis")
    if abs(p["err_yaw_deg"]) > 1e-9:
        bits.append(f"turned {abs(p['err_yaw_deg']):g} deg")
    return " and ".join(bits) if bits else "centred"


def _check_brands(doc: Any, where: str) -> None:
    text = json.dumps(doc, ensure_ascii=False)
    hits = find_brand_tokens(text)
    if hits:
        raise LaneError(
            f"{where} names a brand ({', '.join(sorted(set(hits)))}); generic names only on screen"
        )


def _part_from(tournament: Mapping[str, Any], parts: Mapping[str, PartSpec] | None) -> PartSpec:
    from rtt.parts.library import get_part

    pid = tournament["part"]["part_id"]
    part = parts[pid] if parts and pid in parts else get_part(pid)
    if part.geometry_sha256 != tournament["part"]["geometry_sha256"]:
        raise LaneError(f"part {pid} differs from the one the tournament ran on")
    return part


def export_part(
    run_dir: str | Path,
    web_data: str | Path,
    *,
    parts: Mapping[str, PartSpec] | None = None,
    run_id: str | None = None,
    budget: int = LANE_BUDGET_BYTES,
) -> dict[str, Any]:
    """Record the lanes of one tournament (``runs/effectors/<part>/``) and write its page data.

    Returns the part's entry for ``effectors/index.json``.
    """
    from rtt.sim import objects as objs
    from rtt.sim import replay as rp
    from rtt.sim.task import default_task_v2, task_digest
    from rtt.store.manifest import write_json_atomic
    from rtt.store.runs import read_table

    run_dir, web_data = Path(run_dir), Path(web_data)
    tour = json.loads((run_dir / "tournament.json").read_text(encoding="utf-8"))
    rec = json.loads((run_dir / "recommendation.json").read_text(encoding="utf-8"))
    records = read_table(run_dir / "records.parquet")
    task = default_task_v2()
    if task_digest(task) != tour["provenance"]["task_digest"]:
        raise LaneError("the tournament ran on another task; re-run it")
    part = _part_from(tour, parts)
    pid = part.name
    cands = {c["effector_id"]: c for c in tour["candidates"]}
    conds = {c["condition_id"]: c for c in tour["conditions"]}
    run_id = run_id or f"effectors-{pid}".replace("_", "-")
    code = tour["provenance"]["code_version"]
    if not re.fullmatch(r"[0-9a-f]{7,40}(\+dirty)?", code):  # a run outside git: lanes need a commit
        raise LaneError(f"the tournament's code version {code!r} is not a git commit")
    lane_dir_rel = f"replays/effectors-{pid}"
    lane_dir = web_data / lane_dir_rel
    if lane_dir.exists():
        for old in lane_dir.iterdir():
            if old.is_file():
                old.unlink()
    lanes = []
    for plan in lane_plan(rec, tour):
        c = cands[plan.effector_id]
        spec = spec_from_dict(c["spec"])
        pose = c.get("reference_pose")
        ref = (
            RegistrationReference(tuple(pose["pos"]), tuple(pose["quat_wxyz"]), "nominal_episode")
            if pose
            else None
        )
        cond = conds[plan.condition_id]
        row = next(
            (
                r
                for r in records
                if r["stage"] == "screen"
                and r["effector_id"] == plan.effector_id
                and r["condition_id"] == plan.condition_id
                and int(r["seed"]) == plan.seed
            ),
            None,
        )
        if row is None:
            raise LaneError(f"no screen record for {spec.display_name} at {plan.condition_id}")
        p = cond["placement"]
        placement = objs.Placement(
            x=p["x"],
            y=p["y"],
            yaw=math.radians(p["yaw_deg"]),
            err_x=p["err_x_mm"] / 1000.0,
            err_y=p["err_y_mm"] / 1000.0,
            err_yaw=math.radians(p["err_yaw_deg"]),
        )
        r = record_lane(
            spec,
            part,
            placement,
            plan.seed,
            reference=ref,
            condition_id=plan.condition_id,
            expect_traj_sha=row["traj_sha"],
            task=task,
        )
        replay = lift_lane(
            r,
            part,
            run_id=run_id,
            code_version=code,
            condition_id=plan.condition_id,
            suite=tour["suite"],
            episode=plan.stem,
            role=plan.roles[0],
            created_utc=tour["provenance"]["created_utc"],
        )
        replay = fit_budget(replay, budget)
        _check_brands(replay.header, f"lane {plan.stem}")
        rp.write_replay(replay, lane_dir / plan.stem)
        payload = lane_payload(replay)
        served = sum(f.stat().st_size for f in lane_dir.glob(f"{plan.stem}.*"))
        if served > budget:
            raise LaneError(f"lane {plan.stem} is {served:,} bytes as served, over {budget:,}")
        lanes.append(
            _lane_doc(plan, r, f"{lane_dir_rel}/{plan.stem}", replay, {**payload, "total": served}, cond)
        )

    head = rec["requirement"]
    page = {
        "schema": PAGE_SCHEMA,
        "version": WEB_VERSION,
        "contract": CONTRACT_VERSION,
        "label": PRODUCT_MODE_LABEL,
        "part": {**tour["part"], "title": pid.replace("_", " ").capitalize()},
        "requirement": head,
        "recommendation": _requirement_block(rec["by_requirement"][head]),
        "by_requirement": {k: _requirement_block(v) for k, v in rec["by_requirement"].items()},
        "alternatives": rec["alternatives"],
        "families": _families(tour, rec),
        "groups": [
            {
                "group": g,
                "title": GROUP_TITLES[g],
                "status": st["status"],
                "reason": st["reason"],
                "note": st.get("note"),
                "finalists": st["finalists"],
                "budget": st["budget"],
                "sampled_from": st["sampled_from"],
            }
            for g, st in tour["groups"].items()
        ],
        "candidates": _candidate_rows(tour),
        "certificates": {
            c["effector_id"]: _cert_summary(_load_cert(run_dir, c["effector_id"]))
            for c in tour["candidates"]
            if c["certified"]
        },
        "lanes": lanes,
        "task": _task_doc(task),
        "conditions": [{**c, "seeds": [str(x) for x in c["seeds"]]} for c in tour["conditions"]],
        "fairness": {
            **{k: tour["fairness"][k] for k in ("identical", "problems", "signature_sha256", "conditions")},
            "seeds": {k: [str(x) for x in v] for k, v in tour["fairness"]["seeds"].items()},
        },
        "physics": tour["physics"],
        "design_rounds": [
            {k: r.get(k) for k in ("round", "proposed", "screened", "history_shown", "calls", "cost_usd")}
            for r in tour["design_rounds"]
        ],
        "caveats": {
            "common": COMMON_CAVEAT,
            **{f: FAMILY_CAVEATS[f] for f in sorted({c["family"] for c in tour["candidates"]})},
        },
        "rules": rec["rules"],
        "provenance": {**rec["provenance"], "run_id": run_id, "digest": tour.get("digest")},
    }
    _check_brands(page, f"effectors/{pid}.json")
    write_json_atomic(web_data / "effectors" / f"{pid}.json", page)

    def short(block: Mapping[str, Any]) -> dict[str, Any] | None:
        w = block.get("winner")
        if not w:
            return None
        return {
            "effector_id": w["effector"]["effector_id"],
            "display_name": w["effector"]["display_name"],
            "family": w["effector"]["family"],
            "fidelity": w["fidelity"],
            "score": w["score"],
            "tolerance_mm": w["tolerance_mm"],
            "caveat": w.get("caveat"),
        }

    return {
        "part_id": pid,
        "title": page["part"]["title"],
        "family": tour["part"]["family"],
        "size_mm": tour["part"]["size_mm"],
        "page": f"effectors/{pid}.json",
        "requirement": head,
        "winner": short(rec["by_requirement"][head]),
        "by_requirement": {k: short(v) for k, v in rec["by_requirement"].items()},
        "groups_run": [g for g, st in tour["groups"].items() if st["status"] == "run"],
        "lanes": len(lanes),
        "candidates": len(tour["candidates"]),
        "episodes": tour["physics"]["episodes"] + tour["physics"]["certificate_episodes"],
        "code_version": code,
    }


def _load_cert(run_dir: Path, effector_id: str) -> dict[str, Any] | None:
    path = run_dir / "certificates" / f"{effector_id[:12]}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def export_web(
    runs_root: str | Path,
    web_data: str | Path,
    *,
    parts: Sequence[str] | None = None,
    budget: int = LANE_BUDGET_BYTES,
) -> dict[str, Any]:
    """Lanes and page data for every tournament under ``runs_root`` (``runs/effectors/``), plus
    ``effectors/index.json``. Returns the index document."""
    from rtt.effectors.registry import available_families
    from rtt.store.manifest import write_json_atomic

    root = Path(runs_root)
    names = (
        list(parts) if parts else sorted(p.name for p in root.iterdir() if (p / "tournament.json").is_file())
    )
    if not names:
        raise LaneError(f"no tournaments under {root}")
    entries = [export_part(root / n, web_data, budget=budget) for n in names]
    avail = {f.value for f in available_families()}
    index = {
        "schema": INDEX_SCHEMA,
        "version": WEB_VERSION,
        "contract": CONTRACT_VERSION,
        "label": PRODUCT_MODE_LABEL,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "families": [
            {
                "family": f.value,
                "title": FAMILY_TITLES[f.value],
                "fidelity": FAMILY_FIDELITY[f].value,
                "available": f.value in avail,
            }
            for f in EffectorFamily
        ],
        "parts": entries,
    }
    _check_brands(index, "effectors/index.json")
    write_json_atomic(Path(web_data) / "effectors" / "index.json", index)
    return index
