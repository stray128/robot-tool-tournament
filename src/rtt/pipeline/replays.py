"""Record web replays of scored episodes.

Scoring runs with recording off (it is the hot path). To show an episode, it is run
again with the same finger, condition, controller setting and jitter seed and
recording on; the simulator is deterministic on one machine, so the recorded episode
is the scored one. :func:`record_replays` checks that: a recorded outcome that differs
from the expected one (when given) raises instead of publishing a misleading replay.

Each replay is written as ``<stem>.bin`` + ``<stem>.json``
(:func:`rtt.sim.replay.write_replay`) with extra ``meta``: the design id and
parameters, the condition id and description, and whatever the caller adds.

:func:`lift_to_v2` turns a recorded version 1 replay into a version 2 replay drawn from GLB
assets: the open-source gripper's frame and carriages, each finger's own asset (its v1 boxes
fused into one solid, or the stock blade's CAD) and the object. The v1 geoms stay as the
collision overlay. This is how v1 bouts (the F1 fallback) get the v2 look.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rtt.design.generate import generate_finger
from rtt.design.schema import FingerDesign
from rtt.sim.conditions import Condition
from rtt.sim.episode import EpisodeResult
from rtt.sim.exec_cpu import CpuExecutor, EpisodeJob
from rtt.sim.replay import Replay, write_replay
from rtt.sim.task import TaskConfig, default_task

if TYPE_CHECKING:
    from rtt.render.glb import AssetStore
    from rtt.render.materials import Look

__all__ = ["RecordedReplay", "ReplayMismatchError", "ReplayRequest", "lift_to_v2", "record_replays"]


class ReplayMismatchError(RuntimeError):
    """A recorded episode did not reproduce the outcome it was recorded to show."""


@dataclass(frozen=True)
class ReplayRequest:
    """One episode to record.

    Args:
        stem: File stem (no extension) relative to the output directory.
        design: The finger design.
        condition: Object, placement and jitter seeds.
        controller_setting: Index into the task's grasp heights.
        seed: Jitter seed (default: the condition's first seed, as scored).
        expect_success: The scored outcome; a recorded episode that differs is an error.
        expect_failure_mode: The scored failure mode (``None`` for a success).
        meta: Extra replay-header metadata.
    """

    stem: str
    design: FingerDesign
    condition: Condition
    controller_setting: int
    seed: int | None = None
    expect_success: bool | None = None
    expect_failure_mode: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def episode_seed(self) -> int:
        return int(self.seed if self.seed is not None else self.condition.seeds[0])


@dataclass(frozen=True)
class RecordedReplay:
    """A written replay and the outcome of the recorded episode."""

    request: ReplayRequest
    bin_path: Path
    json_path: Path
    success: bool
    failure_mode: str | None
    max_lift_m: float
    hold_time_s: float
    flags: tuple[str, ...]
    n_frames: int
    n_steps: int
    wall_time_s: float


def _meta(req: ReplayRequest, task: TaskConfig) -> dict[str, Any]:
    fracs = task.controller.grasp_height_fracs
    return {
        "design_id": req.design.design_id(),
        "design": req.design.model_dump(mode="json", exclude={"rationale"}),
        "condition_id": req.condition.condition_id,
        "condition": req.condition.describe(),
        "family": req.condition.family,
        "grasp_height_frac": fracs[req.controller_setting],
        **dict(req.meta),
    }


def _check(req: ReplayRequest, res: EpisodeResult) -> None:
    if req.expect_success is None:
        return
    if res.success != req.expect_success or (not res.success and res.failure_mode != req.expect_failure_mode):
        got = "success" if res.success else res.failure_mode
        want = "success" if req.expect_success else req.expect_failure_mode
        raise ReplayMismatchError(
            f"{req.stem}: recorded episode gave {got}, the scored episode gave {want} "
            f"(design {req.design.design_id()}, {req.condition.condition_id}, k={req.controller_setting}, "
            f"seed {req.episode_seed}); not publishing a replay that disagrees with the score"
        )


def record_replays(
    requests: Sequence[ReplayRequest],
    out_dir: str | Path,
    *,
    task: TaskConfig | None = None,
    executor: CpuExecutor | None = None,
    workers: int | None = None,
) -> list[RecordedReplay]:
    """Run every request with recording on and write its replay under ``out_dir``.

    Episodes run on ``executor`` (or a temporary pool of ``workers``); replays are
    written only after every episode matched its expected outcome.
    """
    task = task or default_task()
    out = Path(out_dir)
    stems = [r.stem for r in requests]
    if len(set(stems)) != len(stems):
        raise ValueError("replay stems must be unique")
    jobs = [
        EpisodeJob(
            generate_finger(r.design),
            r.condition.obj,
            r.condition.placement,
            r.controller_setting,
            r.episode_seed,
            record=True,
            task=task,
        )
        for r in requests
    ]
    own = executor is None
    ex = executor or CpuExecutor(workers)
    try:
        results = ex.run(jobs).results if jobs else []
    finally:
        if own:
            ex.close()
    for req, res in zip(requests, results, strict=True):
        _check(req, res)
    recorded = []
    for req, res in zip(requests, results, strict=True):
        if res.replay is None:
            raise RuntimeError(f"{req.stem}: the episode was not recorded")
        res.replay.header["meta"].update(_meta(req, task))
        bin_path, json_path = write_replay(res.replay, out / req.stem)
        recorded.append(
            RecordedReplay(
                request=req,
                bin_path=bin_path,
                json_path=json_path,
                success=res.success,
                failure_mode=res.failure_mode,
                max_lift_m=float(res.max_lift),
                hold_time_s=float(res.hold_time),
                flags=tuple(f.split("@", 1)[0] for f in res.flags),
                n_frames=res.replay.n_frames,
                n_steps=res.n_steps,
                wall_time_s=float(res.wall_time_s),
            )
        )
    return recorded


# --------------------------------------------------------------------------- v1 -> v2 look

_SLUG_BAD = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(text: str) -> str:
    out = _SLUG_BAD.sub("-", text).strip("-.") or "part"
    return out if out[0].isalnum() else f"p{out}"


def _part_id(meta: dict[str, Any]) -> str:
    obj = meta.get("object") or {}
    if obj.get("name"):
        return _slug(str(obj["name"]))
    size = "x".join(f"{1000.0 * float(v):g}" for v in obj.get("size", []))
    return _slug(f"{obj.get('kind', 'object')}-{size}" if size else str(obj.get("kind", "object")))


def lift_to_v2(
    replay: Replay,
    store: AssetStore,
    *,
    run_id: str,
    code_version: str,
    product_mode: bool = False,
    design_space: str = "v1",
    design_id: str | None = None,
    geometry_id: str | None = None,
    part_id: str | None = None,
    stock_blade: bool = False,
    finger_look: Look | None = None,
    object_look: Look | None = None,
    keep_geoms: bool = True,
    **provenance_extra: Any,
) -> Replay:
    """A version 2 copy of a version 1 replay, drawn from assets written into ``store``.

    Poses are copied unchanged. ``stock_blade=True`` draws the stock blade's CAD on both
    jaws (for the box-approximated v1 stock entrant; the boxes remain the collision overlay).
    ``design_id`` defaults to the replay's ``meta.design_id``; ``part_id`` to the object's
    name, or its kind and size in mm.
    """
    from rtt.render import glb
    from rtt.render import materials as mat
    from rtt.render.gripper import add_gripper_assets
    from rtt.sim import replay as rp

    header = replay.header
    if replay.version != rp.VERSION_V1:
        raise ValueError("lift_to_v2 takes a version 1 replay")
    names = [b["name"] for b in header["bodies"]]
    if names != ["palm", "finger_left", "finger_right", "object"]:
        raise ValueError(f"unexpected replay bodies {names}")
    geoms = {b["name"]: b.get("geoms", []) for b in header["bodies"]}
    grip = add_gripper_assets(store)
    meta = dict(header.get("meta") or {})

    def own(body: str, role: str, look: Look, label: str) -> str:
        mesh = glb.mesh_from_geoms(geoms[body])
        data = glb.glb_bytes(
            [glb.mesh_from_trimesh(mesh, look, name=body)],
            copyright=glb.OURS_ATTRIBUTION,
            extras={"license": glb.OURS_LICENSE, "role": role},
        )
        return store.add(data, role=role, label=label)

    visuals: dict[str, list[dict[str, Any]]] = {"palm": [rp.visual(grip.frame)]}
    for side in ("left", "right"):
        body = f"finger_{side}"
        finger = (
            grip.blade(side)
            if stock_blade
            else own(body, body, finger_look or mat.FINGER_DESIGNED, f"v1 finger, {side} jaw")
        )
        visuals[body] = [rp.visual(grip.carriage(side)), rp.visual(finger)]
    visuals["object"] = [
        rp.visual(own("object", "object", object_look or mat.PART_LOOKS["aluminium_anodised"], "v1 object"))
    ]
    used = {v["asset"] for vs in visuals.values() for v in vs}
    prov = rp.provenance(
        run_id=run_id,
        code_version=code_version,
        product_mode=product_mode,
        seed=meta.get("seed"),
        condition=meta.get("condition_id"),
        **provenance_extra,
    )
    v2_meta = {
        "design_space": design_space,
        "design_id": str(design_id or meta.get("design_id") or "unknown"),
        "geometry_id": geometry_id,
        "part_id": part_id or _part_id(meta),
    }
    if stock_blade:
        v2_meta["finger_visual"] = "stock blade CAD over the v1 box approximation"
    lifted = rp.upgrade_header(
        {**header, "n_frames": replay.n_frames},
        visuals=visuals,
        assets=store.subset(used),
        meta=v2_meta,
        provenance=prov,
        keep_geoms=keep_geoms,
    )
    errs = rp.check_header(lifted, replay.frames, replay.overlays)
    if errs:
        raise ValueError("lifted header is invalid: " + "; ".join(errs))
    return Replay(frames=replay.frames.copy(), header=lifted, overlays=replay.overlays)
