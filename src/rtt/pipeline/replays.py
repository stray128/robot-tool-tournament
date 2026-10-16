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

:func:`lift_freeform_to_v2` does the same for a freeform episode of the v2 scene: each jaw of
the simulated :class:`~rtt.design.freeform.FingerPair` gets its own GLB (its print mesh, or
the union of its convex pieces), the part its own (:func:`part_asset`, or a scan visual from
:func:`scan_asset`), and the episode's per-frame readouts go into the overlay sidecar.

:func:`record_bout_v2` runs a few ``train_v2`` episodes end to end and writes them as a v2 bout
directory (replays, GLB store and ``bout.json`` with each lane's scored outcome), which the web
export publishes as a bout of v2 lanes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

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

__all__ = [
    "BOUT_FILE",
    "ENTRANT_LABELS",
    "RecordedReplay",
    "ReplayMismatchError",
    "ReplayRequest",
    "describe_placement",
    "lift_freeform_to_v2",
    "lift_to_v2",
    "part_asset",
    "record_bout_v2",
    "record_replays",
    "scan_asset",
]


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


def _own_asset(store: AssetStore, mesh: Any, role: str, look: Look, label: str, **entry: Any) -> str:
    """Write one mesh as a GLB asset (our own Apache-2.0 geometry unless ``entry`` says otherwise)."""
    from rtt.render import glb

    license = entry.pop("license", glb.OURS_LICENSE)
    attribution = entry.pop("attribution", glb.OURS_ATTRIBUTION)
    extras = {"license": license, "role": role}
    if entry.get("changes"):
        extras["changes"] = list(entry["changes"])
    data = glb.glb_bytes([glb.mesh_from_trimesh(mesh, look, name=role)], copyright=attribution, extras=extras)
    return store.add(data, role=role, license=license, attribution=attribution, label=label, **entry)


def _finish_v2(
    replay: Replay,
    store: AssetStore,
    visuals: Mapping[str, Sequence[dict[str, Any]]],
    v2_meta: dict[str, Any],
    *,
    run_id: str,
    code_version: str,
    product_mode: bool,
    keep_geoms: bool,
    provenance_extra: Mapping[str, Any],
    overlays: Any = None,
    channels: Sequence[str] = (),
) -> Replay:
    from rtt.sim import replay as rp

    header = replay.header
    meta = dict(header.get("meta") or {})
    used = {v["asset"] for vs in visuals.values() for v in vs}
    extra = {"seed": meta.get("seed"), "condition": meta.get("condition_id"), **provenance_extra}
    prov = rp.provenance(run_id=run_id, code_version=code_version, product_mode=product_mode, **extra)
    v1 = {**header, "n_frames": replay.n_frames}
    if overlays is not None:
        v1["overlay_channels"] = list(channels)
    lifted = rp.upgrade_header(
        v1,
        visuals=visuals,
        assets=store.subset(used),
        meta=v2_meta,
        provenance=prov,
        keep_geoms=keep_geoms,
    )
    errs = rp.check_header(lifted, replay.frames, overlays)
    if errs:
        raise ValueError("lifted header is invalid: " + "; ".join(errs))
    return Replay(frames=replay.frames.copy(), header=lifted, overlays=overlays)


def _check_v1_bodies(replay: Replay) -> dict[str, list[dict[str, Any]]]:
    from rtt.sim import replay as rp

    if replay.version != rp.VERSION_V1:
        raise ValueError("a v2 lift takes a version 1 replay (the recorder's output)")
    names = [b["name"] for b in replay.header["bodies"]]
    if names != ["palm", "finger_left", "finger_right", "object"]:
        raise ValueError(f"unexpected replay bodies {names}")
    return {b["name"]: b.get("geoms", []) for b in replay.header["bodies"]}


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

    geoms = _check_v1_bodies(replay)
    grip = add_gripper_assets(store)
    meta = dict(replay.header.get("meta") or {})

    def own(body: str, role: str, look: Look, label: str) -> str:
        return _own_asset(store, glb.mesh_from_geoms(geoms[body]), role, look, label)

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
    v2_meta = {
        "design_space": design_space,
        "design_id": str(design_id or meta.get("design_id") or "unknown"),
        "geometry_id": geometry_id,
        "part_id": _slug(part_id) if part_id else _part_id(meta),
    }
    if stock_blade:
        v2_meta["finger_visual"] = "stock blade CAD over the v1 box approximation"
    return _finish_v2(
        replay,
        store,
        visuals,
        v2_meta,
        run_id=run_id,
        code_version=code_version,
        product_mode=product_mode,
        keep_geoms=keep_geoms,
        provenance_extra=provenance_extra,
        overlays=replay.overlays,
        channels=replay.header.get("overlay_channels", ()),
    )


# --------------------------------------------------------------------------- freeform (v2) episodes

MAX_JAW_EXTENT_M: Final = 0.25  # a jaw is at most 110 mm long and 40 mm wide; mm meshes fail this

GSO_PIECES_CHANGES: Final = (
    "convex decomposition for collision (CoACD 0.02, seed 0), drawn untextured",
    "re-posed into the object frame",
    "re-encoded as GLB",
)


def _overlay_array(overlays: Any, n_frames: int) -> tuple[Any, tuple[str, ...]]:
    """S's per-channel readouts (``{channel: (n,) array}``) as a float16 ``(n, c)`` array in
    :data:`rtt.sim.replay.OVERLAY_CHANNELS` order, plus the channel names."""
    import numpy as np

    from rtt.sim.replay import OVERLAY_CHANNELS

    if overlays is None:
        return None, ()
    unknown = set(overlays) - set(OVERLAY_CHANNELS)
    if unknown:
        raise ValueError(f"unknown overlay channels {sorted(unknown)}; known: {OVERLAY_CHANNELS}")
    channels = tuple(c for c in OVERLAY_CHANNELS if c in overlays)
    cols = []
    for c in channels:
        col = np.asarray(overlays[c], dtype=np.float64).reshape(-1)
        if len(col) != n_frames:
            raise ValueError(f"overlay channel {c} has {len(col)} values for {n_frames} replay frames")
        cols.append(col)
    return np.stack(cols, axis=1).astype(np.float16), channels


def part_asset(store: AssetStore, part: Any, *, look: Look | None = None, label: str | None = None) -> str:
    """The GLB asset of a part (:class:`rtt.parts.types.PartSpec`): the union of its collision
    pieces in its material's look. A scan (family ``gso``) drawn this way is clay and carries
    the Google Scanned Objects credit; procedural parts are our own (Apache-2.0)."""
    from rtt.render import glb
    from rtt.render.materials import look_from_material

    mesh = glb.mesh_from_pieces(part.pieces)
    if part.family == "gso":
        from rtt.render.materials import PART_LOOKS

        return _own_asset(
            store,
            mesh,
            "object",
            look or PART_LOOKS["clay"],
            label or f"{part.name} (scan, clay)",
            license="CC-BY-4.0",
            attribution=glb.GSO_ATTRIBUTION,
            changes=list(GSO_PIECES_CHANGES),
            brand_visible=False,
        )
    return _own_asset(store, mesh, "object", look or look_from_material(part.material), label or part.name)


def scan_asset(
    store: AssetStore,
    visual: Any,
    fields: Mapping[str, Any],
    *,
    textured: bool,
    look: Look | None = None,
    label: str | None = None,
) -> str:
    """The GLB asset of a scan's visual mesh (``rtt.sim.gso_visual.ScanVisual``, in the object
    frame of its collision pieces). ``fields`` are the licence-side entry fields from
    ``rtt.sim.gso_visual.scan_asset_fields`` (``license``, ``changes``, ``brand_visible``,
    ``source``); the credit line is the short Google Scanned Objects credit."""
    from rtt.render import glb
    from rtt.render.materials import PART_LOOKS

    extras = {"license": fields["license"], "role": "object", "changes": list(fields["changes"])}
    mesh = glb.scan_mesh(visual, look or PART_LOOKS["clay"], textured=textured, name="scan")
    data = glb.glb_bytes([mesh], copyright=glb.GSO_ATTRIBUTION, extras=extras)
    return store.add(
        data,
        role="object",
        license=fields["license"],
        attribution=glb.GSO_ATTRIBUTION,
        changes=list(fields["changes"]),
        label=label or f"{visual.name} (scan{'' if textured else ', clay'})",
        source=fields.get("source"),
        brand_visible=bool(fields.get("brand_visible", textured)),
    )


def lift_freeform_to_v2(
    replay: Replay,
    store: AssetStore,
    pair: Any,
    *,
    run_id: str,
    code_version: str,
    part: Any = None,
    overlays: Any = None,
    jaw_meshes: tuple[Any, Any] | None = None,
    object_asset: str | None = None,
    product_mode: bool = False,
    geometry_id: str | None = None,
    finger_look: Look | None = None,
    keep_geoms: bool = True,
    **provenance_extra: Any,
) -> Replay:
    """A replay v2 of a freeform episode (the v2 scene's recording), each jaw its own asset.

    * ``replay``: the recorder's version 1 replay (bodies palm, both jaws, object);
      ``pair``: the :class:`rtt.design.freeform.FingerPair` that was simulated.
    * Jaws: ``jaw_meshes`` (left, right trimeshes in their jaw frames, metres; for example the
      generator's print meshes) when given, else the union of each jaw's convex pieces, which
      is exactly the collision solid. A stock pair (``print_mesh_sha`` equal to the stock
      blade's) is drawn with the blade CAD. The right jaw is never a mirrored left jaw.
    * Object: ``object_asset`` (an asset already in ``store``, for example a screened
      textured scan from :func:`scan_asset`), else :func:`part_asset` of ``part``, else the
      union of the replay's object geoms.
    * ``overlays``: the episode's per-frame readouts (``EpisodeResultV2.overlays``), written to
      the ``.ovl.bin`` sidecar as float16.
    * ``meta``: ``design_space`` from the recorded meta (default ``freeform-v2``),
      ``design_id`` and ``part_id`` from the pair, ``geometry_id`` the design-level hash
      (argument, else the recorded meta's, else null). The pair's own hash stays in
      ``meta.geometry_hash``.
    """
    import numpy as np
    import trimesh

    from rtt.design.freeform.types import DESIGN_SPACE
    from rtt.render import glb
    from rtt.render import materials as mat
    from rtt.render.gripper import add_gripper_assets, stock_blade_shas
    from rtt.sim import replay as rp

    geoms = _check_v1_bodies(replay)
    meta = dict(replay.header.get("meta") or {})
    grip = add_gripper_assets(store)
    stock = stock_blade_shas()
    is_stock = all(getattr(pair, s).print_mesh_sha == stock[s] for s in ("left", "right"))
    visuals: dict[str, list[dict[str, Any]]] = {"palm": [rp.visual(grip.frame)]}
    for i, side in enumerate(("left", "right")):
        jaw = getattr(pair, side)
        if is_stock:
            finger = grip.blade(side)
        else:
            mesh = jaw_meshes[i] if jaw_meshes is not None else glb.mesh_from_pieces(jaw.pieces)
            if not isinstance(mesh, trimesh.Trimesh):
                raise TypeError("jaw_meshes must be two trimesh.Trimesh (metres, jaw frames)")
            reach = float(np.abs(mesh.bounds).max())
            if reach > MAX_JAW_EXTENT_M:
                raise ValueError(
                    f"the {side} jaw mesh reaches {reach:g} from its frame origin; jaw_meshes are in "
                    "metres (scale a millimetre print STL by 1e-3)"
                )
            finger = _own_asset(
                store,
                mesh,
                f"finger_{side}",
                finger_look or mat.FINGER_DESIGNED,
                f"{pair.part_id}, {side} jaw, design {pair.design_id[:12]}",
            )
        visuals[f"finger_{side}"] = [rp.visual(grip.carriage(side)), rp.visual(finger)]
    if object_asset is not None:
        if object_asset not in store.entries:
            raise ValueError(f"object asset {object_asset} is not in the store")
        obj = object_asset
    elif part is not None:
        obj = part_asset(store, part)
    else:
        obj = _own_asset(
            store, glb.mesh_from_geoms(geoms["object"]), "object", mat.PART_LOOKS["clay"], str(pair.part_id)
        )
    visuals["object"] = [rp.visual(obj)]
    ovl, channels = _overlay_array(overlays, replay.n_frames)
    v2_meta = {
        "design_space": str(meta.get("design_space") or DESIGN_SPACE),
        "design_id": str(pair.design_id),
        "geometry_id": geometry_id if geometry_id is not None else meta.get("geometry_id"),
        "part_id": _slug(str(pair.part_id)),
        "geometry_hash": str(pair.geometry_hash),
    }
    if is_stock:
        v2_meta["finger_visual"] = "stock blade CAD (collision: its committed convex hulls)"
    return _finish_v2(
        replay,
        store,
        visuals,
        v2_meta,
        run_id=run_id,
        code_version=code_version,
        product_mode=product_mode,
        keep_geoms=keep_geoms,
        provenance_extra=provenance_extra,
        overlays=ovl,
        channels=channels,
    )


# --------------------------------------------------------------------------- v2 bouts

BOUT_FORMAT: Final = "rtt-bout"
BOUT_VERSION: Final = 2
BOUT_FILE: Final = "bout.json"

#: What a lane says for each entrant of ``configs/entrants_v2.yaml`` (a recipe says "Designed").
ENTRANT_LABELS: Final[Mapping[str, str]] = {
    "stock": "Stock blade",
    "flat": "Flat pad",
    "imprint_configurator": "Configurator imprint",
    "imprint_swept": "Swept imprint",
    "ref_wrap": "Drafted wrap",
    "ref_hook": "Hooked wrap",
}
ENTRANT_NOTES: Final[Mapping[str, str]] = {
    "stock": "The gripper's own clamp blade, as shipped",
    "ref_wrap": "Reference entrant, scored outside the budget",
    "ref_hook": "Reference entrant, scored outside the budget",
}


def describe_placement(part_name: str, placement: Any) -> str:
    """A lane's condition in words, for example "flanged bushing, 2 mm off centre across the jaw"."""
    import math

    part = part_name.replace("_", " ")
    bits = []
    x, y = 1000.0 * float(placement.err_x), 1000.0 * float(placement.err_y)
    yaw = math.degrees(float(placement.err_yaw))
    if abs(x) > 1e-9:
        bits.append(f"{abs(x):g} mm off centre across the jaw")
    if abs(y) > 1e-9:
        bits.append(f"{abs(y):g} mm off centre along the closing axis")
    if abs(yaw) > 1e-9:
        bits.append(f"turned {abs(yaw):g} deg")
    return f"{part}, {' and '.join(bits) if bits else 'centred'}"


def _part_size_mm(part: Any) -> list[float]:
    import numpy as np

    pts = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in part.pieces])
    return [round(1000.0 * float(e), 3) for e in pts.max(axis=0) - pts.min(axis=0)]


def record_bout_v2(
    out_dir: str | Path,
    episodes: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    title: str | None = None,
    summary: str | None = None,
    repo: str | Path | None = None,
) -> dict[str, Any]:
    """Run v2 episodes with recording on and write them as a v2 bout directory.

    Each episode is ``{"stem", "design", "part", "placement"}`` plus an optional ``"label"``:
    ``design`` is an entrant name of ``configs/entrants_v2.yaml`` or a recipe (a mapping of
    :class:`~rtt.design.freeform.FingerRecipe` fields), ``part`` a ``train_v2`` part and
    ``placement`` a ``train_v2`` placement label. Every episode is that suite condition with
    its first jitter seed, so each lane is an episode the scored suite runs; nothing is staged.

    Writes ``<stem>.json`` (replay v2, from :func:`lift_freeform_to_v2`), ``<stem>.bin``,
    ``<stem>.ovl.bin``, ``assets/`` (the GLB store, with its manifest) and :data:`BOUT_FILE`:
    the bout's lanes with their outcomes from the scorer, which
    :func:`rtt.pipeline.export_web.export_web` (``replay_version=2``) publishes as a bout.
    Returns the bout document.
    """
    import time
    from datetime import UTC, datetime

    import trimesh

    from rtt.design.freeform import FingerRecipe, build_pair, load_entrants_v2
    from rtt.design.freeform.generate import build_pair_full
    from rtt.parts.suite import train_suite
    from rtt.render.glb import AssetStore
    from rtt.sim.episode import run_episode_v2
    from rtt.sim.replay import code_version as _code_version
    from rtt.sim.task import default_task_v2, task_digest
    from rtt.store.manifest import write_json_atomic

    out = Path(out_dir)
    stems = [str(ep["stem"]) for ep in episodes]
    if not stems or len(set(stems)) != len(stems):
        raise ValueError("a bout needs at least one episode, and unique stems")
    task = default_task_v2()
    entrants = {e.name: e for e in load_entrants_v2()}
    suite = train_suite()
    conds = {(c.part.name, c.placement_label): c for c in suite}
    store = AssetStore(out / "assets")
    code = _code_version(repo) if repo is not None else _code_version()
    started = datetime.now(UTC)
    lanes = []
    for ep in episodes:
        stem, part_name, label = str(ep["stem"]), str(ep["part"]), str(ep["placement"])
        if (part_name, label) not in conds:
            raise ValueError(
                f"{stem}: no {suite.name} condition for part {part_name!r} at placement {label!r}"
            )
        cond = conds[(part_name, label)]
        spec = ep["design"]
        if isinstance(spec, str):
            if spec not in entrants:
                raise ValueError(f"{stem}: unknown entrant {spec!r}; known: {sorted(entrants)}")
            design, name, kind = entrants[spec], spec, "entrant"
        else:
            design = FingerRecipe.model_validate(dict(spec))
            name, kind = design.name or "recipe", "recipe"
        seed = int(cond.seeds[0])
        t0 = time.perf_counter()
        if getattr(design, "kind", None) == "stock":
            pair, meshes = build_pair(design, cond.part, task), None
        else:
            built = build_pair_full(design, cond.part, task)
            pair = built.pair
            meshes = tuple(
                trimesh.Trimesh(m.vertices, m.faces, process=False)
                for m in (built.left_mesh, built.right_mesh)
            )
        res = run_episode_v2(pair, cond.part, cond.placement, seed, record=True, task=task)
        if res.replay is None:
            raise RuntimeError(f"{stem}: the episode was not recorded")
        v2 = lift_freeform_to_v2(
            res.replay,
            store,
            pair,
            part=cond.part,
            overlays=res.overlays,
            jaw_meshes=meshes,  # type: ignore[arg-type]
            run_id=run_id,
            code_version=code,
            episode=stem,
            suite=suite.name,
            condition=cond.condition_id,
            seed=seed,
        )
        write_replay(v2, out / stem)
        reg = res.registration
        lanes.append(
            {
                "stem": stem,
                "label": str(ep.get("label") or ENTRANT_LABELS.get(name) or f"Designed: {name}"),
                "design": name,
                "design_kind": kind,
                "design_note": ENTRANT_NOTES.get(name)
                or (
                    "Hand-designed baseline entrant" if kind == "entrant" else "Recipe given in the bout spec"
                ),
                "design_id": pair.design_id,
                "part_id": pair.part_id,
                "part_family": cond.part.family,
                "part_size_mm": _part_size_mm(cond.part),
                "condition_id": cond.condition_id,
                "condition": describe_placement(part_name, cond.placement),
                "placement": label,
                "seed": seed,
                "n_frames": v2.n_frames,
                "success": bool(res.success),
                "lift_success": bool(res.lift_success),
                "failure_mode": res.failure_mode,
                "registered": bool(reg.registered) if reg is not None else False,
                "registration_flags": list(reg.sub_flags) if reg is not None else [],
                "registration": None
                if reg is None
                else {"seat_short_mm": reg.seat_short_mm, "dx_mm": reg.dx_mm, "dy_mm": reg.dy_mm},
                "max_lift_m": float(res.max_lift),
                "hold_time_s": float(res.hold_time),
                "flags": [f.split("@", 1)[0] for f in res.flags],
                "n_steps": int(res.n_steps),
                "wall_time_s": round(time.perf_counter() - t0, 3),
                "q_seat_mm": round(1000.0 * pair.q_seat, 3),
            }
        )
    store.write_manifest()
    finished = datetime.now(UTC)
    doc = {
        "format": BOUT_FORMAT,
        "version": BOUT_VERSION,
        "run_id": run_id,
        "kind": "training",
        "title": title or f"{len(lanes)} recorded v2 episodes from {suite.name}",
        "summary": summary
        or (
            f"Recorded episodes of the v2 training suite ({suite.name}), each the condition's first jitter "
            "seed, replayed from the recording."
        ),
        "task": {"name": task.name, "digest": task_digest(task)},
        "suite": suite.name,
        "code_version": code,
        "started_at": started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "finished_at": finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "wall_s": round((finished - started).total_seconds(), 3),
        "lanes": lanes,
    }
    write_json_atomic(out / BOUT_FILE, doc)
    return doc
