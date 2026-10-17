"""Building, driving and judging a vendored catalog model (shared by both catalog families).

* :func:`vendored_spec`: the :class:`~rtt.effectors.types.EffectorSpec` of one model and
  primitive (``origin = "catalog"``; ``params`` carry the synergy, so its id changes if the
  synergy does).
* :func:`build_vendored`: loads the vendored MJCF (:func:`~rtt.effectors.catalog.vendored.load_child`),
  mounts it palm-down under the wrist with the levelling and centring the open-pose probe
  measured (:func:`mount_geometry`), and describes it as an
  :class:`~rtt.effectors.types.EffectorBuild` (licence records, declared actuation limits).
* :func:`synergy_primitive` and :func:`contact_criterion`: the contract's shared
  :class:`~rtt.effectors.primitives.SynergyPrimitive` and
  :class:`~rtt.effectors.primitives.ContactCriterion`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from rtt.effectors.catalog.attach import (
    OpenPose,
    actuator_limits,
    controls,
    match_options,
    mount_rotation,
    probe,
)
from rtt.effectors.catalog.models import CatalogModel, Synergy, catalog_model
from rtt.effectors.catalog.vendored import load_child, read_model
from rtt.effectors.primitives import ContactCriterion, SynergyPrimitive
from rtt.effectors.types import (
    NAME_PREFIX,
    BuildContext,
    EffectorBuild,
    EffectorFamily,
    EffectorSpec,
    InvalidEffector,
)

if TYPE_CHECKING:
    from rtt.parts.types import PartSpec

__all__ = [
    "OPENING_MARGIN",
    "MountGeometry",
    "build_vendored",
    "contact_criterion",
    "mount_geometry",
    "synergy_params",
    "synergy_primitive",
    "vendored_spec",
]

#: A gripper refuses a part whose closing-axis width leaves less than this much clearance.
OPENING_MARGIN = 0.004  # m


def synergy_params(syn: Synergy) -> dict[str, Any]:
    """The synergy as JSON data (hashed into the effector id)."""
    return {
        "source": "vendored",
        "grip_height_frac": syn.grip_height_frac,
        "tip_offset": syn.tip_offset,
        "rest": syn.rest,
        "open": dict(syn.open),
        "closed": dict(syn.closed),
        "via": [dict(v) for v in syn.via],
    }


def vendored_spec(family: EffectorFamily, cm: CatalogModel, syn: Synergy) -> EffectorSpec:
    label = {"power": "power grasp", "pinch": "pinch"}.get(syn.name, syn.name)
    name = cm.display_name if len(cm.synergies) == 1 else f"{cm.display_name}, {label}"
    return EffectorSpec(
        family=family,
        model=cm.key,
        display_name=name,
        primitive=syn.name,
        params=synergy_params(syn),
        origin="catalog",
    )


@dataclass(frozen=True)
class MountGeometry:
    """Where the model sits on the wrist, and its open pose (wrist frame, metres)."""

    frame_pos: tuple[float, float, float]
    frame_quat: tuple[float, float, float, float]
    pose: OpenPose
    ctrl_open: tuple[float, ...]
    ctrl_closed: tuple[float, ...]
    ctrl_via: tuple[tuple[float, ...], ...]  # waypoints between open and closed
    actuator_names: tuple[str, ...]  # attached names, compile order
    peaks: tuple[float, ...]  # per-actuator peak output (N or N m)
    joint_kinds: tuple[str, ...]  # "hinge" or "slide" of each actuator's driven joint ("tendon" otherwise)

    @property
    def reach(self) -> float:
        return self.pose.reach

    @property
    def tcp(self) -> tuple[float, float, float]:
        """Grasp centre: on the wrist axis, 10 mm above the lowest point of the open pose."""
        return (0.0, 0.0, float(self.frame_pos[2] + self.pose.bottom + 0.010))


def _ctrlrange(model: mujoco.MjModel, acts: list[int]) -> np.ndarray:
    rng = np.array([model.actuator_ctrlrange[a] for a in acts], dtype=np.float64).reshape(-1, 2)
    for i, a in enumerate(acts):
        if not model.actuator_ctrllimited[a]:
            rng[i] = (-np.inf, np.inf)
    return rng


def _upstream(names: list[str]) -> list[str]:
    return [n.removeprefix(NAME_PREFIX) for n in names]


def _attach(
    cm: CatalogModel, spec: mujoco.MjSpec, mount: mujoco.MjsBody, pos: Any, quat: Any
) -> mujoco.MjsBody:
    child, body = load_child(cm.key, root=cm.root, drop_joints=cm.drop_joints)
    match_options(child, spec)
    frame = mount.add_frame(pos=list(pos), quat=list(quat))
    frame.attach_body(body, NAME_PREFIX, "")
    return body


@lru_cache(maxsize=64)
def mount_geometry(key: str, synergy: str) -> MountGeometry:
    """Probe the model in its open pose and level it on the mount plane (module docstring)."""
    cm = catalog_model(key)
    syn = cm.synergy(synergy)
    quat = tuple(float(v) for v in mount_rotation(cm.down, cm.forward))
    info: dict[str, Any] = {}

    def attach(spec: mujoco.MjSpec, mount: mujoco.MjsBody) -> None:
        _attach(cm, spec, mount, (0.0, 0.0, 0.0), quat)

    def ctrl(model: mujoco.MjModel, acts: list[int]) -> tuple[float, ...]:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "" for a in acts]
        rng = _ctrlrange(model, acts)
        info["names"] = tuple(names)
        info["open"] = controls(_upstream(names), syn.open, rng, syn.rest)
        info["closed"] = controls(_upstream(names), syn.closed, rng, syn.rest)
        info["via"] = tuple(controls(_upstream(names), v, rng, syn.rest) for v in syn.via)
        info["peaks"] = actuator_limits(model, acts)
        kinds = []
        for a in acts:
            if int(model.actuator_trntype[a]) == mujoco.mjtTrn.mjTRN_JOINT:
                jt = int(model.jnt_type[int(model.actuator_trnid[a, 0])])
                kinds.append("slide" if jt == mujoco.mjtJoint.mjJNT_SLIDE else "hinge")
            else:
                kinds.append("tendon")
        info["kinds"] = tuple(kinds)
        return info["open"]

    centre = tuple(NAME_PREFIX + b for b in syn.centre_bodies)
    sides = None
    if cm.finger_sides is not None:
        sides = tuple(tuple(NAME_PREFIX + b for b in side) for side in cm.finger_sides)
    pose = probe(attach, ctrl, centre_bodies=centre, finger_sides=sides)  # type: ignore[arg-type]
    off = cm.mount_offset
    frame_pos = (
        -pose.centre_xy[0] + off[0],
        -pose.centre_xy[1] + off[1],
        pose.level() + off[2],
    )
    return MountGeometry(
        frame_pos=frame_pos,
        frame_quat=quat,  # type: ignore[arg-type]
        pose=pose,
        ctrl_open=info["open"],
        ctrl_closed=info["closed"],
        ctrl_via=info["via"],
        actuator_names=info["names"],
        peaks=info["peaks"],
        joint_kinds=info["kinds"],
    )


def _actuation(cm: CatalogModel, geo: MountGeometry) -> dict[str, float]:
    out: dict[str, float] = {"n_actuators": len(geo.peaks)}
    hinge = [p for p, k in zip(geo.peaks, geo.joint_kinds, strict=True) if k == "hinge"]
    linear = [p for p, k in zip(geo.peaks, geo.joint_kinds, strict=True) if k != "hinge"]
    if hinge:
        out["joint_torque_nm_max"] = round(max(hinge), 4)
    if linear:
        out["linear_force_n_max"] = round(max(linear), 4)
    if cm.kind == "gripper" and geo.pose.opening is not None:
        out["opening_mm"] = round(1000.0 * geo.pose.opening, 2)
    return out


def check_fits(cm: CatalogModel, geo: MountGeometry, part: PartSpec) -> None:
    """Refuse a part wider than a gripper's opening along the closing axis (InvalidEffector)."""
    if cm.kind != "gripper" or geo.pose.opening is None:
        return
    width = float(part.size[1])
    if width > geo.pose.opening - OPENING_MARGIN:
        raise InvalidEffector(
            f"the part is {1000 * width:.1f} mm across the closing axis; the gripper opens "
            f"{1000 * geo.pose.opening:.1f} mm"
        )


def build_vendored(ctx: BuildContext, cm: CatalogModel, syn: Synergy) -> EffectorBuild:
    """Attach the model under ``ctx.mount`` and describe it (module docstring)."""
    geo = mount_geometry(cm.key, syn.name)
    check_fits(cm, geo, ctx.part)
    before_bodies = {b.name for b in ctx.spec.bodies}
    before_acts = {a.name for a in ctx.spec.actuators}
    _attach(cm, ctx.spec, ctx.mount, geo.frame_pos, geo.frame_quat)
    bodies = tuple(b.name for b in ctx.spec.bodies if b.name not in before_bodies)
    acts = tuple(a.name for a in ctx.spec.actuators if a.name not in before_acts)
    if acts != geo.actuator_names:
        raise RuntimeError(f"{cm.key}: actuators {acts} differ from the probe's {geo.actuator_names}")
    vm = read_model(cm.key)
    notes = [
        f"vendored model {cm.key}: {vm.license}, {vm.status} licence chain (credits: SOURCE.json)",
        "mounted palm-down on an adapter plate under the shared wrist; upstream wrist or "
        "positioning joints removed: " + (", ".join(cm.drop_joints) or "none"),
        "upstream actuator gains and force ranges unmodified; massless visual geoms not loaded",
        *((cm.card_note,) if cm.card_note else ()),
        *cm.notes,
    ]
    if vm.mesh_files:
        notes.append(f"{len(vm.mesh_files)} collision mesh(es) shipped as their convex hull (SOURCE.json)")
    if vm.inertials:
        notes.append("mass properties of " + ", ".join(vm.inertials) + " pinned at the upstream values")
    return EffectorBuild(
        spec=ctx.effector,
        bodies=bodies,
        actuators=acts,
        ctrl_open=geo.ctrl_open,
        reach=geo.reach,
        tcp=geo.tcp,
        actuation=_actuation(cm, geo),
        sources=vm.sources,
        notes=tuple(notes),
        max_opening=geo.pose.opening if cm.kind == "gripper" else None,
    )


@dataclass(frozen=True)
class PathSynergy(SynergyPrimitive):
    """A synergy through waypoints: the controls follow the piecewise-linear path open ->
    ``via`` -> closed on the shared close ramp (evenly spaced in ``closure``), then stay
    closed. With no waypoints it is the contract's :class:`SynergyPrimitive` exactly."""

    via: tuple[tuple[float, ...], ...] = ()

    def command(
        self, t: float, phase: Any, closure: float, model: mujoco.MjModel, data: mujoco.MjData, scene: Any
    ) -> np.ndarray:
        if not self.via:
            return super().command(t, phase, closure, model, data, scene)
        path = np.array([self.ctrl_open, *self.via, self.ctrl_closed], dtype=np.float64)
        s = min(max(closure, 0.0), 1.0) * (len(path) - 1)
        i = min(int(s), len(path) - 2)
        return path[i] + (s - i) * (path[i + 1] - path[i])


def synergy_primitive(cm: CatalogModel, syn: Synergy) -> SynergyPrimitive:
    geo = mount_geometry(cm.key, syn.name)
    return PathSynergy(
        name=syn.name,
        ctrl_open=geo.ctrl_open,
        ctrl_closed=geo.ctrl_closed,
        grip_height_frac=syn.grip_height_frac,
        tip_offset=syn.tip_offset,
        via=geo.ctrl_via,
    )


def contact_criterion(cm: CatalogModel) -> ContactCriterion:
    return ContactCriterion(min_bodies=cm.min_bodies)
