"""Part-shaped parallel jaws in the Studio (product mode, not the pre-registered result).

**Product mode. Not part of the pre-registered experiment.** The pre-registered v2 entrants
(:mod:`rtt.effectors.parallel_jaw`: stock blade, flat pad, configurator imprint) run only inside
the frozen v2 episode, and that stays exactly as it is. This module gives the same jaws a second,
separate life as Studio tools: the open 84 mm parallel gripper with two v2 fingers, built under
the Studio's tool mount and **built against the object a step touches**. Its numbers are Studio
simulation results under the Studio's controllers, never the experiment's.

**What builds.** A Studio jaw is an :class:`~rtt.effectors.types.EffectorSpec` of the
``parallel_jaw`` family whose ``params`` hold one key, :data:`PART_JAW_KEY`: the entrant (one of
:data:`STUDIO_ENTRANTS`) and the target object (its name, template kind, gripped feature, params
and part reference). :func:`part_jaw_spec` makes one. The family's build path
(:meth:`StudioPartJaw.build`) then adds, under the mount:

* the gripper housing ``eff_pj_housing``: the v2 gripper frame, a 47.6 x 128 x 24 mm box with the
  v2 palm mass (``task_v2`` ``gripper.palm_half_size``, ``gripper.palm_mass``: in the v2 scene the
  palm *is* the gripper frame), hanging from the mount plane. The Studio wrist keeps its own
  0.4 kg (every candidate has it), so the housing is the tool's;
* two jaw bodies ``eff_pj_left`` and ``eff_pj_right`` rooted on the housing's underside, on
  closing slides exactly as in the v2 scene (:func:`rtt.sim.scene.build_spec_v2`): ``q`` from 0
  (open, 84 mm) to the 42 mm stroke, armature and damping of ``task_v2``, the left jaw at
  ``+y``; each jaw is its generated pieces (one convex mesh geom per piece) with its own mass,
  centre of mass and inertia (:class:`~rtt.design.freeform.types.JawGeometry`);
* the v2 drive: a mean-closure tendon ``eff_pj_grip`` driven by a position servo with the
  ``task_v2`` gains (kp 4000, kv 100), force limit 50 N (25 N squeeze per jaw, the sustained
  value of the gripper's protected servo) and control range from ``open_target`` to the stroke
  plus the overdrive, and a joint equality that keeps the jaws coupled (self-centring).

The jaw pair is compiled by the frozen generator (:func:`rtt.design.freeform.generate.build_pair`)
against a :class:`~rtt.parts.types.PartSpec` of the **gripped feature** (:func:`gripped_feature`):

* a ``part``: its own part (:func:`rtt.studio.templates.part.part_spec_of`);
* a template member (a ``threaded_joint`` coupling, nut or cap; a ``port_plug`` plug; a
  ``quarter_turn_latch`` head): the template's own builder is run alone in a scratch scene, and
  the collision geoms of the body the step grips are read from the compiled model, in that body's
  frame. Meshes and boxes are taken exactly; a cylinder becomes the circumscribed 64-gon prism
  (at most 0.12 % of its radius outside the simulated surface, inside the imprint's 0.4 mm
  clearance). The feature is then turned so that the axis the Studio controller closes along
  lies on the jaws' closing axis: a threaded member is gripped along a flat's normal (its local
  ``+x``, the ``flat`` site), so it is turned +90 deg about z; a latch head is gripped across its
  fin (local ``y``) and a plug or part in its own frame, so they are not turned.

**Where the step grips.** S2's controllers grip a template feature at a fixed height, not at the
recipe's grip fraction: a threaded member with the tips 3 mm above its bottom (twist, insert), a
latch head 2 mm above its panel (twist), a plug at :func:`rtt.studio.primitives.insert.grip_tip`.
Built at the recipe's band, the imprint's root block lands on a coupling's top and the jaws stop
on its flats, short of the imprint. So a generated entrant (imprint, flat pad) on a template
feature is compiled from the entrant's own :class:`~rtt.design.freeform.generate.PairSpec` with
its pad band placed where the step grips (``tip_height``, at least the generator's table
clearance); everything else is the entrant's. A part is picked at the pair's own tip height, so it
keeps the recipe's band. The stock blade has no design and is built as it is.

A handwheel valve gets no part-shaped jaw (:data:`NOT_GRIPPED`): S2's twist turns a spoked wheel
with open fingers in its spoke gaps (nothing is gripped), and a solid disc is round (an imprint
gives no hold about its axis). Lids, drawers and bins are not gripped features.

**Primitive.** :class:`PartJawPrimitive` is the v2 grip ramp as a v1-style synergy: the grip
target moves linearly from ``open_target`` to the stroke plus overdrive with the shared closure
(``task_v2`` ``controller``), and the jaws' lowest point goes to the pair's designed tip height
(:attr:`~rtt.design.freeform.types.FingerPair.tip_height`). The criterion is the shared contact
test (two jaw bodies on the part with opposing forces).

**Fidelity.** The family's fidelity is the v2 fingers' (``core``), which the Studio maps to its
``rigid`` tier (:func:`rtt.studio.types.tier_of_effector`); a step on a threaded joint or a latch
is an approximation through its template, with the template's caveat. A Studio jaw is never the
experiment's scene: its records say "Product mode" and its card notes say it is a Studio build.

**Registration.** The ``parallel_jaw`` family slot holds one implementation; importing this module
replaces the frozen reference there with :class:`StudioPartJaw`, a subclass that inherits every
frozen behaviour unchanged (presets, the delegated v2 episode, ``pair_for``, fidelity, analytic
registration) and adds only the Studio build path for specs with :data:`PART_JAW_KEY`. A spec
without it (a pre-registered entrant) still refuses the Studio scene with the frozen message. So
that spawned worker processes find the build path too, a Studio jaw spec pickled for
:mod:`multiprocessing` restores through :func:`_restore_spec`, which imports this module first
(:func:`multiprocessing.reduction.ForkingPickler.register`); every other spec pickles as before.

**Print files.** :func:`jaw_print_files` writes a jaw pair's printed solids through the v2 print
pack (:func:`rtt.part.pack.make_print_pack_v2`): each jaw upright for printing and each jaw fused
with its carriage block, one watertight solid each, CC0-1.0. The stock blade is the open-source
gripper's own part (CC BY 4.0): it is never printed here, and its build carries the licence
records of the blade asset instead.
"""

from __future__ import annotations

import functools
import hashlib
import math
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import mujoco
import numpy as np

from rtt.effectors.parallel_jaw import DISPLAY_NAMES, ParallelJawReference
from rtt.effectors.primitives import ContactCriterion
from rtt.effectors.registry import register_family
from rtt.effectors.types import (
    PRODUCT_MODE_LABEL,
    AssetSource,
    BuildContext,
    EffectorBuild,
    EffectorFamily,
    EffectorSpec,
    InvalidEffector,
    PrimitivePlan,
    _thaw,
    canonical_json,
)

if TYPE_CHECKING:
    from rtt.design.freeform.types import FingerPair
    from rtt.parts.types import PartSpec
    from rtt.sim.controller import Phase
    from rtt.sim.objects import Placement
    from rtt.sim.task import TaskV2
    from rtt.studio.types import ObjectInstance, TaskSpec

__all__ = [
    "GRIPPED_FEATURE",
    "JAW_PREFIX",
    "LATCH_TIP_ABOVE_PANEL",
    "MEMBER_TIP_ABOVE_BOTTOM",
    "NOT_GRIPPED",
    "PART_JAW_KEY",
    "STUDIO_ENTRANTS",
    "GrippedFeature",
    "PartJawPrimitive",
    "StudioPartJaw",
    "feature_part",
    "gripped_feature",
    "is_part_jaw",
    "jaw_print_files",
    "jaw_target",
    "part_jaw_spec",
    "part_jaw_specs",
    "studio_pair",
    "target_instance",
]

#: The params key of a Studio jaw spec (its presence is what makes a spec build in the Studio).
PART_JAW_KEY: Final = "studio_part_jaw"
#: The entrants a Studio jaw can be, the default (deterministic configurator imprint) first; the
#: flat pad and the stock blade stay as baselines.
STUDIO_ENTRANTS: Final[tuple[str, ...]] = ("imprint_configurator", "flat", "stock")
#: Element names of the jaw tool (after the scene's ``eff_`` prefix).
JAW_PREFIX: Final = "pj_"
#: Template kind -> the handle (feature) the step grips, for the kinds that get Studio jaws.
GRIPPED_FEATURE: Final[Mapping[str, str]] = {
    "part": "part",
    "threaded_joint": "member",
    "port_plug": "plug",
    "quarter_turn_latch": "head",
}
#: Kinds without a part-shaped jaw, and why.
NOT_GRIPPED: Final[Mapping[str, str]] = {
    "handwheel_valve": (
        "the twist turns a spoked wheel with open fingers in its spoke gaps (nothing is gripped), "
        "and a solid disc is round, so an imprint gives no hold about its axis"
    ),
    "hinged_lid": "a lid or door is hooked or pushed, not gripped as a part",
    "drawer": "a drawer is hooked or pushed, not gripped as a part",
    "bin": "a bin is a destination, not a gripped part",
}
#: deg about z that puts the controller's closing direction on the jaws' closing axis (y): S2's
#: twist and insert close a threaded member along a flat's normal, its local +x (the ``flat``
#: site); a latch head across its fin (local y); a plug and a part in their own frame.
_CLOSING_TURN_DEG: Final[Mapping[str, float]] = {"threaded_joint": 90.0}
#: Sides of the prism that stands for a simulated cylinder (circumscribed).
CYLINDER_SIDES: Final = 64
_DISPLAY_MAX: Final = 60


# --------------------------------------------------------------------------- specs


def is_part_jaw(spec: EffectorSpec) -> bool:
    """Whether ``spec`` is a Studio part-shaped jaw (it builds into the Studio scene)."""
    return spec.family is EffectorFamily.PARALLEL_JAW and PART_JAW_KEY in spec.params


def jaw_target(spec: EffectorSpec) -> dict[str, Any]:
    """The plain target record of a Studio jaw spec (entrant, object, kind, handle, params, ref)."""
    if not is_part_jaw(spec):
        raise ValueError(f"{spec.display_name!r} is not a Studio part-shaped jaw")
    return dict(_thaw(spec.params[PART_JAW_KEY]))


def _display(entrant: str, label: str) -> str:
    base = DISPLAY_NAMES[entrant]
    room = _DISPLAY_MAX - len(base) - 2
    return f"{base}: {label[:room].rstrip()}" if room > 3 else base


def part_jaw_spec(obj: ObjectInstance, entrant: str = STUDIO_ENTRANTS[0]) -> EffectorSpec:
    """The Studio jaw ``entrant`` built for ``obj`` (its gripped feature, :data:`GRIPPED_FEATURE`).

    Raises ``ValueError`` for an unknown entrant or an object kind without a gripped feature
    (:data:`NOT_GRIPPED` says why)."""
    if entrant not in STUDIO_ENTRANTS:
        raise ValueError(f"unknown Studio jaw entrant {entrant!r}; known: {STUDIO_ENTRANTS}")
    kind = obj.kind.value
    if kind not in GRIPPED_FEATURE:
        raise ValueError(
            f"{obj.name} ({kind}) gets no part-shaped jaw: {NOT_GRIPPED.get(kind, 'no gripped feature')}"
        )
    target = {
        "entrant": entrant,
        "object": obj.name,
        "kind": kind,
        "handle": GRIPPED_FEATURE[kind],
        "params": _thaw(obj.params),
        "ref": obj.ref,
    }
    return EffectorSpec(
        family=EffectorFamily.PARALLEL_JAW,
        model=f"studio-{entrant.replace('_', '-')}",
        display_name=_display(entrant, obj.label),
        primitive="jaw",
        params={PART_JAW_KEY: target},
        origin="manual",
    )


def part_jaw_specs(task_spec: TaskSpec) -> list[EffectorSpec]:
    """Every Studio jaw of a task: the :data:`STUDIO_ENTRANTS` for each object that a step touches
    and that has a gripped feature, in task order (objects, then entrants)."""
    touched = {s.object for s in task_spec.steps}
    out = []
    for obj in task_spec.objects:
        if obj.name in touched and obj.kind.value in GRIPPED_FEATURE:
            out += [part_jaw_spec(obj, e) for e in STUDIO_ENTRANTS]
    return out


def target_instance(spec: EffectorSpec) -> ObjectInstance:
    """The :class:`~rtt.studio.types.ObjectInstance` a Studio jaw is built for (at the origin)."""
    from rtt.studio.types import ObjectInstance

    t = jaw_target(spec)
    return ObjectInstance(str(t["object"]), t["kind"], "Gripped feature", dict(t["params"]), ref=t["ref"])


# --------------------------------------------------------------------------- the gripped feature


def _hull_piece(points: np.ndarray) -> Any:
    """A :class:`rtt.parts.pieces.Piece` of the convex hull of ``points`` (m)."""
    from scipy.spatial import ConvexHull

    from rtt.parts.library import piece_from_mesh

    pts = np.round(np.asarray(points, dtype=np.float64), 9)
    hull = ConvexHull(pts)
    keep = np.asarray(hull.vertices, dtype=np.int64)
    index = {int(v): i for i, v in enumerate(keep)}
    verts = pts[keep]
    centre = verts.mean(axis=0)
    tris = []
    for simplex in hull.simplices:
        a, b, c = (index[int(v)] for v in simplex)
        n = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        tris.append((a, b, c) if float(n @ (verts[a] - centre)) > 0.0 else (a, c, b))
    return piece_from_mesh(verts, np.asarray(tris, dtype=np.int64))


def _geom_points(model: mujoco.MjModel, g: int) -> np.ndarray:
    """A collision geom's convex vertex set in its body's frame (module docstring)."""
    t = int(model.geom_type[g])
    size = np.asarray(model.geom_size[g], dtype=np.float64)
    gt = mujoco.mjtGeom
    if t == gt.mjGEOM_MESH:
        m = int(model.geom_dataid[g])
        a, n = int(model.mesh_vertadr[m]), int(model.mesh_vertnum[m])
        local = np.asarray(model.mesh_vert[a : a + n], dtype=np.float64)
    elif t == gt.mjGEOM_BOX:
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
        local = signs * size
    elif t == gt.mjGEOM_CYLINDER:
        rc = float(size[0]) / math.cos(math.pi / CYLINDER_SIDES)
        ang = 2.0 * math.pi * np.arange(CYLINDER_SIDES) / CYLINDER_SIDES
        ring = np.c_[rc * np.cos(ang), rc * np.sin(ang)]
        local = np.vstack([np.c_[ring, np.full(len(ring), z)] for z in (-size[1], size[1])])
    else:
        raise ValueError(f"geom {g} of type {t} has no convex piece model here")
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(model.geom_quat[g], dtype=np.float64))
    return local @ rot.reshape(3, 3).T + np.asarray(model.geom_pos[g], dtype=np.float64)


#: Where S2's controllers put the fingers' lowest point on a template feature (m), copied from
#: :mod:`rtt.studio.primitives.twist` and :mod:`rtt.studio.primitives.insert` so that a Studio jaw
#: is designed for the grip the step makes (a test checks the jaws seat into the imprint there):
#: a threaded member is gripped with the tips 3 mm above its bottom (twist and insert), a latch
#: head with the tips 2 mm above its panel (twist); a plug is picked at
#: :func:`rtt.studio.primitives.insert.grip_tip` above the table.
MEMBER_TIP_ABOVE_BOTTOM: Final = 0.003
LATCH_TIP_ABOVE_PANEL: Final = 0.002


@dataclass(frozen=True, eq=False)
class GrippedFeature:
    """The feature a step grips: its PartSpec in the jaws' frame and the grip height the step's
    controller uses (the pad band's bottom above the feature's lowest point; ``None`` for a part,
    which the pick lowers to the pair's own designed tip height), plus notes."""

    part: PartSpec
    grip_tip_m: float | None
    notes: tuple[str, ...] = ()


def _template_feature(inst: ObjectInstance, task: TaskV2) -> GrippedFeature:
    """The gripped body of a template instance as a PartSpec, and its grip height (module docstring)."""
    from rtt.design.freeform.validate import TABLE_CLEARANCE
    from rtt.effectors.scene import REPO_ROOT
    from rtt.parts.library import Shape, assemble
    from rtt.parts.types import Material
    from rtt.sim.scene import _base_spec
    from rtt.studio.templates import TemplateContext, get_template_builder, object_prefix

    spec, _palm = _base_spec(task)
    handles = get_template_builder(inst.kind).add(
        TemplateContext(
            spec=spec, instance=inst, prefix=object_prefix(inst.name), task=task, repo_root=REPO_ROOT
        )
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    body = model.body(handles.bodies[-1]).id  # the body a pick, insert or twist acts on
    turn = math.radians(_CLOSING_TURN_DEG.get(inst.kind.value, 0.0))
    rz = np.array([[math.cos(turn), -math.sin(turn), 0.0], [math.sin(turn), math.cos(turn), 0.0], [0, 0, 1]])
    pieces, bottom = [], math.inf
    r_body = data.xmat[body].reshape(3, 3)
    for g in np.flatnonzero(model.geom_bodyid == body):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        pts = _geom_points(model, int(g))
        bottom = min(bottom, float((pts @ r_body.T + data.xpos[body])[:, 2].min()))
        pieces.append(_hull_piece(pts @ rz.T))
    if not pieces:
        raise InvalidEffector(
            f"{inst.name}: the gripped {GRIPPED_FEATURE[inst.kind.value]} has no collision geometry"
        )
    name = f"studio_{inst.name}_{GRIPPED_FEATURE[inst.kind.value]}"
    probe = assemble(
        name,
        f"studio_{inst.kind.value}",
        Shape(pieces=tuple(pieces)),
        Material(name, 1000.0, (0.2, 0.5, 0.9)),
    )
    density = float(model.body_mass[body]) / max(float(probe.volume), 1e-12)  # the simulated mass
    part = replace(probe, material=Material(name, density, (0.2, 0.5, 0.9)))
    info = handles.info
    notes: list[str] = []
    kind = inst.kind.value
    if kind == "threaded_joint":
        tip = MEMBER_TIP_ABOVE_BOTTOM
    elif kind == "quarter_turn_latch":
        tip = float(info["base_z"]) + LATCH_TIP_ABOVE_PANEL - bottom
    else:  # port_plug: the insert's pick grips the standing plug at grip_tip above the table
        from rtt.studio.primitives.insert import grip_tip

        tip = grip_tip(
            info,
            float(info["plug_len_m"]),
            float(task.physics.friction),
            float(task.controller.min_tip_clearance),
        )
    if tip < TABLE_CLEARANCE:
        notes.append(
            f"the step grips {1000 * tip:.1f} mm above the {GRIPPED_FEATURE[kind]}'s bottom; the jaw is "
            f"designed {1000 * TABLE_CLEARANCE:.1f} mm above it (the generator's table clearance)"
        )
        tip = TABLE_CLEARANCE
    notes.append(
        f"pad band designed {1000 * tip:.1f} mm above the {GRIPPED_FEATURE[kind]}'s bottom, where the step grips"
    )
    return GrippedFeature(part, round(tip, 7), tuple(notes))


def _instance_key(inst: ObjectInstance) -> str:
    return canonical_json(
        {"kind": inst.kind.value, "name": inst.name, "params": _thaw(inst.params), "ref": inst.ref}
    )


_FEATURES: dict[tuple[str, str], GrippedFeature] = {}


def gripped_feature(inst: ObjectInstance, task: TaskV2 | None = None) -> GrippedFeature:
    """The feature a step grips on ``inst`` (:class:`GrippedFeature`, module docstring), in the
    jaws' frame (closing axis = y). Cached per instance geometry and task."""
    from rtt.sim.task import default_task_v2, task_digest

    task = task or default_task_v2()
    kind = inst.kind.value
    if kind not in GRIPPED_FEATURE:
        raise ValueError(f"{inst.name} ({kind}) has no gripped feature: {NOT_GRIPPED.get(kind, '')}")
    key = (_instance_key(inst), task_digest(task))
    hit = _FEATURES.get(key)
    if hit is None:
        if kind == "part":
            from rtt.studio.templates.part import part_spec_of

            hit = GrippedFeature(part_spec_of(inst), None)
        else:
            hit = _template_feature(inst, task)
        if len(_FEATURES) > 64:
            _FEATURES.clear()
        _FEATURES[key] = hit
    return hit


def feature_part(inst: ObjectInstance, task: TaskV2 | None = None) -> PartSpec:
    """The :class:`~rtt.parts.types.PartSpec` of the feature a step grips on ``inst``."""
    return gripped_feature(inst, task).part


@functools.lru_cache(maxsize=4)
def _entrants() -> dict[str, Any]:
    from rtt.design.freeform.baselines import load_entrants_v2

    return {b.name: b for b in load_entrants_v2()}


def _design(entrant: str, feature: GrippedFeature) -> Any:
    """What the generator compiles: the entrant itself, or (a generated entrant on a template
    feature) the entrant's own :class:`~rtt.design.freeform.generate.PairSpec` with its pad band
    placed where the step grips (``tip_height``); everything else is the entrant's."""
    baseline = _entrants()[entrant]
    if baseline.kind == "stock" or feature.grip_tip_m is None:
        return baseline
    spec, _repairs = baseline.pair_spec()
    return replace(spec, tip_height=feature.grip_tip_m)


_PAIRS: dict[tuple[str, str], Any] = {}


def studio_pair(spec: EffectorSpec, task: TaskV2 | None = None) -> FingerPair:
    """The jaw pair of a Studio jaw: its entrant compiled by the frozen generator against the
    gripped feature (cached). Raises :class:`~rtt.effectors.types.InvalidEffector` with the
    generator's own reason when the pair cannot be built (an opening or seat rule, for example)."""
    from rtt.design.freeform.generate import build_pair
    from rtt.sim.fingers_v2 import check_pair
    from rtt.sim.task import default_task_v2, task_digest
    from rtt.studio.templates import TemplateUnavailable

    task = task or default_task_v2()
    key = (spec.effector_id, task_digest(task))
    hit = _PAIRS.get(key)
    if hit is not None:
        return hit
    t = jaw_target(spec)
    entrant = str(t["entrant"])
    try:
        feature = gripped_feature(target_instance(spec), task)
        pair = build_pair(_design(entrant, feature), feature.part, task)
        check_pair(pair)
    except TemplateUnavailable as exc:
        raise InvalidEffector(f"{t['object']}: {exc}") from exc
    except InvalidEffector:
        raise
    except ValueError as exc:  # InvalidDesign is a ValueError and carries the rule that failed
        raise InvalidEffector(f"{DISPLAY_NAMES[entrant]} for {t['object']}: {exc}") from exc
    if len(_PAIRS) > 128:
        _PAIRS.clear()
    _PAIRS[key] = pair
    return pair


def _pad_height(entrant: str) -> float:
    if entrant == "stock":
        from rtt.design.freeform.baselines import STOCK_PAD_HEIGHT

        return float(STOCK_PAD_HEIGHT)
    recipe = _entrants()[entrant].recipe
    return float(recipe.pad_height_mm) / 1000.0


def _stock_sources() -> tuple[AssetSource, ...]:
    """Licence records of the stock blade: its committed hulls (derived from the gripper's clamp
    mesh) and the licence text, from the gripper asset lock (attribution: credits only)."""
    from rtt.effectors.scene import REPO_ROOT
    from rtt.render.gripper import ASSET_DIR, load_lock

    lock = load_lock()
    root = ASSET_DIR.relative_to(REPO_ROOT).as_posix()
    files = lock["files"]

    def sha(rel: str) -> str:
        return hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest()

    clamp = files["meshes/clamp1_1.stl"]
    lic_name = next(k for k in files if k.upper().startswith("LICENSE"))
    lic = files[lic_name]
    common = {
        "upstream_url": lock["repository"],
        "upstream_commit": lock["commit"],
        "license": lock["license"],
        "copyright": lock["attribution"].split(",")[0].strip(),
        "attribution": lock["attribution"],
    }
    return (
        AssetSource(
            path=f"{root}/stock_blade_hulls.npz",
            sha256=sha(f"{root}/stock_blade_hulls.npz"),
            upstream_path=clamp["path"],
            upstream_sha256=clamp["sha256"],
            changes=(
                "convex decomposition of the clamp blade mesh (CoACD 0.02, seed 0), left and right jaws",
            ),
            **common,
        ),
        AssetSource(
            path=f"{root}/{lic_name}",
            sha256=sha(f"{root}/{lic_name}"),
            upstream_path=lic["path"],
            upstream_sha256=lic["sha256"],
            **common,
        ),
    )


# --------------------------------------------------------------------------- primitive


@dataclass(frozen=True)
class PartJawPrimitive:
    """The v2 grip ramp as a synergy (module docstring): one control, the grip target. ``plan``
    puts the jaws' lowest point at the pair's designed tip height for the task it is given."""

    spec: EffectorSpec
    ctrl_open: float
    ctrl_closed: float
    name: str = "jaw"

    def plan(self, part: PartSpec, placement: Placement, task: TaskV2, build: EffectorBuild) -> PrimitivePlan:
        if build.n_actuators != 1:
            raise ValueError("a Studio jaw has one grip actuator")
        tip = float(studio_pair(self.spec, task).tip_height)
        return PrimitivePlan(tip_height=tip, note="the jaw pair's designed tip height")

    def command(
        self, t: float, phase: Phase, closure: float, model: mujoco.MjModel, data: mujoco.MjData, scene: Any
    ) -> np.ndarray:
        return np.array([self.ctrl_open + float(closure) * (self.ctrl_closed - self.ctrl_open)])


# --------------------------------------------------------------------------- the family


class StudioPartJaw(ParallelJawReference):
    """The frozen reference plus the Studio build path for part-shaped jaws (module docstring)."""

    def build(self, ctx: BuildContext) -> EffectorBuild:
        spec = ctx.effector
        if not is_part_jaw(spec):
            return super().build(ctx)  # a pre-registered entrant: the frozen v2 episode only
        from rtt.sim.fingers_v2 import jaw_bodies, lowest_point
        from rtt.sim.scene import FINGER_RGBA, PALM_RGBA, _add_position_actuator

        task = ctx.task
        g, c = task.gripper, task.controller
        t = jaw_target(spec)
        pair = studio_pair(spec, task)
        left, right = jaw_bodies(pair)
        p = f"{ctx.prefix}{JAW_PREFIX}"
        hx, hy, hz = (float(v) for v in g.palm_half_size)
        housing = ctx.mount.add_body(name=f"{p}housing", pos=[0.0, 0.0, ctx.mount_z - hz])
        housing.add_geom(
            name=f"{p}housing",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[hx, hy, hz],
            mass=float(g.palm_mass),
            rgba=list(PALM_RGBA),
        )
        joints = []
        for side_name, jaw, side in (("left", left, 1.0), ("right", right, -1.0)):
            body = housing.add_body(name=f"{p}{side_name}", pos=[0.0, side * float(g.open_half_gap), -hz])
            body.add_joint(
                name=f"{p}{side_name}",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=[0.0, -side, 0.0],
                range=[0.0, float(g.stroke)],
                armature=float(g.finger_armature),
                damping=float(g.finger_damping),
            )
            joints.append(f"{p}{side_name}")
            body.explicitinertial = True
            body.mass = jaw.props.mass
            body.ipos = jaw.props.com.tolist()
            body.fullinertia = jaw.props.fullinertia()
            if jaw.boxes:  # the v1 box approximation of the stock blade (only when its hulls are missing)
                for i, b in enumerate(jaw.boxes):
                    body.add_geom(
                        name=f"{p}{side_name}_{i:02d}",
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=list(b.half_size),
                        pos=list(b.center),
                        quat=list(b.quat_wxyz),
                        rgba=list(FINGER_RGBA),
                    )
            for i, verts in enumerate(jaw.pieces):
                mesh = f"{p}{side_name}_piece{i:02d}"
                ctx.spec.add_mesh(name=mesh, uservert=np.asarray(verts, dtype=np.float64).ravel().tolist())
                body.add_geom(
                    name=f"{p}{side_name}_{i:02d}",
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=mesh,
                    rgba=list(FINGER_RGBA),
                )
        tendon = ctx.spec.add_tendon(name=f"{p}grip")
        for j in joints:
            tendon.wrap_joint(j, 0.5)
        ctx.spec.add_equality(
            name=f"{p}coupling",
            type=mujoco.mjtEq.mjEQ_JOINT,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            name1=joints[0],
            name2=joints[1],
            solref=list(g.equality_solref),
        )
        _add_position_actuator(
            ctx.spec,
            f"{p}grip",
            f"{p}grip",
            float(g.grip_kp),
            float(g.grip_kv),
            float(g.grip_force),
            (float(c.open_target), float(g.stroke + g.grip_overdrive)),
            trntype=mujoco.mjtTrn.mjTRN_TENDON,
        )
        low = lowest_point(pair)
        root = ctx.mount_z - 2.0 * hz
        y_left = float(min(p_[:, 1].min() for p_ in pair.left.pieces))
        y_right = float(max(p_[:, 1].max() for p_ in pair.right.pieces))
        opening = (float(g.open_half_gap) + y_left) - (-float(g.open_half_gap) + y_right)
        entrant = str(t["entrant"])
        what = DISPLAY_NAMES[entrant].split(", ", 1)[1]
        shaped = f"{t['object']} ({t['kind']} {t['handle']})"
        force = float(g.grip_force)
        notes = [
            PRODUCT_MODE_LABEL,
            (
                f"Studio build of the v2 {what} on the open 84 mm parallel gripper, shaped for {shaped}; "
                "not the pre-registered result"
            ),
            (
                f"jaw force {0.5 * force:g} N per jaw ({force:g} N on the mean-closure servo, the sustained "
                "value of the gripper's protected servo); stroke 42 mm per jaw"
            ),
            *gripped_feature(target_instance(spec), task).notes,
            *pair.auto_repairs,
        ]
        return EffectorBuild(
            spec=spec,
            bodies=(f"{p}housing", f"{p}left", f"{p}right"),
            actuators=(f"{p}grip",),
            ctrl_open=(float(c.open_target),),
            reach=2.0 * hz + low,
            tcp=(0.0, 0.0, root - low + 0.5 * _pad_height(entrant)),
            actuation={
                "n_actuators": 1,
                "linear_force_n_max": 0.5 * float(g.grip_force),
                "grip_force_n": float(g.grip_force),
                "opening_mm": round(1000.0 * opening, 2),
                "stroke_mm": round(1000.0 * float(g.stroke), 2),
            },
            sources=_stock_sources() if entrant == "stock" else (),
            notes=tuple(notes),
            max_opening=opening,
        )

    def primitive(self, spec: EffectorSpec, build: EffectorBuild) -> PartJawPrimitive:
        if not is_part_jaw(spec):
            return super().primitive(spec, build)
        from rtt.sim.task import default_task_v2

        g = default_task_v2().gripper  # the closed target saturates the servo; ctrlrange clips it
        return PartJawPrimitive(
            spec=spec, ctrl_open=float(build.ctrl_open[0]), ctrl_closed=float(g.stroke + g.grip_overdrive)
        )

    def criterion(self, spec: EffectorSpec, build: EffectorBuild) -> ContactCriterion:
        if not is_part_jaw(spec):
            return super().criterion(spec, build)
        return ContactCriterion(name="jaw_contacts")


register_family(StudioPartJaw(), replace=True)


# --------------------------------------------------------------------------- worker processes


def _restore_spec(state: dict[str, Any]) -> EffectorSpec:
    """Unpickle a Studio jaw spec (importing this module registered :class:`StudioPartJaw`)."""
    spec = object.__new__(EffectorSpec)
    spec.__dict__.update(state)
    return spec


def _reduce_spec(spec: EffectorSpec) -> Any:
    if is_part_jaw(spec):
        return (_restore_spec, (dict(spec.__dict__),))
    return spec.__reduce_ex__(4)  # every other spec pickles exactly as before


ForkingPickler.register(EffectorSpec, _reduce_spec)


# --------------------------------------------------------------------------- print files


def jaw_print_files(
    spec: EffectorSpec, out_dir: str | Path, *, prefix: str = "", task: TaskV2 | None = None
) -> dict[str, dict[str, Any]]:
    """Write a Studio jaw's printed solids into ``out_dir`` through the v2 print pack (module
    docstring): ``<prefix>left.stl``, ``<prefix>right.stl`` (each jaw upright for printing) and
    ``<prefix>mount_left.stl``, ``<prefix>mount_right.stl`` (each jaw fused with its carriage
    block). Returns file name -> facts (sha256, volume, watertight, what it is). Refuses the
    stock blade (``ValueError``): it is the gripper's own part, not a generated jaw."""
    import trimesh

    from rtt.design.freeform.export import stl_bytes
    from rtt.design.freeform.generate import build_pair_full
    from rtt.part.pack import make_print_pack_v2
    from rtt.sim.task import default_task_v2

    task = task or default_task_v2()
    t = jaw_target(spec)
    entrant = str(t["entrant"])
    if entrant == "stock":
        raise ValueError(
            "the stock blade is the open-source gripper's own part (CC BY 4.0): it is not printed here"
        )
    pair = studio_pair(spec, task)
    feature = gripped_feature(target_instance(spec), task)
    build = build_pair_full(_design(entrant, feature), feature.part, task)
    if build.pair.geometry_hash != pair.geometry_hash:
        raise RuntimeError("the print build differs from the simulated jaw pair")
    left = stl_bytes(build.left_mesh.vertices, build.left_mesh.faces)
    right = stl_bytes(build.right_mesh.vertices, build.right_mesh.faces)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    facts: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        make_print_pack_v2(
            pair,
            left,
            right,
            tmp,
            recipe=build.recipe_record(),
            part={"object": t["object"], "kind": t["kind"], "feature": t["handle"]},
            name="jaws",
        )
        src = Path(tmp) / "jaws"
        for name, what in (
            ("left.stl", "left jaw, upright for printing (root plane on the bed, tip up)"),
            ("right.stl", "right jaw, upright for printing (its own part, never the left one mirrored)"),
            ("mount_left.stl", "left jaw fused with its carriage block for the guide rods"),
            ("mount_right.stl", "right jaw fused with its carriage block for the guide rods"),
        ):
            data = (src / name).read_bytes()
            (out / f"{prefix}{name}").write_bytes(data)
            mesh = trimesh.load(out / f"{prefix}{name}", file_type="stl", force="mesh", process=True)
            mesh.merge_vertices()
            facts[f"{prefix}{name}"] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "watertight": bool(mesh.is_watertight and mesh.is_volume and mesh.body_count == 1),
                "volume_mm3": round(float(mesh.volume), 3),
                "what": what,
            }
    return facts
