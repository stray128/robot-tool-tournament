"""Combination search of the Studio (track S5, product mode): from candidate tools to finalist setups.

**Product mode, not part of the pre-registered experiment.** Every record carries
:data:`~rtt.studio.types.PRODUCT_MODE_LABEL`. The pipeline (``docs/STUDIO_SEARCH.md``):

1. **Candidates** (:func:`default_candidates`): every preset of every effector family that loads,
   plus the open library's entrants when the library package is installed, plus any
   caller-supplied effectors. Each carries capability tags (the primitives it can serve).
2. **Capability pruning** (:func:`prune`), each with a reason: a family that cannot build into the
   Studio scene; a tool heavier than the arm payload limit (:data:`ARM_PAYLOAD_KG`, [assumed]);
   a step whose primitive the tool's mechanism cannot serve; a tool whose opening, cup or reach
   does not fit **every** object it would touch (orchestrator ruling 4: the family's own build
   check is run against each picked part, plug and threaded member, not just the first part).
   ``grasp``, ``suction`` and ``wrap`` are one **acquire** class (ruling 6): on a step with any of
   them, every acquire-capable tool competes with its own mechanism (the Studio's pick controllers
   drive the head's own v1 primitive), and the screen records the primitive it actually used
   (:func:`acquire_primitive`).
3. **Per-step screening** (:func:`screen`): the steps are grouped into **hold chains**
   (consecutive steps joined by ``hold_after``, which one tool must do). Each surviving candidate
   runs each chain as a single-tool setup on the chain's sub-task, on the first
   ``n_screen`` conditions of the grid, starting from the world state the best tool of the earlier
   chains left behind under that condition (:func:`rtt.studio.runner.run_segments`, the contract's
   tool-change model). The **success matrix** is candidates x steps.
4. **Combination search** (:func:`combine`): ``single`` setups, ``tool_change`` setups (each chain's
   top candidates, 2 to 4 distinct tools) and ``multi_head`` setups (pairs of the chains' top
   candidates) are scored from the matrix (product of the chain rates, summed durations plus
   change times, complexity). Multi-tool heads get a **bracket geometry search** within the
   :class:`~rtt.studio.types.HeadMount` bounds and a **swept-volume check** (ruling 2): the idle
   head and the bracket must clear every other object and the table at each step's engagement
   pose, over the headings a controller may use (a twist turns the whole bracket), and both heads
   must clear everything while the head switch swings them at the home pose. A rejected geometry
   is reported with its reason. Setups over the payload limit are refused (ruling 5).
5. **Finalists**: the best predicted setups (3 to 5), always including the best single
   off-the-shelf effector, run the full task on the whole condition grid
   (:func:`rtt.studio.runner.run_full_tasks`); :mod:`rtt.studio.recommend` ranks them after
   :func:`~rtt.studio.fairness.assert_fair`.

:func:`run_task_tournament` runs all of it and returns a :class:`TournamentReport` (the plain
function behind the agent tool of the same name).
"""

from __future__ import annotations

import hashlib
import itertools
import math
import statistics
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal

import mujoco
import numpy as np

from rtt.effectors.registry import available_families, get_family
from rtt.effectors.types import (
    PRODUCT_MODE_LABEL,
    EffectorFamily,
    EffectorSpec,
    InvalidEffector,
    canonical_json,
)
from rtt.sim.task import TaskV2, default_task_v2
from rtt.studio.episode import HOME_HEIGHT, StudioCondition, WorldState
from rtt.studio.plan import ControllerSet
from rtt.studio.runner import (
    MAX_WORKERS,
    SegmentJob,
    WorkerPool,
    condition_dict,
    condition_grid,
    run_full_tasks,
    run_segment_job,
)
from rtt.studio.templates import TemplateUnavailable
from rtt.studio.types import (
    CONTRACT_VERSION,
    MAX_TOOLS,
    PRESENT_S,
    BracketSpec,
    CertificateRef,
    ComplexityMetrics,
    HeadMount,
    ObjectInstance,
    PrimitiveKind,
    Setup,
    SetupKind,
    SetupResult,
    Step,
    SuccessCriterion,
    TaskResult,
    TaskSpec,
    TemplateKind,
    ToolChanger,
)

__all__ = [
    "ACQUIRE",
    "ARM_PAYLOAD_KG",
    "MECHANISM_CAPABILITIES",
    "BracketGeometry",
    "Candidate",
    "Cell",
    "Pruned",
    "Screen",
    "SetupPlan",
    "ToolInfo",
    "TournamentReport",
    "acquire_primitive",
    "bracket_search",
    "candidate_of",
    "combine",
    "default_candidates",
    "designed_candidates",
    "hold_chains",
    "last_link_intrusion",
    "mechanism_of",
    "probe_tool",
    "prune",
    "run_task_tournament",
    "screen",
    "sub_task",
    "swept_volume_problems",
]

#: The acquire class (ruling 6): a step with any of these is open to every acquire-capable tool.
ACQUIRE: Final = frozenset({PrimitiveKind.GRASP, PrimitiveKind.SUCTION, PrimitiveKind.WRAP})
#: Largest tool stack the arm may carry (every head, bracket or changer plates) [assumed: a small
#: collaborative arm's rated payload]. Setups above it are refused, never run.
ARM_PAYLOAD_KG: Final = 3.0
#: Margin a gripper's opening keeps over the width it closes on (the families' own margin).
OPENING_MARGIN_MM: Final = 5.0
#: Clearance each side of a handle a pre-shaped hand needs (the S2 pre-shape rule, [assumed]).
HOOK_CLEARANCE_MM: Final = 6.0
#: The swept-volume check keeps the idle head and bracket this far from objects and the table [assumed].
SWEEP_CLEARANCE_M: Final = 0.005
#: The flange's mounting face in the flange frame (m): only the hub may reach above it (the
#: flange and the arm's last link are there) [assumed keep-out].
FLANGE_FACE_M: Final = -0.012
#: Radius of the arm's last link above the flange: the flange's plus 10 mm [assumed].
LAST_LINK_RADIUS_M: Final = 0.0315 + 0.010
#: Default screening and search sizes.
N_SCREEN: Final = 3
TOP_K: Final = 3
N_FINALISTS: Final = 4
MULTI_HEAD_PAIRS: Final = 6
_MAX_ASSIGNMENTS: Final = 4000

Mechanism = Literal["fingers", "suction", "soft_arm"]

#: What each mechanism can serve [assumed, documented in docs/STUDIO_SEARCH.md section 2].
MECHANISM_CAPABILITIES: Final[Mapping[str, frozenset[PrimitiveKind]]] = {
    "fingers": frozenset(
        {
            PrimitiveKind.GRASP,
            PrimitiveKind.HOOK_PULL,
            PrimitiveKind.TWIST,
            PrimitiveKind.INSERT,
            PrimitiveKind.PUSH,
        }
    ),
    "suction": frozenset(
        {PrimitiveKind.SUCTION, PrimitiveKind.TWIST, PrimitiveKind.INSERT, PrimitiveKind.PUSH}
    ),
    "soft_arm": frozenset({PrimitiveKind.WRAP, PrimitiveKind.HOOK_PULL, PrimitiveKind.TWIST}),
}
_NO_CAPABILITY: Final[Mapping[tuple[str, PrimitiveKind], str]] = {
    ("suction", PrimitiveKind.HOOK_PULL): "a suction cup has no hook or fingers to engage a handle or lip",
    ("soft_arm", PrimitiveKind.INSERT): "a soft arm cannot hold a member stiffly along the seat axis",
    ("soft_arm", PrimitiveKind.PUSH): "a slack soft arm has no stiff face to push with",
}
#: Library capability tags (``rtt.library.taxonomy``) -> the primitives they serve.
_LIBRARY_TAGS: Final[Mapping[str, PrimitiveKind]] = {
    "pinch": PrimitiveKind.GRASP,
    "power": PrimitiveKind.GRASP,
    "suction": PrimitiveKind.SUCTION,
    "wrap": PrimitiveKind.WRAP,
    "hook": PrimitiveKind.HOOK_PULL,
    "push": PrimitiveKind.PUSH,
    "twist": PrimitiveKind.TWIST,
    "insert": PrimitiveKind.INSERT,
}


# --------------------------------------------------------------------------- candidates


def mechanism_of(spec: EffectorSpec) -> Mechanism:
    """``suction`` (cups), ``soft_arm`` (tentacles) or ``fingers`` (everything else)."""
    if spec.family is EffectorFamily.SUCTION:
        return "suction"
    if spec.family is EffectorFamily.SOFT_ARM:
        return "soft_arm"
    return "fingers"


def acquire_primitive(spec: EffectorSpec) -> PrimitiveKind:
    """The acquire primitive a tool actually performs (ruling 6): its own mechanism's."""
    return {
        "suction": PrimitiveKind.SUCTION,
        "soft_arm": PrimitiveKind.WRAP,
        "fingers": PrimitiveKind.GRASP,
    }[mechanism_of(spec)]


@dataclass(frozen=True, eq=False)
class Candidate:
    """One candidate tool: its spec, where it came from and what it can serve.

    ``off_the_shelf``: a preset or library item (not a designed recipe); the best single
    off-the-shelf effector is always reported next to the recommendation. ``item``: the
    library item key, when it came from the library.
    """

    spec: EffectorSpec
    source: Literal["preset", "library", "designed", "manual"]
    capabilities: frozenset[PrimitiveKind]
    off_the_shelf: bool = True
    item: str | None = None

    @property
    def effector_id(self) -> str:
        return self.spec.effector_id

    @property
    def name(self) -> str:
        return self.spec.display_name

    @property
    def mechanism(self) -> Mechanism:
        return mechanism_of(self.spec)

    def to_dict(self) -> dict[str, Any]:
        return {
            "effector_id": self.effector_id,
            "display_name": self.name,
            "family": self.spec.family.value,
            "model": self.spec.model,
            "primitive": self.spec.primitive,
            "source": self.source,
            "off_the_shelf": self.off_the_shelf,
            "item": self.item,
            "mechanism": self.mechanism,
            "capabilities": sorted(k.value for k in self.capabilities),
            "acquire_primitive": acquire_primitive(self.spec).value,
            "fidelity": self.spec.fidelity.value,
        }


def candidate_of(
    spec: EffectorSpec,
    source: Literal["preset", "library", "designed", "manual"] = "manual",
    *,
    tags: Iterable[str] | None = None,
    item: str | None = None,
) -> Candidate:
    """A candidate from a spec: capabilities from library ``tags`` when given, else its mechanism."""
    if tags is not None:
        caps = {_LIBRARY_TAGS[t] for t in tags if t in _LIBRARY_TAGS}
    else:
        caps = set(MECHANISM_CAPABILITIES[mechanism_of(spec)])
    if caps & ACQUIRE:
        caps |= ACQUIRE  # ruling 6: one acquire class
    designed = spec.origin in ("nemotron", "random") or spec.family is EffectorFamily.DESIGNED
    return Candidate(
        spec=spec,
        source="designed" if designed and source in ("preset", "manual") else source,
        capabilities=frozenset(caps),
        off_the_shelf=not designed,
        item=item,
    )


def _library_candidates() -> list[Candidate]:
    """The open library's entrants, when the library package is installed (track S3)."""
    try:
        from rtt.library.entrants import item_specs  # type: ignore[import-not-found]
        from rtt.library.records import item_keys, read_item  # type: ignore[import-not-found]
    except ImportError:
        return []
    out = []
    for key in item_keys():
        item = read_item(key)
        if str(getattr(item.rating, "value", item.rating)) != "green":
            continue  # amber and red items never compete or ship (the library's licence gate)
        for spec in item_specs(key):
            out.append(candidate_of(spec, "library", tags=item.capabilities, item=key))
    return out


#: The recorded Nemotron designs (the effector layer's morphology smoke run, cached: no spend).
RECORDED_DESIGNS: Final = "src/rtt/effectors/morphology/recorded/smoke-2026-10-17/manifest.json"


def designed_candidates(*, recorded: bool = True) -> list[Candidate]:
    """Designed effectors: the designed family's hand-written reference recipes and, with
    ``recorded``, the Nemotron designs of the recorded morphology run (read from its manifest;
    no model is called). Designs can fill any slot; they are never "off the shelf"."""
    import json

    from rtt.effectors.designed import reference_recipes
    from rtt.effectors.scene import REPO_ROOT
    from rtt.effectors.types import MorphologyRecipe

    recipes = list(reference_recipes())
    path = REPO_ROOT / RECORDED_DESIGNS
    if recorded and path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        for d in data.get("arms", {}).get("nemotron", {}).get("designs", []):
            r = d["recipe"]
            recipes.append(
                MorphologyRecipe(
                    family=r["family"],
                    kind=r["kind"],
                    params=r["params"],
                    hypothesis=r.get("hypothesis", ""),
                    origin=r.get("origin", "nemotron"),
                    parents=tuple(r.get("parents", ())),
                    schema=r.get("schema", "rtt-morphology/1"),
                )
            )
    out = []
    for recipe in recipes:
        spec = get_family(recipe.family).spec_from_recipe(recipe)
        out.append(candidate_of(spec, "designed"))
    return out


def default_candidates(*, library: bool = True, designed: bool = True) -> tuple[Candidate, ...]:
    """Every family preset (in family order), then the library's entrants, then the designed
    effectors (:func:`designed_candidates`), deduplicated by effector id."""
    out: list[Candidate] = []
    seen: set[str] = set()

    def add(c: Candidate) -> None:
        if c.effector_id not in seen:
            seen.add(c.effector_id)
            out.append(c)

    for fam in available_families():
        for spec in get_family(fam).presets():
            add(candidate_of(spec, "preset"))
    if library:
        for c in _library_candidates():
            add(c)
    if designed:
        for c in designed_candidates():
            add(c)
    return tuple(out)


# --------------------------------------------------------------------------- tool probes


@dataclass(frozen=True)
class ToolInfo:
    """A tool as built in the Studio scene (measured on a probe task), or why it cannot be."""

    effector_id: str
    buildable: bool
    reason: str | None = None
    mass_kg: float = math.nan
    n_actuators: int = 0
    max_opening_mm: float | None = None
    reach_mm: float = math.nan
    tcp_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    actuation: Mapping[str, float] = field(default_factory=dict)
    licences: tuple[str, ...] = ()
    amber: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "effector_id": self.effector_id,
            "buildable": self.buildable,
            "reason": self.reason,
            "mass_kg": None if math.isnan(self.mass_kg) else round(self.mass_kg, 5),
            "n_actuators": self.n_actuators,
            "max_opening_mm": None if self.max_opening_mm is None else round(self.max_opening_mm, 2),
            "reach_mm": None if math.isnan(self.reach_mm) else round(self.reach_mm, 2),
            "actuation": dict(self.actuation),
            "licences": list(self.licences),
            "amber": self.amber,
        }


def _probe_object(name: str, kind: str, **params: Any) -> ObjectInstance:
    return ObjectInstance.of(name, kind, "Probe part", **params)


def _probe_task(obj: ObjectInstance) -> TaskSpec:
    """One pick of ``obj`` (a part): the scene builds the tool against exactly that part."""
    step = Step.of(
        "probe_pick", "grasp", obj.name, "part", "Probe pick", [SuccessCriterion("lift_mm", at_least=1.0)]
    )
    return TaskSpec("Probe", "", (obj,), (step,), source="test")


#: Probe parts a tool is measured on, in order (the first it can be built for): a 20 mm cube, the
#: library block (what a tool that picks nothing is built against), a 50 mm cube.
_PROBE_PARTS: Final = (
    _probe_object("probe", "part", shape="box", size_x_mm=20, size_y_mm=20, size_z_mm=20, mass_g=50),
    ObjectInstance.of("probe", "part", "Probe part", ref="library:block", shape="mesh"),
    _probe_object("probe", "part", shape="box", size_x_mm=50, size_y_mm=50, size_z_mm=50, mass_g=200),
)
_PROBE_PART = _PROBE_PARTS[0]
_TOOLS: dict[tuple[str, str], ToolInfo] = {}
_FITS: dict[tuple[str, str], str | None] = {}


def probe_scene(spec: EffectorSpec, task: TaskV2 | None = None) -> tuple[Any, str | None]:
    """The tool alone in a Studio scene on the first probe part it can be built for, or ``None``
    and why it cannot be built at all."""
    from rtt.studio.scene import build_studio_scene

    task = task or default_task_v2()
    reason = None
    for part in _PROBE_PARTS:
        probe = _probe_task(part)
        try:
            return build_studio_scene(Setup.single(spec, probe.step_ids), probe, 0, task=task), None
        except NotImplementedError:
            return (
                None,
                "this family runs only inside the frozen v2 episode; it does not build into the Studio scene",
            )
        except InvalidEffector as exc:
            reason = reason or f"cannot be built: {exc}"
    return None, reason


def probe_tool(spec: EffectorSpec, task: TaskV2 | None = None) -> ToolInfo:
    """Build ``spec`` alone in the Studio scene (:func:`probe_scene`) and measure it (cached)."""
    task = task or default_task_v2()
    key = (spec.effector_id, _task_key(task))
    if key in _TOOLS:
        return _TOOLS[key]
    scene, reason = probe_scene(spec, task)
    if scene is None:
        info = ToolInfo(spec.effector_id, False, reason)
    else:
        b = scene.active.build
        info = ToolInfo(
            spec.effector_id,
            True,
            mass_kg=float(b.mass or scene.tool_mass_kg),
            n_actuators=b.n_actuators,
            max_opening_mm=None if b.max_opening is None else 1000.0 * float(b.max_opening),
            reach_mm=1000.0 * float(b.reach),
            tcp_mm=tuple(round(1000.0 * v, 3) for v in scene.active.tcp_wrist),  # type: ignore[arg-type]
            actuation=dict(b.actuation),
            licences=tuple(sorted({s.license for s in b.sources})),
            amber=any(s.status == "amber" for s in b.sources),
        )
    _TOOLS[key] = info
    return info


def _task_key(task: TaskV2) -> str:
    from rtt.sim.task import task_digest

    return task_digest(task)


def _fits_part(spec: EffectorSpec, obj: ObjectInstance, task: TaskV2) -> str | None:
    """The family's own build check against one part (``None``: it fits)."""
    from rtt.studio.scene import build_studio_scene

    probe = ObjectInstance(
        "probe", obj.kind, obj.label, dict(obj.params), ref=obj.ref
    )  # same geometry, the probe's name
    key = (spec.effector_id, canonical_json(probe.canonical_dict()) + _task_key(task))
    if key not in _FITS:
        pt = _probe_task(probe)
        try:
            build_studio_scene(Setup.single(spec, pt.step_ids), pt, 0, task=task)
            _FITS[key] = None
        except InvalidEffector as exc:
            _FITS[key] = str(exc)
        except TemplateUnavailable as exc:
            _FITS[key] = f"the part has no physics yet ({exc})"
    return _FITS[key]


def _slug_instance(diameter_mm: float, length_mm: float) -> ObjectInstance:
    """A cylinder part of a plug's or threaded member's size (what the pick plans with)."""
    return _probe_object(
        "probe",
        "part",
        shape="cylinder",
        size_x_mm=max(5.0, diameter_mm),
        size_y_mm=max(5.0, diameter_mm),
        size_z_mm=max(2.0, length_mm),
        mass_g=50,
    )


# --------------------------------------------------------------------------- pruning


@dataclass(frozen=True)
class Pruned:
    """A candidate left out of a step (``step_id``) or out of everything (``None``), and why."""

    effector_id: str
    display_name: str
    step_id: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "effector_id": self.effector_id,
            "display_name": self.display_name,
            "step_id": self.step_id,
            "reason": self.reason,
        }


def hold_chains(task_spec: TaskSpec) -> tuple[tuple[str, ...], ...]:
    """Consecutive steps joined by ``hold_after`` (one tool must do a whole chain)."""
    out: list[list[str]] = []
    for i, s in enumerate(task_spec.steps):
        if i > 0 and task_spec.steps[i - 1].hold_after:
            out[-1].append(s.id)
        else:
            out.append([s.id])
    return tuple(tuple(c) for c in out)


def capability_problem(cand: Candidate, step: Step) -> str | None:
    """Why ``cand`` cannot serve ``step``'s primitive (``None``: it can)."""
    if step.primitive in ACQUIRE:
        if cand.capabilities & ACQUIRE:
            return None
        return f"it cannot pick or hold a part ({step.primitive.value} step)"
    if step.primitive in cand.capabilities:
        return None
    return _NO_CAPABILITY.get(
        (cand.mechanism, step.primitive), f"its capability tags do not include {step.primitive.value}"
    )


def _cup_mm(spec: EffectorSpec) -> float | None:
    if mechanism_of(spec) != "suction":
        return None
    d = spec.params.get("diameter_mm")
    return float(d) if isinstance(d, int | float) else None


def fit_problem(cand: Candidate, info: ToolInfo, step: Step, task_spec: TaskSpec, task: TaskV2) -> str | None:
    """Whether the tool fits the object the step touches (ruling 4), ``None`` when it does."""
    obj = task_spec.object(step.object)
    p = obj.params
    cup = _cup_mm(cand.spec)
    opening = info.max_opening_mm
    kind = obj.kind
    picks_plug = step.primitive is PrimitiveKind.INSERT and kind is TemplateKind.PORT_PLUG
    if step.primitive is PrimitiveKind.INSERT and kind is TemplateKind.THREADED_JOINT:
        d = float(p["diameter_mm"])  # fingers grip a member in place across its flats; cups press its top
        if opening is not None and opening - OPENING_MARGIN_MM < d:
            return f"it opens {opening:.0f} mm; the {p['member']} {obj.name} is {d:g} mm across"
        return None
    if step.primitive in ACQUIRE or picks_plug:  # an insert picks its plug off the table
        if kind is TemplateKind.PART:
            if cup is not None and p["shape"] in ("box", "cylinder"):
                face = (
                    min(float(p["size_x_mm"]), float(p["size_y_mm"]))
                    if p["shape"] == "box"
                    else float(p["size_x_mm"])
                )
                if cup > face:
                    return f"its {cup:g} mm cup is wider than the {face:g} mm top face of {obj.name}"
            reason = _fits_part(cand.spec, obj, task)
            return None if reason is None else f"{obj.name}: {reason}"
        if kind is TemplateKind.PORT_PLUG:
            d, length = float(p["plug_diameter_mm"]), float(p["plug_length_mm"])
            if cup is not None and cup > d:
                return f"its {cup:g} mm cup is wider than the {d:g} mm top of the plug {obj.name}"
        elif kind is TemplateKind.THREADED_JOINT:
            d, length = float(p["diameter_mm"]), float(p["length_mm"])
            if cup is not None and p["member"] != "cap":
                return f"{obj.name} (a {p['member']}) has an open top; a cup cannot seal on it"
            if cup is not None and cup > d:
                return f"its {cup:g} mm cup is wider than the {d:g} mm cap {obj.name}"
        else:
            return None
        reason = _fits_part(cand.spec, _slug_instance(d, length), task)
        return None if reason is None else f"{obj.name}: {reason}"
    if step.primitive is PrimitiveKind.TWIST:
        if kind is TemplateKind.THREADED_JOINT:
            d = float(p["diameter_mm"])
            if opening is not None and opening - OPENING_MARGIN_MM < d:
                return f"it opens {opening:.0f} mm; the {p['member']} {obj.name} is {d:g} mm across"
            if cup is not None and p["member"] != "cap":
                return f"{obj.name} (a {p['member']}) has an open top; a cup cannot seal on it"
        if kind is TemplateKind.QUARTER_TURN_LATCH and cup is not None:
            if p["head"] == "slot":
                return f"the slot across the head of {obj.name} breaks a cup's seal"
            fin = max(4.0, 0.2 * float(p["head_diameter_mm"]))
            if cup > fin:
                return f"its {cup:g} mm cup is wider than the {fin:g} mm wing of {obj.name}"
        return None
    if step.primitive is PrimitiveKind.HOOK_PULL and opening is not None:
        if kind is TemplateKind.HINGED_LID:
            feature = 10.0 if p["feature"] == "handle" else float(p["thickness_mm"])
        elif kind is TemplateKind.DRAWER:
            feature = {"bar": 10.0, "knob": 12.0, "lip": 4.0}[str(p["handle"])]
        else:
            return None
        need = feature + 2.0 * HOOK_CLEARANCE_MM
        if opening < need:
            return (
                f"it opens {opening:.0f} mm; pre-shaping around the {feature:g} mm feature needs {need:g} mm"
            )
    return None


def prune(
    task_spec: TaskSpec,
    candidates: Sequence[Candidate],
    *,
    task: TaskV2 | None = None,
    payload_kg: float = ARM_PAYLOAD_KG,
) -> tuple[dict[int, list[Candidate]], list[Pruned], dict[str, ToolInfo]]:
    """Capability pruning (module docstring): chain index -> the candidates that may run it,
    every pruned (candidate, step) with its reason, and the tools' probe measurements."""
    task = task or default_task_v2()
    chains = hold_chains(task_spec)
    eligible: dict[int, list[Candidate]] = {j: [] for j in range(len(chains))}
    pruned: list[Pruned] = []
    infos: dict[str, ToolInfo] = {}
    for cand in candidates:
        info = probe_tool(cand.spec, task)
        infos[cand.effector_id] = info
        if not info.buildable:
            pruned.append(Pruned(cand.effector_id, cand.name, None, info.reason or "cannot be built"))
            continue
        if info.amber:
            pruned.append(
                Pruned(cand.effector_id, cand.name, None, "an amber licence: never competes or ships")
            )
            continue
        if info.mass_kg > payload_kg:
            pruned.append(
                Pruned(
                    cand.effector_id,
                    cand.name,
                    None,
                    f"it weighs {info.mass_kg:.2f} kg; the arm carries at most {payload_kg:g} kg [assumed]",
                )
            )
            continue
        for j, chain in enumerate(chains):
            ok = True
            if not any(task_spec.step(sid).primitive in ACQUIRE for sid in chain):
                reason = _fits_part(cand.spec, _PROBE_PARTS[1], task)
                if reason is not None:  # a tool that picks nothing is built against the block
                    pruned.append(
                        Pruned(cand.effector_id, cand.name, chain[0], f"built against the block: {reason}")
                    )
                    continue
            for sid in chain:
                step = task_spec.step(sid)
                reason = capability_problem(cand, step) or fit_problem(cand, info, step, task_spec, task)
                if reason is not None:
                    pruned.append(Pruned(cand.effector_id, cand.name, sid, reason))
                    ok = False
                    break
            if ok:
                eligible[j].append(cand)
    return eligible, pruned, infos


# --------------------------------------------------------------------------- screening


def sub_task(task_spec: TaskSpec, step_ids: Sequence[str]) -> TaskSpec:
    """The task restricted to ``step_ids`` (every object kept; dependencies on dropped steps removed)."""
    keep = set(step_ids)
    steps = tuple(
        replace(s, after=tuple(a for a in s.after if a in keep)) for s in task_spec.steps if s.id in keep
    )
    return TaskSpec(
        task_spec.title,
        task_spec.description,
        task_spec.objects,
        steps,
        assumptions=task_spec.assumptions,
        source=task_spec.source,
    )


@dataclass(frozen=True)
class Cell:
    """One candidate on one step in the screen (one entry of the success matrix).

    ``primitive_used``: the primitive the tool actually performed (its own acquire primitive on a
    grasp, suction or wrap step, ruling 6). ``context``: ``start`` (the task's first chain) or
    ``carried`` (the state the earlier chains' best tools left); ``context_ok``: conditions whose
    earlier chains all succeeded in that context.
    """

    effector_id: str
    step_id: str
    chain: int
    primitive: str
    primitive_used: str
    n: int
    successes: int
    failure_modes: Mapping[str, int]
    median_duration_s: float
    fidelity: str
    caveats: tuple[str, ...]
    controller: str
    context: Literal["start", "carried"]
    context_ok: int
    traj_shas: tuple[str, ...] = ()

    @property
    def rate(self) -> float:
        return self.successes / self.n if self.n else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "effector_id": self.effector_id,
            "step_id": self.step_id,
            "chain": self.chain,
            "primitive": self.primitive,
            "primitive_used": self.primitive_used,
            "n": self.n,
            "successes": self.successes,
            "rate": round(self.rate, 4),
            "failure_modes": dict(sorted(self.failure_modes.items())),
            "median_duration_s": round(self.median_duration_s, 3),
            "fidelity": self.fidelity,
            "caveats": list(self.caveats),
            "controller": self.controller,
            "context": self.context,
            "context_ok": self.context_ok,
        }


@dataclass
class Screen:
    """The per-step screen: the success matrix and what the combination search reads."""

    task_spec: TaskSpec
    chains: tuple[tuple[str, ...], ...]
    conditions: tuple[StudioCondition, ...]
    candidates: dict[str, Candidate]
    infos: dict[str, ToolInfo]
    eligible: dict[int, list[Candidate]]
    pruned: list[Pruned]
    cells: dict[tuple[str, str], Cell] = field(default_factory=dict)
    chain_rate: dict[tuple[str, int], float] = field(default_factory=dict)
    chain_duration: dict[tuple[str, int], float] = field(default_factory=dict)
    context_tools: list[str | None] = field(default_factory=list)
    controllers: ControllerSet = "default"
    runs: int = 0
    wall_s: float = 0.0

    def rate(self, effector_id: str, chain: int) -> float:
        return self.chain_rate.get((effector_id, chain), 0.0)

    def ranked(self, chain: int) -> list[Candidate]:
        """The chain's screened candidates, best first (rate, then simpler, then faster, then id)."""
        return sorted(
            (c for c in self.eligible[chain] if (c.effector_id, chain) in self.chain_rate),
            key=lambda c: (
                -self.rate(c.effector_id, chain),
                self.infos[c.effector_id].n_actuators,
                round(self.infos[c.effector_id].mass_kg, 4),
                round(self.chain_duration.get((c.effector_id, chain), math.inf), 3),
                c.effector_id,
            ),
        )

    def matrix(self) -> dict[str, Any]:
        """Candidates x steps: success rate per cell (``null``: pruned), with the pruning reasons."""
        steps = [s for chain in self.chains for s in chain]
        reasons = {(p.effector_id, p.step_id): p.reason for p in self.pruned}
        rows = []
        for eid, cand in self.candidates.items():
            row: dict[str, Any] = {"effector_id": eid, "display_name": cand.name, "cells": {}}
            for sid in steps:
                cell = self.cells.get((eid, sid))
                if cell is not None:
                    row["cells"][sid] = {
                        "rate": round(cell.rate, 4),
                        "primitive_used": cell.primitive_used,
                        "failure_modes": dict(cell.failure_modes),
                    }
                else:
                    row["cells"][sid] = {"rate": None, "pruned": self._why_not(eid, sid, reasons)}
            rows.append(row)
        return {"steps": steps, "rows": rows}

    def _why_not(self, eid: str, sid: str, reasons: Mapping[tuple[str, str | None], str]) -> str:
        if (eid, sid) in reasons:
            return reasons[(eid, sid)]
        if (eid, None) in reasons:
            return reasons[(eid, None)]
        chain = next(c for c in self.chains if sid in c)
        for other in chain:
            if (eid, other) in reasons:
                return f"its hold chain is pruned at {other}: {reasons[(eid, other)]}"
        return "not screened"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": PRODUCT_MODE_LABEL,
            "controllers": self.controllers,
            "conditions": [condition_dict(c) for c in self.conditions],
            "chains": [list(c) for c in self.chains],
            "candidates": [c.to_dict() for c in self.candidates.values()],
            "tools": {k: v.to_dict() for k, v in self.infos.items()},
            "pruned": [p.to_dict() for p in self.pruned],
            "cells": [c.to_dict() for c in self.cells.values()],
            "chain_rates": [
                {
                    "effector_id": e,
                    "chain": j,
                    "rate": round(r, 4),
                    "duration_s": round(self.chain_duration.get((e, j), math.nan), 3)
                    if math.isfinite(self.chain_duration.get((e, j), math.nan))
                    else None,
                }
                for (e, j), r in sorted(self.chain_rate.items(), key=lambda kv: (kv[0][1], kv[0][0]))
            ],
            "context_tools": list(self.context_tools),
            "matrix": self.matrix(),
            "runs": self.runs,
            "wall_s": round(self.wall_s, 2),
        }


def screen(
    task_spec: TaskSpec,
    candidates: Sequence[Candidate],
    conditions: Sequence[StudioCondition],
    *,
    pool: WorkerPool,
    task: TaskV2 | None = None,
    controllers: ControllerSet = "default",
    payload_kg: float = ARM_PAYLOAD_KG,
) -> Screen:
    """Prune, then screen every surviving candidate on every hold chain (module docstring)."""
    t0 = time.perf_counter()
    task = task or default_task_v2()
    conditions = tuple(conditions)
    eligible, pruned, infos = prune(task_spec, candidates, task=task, payload_kg=payload_kg)
    chains = hold_chains(task_spec)
    out = Screen(
        task_spec=task_spec,
        chains=chains,
        conditions=conditions,
        candidates={c.effector_id: c for c in candidates},
        infos=infos,
        eligible=eligible,
        pruned=pruned,
        controllers=controllers,
    )
    context: dict[str, WorldState | None] = {c.condition_id: None for c in conditions}
    context_ok: dict[str, bool] = {c.condition_id: True for c in conditions}
    for j, chain in enumerate(chains):
        sub = sub_task(task_spec, chain)
        cands = eligible[j]
        jobs = [
            SegmentJob(
                Setup.single(c.spec, chain),
                sub,
                cond,
                initial=context[cond.condition_id],
                task=task,
                controllers=controllers,
                tag=(c.effector_id, j, cond.condition_id),
            )
            for c in cands
            for cond in conditions
        ]
        results = pool.map(run_segment_job, jobs, chunksize=len(conditions))
        out.runs += len(jobs)
        by_cand: dict[str, list[tuple[StudioCondition, TaskResult, WorldState | None]]] = {}
        for job, (res, state) in zip(jobs, results, strict=True):
            by_cand.setdefault(job.tag[0], []).append((job.condition, res, state))
        for cand in cands:
            rows = by_cand.get(cand.effector_id, [])
            _cells(out, cand, j, chain, rows, sum(context_ok.values()))
        ranked = out.ranked(j)
        best = ranked[0] if ranked else None
        out.context_tools.append(best.effector_id if best else None)
        if best is not None:
            for cond, res, state in by_cand[best.effector_id]:
                context[cond.condition_id] = state
                context_ok[cond.condition_id] = context_ok[cond.condition_id] and res.completed
    out.wall_s = time.perf_counter() - t0
    return out


def _cells(
    out: Screen,
    cand: Candidate,
    j: int,
    chain: Sequence[str],
    rows: Sequence[tuple[StudioCondition, TaskResult, WorldState | None]],
    context_ok: int,
) -> None:
    n = len(rows)
    done = [r for _, r, _ in rows if r.completed]
    out.chain_rate[(cand.effector_id, j)] = len(done) / n if n else 0.0
    durations = [r.cycle_time_s for r in done] or [r.cycle_time_s for _, r, _ in rows]
    out.chain_duration[(cand.effector_id, j)] = float(statistics.median(durations)) if durations else math.inf
    for sid in chain:
        outs = [next(o for o in r.steps if o.step_id == sid) for _, r, _ in rows]
        step = out.task_spec.step(sid)
        used = acquire_primitive(cand.spec) if step.primitive in ACQUIRE else step.primitive
        ok = [o for o in outs if o.success]
        dur = [o.duration_s for o in ok] or [o.duration_s for o in outs if o.duration_s > 0] or [0.0]
        out.cells[(cand.effector_id, sid)] = Cell(
            effector_id=cand.effector_id,
            step_id=sid,
            chain=j,
            primitive=step.primitive.value,
            primitive_used=used.value,
            n=n,
            successes=len(ok),
            failure_modes=dict(Counter(o.failure_mode for o in outs if o.failure_mode)),
            median_duration_s=float(statistics.median(dur)),
            fidelity=outs[0].fidelity.value if outs else "not_simulated",
            caveats=tuple(dict.fromkeys(c for o in outs for c in o.caveats)),
            controller=next((o.controller for o in outs if o.controller), ""),
            context="start" if j == 0 else "carried",
            context_ok=context_ok,
            traj_shas=tuple(o.traj_sha for o in outs),
        )


# --------------------------------------------------------------------------- multi-head geometry


@dataclass(frozen=True)
class BracketGeometry:
    """A feasible multi-tool head geometry and its clearances (or why none was found)."""

    mounts: tuple[HeadMount, HeadMount]
    bracket: BracketSpec
    clearance_mm: float
    stack_mass_kg: float
    carried_kg: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mounts": [m.to_dict() for m in self.mounts],
            "bracket": self.bracket.to_dict(),
            "clearance_mm": round(self.clearance_mm, 2),
            "stack_mass_kg": round(self.stack_mass_kg, 5),
            "carried_kg": round(self.carried_kg, 5),
        }


def _rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _geom_spheres(model: mujoco.MjModel, data: mujoco.MjData, geoms: Iterable[int]) -> np.ndarray:
    """Spheres (n, 4: world centre, radius) covering each geom's oriented bounding box."""
    out = []
    for g in geoms:
        aabb = model.geom_aabb[g]
        c, h = np.asarray(aabb[:3]), np.maximum(np.asarray(aabb[3:]), 1e-4)
        r = data.geom_xmat[g].reshape(3, 3)
        base = data.geom_xpos[g] + r @ c
        h_min = max(float(h.min()), 0.004)
        n = np.clip(np.ceil(h / h_min), 1, 6).astype(int)
        piece = h / n
        rad = float(np.linalg.norm(piece))
        grids = [((np.arange(k) + 0.5) / k * 2.0 - 1.0) * hk for k, hk in zip(n, h, strict=True)]
        for x in grids[0]:
            for y in grids[1]:
                for z in grids[2]:
                    out.append([*(base + r @ np.array([x, y, z])), rad])
    return np.array(out, dtype=np.float64).reshape(-1, 4)


def _object_boxes(scene: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """World AABBs of every object geom at the start and, for hinges and slides, at their far
    limit too (an opened lid or drawer), with the owning object's name."""
    model = scene.model
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    poses = [data]
    moved = mujoco.MjData(model)
    any_moved = False
    for obj in scene.objects.values():
        for joint in obj.handles.joints.values():
            j = model.joint(joint).id
            if int(model.jnt_type[j]) not in (2, 3):  # slide, hinge
                continue
            adr = int(model.jnt_qposadr[j])
            lo, hi = model.jnt_range[j]
            if model.jnt_limited[j] and math.isfinite(hi):
                moved.qpos[adr] = hi if abs(hi) >= abs(lo) else lo
                any_moved = True
    if any_moved:
        mujoco.mj_kinematics(model, moved)
        poses.append(moved)
    lows, highs, names = [], [], []
    for name, obj in scene.objects.items():
        for g in obj.geom_ids:
            g = int(g)
            if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                continue
            for d in poses:
                c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
                r = d.geom_xmat[g].reshape(3, 3)
                centre = d.geom_xpos[g] + r @ c
                half = np.abs(r) @ h
                lows.append(centre - half)
                highs.append(centre + half)
                names.append(name)
    return np.array(lows).reshape(-1, 3), np.array(highs).reshape(-1, 3), names


def _gap(spheres: np.ndarray, lo: np.ndarray, hi: np.ndarray, table_z: float) -> tuple[float, int]:
    """Smallest clearance (m) of the spheres to the boxes and the table top, and the box index
    it is to (``-1``: the table)."""
    if not len(spheres):
        return math.inf, -1
    c, r = spheres[:, :3], spheres[:, 3]
    table = float(np.min(c[:, 2] - r - table_z))
    best, which = table, -1
    if len(lo):
        closest = np.clip(c[:, None, :], lo[None], hi[None])
        d = np.linalg.norm(c[:, None, :] - closest, axis=2) - r[:, None]
        k = int(np.argmin(d))
        if float(d.flat[k]) < best:
            best, which = float(d.flat[k]), k % len(lo)
    return best, which


def _flange_frame(scene: Any, data: mujoco.MjData, geoms: Iterable[int], angle_active: float) -> np.ndarray:
    """Spheres of ``geoms`` in the flange frame: the scene presents the active head by turning the
    flange ``-angle_active`` about x in the palm frame, so ``flange = Rx(+angle_active) palm``."""
    sp = _geom_spheres(scene.model, data, geoms)
    rp, pp = data.xmat[scene.ids.palm_body].reshape(3, 3), data.xpos[scene.ids.palm_body]
    palm = (sp[:, :3] - pp) @ rp  # rows: rp.T @ (x - pp)
    flange = palm @ _rot_x(angle_active).T
    return np.column_stack([flange, sp[:, 3]])


def last_link_intrusion(spheres: np.ndarray) -> float:
    """How far (m) any sphere (flange frame: centre, radius) enters the arm's last link: a cylinder
    of :data:`LAST_LINK_RADIUS_M` upward from the flange's mounting face (the floating wrist does
    not model the arm, so this is its keep-out) [assumed]; 0 when none does."""
    if not len(spheres):
        return 0.0
    c, r = spheres[:, :3], spheres[:, 3]
    d_rad = np.maximum(0.0, np.linalg.norm(c[:, :2], axis=1) - LAST_LINK_RADIUS_M)
    d_z = np.maximum(0.0, FLANGE_FACE_M - c[:, 2])
    inside = np.hypot(d_rad, d_z) < r - 0.002
    if not np.any(inside):
        return 0.0
    return float(np.max((c[:, 2] + r - FLANGE_FACE_M)[inside]))


def swept_volume_problems(
    setup: Setup,
    task_spec: TaskSpec,
    *,
    task: TaskV2 | None = None,
    clearance_m: float = SWEEP_CLEARANCE_M,
) -> tuple[list[str], float, Any]:
    """The swept-volume check of a multi-tool head (ruling 2; module docstring).

    Returns the problems (empty when it passes), the smallest clearance found (m) and the scene
    built with head 0 active (its bracket and masses are measured from it).
    """
    from rtt.studio.scene import STUDIO_PREFIX, build_studio_scene

    if setup.kind is not SetupKind.MULTI_HEAD:
        raise ValueError("the swept-volume check is for multi-tool heads")
    task = task or default_task_v2()
    scene = build_studio_scene(setup, task_spec, 0, task=task)
    model = scene.model
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    a = [m.angle_deg for m in setup.mounts]
    head_geoms = {0: scene.ids.active_geoms, 1: scene.ids.idle_geoms}
    eff = {int(g) for g in scene.ids.effector_geoms}
    hub = model.geom(f"{STUDIO_PREFIX}hub").id
    stack_geoms = sorted(eff - set(map(int, head_geoms[0])) - set(map(int, head_geoms[1])) - {hub})
    adapters = [
        g
        for g in stack_geoms
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith(f"{STUDIO_PREFIX}adapter")
    ]
    heads = {k: _flange_frame(scene, data, head_geoms[k], a[0]) for k in (0, 1)}
    stack = _flange_frame(scene, data, stack_geoms, a[0])
    tcp = {
        h.tool: _rot_x(a[0]) @ np.asarray(h.tcp_wrist, dtype=np.float64) for h in scene.heads
    }  # flange frame
    lo, hi, names = _object_boxes(scene)
    table_z = float(task.physics.table_top_z)
    problems: list[str] = []
    worst = math.inf
    # 0. the heads never touch each other
    d = np.linalg.norm(heads[0][:, None, :3] - heads[1][None, :, :3], axis=2)
    gap_heads = float(np.min(d - heads[0][:, None, 3] - heads[1][None, :, 3]))
    if gap_heads < 0.002:
        problems.append("the two heads overlap on the bracket")
    # 0b. no head or adapter enters the arm's last link (:func:`last_link_intrusion`)
    depth = last_link_intrusion(np.vstack([heads[0], heads[1], _flange_frame(scene, data, adapters, a[0])]))
    if depth > 0.0:
        problems.append(
            f"a head or an adapter reaches {1000 * depth:.0f} mm above the flange's mounting face "
            "within the arm's last link"
        )
    segs = setup.segments(task_spec)
    home = np.array([0.0, 0.0, table_z + HOME_HEIGHT])
    # 1. the head switch swings both heads (and the bracket) about the wrist at the home pose
    for prev, seg in itertools.pairwise(segs):
        a0, a1 = -a[prev.tool], -a[seg.tool]
        n = max(2, int(abs(a1 - a0) / 5.0) + 1)
        body = np.vstack([heads[0], heads[1], stack])
        for theta in np.linspace(a0, a1, n)[1:-1]:
            pts = (_rot_x(theta) @ body[:, :3].T).T + home
            g, k = _gap(np.column_stack([pts, body[:, 3]]), lo, hi, table_z)
            worst = min(worst, g)
            if g < clearance_m:
                what = "the table" if k < 0 else names[k]
                problems.append(
                    f"switching to head {seg.tool} at the home pose swings the heads through {what}"
                )
                break
    # 2. at every step's engagement pose the idle head and the bracket clear everything else
    for seg in segs:
        k, idle = seg.tool, 1 - seg.tool
        r_present = _rot_x(-a[k])
        tcp_palm = r_present @ tcp[k]
        body = np.vstack([heads[idle], stack])
        for sid in seg.step_ids:
            step = task_spec.step(sid)
            obj = scene.objects[step.object]
            site = obj.site_pos(data, step.handle)
            skip = {step.object} | ({step.target} if step.target else set())
            keep = np.array([nm not in skip for nm in names], dtype=bool)
            yaw0 = task_spec.object(step.object).pose.yaw_deg
            yaws = (
                [yaw0 + 15.0 * i for i in range(24)]
                if step.info.uses_roll
                else [yaw0 + 90.0 * i for i in range(4)]
            )
            hit = None
            for yaw in yaws:
                rz = _rot_z(yaw)
                wrist = site - rz @ tcp_palm
                pts = (rz @ r_present @ body[:, :3].T).T + wrist
                g, kk = _gap(np.column_stack([pts, body[:, 3]]), lo[keep], hi[keep], table_z)
                worst = min(worst, g)
                if g < clearance_m:
                    hit = (
                        "the table" if kk < 0 else [nm for nm, kp in zip(names, keep, strict=True) if kp][kk]
                    )
                    break
            if hit is not None:
                problems.append(
                    f"during {sid} the idle head {idle} or the bracket comes within "
                    f"{1000 * clearance_m:g} mm of {hit}"
                )
    return problems, worst, scene


#: Bracket layouts searched: (angle of head 0, angle of head 1), within the HeadMount bounds.
BRACKET_LAYOUTS: Final[tuple[tuple[float, float], ...]] = (
    (-30.0, 30.0),
    (-45.0, 45.0),
    (-60.0, 60.0),
    (-75.0, 75.0),
    (0.0, 60.0),
    (0.0, 75.0),
)
BRACKET_STANDOFFS_MM: Final[tuple[float, ...]] = (50.0, 70.0, 100.0, 150.0)


def bracket_search(
    heads: tuple[EffectorSpec, EffectorSpec],
    assignment: Mapping[str, int],
    task_spec: TaskSpec,
    *,
    task: TaskV2 | None = None,
    payload_kg: float = ARM_PAYLOAD_KG,
    present_s: float = PRESENT_S,
) -> tuple[BracketGeometry | None, list[str]]:
    """The lightest feasible bracket geometry for two heads (module docstring), or ``None`` and
    the reasons every geometry was rejected (the first per layout)."""
    task = task or default_task_v2()
    bracket = BracketSpec()
    found: list[tuple[tuple[float, float, float], BracketGeometry]] = []
    reasons: list[str] = []
    for (a0, a1), standoff in itertools.product(BRACKET_LAYOUTS, BRACKET_STANDOFFS_MM):
        mounts = (HeadMount(a0, standoff), HeadMount(a1, standoff))
        setup = Setup(
            SetupKind.MULTI_HEAD,
            heads,
            dict(assignment),
            mounts=mounts,
            bracket=bracket,
            present_s=present_s,
        )
        if setup.problems(task_spec):
            reasons.append("; ".join(setup.problems(task_spec)))
            continue
        try:
            problems, gap, scene = swept_volume_problems(setup, task_spec, task=task)
        except InvalidEffector as exc:
            reasons.append(f"a head cannot be built: {exc}")
            continue
        carried = float(scene.tool_mass_kg)
        if carried > payload_kg:
            problems.append(f"the head weighs {carried:.2f} kg; the arm carries at most {payload_kg:g} kg")
        if problems:
            reasons.append(f"heads at {a0:g}/{a1:g} deg, {standoff:g} mm: {problems[0]}")
            continue
        geo = BracketGeometry(mounts, bracket, 1000.0 * gap, float(scene.stack_mass_kg), carried)
        found.append(((standoff, abs(a1 - a0), -gap), geo))
    if not found:
        return None, list(dict.fromkeys(reasons))
    found.sort(key=lambda kv: kv[0])
    return found[0][1], []


def _summary_of(reasons: Sequence[str]) -> str:
    """Why no bracket layout works, by problem: ``"<problem> (n layouts); ..."``."""
    problems = Counter(r.split(": ", 1)[-1] for r in reasons)
    total = len(reasons)
    parts = [f"{p} ({n} of {total} layouts)" for p, n in problems.most_common(3)]
    return "no bracket layout clears: " + "; ".join(parts)


# --------------------------------------------------------------------------- combinations


@dataclass(frozen=True, eq=False)
class SetupPlan:
    """A setup scored from the screen: predicted completion, cycle time and complexity."""

    setup: Setup
    predicted: float
    weakest: float
    cycle_s: float
    complexity: ComplexityMetrics
    payload_kg: float
    chain_tools: tuple[str, ...]
    geometry: BracketGeometry | None = None
    progress: float = 0.0  # mean screened success of the steps (orders setups that all fail)

    def key(self) -> tuple[float, ...]:
        return (
            -round(self.predicted, 9),
            -round(self.weakest, 9),
            -round(self.progress, 9),
            round(self.cycle_s, 3),
            *self.complexity.key(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "setup_id": self.setup.setup_id,
            "kind": self.setup.kind.value,
            "label": self.setup.label,
            "tools": [t.display_name for t in self.setup.tools],
            "effector_ids": [t.effector_id for t in self.setup.tools],
            "chain_tools": list(self.chain_tools),
            "predicted_completion": round(self.predicted, 4),
            "weakest_chain_rate": round(self.weakest, 4),
            "predicted_cycle_s": round(self.cycle_s, 3),
            "predicted_step_progress": round(self.progress, 4),
            "complexity": self.complexity.to_dict(),
            "payload_kg": round(self.payload_kg, 4),
            "geometry": self.geometry.to_dict() if self.geometry else None,
        }


def _label(kind: SetupKind, names: Sequence[str]) -> str:
    """A short brand-free label (at most 60 characters): the effectors' names before the comma,
    shortened to fit."""
    base = [n.split(",")[0].strip() for n in names]
    head = {
        SetupKind.SINGLE: "",
        SetupKind.MULTI_HEAD: "Two-tool head: ",
        SetupKind.TOOL_CHANGE: "Tool change: ",
    }[kind]
    sep = " + " if kind is SetupKind.MULTI_HEAD else " > "
    for words in (99, 3, 2, 1):
        short = [" ".join(b.split()[:words]) for b in base]
        text = head + sep.join(short) if kind is not SetupKind.SINGLE else names[0]
        if len(text) <= 60:
            return text
    return (head + f"{len(names)} tools").strip()[:60]


def _complexity(
    setup: Setup, infos: Mapping[str, ToolInfo], geometry: BracketGeometry | None, task_spec: TaskSpec
) -> tuple[ComplexityMetrics, float]:
    """Complexity and the heaviest carried stack (the payload) of a setup, from the probes."""
    masses = [infos[t.effector_id].mass_kg for t in setup.tools]
    n_act = sum(infos[t.effector_id].n_actuators for t in setup.tools)
    n_changes = len(setup.segments(task_spec)) - 1
    if setup.kind is SetupKind.SINGLE:
        carried = total = masses[0]
    elif setup.kind is SetupKind.TOOL_CHANGE:
        ch = setup.changer or ToolChanger()
        carried = max(m + ch.master_mass_kg + ch.tool_plate_mass_kg for m in masses)
        total = sum(masses) + ch.master_mass_kg + len(masses) * ch.tool_plate_mass_kg
    else:
        carried = total = geometry.carried_kg if geometry else sum(masses)
    return (
        ComplexityMetrics(
            n_tools=len(setup.tools),
            n_actuators=n_act,
            tool_mass_kg=carried,
            total_mass_kg=total,
            n_changes=n_changes,
            bracket=setup.kind is SetupKind.MULTI_HEAD,
        ),
        carried,
    )


def _predict(
    sc: Screen,
    setup: Setup,
    chain_tools: Sequence[str],
    *,
    geometry: BracketGeometry | None = None,
) -> SetupPlan:
    rates = [sc.rate(e, j) for j, e in enumerate(chain_tools)]
    durations = [sc.chain_duration.get((e, j), math.inf) for j, e in enumerate(chain_tools)]
    segs = setup.segments(sc.task_spec)
    cycle = sum(d for d in durations if math.isfinite(d)) + sum(s.change_s for s in segs)
    complexity, carried = _complexity(setup, sc.infos, geometry, sc.task_spec)
    steps = [
        sc.cells[(e, sid)].rate if (e, sid) in sc.cells else 0.0
        for chain, e in zip(sc.chains, chain_tools, strict=True)
        for sid in chain
    ]
    return SetupPlan(
        setup=setup,
        progress=sum(steps) / len(steps) if steps else 0.0,
        predicted=math.prod(rates) if rates else 0.0,
        weakest=min(rates) if rates else 0.0,
        cycle_s=cycle,
        complexity=complexity,
        payload_kg=carried,
        chain_tools=tuple(chain_tools),
        geometry=geometry,
    )


def _assignment(sc: Screen, chain_tools: Sequence[str], tools: Sequence[str]) -> dict[str, int]:
    index = {e: i for i, e in enumerate(tools)}
    return {sid: index[e] for chain, e in zip(sc.chains, chain_tools, strict=True) for sid in chain}


@dataclass
class Combinations:
    """What the combination search considered, chose and rejected."""

    plans: list[SetupPlan]
    rejected: list[dict[str, Any]]
    counts: dict[str, int]

    def to_dict(self, top: int = 15) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "top": [p.to_dict() for p in sorted(self.plans, key=lambda p: p.key())[:top]],
            "rejected": list(self.rejected),
        }


def combine(
    sc: Screen,
    *,
    task: TaskV2 | None = None,
    top_k: int = TOP_K,
    multi_head_pairs: int = MULTI_HEAD_PAIRS,
    payload_kg: float = ARM_PAYLOAD_KG,
    changer: ToolChanger | None = None,
    present_s: float = PRESENT_S,
    kinds: Iterable[SetupKind] = tuple(SetupKind),
) -> Combinations:
    """Score single, tool-change and multi-head setups from the screen (module docstring)."""
    task = task or default_task_v2()
    kinds = set(kinds)
    changer = changer or ToolChanger()
    n = len(sc.chains)
    plans: list[SetupPlan] = []
    rejected: list[dict[str, Any]] = []
    counts = {"single": 0, "tool_change": 0, "multi_head": 0, "multi_head_geometry_searched": 0}
    step_ids = sc.task_spec.step_ids
    everywhere = [c for c in sc.candidates.values() if all(c in sc.eligible[j] for j in range(n)) and n > 0]
    if SetupKind.SINGLE in kinds:
        for c in everywhere:
            setup = Setup.single(c.spec, step_ids, label=_label(SetupKind.SINGLE, [c.name]))
            plans.append(_predict(sc, setup, [c.effector_id] * n))
            counts["single"] += 1
    tops = [[c.effector_id for c in sc.ranked(j)[:top_k]] for j in range(n)]
    if any(not t for t in tops):
        empty = [sc.chains[j][0] for j, t in enumerate(tops) if not t]
        rejected.append(
            {
                "kind": "all",
                "reason": f"no candidate survived pruning for steps {empty}: no setup can run them",
            }
        )
        return Combinations(plans, rejected, counts)
    if SetupKind.TOOL_CHANGE in kinds and n >= 2:
        k = top_k
        while k > 1 and math.prod(min(k, len(t)) for t in tops) > _MAX_ASSIGNMENTS:
            k -= 1
        seen: set[str] = set()
        for chain_tools in itertools.product(*[t[:k] for t in tops]):
            tools = list(dict.fromkeys(chain_tools))
            if not 2 <= len(tools) <= MAX_TOOLS:
                continue
            per_step = {
                sid: sc.candidates[e].spec
                for chain, e in zip(sc.chains, chain_tools, strict=True)
                for sid in chain
            }
            names = [sc.candidates[e].name for e in tools]
            setup = Setup.tool_change(per_step, changer, label=_label(SetupKind.TOOL_CHANGE, names))
            if setup.setup_id in seen:
                continue
            seen.add(setup.setup_id)
            plan = _predict(sc, setup, chain_tools)
            counts["tool_change"] += 1
            if plan.payload_kg > payload_kg:
                rejected.append(
                    {
                        "kind": "tool_change",
                        "tools": names,
                        "reason": f"carries {plan.payload_kg:.2f} kg; the arm carries at most {payload_kg:g} kg",
                    }
                )
                continue
            plans.append(plan)
    if SetupKind.MULTI_HEAD in kinds and n >= 2:
        pool_ids = list(dict.fromkeys(e for t in tops for e in t))
        pairs: list[tuple[SetupPlan, tuple[EffectorSpec, EffectorSpec], dict[str, int]]] = []
        for a, b in itertools.combinations(pool_ids, 2):
            chain_tools = []
            for j in range(n):
                options = [e for e in (a, b) if sc.candidates[e] in sc.eligible[j]]
                if not options:
                    break
                chain_tools.append(
                    min(options, key=lambda e, j=j: (-sc.rate(e, j), sc.infos[e].n_actuators, e))
                )
            if len(chain_tools) != n or len(set(chain_tools)) < 2:
                continue
            first = chain_tools[0]
            order = [first, b if first == a else a]
            heads = (sc.candidates[order[0]].spec, sc.candidates[order[1]].spec)
            assignment = _assignment(sc, chain_tools, order)
            draft = Setup.multi_head(
                [(heads[0], HeadMount(-90.0)), (heads[1], HeadMount(90.0))],
                assignment,
                label=_label(SetupKind.MULTI_HEAD, [h.display_name for h in heads]),
            )
            pairs.append((_predict(sc, draft, chain_tools), heads, assignment))
        pairs.sort(key=lambda p: p[0].key())
        for draft_plan, heads, assignment in pairs[:multi_head_pairs]:
            counts["multi_head_geometry_searched"] += 1
            names = [h.display_name for h in heads]
            geo, reasons = bracket_search(
                heads, assignment, sc.task_spec, task=task, payload_kg=payload_kg, present_s=present_s
            )
            if geo is None:
                rejected.append(
                    {
                        "kind": "multi_head",
                        "tools": names,
                        "reason": _summary_of(reasons),
                        "all": reasons[:24],
                    }
                )
                continue
            setup = Setup(
                SetupKind.MULTI_HEAD,
                heads,
                assignment,
                mounts=geo.mounts,
                bracket=geo.bracket,
                present_s=present_s,
                label=_label(SetupKind.MULTI_HEAD, names),
            )
            plans.append(_predict(sc, setup, draft_plan.chain_tools, geometry=geo))
            counts["multi_head"] += 1
    return Combinations(plans, rejected, counts)


def choose_finalists(
    combos: Combinations, sc: Screen, n_finalists: int = N_FINALISTS
) -> tuple[list[SetupPlan], SetupPlan | None]:
    """The finalists (3 to 5): the best predicted setup, the best single off-the-shelf effector,
    the best tool change and the best multi-tool head, then the next best; and that best single."""
    if not 1 <= n_finalists <= 5:
        raise ValueError("between 1 and 5 finalists")
    order = sorted(combos.plans, key=lambda p: p.key())
    singles = [
        p for p in order if p.setup.kind is SetupKind.SINGLE and sc.candidates[p.chain_tools[0]].off_the_shelf
    ]
    best_single = singles[0] if singles else None
    picks: list[SetupPlan] = []

    def add(p: SetupPlan | None) -> None:
        if (
            p is not None
            and all(q.setup.setup_id != p.setup.setup_id for q in picks)
            and len(picks) < n_finalists
        ):
            picks.append(p)

    add(order[0] if order else None)
    add(best_single)
    for kind in (SetupKind.TOOL_CHANGE, SetupKind.MULTI_HEAD):
        add(next((p for p in order if p.setup.kind is kind), None))
    for p in order:
        add(p)
    return picks, best_single


# --------------------------------------------------------------------------- results


def setup_result(plan: SetupPlan, task_spec: TaskSpec, runs: Sequence[TaskResult]) -> SetupResult:
    """A finalist's :class:`SetupResult`: measured carried mass, step certificates (sha256 of the
    step's outcomes across the grid, scored by its success rate)."""
    carried = max((m for r in runs for m in r.tool_mass_kg), default=plan.complexity.tool_mass_kg)
    complexity = replace(plan.complexity, tool_mass_kg=carried)
    certs = []
    for s in task_spec.steps:
        outs = [next(o for o in r.steps if o.step_id == s.id) for r in runs]
        digest = canonical_json([o.to_dict() for o in outs])
        certs.append(
            CertificateRef(
                step_id=s.id,
                effector_id=plan.setup.effector_for(s.id).effector_id,
                kind="step",
                ref=hashlib.sha256(digest.encode()).hexdigest(),
                score=sum(o.success for o in outs) / len(outs),
            )
        )
    return SetupResult(plan.setup, task_spec.task_id, tuple(runs), complexity, tuple(certs))


@dataclass
class TournamentReport:
    """Everything :func:`run_task_tournament` did, JSON-ready through :meth:`to_dict`."""

    task_spec: TaskSpec
    conditions: tuple[StudioCondition, ...]
    screen: Screen
    combinations: Combinations
    finalists: list[SetupPlan]
    results: list[SetupResult]
    best_single: SetupPlan | None
    recommendation: dict[str, Any]
    sensitivity: dict[str, Any]
    controllers: ControllerSet
    timings: dict[str, float]
    workers: int
    payload_kg: float = ARM_PAYLOAD_KG

    def result(self, setup_id: str) -> SetupResult:
        return next(r for r in self.results if r.setup.setup_id == setup_id)

    def to_dict(self) -> dict[str, Any]:
        from rtt.studio.recommend import ASSUMED_CONSTANTS, STUDIO_CAVEATS

        plans = {p.setup.setup_id: p for p in self.finalists}
        return {
            "format": "rtt-studio-tournament",
            "version": 1,
            "contract": CONTRACT_VERSION,
            "label": PRODUCT_MODE_LABEL,
            "controllers": self.controllers,
            "evidence": self.controllers == "default",
            "task": self.task_spec.to_dict(),
            "task_id": self.task_spec.task_id,
            "conditions": [condition_dict(c) for c in self.conditions],
            "screen": self.screen.to_dict(),
            "combinations": self.combinations.to_dict(),
            "finalists": [
                {
                    "plan": plans[r.setup.setup_id].to_dict(),
                    "setup": r.setup.to_dict(),
                    "result": r.to_dict(),
                    "runs": [run.to_record() for run in r.runs],
                }
                for r in self.results
            ],
            "best_single": self.best_single.setup.setup_id if self.best_single else None,
            "recommendation": self.recommendation,
            "sensitivity": self.sensitivity,
            "constants": {
                **ASSUMED_CONSTANTS,
                "arm_payload_kg": {**ASSUMED_CONSTANTS["arm_payload_kg"], "value": self.payload_kg},
            },
            "fixed_caveats": list(STUDIO_CAVEATS),
            "timings": {k: round(v, 2) for k, v in self.timings.items()},
            "workers": self.workers,
        }


def run_task_tournament(
    task_spec: TaskSpec | Mapping[str, Any],
    *,
    candidates: Sequence[Candidate | EffectorSpec] | None = None,
    library: bool = True,
    n_conditions: int = 8,
    n_screen: int = N_SCREEN,
    n_finalists: int = N_FINALISTS,
    top_k: int = TOP_K,
    multi_head_pairs: int = MULTI_HEAD_PAIRS,
    workers: int | None = MAX_WORKERS,
    controllers: ControllerSet = "default",
    task: TaskV2 | None = None,
    payload_kg: float = ARM_PAYLOAD_KG,
    kinds: Iterable[SetupKind | str] = tuple(SetupKind),
    pool: WorkerPool | None = None,
) -> TournamentReport:
    """Run the whole Studio search on one task (the plain function behind the agent tool).

    ``task_spec``: a :class:`TaskSpec` or its tool-call form (``TaskSpec.to_tool_args()`` or
    ``to_dict()``). ``controllers="reference"`` runs the S1 smoke controllers: everything runs,
    but :mod:`rtt.studio.recommend` refuses those results as evidence and ranks nothing.
    """
    from rtt.studio.recommend import recommend_setups, sensitivity
    from rtt.studio.types import task_spec_from_tool_args

    t0 = time.perf_counter()
    if not isinstance(task_spec, TaskSpec):
        args = {
            k: task_spec[k]
            for k in task_spec
            if k not in ("contract", "label", "task_id", "source", "fidelity")
        }
        task_spec = task_spec_from_tool_args(args, source=str(task_spec.get("source") or "manual"))  # type: ignore[arg-type]
    if task_spec.refusal is not None or not task_spec.steps:
        raise ValueError("a refused task has nothing to run")
    task = task or default_task_v2()
    if not 1 <= n_screen <= n_conditions:
        raise ValueError("n_screen must be between 1 and n_conditions")
    cands = (
        [c if isinstance(c, Candidate) else candidate_of(c) for c in candidates]
        if candidates
        else list(default_candidates(library=library))
    )
    grid = condition_grid(task_spec, n_conditions)
    own = pool is None
    pool = pool or WorkerPool(workers)
    timings: dict[str, float] = {}
    try:
        t = time.perf_counter()
        sc = screen(
            task_spec,
            cands,
            grid[:n_screen],
            pool=pool,
            task=task,
            controllers=controllers,
            payload_kg=payload_kg,
        )
        timings["screen_s"] = time.perf_counter() - t
        t = time.perf_counter()
        combos = combine(
            sc,
            task=task,
            top_k=top_k,
            multi_head_pairs=multi_head_pairs,
            payload_kg=payload_kg,
            kinds=[SetupKind(k) for k in kinds],
        )
        finalists, best_single = choose_finalists(combos, sc, n_finalists)
        timings["combine_s"] = time.perf_counter() - t
        t = time.perf_counter()
        runs = run_full_tasks(
            [p.setup for p in finalists], task_spec, grid, pool=pool, task=task, controllers=controllers
        )
        results = [setup_result(p, task_spec, runs[p.setup.setup_id]) for p in finalists]
        timings["finalists_s"] = time.perf_counter() - t
    finally:
        if own:
            pool.close()
    best_id = best_single.setup.setup_id if best_single else None
    rec = recommend_setups(
        results, best_single_id=best_id, payload={p.setup.setup_id: p.payload_kg for p in finalists}
    )
    sens = sensitivity(results) if rec.get("evidence") == "accepted" else {"skipped": rec.get("reason")}
    timings["total_s"] = time.perf_counter() - t0
    return TournamentReport(
        task_spec=task_spec,
        conditions=grid,
        screen=sc,
        combinations=combos,
        finalists=finalists,
        results=results,
        best_single=best_single,
        recommendation=rec,
        sensitivity=sens,
        controllers=controllers,
        timings=timings,
        workers=pool.workers,
        payload_kg=payload_kg,
    )


# --------------------------------------------------------------------------- command line


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m rtt.studio.search TASK.json --out DIR [--workers 6] [--pack]``.

    ``TASK.json``: a ``TaskSpec.to_dict()`` (or its tool-call form). Writes ``DIR/report.json``
    and, with ``--pack``, ``DIR/build_pack/`` for the recommendation (or the best finalist when
    there is none, labelled as such in its certificate).
    """
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(prog="python -m rtt.studio.search", description=main.__doc__)
    ap.add_argument("task", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--conditions", type=int, default=8)
    ap.add_argument("--screen", type=int, default=N_SCREEN)
    ap.add_argument("--finalists", type=int, default=N_FINALISTS)
    ap.add_argument("--no-library", action="store_true")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--controllers", choices=("default", "reference"), default="default")
    a = ap.parse_args(argv)
    task = json.loads(a.task.read_text(encoding="utf-8"))
    rep = run_task_tournament(
        task,
        workers=a.workers,
        n_conditions=a.conditions,
        n_screen=a.screen,
        n_finalists=a.finalists,
        library=not a.no_library,
        controllers=a.controllers,
    )
    a.out.mkdir(parents=True, exist_ok=True)
    data = rep.to_dict()
    (a.out / "report.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    rec = data["recommendation"]
    pick = (rec.get("recommendation") or {}).get("setup_id") or (rec["ranking"] or [{}])[0].get("setup_id")
    print(
        json.dumps({"report": str(a.out / "report.json"), "recommendation": pick, "timings": data["timings"]})
    )
    if a.pack and pick:
        from rtt.studio.buildpack import export_build_pack

        bp = export_build_pack(data, a.out / "build_pack", setup_id=pick, overwrite=True)
        print(
            json.dumps(
                {"build_pack": str(bp.out_dir), "zip": str(bp.zip_path), "files": len(bp.manifest.files)}
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
