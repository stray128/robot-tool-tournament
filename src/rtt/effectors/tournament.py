"""The per-part effector tournament (product mode, not pre-registered; ``docs/EFFECTORS.md`` section 4).

For one part, every registered effector family enters its presets (and the designed groups
their recipes); each candidate is screened on the same conditions, the best two per group are
certified, and :mod:`rtt.effectors.recommend` turns the certificates into one recommendation.

Stages (:func:`run_tournament`):

1. **Entrants.** Every preset of every available family (:func:`rtt.effectors.registry.available_families`,
   so a family that registers itself is picked up without a list to edit), bucketed by
   :attr:`~rtt.effectors.types.EffectorSpec.group`. The designed group adds Nemotron proposals in
   :data:`DESIGN_ROUNDS` rounds of :data:`DESIGNS_PER_ROUND` (round 2 sees round 1's physics). The
   random control is matched to them: after the rounds it draws, from the same spaces Nemotron was
   offered and with the same validators, exactly as many screened designs as Nemotron's valid
   designs screened on this part (:func:`_match_random`). Without the Nemotron rounds it draws
   :data:`RANDOM_RECIPES`. Both come from a :class:`MorphologyDesigner` (track B3's
   :mod:`rtt.effectors.morphology` by default).
2. **Build check.** Each candidate is built into the shared scene; an
   :class:`~rtt.effectors.types.InvalidEffector` (or an unbuildable jaw) is listed with its reason and
   scores 0. The compiled model gives the candidate's fairness signature (wrist servos, solver,
   part contacts, phase timings).
3. **Budget.** At most :data:`MAX_SCREEN_CANDIDATES` buildable candidates per group are screened
   (a larger group is sampled with a fixed seed), so no group spends more than
   :data:`MAX_SCREEN_EPISODES` screen episodes.
4. **Reference.** A ``nominal_episode`` family's registration reference is the pose its effector
   produces in the certificate's nominal episode (no placement error, no jitter); that episode runs
   first, exactly as :func:`rtt.effectors.certificate.certify_effector` runs it. If it does not hold,
   nothing of that candidate can be registered. Reference episodes are listed in the budget
   separately from the screen.
5. **Screen.** Every screened candidate on the part's four ``train_v2`` placements x 2 jitter seeds
   (:func:`training_conditions`; :func:`upload_conditions` for an uploaded part): 8 episodes each,
   the same condition ids and seeds for every candidate (common random numbers).
6. **Finalists.** The top :data:`CERTIFICATES_PER_GROUP` per group by screen success for the
   requirement (ties: held rate, fewer actuators, lower mass, ``effector_id``).
7. **Certificate.** :func:`~rtt.effectors.certificate.certify_effector`, unchanged, per finalist.
8. **Recommendation.** :func:`rtt.effectors.recommend.recommend` for both requirements.

The result is a pure function of (part, groups, requirement, seed, code, cached Nemotron
responses): a re-run gives identical records, ``traj_sha`` values and recommendation. Episodes run
in-process (``workers = 1``) or on one spawned process pool kept open for the whole tournament.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import math
import multiprocessing as mp
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, Self

import numpy as np

from rtt.effectors.registry import EffectorFamilyImpl, UnknownFamilyError, get_family
from rtt.effectors.types import (
    CONTRACT_VERSION,
    PRODUCT_MODE_LABEL,
    ContractError,
    EffectorEpisodeResult,
    EffectorFamily,
    EffectorSpec,
    EntrantGroup,
    InvalidEffector,
    MorphologyRecipe,
    RegistrationReference,
    Requirement,
    canonical_json,
)

if TYPE_CHECKING:
    import mujoco

    from rtt.effectors.certificate import EffectorCertificate
    from rtt.effectors.runner import EffectorJob
    from rtt.parts.types import PartSpec
    from rtt.sim.objects import Placement
    from rtt.sim.task import TaskV2

__all__ = [
    "CERTIFICATES_PER_GROUP",
    "DESIGNS_PER_ROUND",
    "DESIGN_ROUNDS",
    "GROUP_ORDER",
    "LANE_CONDITION",
    "MAX_SCREEN_CANDIDATES",
    "MAX_SCREEN_EPISODES",
    "RANDOM_RECIPES",
    "SCREEN_SEEDS",
    "Candidate",
    "DesignBatch",
    "DesignFeedback",
    "Executor",
    "GroupState",
    "MorphologyDesigner",
    "ScreenCondition",
    "TournamentExecutor",
    "TournamentResult",
    "default_designer",
    "fairness_problems",
    "family_status",
    "jsonable",
    "run_tournament",
    "spec_from_dict",
    "training_conditions",
    "upload_conditions",
    "wrist_signature",
    "write_outputs",
]

#: Groups in the order they are reported (the order of :class:`EntrantGroup`).
GROUP_ORDER: Final[tuple[EntrantGroup, ...]] = tuple(EntrantGroup)
#: Jitter seeds per screen condition (the ``train_v2`` suite has 2).
SCREEN_SEEDS: Final = 2
#: Screen candidates per group: 12 x 8 = 96 screen episodes at most.
MAX_SCREEN_CANDIDATES: Final = 12
MAX_SCREEN_EPISODES: Final = 96
#: Finalists certified per group.
CERTIFICATES_PER_GROUP: Final = 2
#: Nemotron proposals: rounds and designs per round (the second round sees the first's physics).
DESIGN_ROUNDS: Final = 2
DESIGNS_PER_ROUND: Final = 4
#: Recipes the random control draws when no Nemotron rounds run (with them, it is matched to
#: the number of Nemotron designs screened on the part).
RANDOM_RECIPES: Final = 8
#: A matched random draw stops after this many recipes per design wanted (unbuildable draws).
RANDOM_DRAW_FACTOR: Final = 4
#: The condition every lane of a part plays (the part 3 mm off centre), first seed.
LANE_CONDITION: Final = "c2"
_FIDELITY_RANK: Final = {"core": 0, "rigid": 1, "approximation": 2}

Status = Literal["pending", "screened", "invalid_build", "contract_error", "not_sampled"]


# --------------------------------------------------------------------------- conditions


@dataclass(frozen=True)
class ScreenCondition:
    """One screen condition: a placement and its jitter seeds (the same for every candidate)."""

    condition_id: str
    label: str
    placement: Placement
    seeds: tuple[int, ...]

    def describe(self) -> dict[str, Any]:
        p = self.placement
        return {
            "condition_id": self.condition_id,
            "label": self.label,
            "placement": {
                "x": round(p.x, 9),
                "y": round(p.y, 9),
                "yaw_deg": round(math.degrees(p.yaw), 9),
                "err_x_mm": round(1000.0 * p.err_x, 6),
                "err_y_mm": round(1000.0 * p.err_y, 6),
                "err_yaw_deg": round(math.degrees(p.err_yaw), 6),
            },
            "seeds": [int(s) for s in self.seeds],
        }


def training_conditions(part: PartSpec) -> tuple[ScreenCondition, ...]:
    """The part's ``train_v2`` conditions: c0 to c3 with the suite's 2 jitter seeds each."""
    from rtt.parts.suite import train_suite

    conds = [c for c in train_suite() if c.part.name == part.name]
    if not conds:
        raise ValueError(f"{part.name!r} is not a train_v2 part; use upload_conditions for an uploaded part")
    if conds[0].part.geometry_sha256 != part.geometry_sha256:
        raise ValueError(f"{part.name!r} differs from the train_v2 part of that name")
    return tuple(
        ScreenCondition(
            c.condition_id, c.placement_label, c.placement, tuple(int(s) for s in c.seeds[:SCREEN_SEEDS])
        )
        for c in conds
    )


def upload_conditions(
    part: PartSpec,
    placements: Mapping[str, Placement] | None = None,
    *,
    suite: str = "upload",
    seeds: int = SCREEN_SEEDS,
) -> tuple[ScreenCondition, ...]:
    """Screen conditions for a part outside ``train_v2``.

    ``placements`` default to ``train_v2``'s four (c0 centred, c1 +2 mm, c2 -3 mm, c3 +1 mm and
    +6 deg), so an upload is screened like a training part; its seeds derive from the part's
    geometry hash and the condition id (identical on every machine).
    """
    from rtt.parts.suite import load_train_spec
    from rtt.sim.conditions import derive_seed

    table = dict(placements) if placements is not None else dict(load_train_spec().placements)
    if not table:
        raise ValueError("needs at least one placement")
    base = int(part.geometry_sha256[:15], 16)
    out = []
    for label, placement in table.items():
        cid = f"{suite}/{part.name}/{label}"
        out.append(
            ScreenCondition(cid, label, placement, tuple(derive_seed(base, cid, r) for r in range(seeds)))
        )
    return tuple(out)


# --------------------------------------------------------------------------- designed groups


@dataclass(frozen=True)
class DesignFeedback:
    """Physics feedback on one recipe, shown to the designer in the next round."""

    recipe: MorphologyRecipe
    held_rate: float
    registered_rate: float
    failure_modes: Mapping[str, int]
    n_episodes: int = 0  # screen episodes behind the rates (0: unknown)


@dataclass(frozen=True)
class DesignBatch:
    """One round of proposals: valid recipes, rejections with reasons, and what the calls cost."""

    recipes: tuple[MorphologyRecipe, ...]
    rejections: tuple[Mapping[str, Any], ...] = ()
    calls: int = 0
    cost_usd: float = 0.0
    analysis: str = ""
    kinds: tuple[str, ...] = ()  # the spaces the proposer was offered (labels; empty: unknown)


class MorphologyDesigner(Protocol):
    """What the designed and random groups need (track B3's :mod:`rtt.effectors.morphology`)."""

    def random_recipes(self, part: PartSpec, n: int, seed: int) -> Sequence[MorphologyRecipe]:
        """``n`` valid recipes; the first ``n`` of a larger draw with the same seed are the same.
        An implementation may accept ``kinds=`` (space labels) to draw from those spaces only."""
        ...

    def propose(self, part: PartSpec, history: Sequence[DesignFeedback], n: int) -> DesignBatch: ...

    def spec_for(self, recipe: MorphologyRecipe) -> EffectorSpec: ...


class _MorphologyAdapter:
    """:class:`MorphologyDesigner` over :mod:`rtt.effectors.morphology` (the B3 interface)."""

    def __init__(self, module: Any, client: Any = None) -> None:
        self.module = module
        self.client = client

    def random_recipes(
        self, part: PartSpec, n: int, seed: int, *, kinds: Sequence[str] | None = None
    ) -> Sequence[MorphologyRecipe]:
        return list(self.module.random_recipes(part, n, seed, kinds=list(kinds) if kinds else None))

    def propose(self, part: PartSpec, history: Sequence[DesignFeedback], n: int) -> DesignBatch:
        if self.client is None:
            raise RuntimeError("Nemotron proposals need a model client (design = true)")
        outcome = self.module.DesignOutcome
        past = [
            outcome(
                recipe=h.recipe,
                held_rate=h.held_rate,
                registered_rate=h.registered_rate,
                failure_modes=dict(h.failure_modes),
                n_episodes=h.n_episodes,
            )
            for h in history
        ]
        batch = self.module.propose_recipes(part, past, n, client=self.client)
        recipes = _first_attr(batch, ("recipes", "valid", "accepted"), ())
        rejections = _first_attr(batch, ("rejections", "rejected"), ())
        entries = _first_attr(batch, ("ledger", "entries", "calls", "ledger_entries"), ())
        cost = sum(
            float(getattr(e, "cost_usd", 0.0) or 0.0) for e in entries if not getattr(e, "cached", False)
        )
        return DesignBatch(
            tuple(recipes),
            tuple(_rejection(r) for r in rejections),
            calls=len(entries),
            cost_usd=cost,
            analysis=str(getattr(batch, "analysis", "") or ""),
            kinds=tuple(str(k) for k in (getattr(batch, "kinds", None) or ())),
        )

    def spec_for(self, recipe: MorphologyRecipe) -> EffectorSpec:
        return self.module.spec_for(recipe)


def _rejection(r: Any) -> dict[str, Any]:
    """A rejection as JSON data: B3's ``Rejection`` (``to_dict``), a mapping, or its text."""
    if isinstance(r, Mapping):
        return dict(r)
    to_dict = getattr(r, "to_dict", None)
    return dict(to_dict()) if callable(to_dict) else {"reason": str(r)}


def _random_draw(
    designer: MorphologyDesigner, part: PartSpec, n: int, seed: int, kinds: Sequence[str]
) -> list[MorphologyRecipe]:
    if kinds:
        return list(designer.random_recipes(part, n, seed, kinds=kinds))  # type: ignore[call-arg]
    return list(designer.random_recipes(part, n, seed))


def plain_space_name(label: str) -> str:
    """A design-space label in plain words, for notes people read: ``"fingered_hand"`` ->
    ``"fingered hand"``, ``"soft_arm/tentacle"`` -> ``"soft arm (tentacle)"``,
    ``"suction/suction_cup"`` -> ``"suction cup"`` (the kind already names the family)."""
    fam, sep, kind = str(label).partition("/")
    if not sep:
        return fam.replace("_", " ")
    fam_w, kind_w = fam.replace("_", " "), kind.replace("_", " ")
    return kind_w if fam_w in kind_w else f"{fam_w} ({kind_w})"


def _match_random(
    designer: MorphologyDesigner,
    st: GroupState,
    target: int,
    kinds: Sequence[str],
    seed: int,
    add: Callable[[EntrantGroup, Sequence[MorphologyRecipe], str], list[Candidate]],
    build_check: Callable[[Candidate], None],
    screen: Callable[[list[Candidate]], None],
    part: PartSpec,
) -> None:
    """Draw the random control after the Nemotron rounds: exactly ``target`` buildable recipes
    (the Nemotron designs screened on this part), from the spaces Nemotron was offered, with the
    same validators. A draw is the prefix of a larger one with the same seed, so the draw grows
    until enough recipes build (at most :data:`RANDOM_DRAW_FACTOR` x ``target`` + 8 recipes)."""
    pool = ", ".join(plain_space_name(k) for k in kinds) if kinds else "every design space"
    st.note = (
        f"matched to the {target} Nemotron design{'s' if target != 1 else ''} screened, "
        f"drawn from the same spaces: {pool}"
    )
    if target <= 0:
        st.reason = "no Nemotron design was screened on this part, so the matched control is empty"
        return
    drawn: list[MorphologyRecipe] = []
    buildable: list[Candidate] = []
    n = target
    limit = RANDOM_DRAW_FACTOR * target + RANDOM_RECIPES
    while len(buildable) < target and n <= limit:
        recipes = _random_draw(designer, part, n, seed, kinds)
        if [r.recipe_id for r in recipes[: len(drawn)]] != [r.recipe_id for r in drawn]:
            raise ContractError("the random draw is not a prefix of a larger draw with the same seed")
        new = recipes[len(drawn) :]
        drawn = recipes
        for cand in add(EntrantGroup.RANDOM, new, "random"):
            build_check(cand)
            if cand.status == "pending":
                if len(buildable) < target:
                    buildable.append(cand)
                else:  # built, but the control is already full
                    cand.status, cand.reason = "not_sampled", "the matched control already had its designs"
        if len(recipes) < n:  # the draw cap of the spaces was reached
            break
        n += target - len(buildable) if len(buildable) < target else 0
        if not new:
            break
    st.sampled_from = len(drawn)
    if len(buildable) < target:
        st.note += f"; only {len(buildable)} buildable random designs could be drawn"
    screen(buildable)


def _first_attr(obj: Any, names: Sequence[str], default: Any) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def default_designer(client: Any = None) -> MorphologyDesigner | None:
    """Track B3's designer (:mod:`rtt.effectors.morphology`), or ``None`` while it is not built."""
    import importlib

    name = "rtt.effectors.morphology"
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        return None
    return _MorphologyAdapter(module, client)


# --------------------------------------------------------------------------- candidates


@dataclass(eq=False)
class Candidate:
    """One entrant and everything the tournament learned about it."""

    spec: EffectorSpec
    source: str  # "preset", "nemotron-1", "nemotron-2", "random"
    status: Status = "pending"
    reason: str | None = None
    n_actuators: int | None = None
    mass_kg: float | None = None
    actuation: dict[str, float] = field(default_factory=dict)
    reach_m: float | None = None
    max_opening_m: float | None = None
    notes: tuple[str, ...] = ()
    fairness: dict[str, Any] | None = None
    reference: RegistrationReference | None = None
    reference_source: str = "analytic"  # "analytic" | "nominal_episode" | "none"
    reference_result: EffectorEpisodeResult | None = None
    screen: list[EffectorEpisodeResult] = field(default_factory=list)
    finalist: bool = False
    certificate: EffectorCertificate | None = None

    @property
    def effector_id(self) -> str:
        return self.spec.effector_id

    @property
    def group(self) -> EntrantGroup:
        return self.spec.group

    @property
    def fidelity(self) -> str:
        return self.spec.fidelity.value

    def screen_registered(self, r: EffectorEpisodeResult) -> bool:
        """Registered in the screen: success against a reference the candidate has."""
        return bool(r.success) and self.reference_source != "none"

    def screen_rates(self) -> dict[str, Any]:
        n = len(self.screen)
        held = sum(bool(r.held) for r in self.screen)
        reg = sum(self.screen_registered(r) for r in self.screen)
        modes = collections.Counter(r.failure_mode or "none" for r in self.screen if r.failure_mode)
        return {
            "n": n,
            "held": held / n if n else 0.0,
            "registered": reg / n if n else 0.0,
            "held_count": held,
            "registered_count": reg,
            "failure_modes": dict(sorted(modes.items())),
        }

    def rate(self, requirement: Requirement | str) -> float:
        return float(self.screen_rates()[Requirement(requirement).value])

    def episodes(self) -> list[EffectorEpisodeResult]:
        out = list(self.screen)
        if self.reference_result is not None:
            out.insert(0, self.reference_result)
        return out

    def summary(self) -> dict[str, Any]:
        s = self.spec
        out: dict[str, Any] = {
            "effector_id": s.effector_id,
            "display_name": s.display_name,
            "family": s.family.value,
            "model": s.model,
            "primitive": s.primitive,
            "origin": s.origin,
            "group": s.group.value,
            "fidelity": s.fidelity.value,
            "source": self.source,
            "status": self.status,
            "reason": self.reason,
            "n_actuators": self.n_actuators,
            "mass_kg": _num(self.mass_kg, 4),
            "actuation": dict(self.actuation),
            "reach_mm": _num(1000.0 * self.reach_m, 2) if self.reach_m is not None else None,
            "max_opening_mm": _num(1000.0 * self.max_opening_m, 2)
            if self.max_opening_m is not None
            else None,
            "notes": list(self.notes),
            "reference": self.reference_source,
            "reference_pose": (
                {"pos": list(self.reference.pos), "quat_wxyz": list(self.reference.quat_wxyz)}
                if self.reference is not None
                else None
            ),
            "finalist": self.finalist,
            "certified": self.certificate is not None,
            "spec": s.to_dict(),
        }
        if self.status == "screened":
            out["screen"] = self.screen_rates()
        return out


def _num(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


# --------------------------------------------------------------------------- fairness


def wrist_signature(model: mujoco.MjModel) -> dict[str, Any]:
    """What every family must share, read from a compiled model: wrist joints and servos, the
    wrist body's mass, the solver options and the part's contact parameters."""
    from rtt.sim.scene import BODY_OBJECT, BODY_PALM, PALM_JOINTS

    def vec(a: Any, digits: int = 9) -> list[float]:
        return [round(float(v), digits) for v in np.asarray(a).ravel()]

    out: dict[str, Any] = {}
    for name in PALM_JOINTS:
        j = model.joint(name)
        a = model.actuator(name)
        out[name] = {
            "range": vec(j.range),
            "armature": vec(j.armature),
            "damping": vec(j.damping),
            "stiffness": vec(j.stiffness),
            "frictionloss": vec(j.frictionloss),
            "gain": vec(a.gainprm[:3]),
            "bias": vec(a.biasprm[:3]),
            "forcerange": vec(a.forcerange),
            "ctrlrange": vec(a.ctrlrange),
            "gear": vec(a.gear[:1]),
        }
    palm = model.body(BODY_PALM).id
    obj = model.body(BODY_OBJECT).id
    geoms = np.flatnonzero(model.geom_bodyid == obj)
    opt = model.opt
    out["palm"] = {
        "mass": round(float(model.body_mass[palm]), 9),
        "gravcomp": float(model.body_gravcomp[palm]),
    }
    out["option"] = {
        "timestep": float(opt.timestep),
        "integrator": int(opt.integrator),
        "cone": int(opt.cone),
        "impratio": float(opt.impratio),
        "iterations": int(opt.iterations),
        "ls_iterations": int(opt.ls_iterations),
        "tolerance": float(opt.tolerance),
        "gravity": vec(opt.gravity),
        "disableflags": int(opt.disableflags),
        "enableflags": int(opt.enableflags),
    }
    out["part"] = {
        "mass": round(float(model.body_mass[obj]), 9),
        "gravcomp": float(model.body_gravcomp[obj]),
        "priority": sorted({int(p) for p in model.geom_priority[geoms]}),
        "friction": vec(model.geom_friction[geoms[0]]),
        "condim": sorted({int(c) for c in model.geom_condim[geoms]}),
        "solref": vec(model.geom_solref[geoms[0]]),
        "solimp": vec(model.geom_solimp[geoms[0]]),
    }
    return out


def _phase_signature(task: TaskV2, h: float, reach: float, tip: float) -> dict[str, float]:
    from rtt.sim import objects as objs
    from rtt.sim.controller import _plan, reach_floor

    plan = _plan(task, objs.Placement(), h, reach, tip, reach_floor(reach, h, task))
    return {
        "t_descend": round(plan.t_descend, 9),
        "t_close": round(plan.t_close, 9),
        "close_time": round(plan.close_time, 9),
        "t_lift": round(plan.t_lift, 9),
        "t_hold": round(plan.t_hold, 9),
        "lift_m": round(plan.lift_z - plan.grasp_z, 9),
        "control_period": float(task.controller.control_period),
        "check_period": float(task.score.check_period),
        "success_lift_m": float(task.score.success_lift),
        "success_hold_s": float(task.score.success_hold),
        "max_time_s": float(task.score.max_time),
    }


def _is_native(impl: EffectorFamilyImpl) -> bool:
    return type(impl).run_episode is EffectorFamilyImpl.run_episode


def _build_check(cand: Candidate, part: PartSpec, task: TaskV2) -> None:
    """Build ``cand`` into its scene: mark it invalid (with the reason) or record its build facts."""
    from rtt.effectors.scene import build_effector_scene
    from rtt.sim import objects as objs

    impl = get_family(cand.spec.family)
    h = float(objs.height(part.to_mesh_object()))
    try:
        if _is_native(impl):
            scene = build_effector_scene(impl, cand.spec, part, task)
            build = scene.build
            pplan = impl.primitive(cand.spec, build).plan(part, objs.Placement(), task, build)
            cand.n_actuators = build.n_actuators
            cand.mass_kg = build.mass
            cand.actuation = {k: float(v) for k, v in build.actuation.items()}
            cand.reach_m = build.reach
            cand.max_opening_m = build.max_opening
            cand.notes = tuple(build.notes)
            model = scene.model
            tip, reach = pplan.tip_height, build.reach
        elif hasattr(impl, "pair_for"):  # the delegated reference: the frozen jaw scene
            from rtt.sim.scene import BODY_PALM, build_model_v2

            pair = impl.pair_for(cand.spec, part, task)
            model, _ = build_model_v2(pair, part.to_mesh_object(), task)
            palm = model.body(BODY_PALM).id
            cand.n_actuators = int(model.nu) - 4
            cand.mass_kg = float(model.body_subtreemass[palm] - model.body_mass[palm])
            cand.actuation = {"grip_force_n": float(task.gripper.grip_force)}
            cand.reach_m = float(pair.finger_length)
            cand.max_opening_m = 2.0 * float(task.gripper.open_half_gap)
            cand.notes = tuple(pair.auto_repairs)
            tip, reach = float(pair.tip_height), float(pair.finger_length)
        else:  # another delegated family: no scene to inspect here
            return
    except InvalidEffector as exc:
        cand.status, cand.reason = "invalid_build", str(exc) or type(exc).__name__
        return
    except ContractError as exc:
        cand.status, cand.reason = "contract_error", f"ContractError: {exc}"[:400]
        return
    except ValueError as exc:  # an unbuildable jaw (InvalidDesign is a ValueError)
        if _is_native(impl):
            raise
        cand.status, cand.reason = "invalid_build", str(exc) or type(exc).__name__
        return
    cand.fairness = {
        "wrist": wrist_signature(model),
        "phases": _phase_signature(task, h, reach, tip),
    }


def fairness_problems(candidates: Iterable[Candidate], conditions: Sequence[ScreenCondition]) -> list[str]:
    """Everything that differs between candidates and must not: wrist, solver, part contacts,
    phase timings, and the screen's condition ids and seeds (empty when the tournament is fair)."""
    problems: list[str] = []
    want_conditions = [(c.condition_id, s) for c in conditions for s in c.seeds]
    first: tuple[str, dict[str, Any]] | None = None
    for cand in candidates:
        if cand.fairness is not None:
            if first is None:
                first = (cand.spec.display_name, cand.fairness)
            elif cand.fairness != first[1]:
                keys = sorted(
                    f"{section}.{k}"
                    for section in cand.fairness
                    for k in cand.fairness[section]
                    if cand.fairness[section].get(k) != first[1].get(section, {}).get(k)
                )
                problems.append(f"{cand.spec.display_name} differs from {first[0]} in {', '.join(keys[:6])}")
        if cand.status == "screened":
            seen = [(r.condition_id, int(r.seed)) for r in cand.screen]
            if seen != want_conditions:
                problems.append(f"{cand.spec.display_name} was screened on other conditions or seeds")
    return problems


# --------------------------------------------------------------------------- execution


def _certify_job(args: tuple[EffectorSpec, Any, Any]) -> EffectorCertificate:
    """One whole certificate, in-process (the unit a worker runs; picklable)."""
    from rtt.effectors.certificate import certify_effector

    spec, part, task = args
    return certify_effector(spec, part, task=task, workers=1)


class TournamentExecutor:
    """Maps a picklable function in-process (``workers <= 1``) or on one spawned pool kept open."""

    def __init__(self, workers: int | None) -> None:
        if workers is None:
            workers = max(1, (mp.cpu_count() or 2) - 1)
        self.workers = int(workers)
        self._pool: Any = None

    def map(self, fn: Callable[[Any], Any], items: Sequence[Any]) -> list[Any]:
        items = list(items)
        if not items:
            return []
        if self.workers <= 1 or len(items) == 1:
            return [fn(i) for i in items]
        if self._pool is None:
            self._pool = mp.get_context("spawn").Pool(processes=self.workers)
        return list(self._pool.map(fn, items, chunksize=1))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Executor(Protocol):
    """Anything with ``map(fn, items) -> list`` in order (a test can fake the physics with one)."""

    def map(self, fn: Callable[[Any], Any], items: Sequence[Any]) -> list[Any]: ...


# --------------------------------------------------------------------------- the tournament


@dataclass
class GroupState:
    """One budget group: its candidates and how the budget was spent."""

    group: EntrantGroup
    requested: bool
    status: str = "run"  # "run" | "empty" | "skipped"
    reason: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    sampled_from: int = 0
    finalists: list[Candidate] = field(default_factory=list)
    note: str | None = None  # how the group was drawn, when that is not the plain rule

    def screened(self) -> list[Candidate]:
        return [c for c in self.candidates if c.status == "screened"]

    def budget(self) -> dict[str, int]:
        screened = self.screened()
        return {
            "candidates": len(self.candidates),
            "buildable": sum(c.status in ("screened", "not_sampled") for c in self.candidates),
            "screened": len(screened),
            "screen_episodes": sum(len(c.screen) for c in screened),
            "reference_episodes": sum(c.reference_result is not None for c in screened),
            "certificates": sum(c.certificate is not None for c in self.candidates),
            "certificate_episodes": sum(
                len(c.certificate.tests) + (1 if c.certificate.reference != "analytic" else 0)
                for c in self.candidates
                if c.certificate is not None
            ),
            "max_screen_episodes": MAX_SCREEN_EPISODES,
            "max_certificates": CERTIFICATES_PER_GROUP,
        }


@dataclass
class TournamentResult:
    """Everything one tournament produced (written by :func:`write_outputs`)."""

    part: PartSpec
    requirement: Requirement
    seed: int
    suite: str
    conditions: tuple[ScreenCondition, ...]
    groups: dict[EntrantGroup, GroupState]
    families: dict[str, str]  # family -> "available" or why not
    recommendation: dict[str, Any]
    design_rounds: list[dict[str, Any]]
    llm: dict[str, Any]
    task_digest: str
    code_version: str
    created_utc: str
    wall_s: float

    def candidates(self) -> list[Candidate]:
        return [c for g in GROUP_ORDER if g in self.groups for c in self.groups[g].candidates]

    def candidate(self, effector_id: str) -> Candidate:
        return next(c for c in self.candidates() if c.effector_id == effector_id)

    def records(self) -> list[dict[str, Any]]:
        """Every episode as a flat record: reference and screen episodes (``to_record()``) and
        certificate tests, with the stage and the tournament's own registered verdict."""
        rows = []
        for cand in self.candidates():
            for r in cand.episodes():
                rec = r.to_record()
                stage = "reference" if r is cand.reference_result else "screen"
                rec.update(
                    stage=stage,
                    source=cand.source,
                    # the reference episode is the pose itself: it has no registered verdict
                    registered=bool(cand.screen_registered(r)) if stage == "screen" else None,
                    reference_source=cand.reference_source,
                )
                rows.append(rec)
            if cand.certificate is not None:
                for t in cand.certificate.tests:
                    rows.append(_certificate_record(cand, t))
        return rows

    def fairness(self) -> dict[str, Any]:
        cands = [c for c in self.candidates() if c.fairness is not None]
        problems = fairness_problems(self.candidates(), self.conditions)
        ref = cands[0].fairness if cands else None
        return {
            "identical": not problems,
            "problems": problems,
            "signature_sha256": hashlib.sha256(canonical_json(ref).encode()).hexdigest() if ref else None,
            "signature": ref,
            "candidates_checked": len(cands),
            "conditions": [c.condition_id for c in self.conditions],
            "seeds": {c.condition_id: [int(s) for s in c.seeds] for c in self.conditions},
        }

    def physics(self) -> dict[str, Any]:
        eps = [r for c in self.candidates() for r in c.episodes()]
        held = [r for r in eps if r.held]
        modes = collections.Counter(r.failure_mode for r in eps if r.failure_mode)
        cert_tests = [t for c in self.candidates() if c.certificate is not None for t in c.certificate.tests]
        unreliable_held = sum(1 for r in eps if r.failure_mode == "physics_unreliable")
        return {
            "episodes": len(eps),
            "certificate_episodes": len(cert_tests),
            "bad_qacc": int(sum(r.bad_qacc for r in eps) + sum(t.bad_qacc for t in cert_tests)),
            "physics_invalid": int(modes.get("physics_invalid", 0))
            + sum(t.failure_mode == "physics_invalid" for t in cert_tests),
            "physics_unreliable": unreliable_held
            + sum(t.failure_mode == "physics_unreliable" for t in cert_tests),
            "held_episodes": len(held),
            "failure_modes": dict(sorted(modes.items())),
        }

    def tournament_doc(self) -> dict[str, Any]:
        return {
            "contract": CONTRACT_VERSION,
            "label": PRODUCT_MODE_LABEL,
            "part": _part_doc(self.part),
            "requirement": self.requirement.value,
            "seed": self.seed,
            "suite": self.suite,
            "conditions": [c.describe() for c in self.conditions],
            "families": dict(self.families),
            "stages": [
                "build_check",
                "budget",
                "reference",
                "screen",
                "finalists",
                "certificate",
                "recommendation",
            ],
            "groups": {
                g.value: {
                    "requested": s.requested,
                    "status": s.status,
                    "reason": s.reason,
                    "note": s.note,
                    "sampled_from": s.sampled_from,
                    "candidates": [c.effector_id for c in s.candidates],
                    "finalists": [c.effector_id for c in s.finalists],
                    "budget": s.budget(),
                }
                for g, s in self.groups.items()
            },
            "candidates": [c.summary() for c in self.candidates()],
            "design_rounds": self.design_rounds,
            "llm": dict(self.llm),
            "fairness": self.fairness(),
            "physics": self.physics(),
            "provenance": self.provenance(),
            "wall_s": round(self.wall_s, 3),
        }

    def provenance(self) -> dict[str, Any]:
        import mujoco

        return {
            "code_version": self.code_version,
            "task_digest": self.task_digest,
            "engine": f"MuJoCo {mujoco.__version__}",
            "executor": "cpu",
            "created_utc": self.created_utc,
        }

    def digest(self) -> str:
        """sha256 over what a re-run must reproduce: every record (no wall times) and the
        recommendation (no creation time)."""
        drop = {"wall_time_s"}
        rows = [{k: v for k, v in r.items() if k not in drop} for r in self.records()]
        rec = {k: v for k, v in self.recommendation.items() if k != "provenance"}
        return hashlib.sha256(canonical_json({"records": rows, "recommendation": rec}).encode()).hexdigest()


def _certificate_record(cand: Candidate, t: Any) -> dict[str, Any]:
    s = cand.spec
    return {
        "contract": CONTRACT_VERSION,
        "label": PRODUCT_MODE_LABEL,
        "effector_id": s.effector_id,
        "family": s.family.value,
        "model": s.model,
        "display_name": s.display_name,
        "primitive": s.primitive,
        "origin": s.origin,
        "group": s.group.value,
        "fidelity": s.fidelity.value,
        "recipe_id": s.recipe.recipe_id if s.recipe is not None else None,
        "part_id": cand.certificate.part_id if cand.certificate is not None else None,
        "condition_id": f"certificate/{t.name}",
        "seed": 0,
        "perturbation": t.params.get("perturbation") if isinstance(t.params, Mapping) else None,
        "success": bool(t.registered),
        "held": bool(t.held),
        "engaged": bool(t.engaged),
        "failure_mode": t.failure_mode,
        "registration_flags": list(t.registration_flags),
        "flags": list(t.flags),
        "max_lift": float(t.max_lift),
        "hold_time": float(t.hold_time),
        "bad_qacc": int(t.bad_qacc),
        "penetration_mm": _num(t.penetration_mm, 6),
        "reference": cand.certificate.reference if cand.certificate is not None else None,
        "traj_sha": t.traj_sha,
        "stage": "certificate",
        "source": cand.source,
        "registered": bool(t.registered),
        "reference_source": cand.certificate.reference if cand.certificate is not None else None,
        "m_offset_mm": _num(t.offset_mm, 6),
        "certificate_category": t.category,
    }


def _part_doc(part: PartSpec) -> dict[str, Any]:
    return {
        "part_id": part.name,
        "family": part.family,
        "size_mm": [round(1000.0 * float(v), 3) for v in part.size],
        "mass_kg": round(float(part.mass), 5),
        "geometry_sha256": part.geometry_sha256,
    }


def family_status(families: Sequence[EffectorFamily] | None) -> tuple[list[EffectorFamily], dict[str, str]]:
    status: dict[str, str] = {}
    out = []
    pool = list(families) if families is not None else list(EffectorFamily)
    for fam in EffectorFamily:
        if fam not in pool:
            status[fam.value] = "not requested"
            continue
        try:
            get_family(fam)
        except UnknownFamilyError as exc:
            status[fam.value] = f"not available: {exc}"
            continue
        status[fam.value] = "available"
        out.append(fam)
    return out, status


#: Families whose presets enter each budget group (origin "nemotron"/"random" recipes aside).
_GROUP_FAMILIES: Final[Mapping[EntrantGroup, tuple[EffectorFamily, ...]]] = {
    EntrantGroup.REFERENCE: (EffectorFamily.PARALLEL_JAW,),
    EntrantGroup.CATALOG: (EffectorFamily.CATALOG_HAND, EffectorFamily.CATALOG_GRIPPER),
    EntrantGroup.SOFT_ARM: (EffectorFamily.SOFT_ARM,),
    EntrantGroup.SUCTION: (EffectorFamily.SUCTION,),
    EntrantGroup.DESIGNED: (EffectorFamily.DESIGNED,),
    EntrantGroup.RANDOM: (EffectorFamily.DESIGNED,),
}


def _empty_reason(group: EntrantGroup, status: Mapping[str, str]) -> str:
    missing = [f.value for f in _GROUP_FAMILIES[group] if status.get(f.value) != "available"]
    if missing:
        return f"no family of this group is available ({', '.join(missing)} not built yet)"
    return "the group's families offered no preset for this part"


def _sample(
    cands: list[Candidate], k: int, seed: int, part: PartSpec, group: EntrantGroup
) -> list[Candidate]:
    """``k`` of ``cands`` chosen with a fixed seed (in entry order), the rest marked not sampled."""
    from rtt.sim.conditions import derive_seed

    if len(cands) <= k:
        return cands
    rng = np.random.default_rng(derive_seed(seed, part.name, part.geometry_sha256, group.value, "sample"))
    order = sorted(range(len(cands)), key=lambda i: cands[i].effector_id)
    keep = {order[i] for i in rng.choice(len(order), size=k, replace=False)}
    for i, cand in enumerate(cands):
        if i not in keep:
            cand.status, cand.reason = "not_sampled", f"the group had more than {k} buildable candidates"
    return [c for i, c in enumerate(cands) if i in keep]


def _finalist_key(cand: Candidate, requirement: Requirement) -> tuple[Any, ...]:
    rates = cand.screen_rates()
    mass = cand.mass_kg if cand.mass_kg is not None and math.isfinite(cand.mass_kg) else math.inf
    return (
        -rates[requirement.value],
        -rates["held"],
        cand.n_actuators if cand.n_actuators is not None else 10**6,
        mass,
        cand.effector_id,
    )


def _reference_jobs(
    cands: Sequence[Candidate], part: PartSpec, task: TaskV2
) -> list[tuple[Candidate, EffectorJob]]:
    """The certificate's nominal episode for every ``nominal_episode`` candidate (its reference)."""
    from rtt.effectors.runner import EffectorJob
    from rtt.sim import objects as objs
    from rtt.sim.certify import certificate_task, plan_tests

    ctask = certificate_task(task)
    nominal = plan_tests(part.to_mesh_object(), objs.Placement(), ctask)[0]
    if nominal.name != "nominal" or nominal.perturbation is not None:
        raise RuntimeError("the certificate battery no longer starts with the nominal test")
    out = []
    for cand in cands:
        if get_family(cand.spec.family).reference_mode != "nominal_episode":
            cand.reference_source = "analytic"
            continue
        out.append(
            (
                cand,
                EffectorJob(
                    cand.spec,
                    part,
                    nominal.placement,
                    0,
                    task=ctask,  # type: ignore[arg-type]
                    condition_id="certificate/nominal",
                ),
            )
        )
    return out


def _screen_jobs(
    cands: Sequence[Candidate], part: PartSpec, task: TaskV2, conditions: Sequence[ScreenCondition]
) -> list[tuple[Candidate, EffectorJob]]:
    from rtt.effectors.runner import EffectorJob

    return [
        (
            cand,
            EffectorJob(
                cand.spec,
                part,
                cond.placement,
                int(seed),
                task=task,
                reference=cand.reference,
                condition_id=cond.condition_id,
            ),
        )
        for cand in cands
        for cond in conditions
        for seed in cond.seeds
    ]


def _run_screen(
    cands: Sequence[Candidate],
    part: PartSpec,
    task: TaskV2,
    conditions: Sequence[ScreenCondition],
    executor: Executor,
    progress: Callable[[str, int, int], None] | None,
) -> None:
    from rtt.effectors.runner import run_effector_job

    refs = _reference_jobs(cands, part, task)
    if progress:
        progress("reference", 0, len(refs))
    for (cand, _), r in zip(refs, executor.map(run_effector_job, [j for _, j in refs]), strict=True):
        cand.reference_result = r
        ref = r.reference_pose() if r.held else None
        cand.reference = ref
        cand.reference_source = "nominal_episode" if ref is not None else "none"
    jobs = _screen_jobs(cands, part, task, conditions)
    if progress:
        progress("screen", 0, len(jobs))
    for (cand, _), r in zip(jobs, executor.map(run_effector_job, [j for _, j in jobs]), strict=True):
        cand.screen.append(r)
    for cand in cands:
        cand.status = "screened"
        if cand.mass_kg is None or not math.isfinite(cand.mass_kg):
            masses = [r.effector_mass_kg for r in cand.screen if math.isfinite(r.effector_mass_kg)]
            cand.mass_kg = masses[0] if masses else None


def _feedback(cand: Candidate) -> DesignFeedback:
    assert cand.spec.recipe is not None
    rates = cand.screen_rates()
    return DesignFeedback(
        cand.spec.recipe, rates["held"], rates["registered"], rates["failure_modes"], n_episodes=rates["n"]
    )


def run_tournament(
    part: PartSpec,
    *,
    requirement: Requirement | str = Requirement.REGISTERED,
    groups: Sequence[EntrantGroup | str] | None = None,
    seed: int = 0,
    workers: int | None = 1,
    conditions: Sequence[ScreenCondition] | None = None,
    suite: str | None = None,
    task: TaskV2 | None = None,
    families: Sequence[EffectorFamily | str] | None = None,
    design: bool = False,
    designer: MorphologyDesigner | None = None,
    executor: Executor | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> TournamentResult:
    """Run the tournament for ``part`` (module docstring) and recommend an effector.

    * ``requirement``: the headline (``registered`` by default); both are always computed.
    * ``groups``: budget groups to run (default: all six).
    * ``conditions``: the screen (default :func:`training_conditions`).
    * ``families``: restrict the families considered (default: every available one).
    * ``design``: run the Nemotron proposals of the designed group (``designer`` must hold a
      model client); without it the designed group holds the designed family's presets only.
    * ``designer``: the :class:`MorphologyDesigner` (default :func:`default_designer`, ``None``
      while track B3 is not built: the designed and random groups are then skipped).
    * ``executor``: runs the episodes and certificates (default: in-process for one worker,
      else one spawned pool for the whole tournament).
    """
    from rtt.effectors.recommend import recommend
    from rtt.sim.conditions import derive_seed
    from rtt.sim.replay import code_version as _code_version
    from rtt.sim.task import default_task_v2, task_digest

    t0 = time.perf_counter()
    req = Requirement(requirement)
    task = task or default_task_v2()
    wanted = tuple(EntrantGroup(g) for g in (groups if groups is not None else GROUP_ORDER))
    if not wanted:
        raise ValueError("run at least one group")
    if conditions is None:
        conditions = training_conditions(part)
        suite = suite or "train_v2"
    conditions = tuple(conditions)
    ids = [c.condition_id for c in conditions]
    if not conditions or len(set(ids)) != len(ids):
        raise ValueError("the screen needs conditions with unique ids")
    n_screen = sum(len(c.seeds) for c in conditions)
    if n_screen * MAX_SCREEN_CANDIDATES > MAX_SCREEN_EPISODES:
        raise ValueError(f"{n_screen} screen episodes per candidate would exceed the group budget")
    fams, fam_status = family_status([EffectorFamily(f) for f in families] if families is not None else None)
    designer = designer if designer is not None else default_designer()

    states = {g: GroupState(g, requested=g in wanted) for g in GROUP_ORDER}
    for g in GROUP_ORDER:
        if g not in wanted:
            states[g].status, states[g].reason = "skipped", "not requested"
    # 1. presets of every available family, bucketed by their budget group
    seen: set[str] = set()
    for fam in fams:
        for spec in get_family(fam).presets():
            if spec.effector_id in seen or spec.group not in wanted:
                continue
            seen.add(spec.effector_id)
            states[spec.group].candidates.append(Candidate(spec, "preset"))
    rounds: list[dict[str, Any]] = []
    llm = {"calls": 0, "usd": 0.0}

    def add_recipes(group: EntrantGroup, recipes: Sequence[MorphologyRecipe], source: str) -> list[Candidate]:
        added = []
        assert designer is not None
        for recipe in recipes:
            spec = designer.spec_for(recipe)
            if spec.group is not group:
                raise ContractError(
                    f"a {source} recipe made a spec of group {spec.group.value}, not {group.value}"
                )
            if spec.effector_id in seen:
                continue
            seen.add(spec.effector_id)
            cand = Candidate(spec, source)
            states[group].candidates.append(cand)
            added.append(cand)
        return added

    def designable() -> bool:
        return designer is not None and EffectorFamily.DESIGNED in fams

    design_on = EntrantGroup.DESIGNED in wanted and design
    if EntrantGroup.DESIGNED in wanted:
        if not designable():
            states[EntrantGroup.DESIGNED].reason = "the morphology designer (track B3) is not available"
            design_on = False
        elif not design:
            states[EntrantGroup.DESIGNED].reason = "Nemotron proposals not requested (design = false)"
    rseed = derive_seed(seed, part.name, part.geometry_sha256, "random")
    if EntrantGroup.RANDOM in wanted:
        if not designable():
            states[EntrantGroup.RANDOM].reason = "the morphology designer (track B3) is not available"
        elif not design_on:  # no Nemotron rounds to match: the fixed draw, screened with the presets
            assert designer is not None
            add_recipes(EntrantGroup.RANDOM, designer.random_recipes(part, RANDOM_RECIPES, rseed), "random")

    own_executor = executor is None
    ex: Executor = executor if executor is not None else TournamentExecutor(workers)
    try:
        # 2-3. build check, then the budget (presets of a designed group leave room for proposals)
        for g in GROUP_ORDER:
            st = states[g]
            for cand in st.candidates:
                _build_check(cand, part, task)
            buildable = [c for c in st.candidates if c.status == "pending"]
            room = MAX_SCREEN_CANDIDATES - (
                DESIGN_ROUNDS * DESIGNS_PER_ROUND if g is EntrantGroup.DESIGNED and design_on else 0
            )
            st.sampled_from = len(buildable)
            _sample(buildable, max(0, room), seed, part, g)
        # 4-5. references and the screen, every group in one batch
        first = [c for g in GROUP_ORDER for c in states[g].candidates if c.status == "pending"]
        _run_screen(first, part, task, conditions, ex, progress)
        # the designed group: Nemotron in rounds, each round screened before the next; then the
        # random control, matched to the Nemotron designs screened
        if design_on:
            assert designer is not None
            st = states[EntrantGroup.DESIGNED]
            offered: tuple[str, ...] = ()
            for rnd in range(1, DESIGN_ROUNDS + 1):
                history = [
                    _feedback(c)
                    for c in st.screened()
                    if c.spec.recipe is not None and c.source.startswith("nemotron")
                ]
                try:
                    batch = designer.propose(part, history, DESIGNS_PER_ROUND)
                except Exception as exc:  # noqa: BLE001 - a failed or refused call ends the rounds
                    error = f"{type(exc).__name__}: {exc}"[:400]
                    rounds.append({"round": rnd, "error": error, "history_shown": len(history)})
                    st.reason = f"Nemotron round {rnd} did not complete ({error}); the group keeps its earlier designs"
                    break
                llm["calls"] += int(batch.calls)
                llm["usd"] = round(llm["usd"] + float(batch.cost_usd), 6)
                added = add_recipes(
                    EntrantGroup.DESIGNED, batch.recipes[:DESIGNS_PER_ROUND], f"nemotron-{rnd}"
                )
                for cand in added:
                    _build_check(cand, part, task)
                fresh = [c for c in added if c.status == "pending"]
                _run_screen(fresh, part, task, conditions, ex, progress)
                rounds.append(
                    {
                        "round": rnd,
                        "proposed": [c.effector_id for c in added],
                        "screened": [c.effector_id for c in fresh],
                        "rejections": [dict(r) for r in batch.rejections],
                        "history_shown": len(history),
                        "calls": int(batch.calls),
                        "cost_usd": round(float(batch.cost_usd), 6),
                        "analysis": batch.analysis[:2000],
                    }
                )
                offered = offered or batch.kinds
            if EntrantGroup.RANDOM in wanted:
                target = sum(c.source.startswith("nemotron") for c in st.screened())
                _match_random(
                    designer,
                    states[EntrantGroup.RANDOM],
                    target,
                    offered,
                    rseed,
                    add_recipes,
                    lambda c: _build_check(c, part, task),
                    lambda cs: _run_screen(cs, part, task, conditions, ex, progress),
                    part,
                )
        for g, st in states.items():
            if st.requested and not st.candidates:
                st.status = "empty"
                st.reason = st.reason or _empty_reason(g, fam_status)
        # 6. finalists, 7. certificates
        finalists: list[Candidate] = []
        for g in GROUP_ORDER:
            st = states[g]
            ranked = sorted(st.screened(), key=lambda c: _finalist_key(c, req))
            st.finalists = ranked[:CERTIFICATES_PER_GROUP]
            for cand in st.finalists:
                cand.finalist = True
            finalists += st.finalists
        if progress:
            progress("certificate", 0, len(finalists))
        certs = ex.map(_certify_job, [(c.spec, part, task) for c in finalists])
        for cand, cert in zip(finalists, certs, strict=True):
            cand.certificate = cert
    finally:
        if own_executor:
            ex.close()  # type: ignore[attr-defined]

    created = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        code = _code_version()
    except ValueError:
        code = "unknown"
    result = TournamentResult(
        part=part,
        requirement=req,
        seed=int(seed),
        suite=suite or "custom",
        conditions=conditions,
        groups=states,
        families=fam_status,
        recommendation={},
        design_rounds=rounds,
        llm=llm,
        task_digest=task_digest(task),
        code_version=code,
        created_utc=created,
        wall_s=0.0,
    )
    result.recommendation = recommend(result)
    result.wall_s = time.perf_counter() - t0
    return result


# --------------------------------------------------------------------------- outputs


def spec_from_dict(d: Mapping[str, Any]) -> EffectorSpec:
    """An :class:`EffectorSpec` back from :meth:`EffectorSpec.to_dict` (its id must match)."""
    r = d.get("recipe")
    recipe = None
    if r:
        recipe = MorphologyRecipe(
            family=r["family"],
            kind=r["kind"],
            params=r["params"],
            hypothesis=r.get("hypothesis", ""),
            origin=r.get("origin", "manual"),
            parents=tuple(r.get("parents", ())),
            schema=r.get("schema", "rtt-morphology/1"),
        )
    spec = EffectorSpec(
        family=d["family"],
        model=d["model"],
        display_name=d["display_name"],
        primitive=d["primitive"],
        params=d.get("params") or {},
        recipe=recipe,
        origin=d["origin"],
    )
    if "effector_id" in d and spec.effector_id != d["effector_id"]:
        raise ValueError(f"spec {d.get('display_name')!r} does not hash to its recorded effector_id")
    return spec


def write_outputs(result: TournamentResult, out_dir: str | Path) -> dict[str, Path]:
    """Write ``records.parquet``, ``certificates/*.json``, ``tournament.json`` and
    ``recommendation.json`` under ``out_dir`` (``runs/effectors/<part>/``)."""
    from rtt.store.manifest import write_json_atomic
    from rtt.store.runs import write_parquet

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cert_dir = out / "certificates"
    if cert_dir.exists():
        for old in cert_dir.glob("*.json"):
            old.unlink()
    cert_dir.mkdir(exist_ok=True)
    rows = result.records()
    paths = {"records": write_parquet(out / "records.parquet", _parquet_rows(rows))}
    for cand in result.candidates():
        if cand.certificate is not None:
            path = cert_dir / f"{cand.spec.short_id}.json"
            write_json_atomic(path, cand.certificate.to_dict())
    paths["certificates"] = cert_dir
    tour = out / "tournament.json"
    write_json_atomic(tour, {**result.tournament_doc(), "digest": result.digest()})
    paths["tournament"] = tour
    rec = out / "recommendation.json"
    write_json_atomic(rec, result.recommendation)
    paths["recommendation"] = rec
    return paths


def _parquet_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rows with one column set (missing keys as null; lists as JSON text so types stay uniform)."""
    import json

    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    out = []
    for r in rows:
        row = {}
        for k in keys:
            v = r.get(k)
            if isinstance(v, list | tuple | dict):
                v = json.dumps(v, sort_keys=True)
            row[k] = v
        out.append(row)
    return out


def jsonable(value: Any) -> Any:
    """``value`` with dataclasses and tuples turned into JSON data (for debugging and the CLI)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    return value
