"""The per-part effector tournament, the recommendation and the agent's ``recommend_effector`` tool.

Product mode, not pre-registered (``docs/EFFECTORS.md`` sections 4, 5 and 7). Physics runs on the
toy native family of the contract tests (``tests/effector_toy.py``) and the parallel-jaw reference;
the budget, designed-group and recommendation rules run on a fake executor that answers episodes
and certificates from a table (no MuJoCo), so every rule is checked quickly and exactly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from effector_toy import ToyTwoPad
from rtt.agent import effector_tool as et
from rtt.agent.caps import AgentCaps, DailySpendBook, GuardedLedger, SpendGuard
from rtt.agent.engineer import AgentLLM, GraspEngineer, system_prompt
from rtt.agent.fakes import FakePartBackend, _response
from rtt.agent.jobs import LocalJobRunner
from rtt.agent.recorder import PRODUCT_MODE_LABEL, SessionRecorder, load_session
from rtt.agent.schemas import (
    EFFECTOR_TOOL_SPECS,
    TOOL_NAMES,
    ArgsError,
    openai_tools,
    validate_args,
)
from rtt.agent.store import AgentStore
from rtt.agent.tools import CERT_PASS_SCORE, Toolbox
from rtt.effectors import registry
from rtt.effectors.certificate import EffectorCertificate, EffectorCertificateTest
from rtt.effectors.parallel_jaw import jaw_spec
from rtt.effectors.recommend import APPROXIMATION_CAVEAT, recommend_for
from rtt.effectors.runner import EffectorJob, run_effector_job
from rtt.effectors.tournament import (
    CERTIFICATES_PER_GROUP,
    MAX_SCREEN_CANDIDATES,
    MAX_SCREEN_EPISODES,
    Candidate,
    DesignBatch,
    _build_check,
    _certify_job,
    _finalist_key,
    run_tournament,
    spec_from_dict,
    training_conditions,
    upload_conditions,
)
from rtt.effectors.types import (
    PRODUCT_MODE_LABEL as EFF_LABEL,
)
from rtt.effectors.types import (
    EffectorEpisodeResult,
    EffectorFamily,
    EffectorSpec,
    EntrantGroup,
    MorphologyRecipe,
    Requirement,
    find_brand_tokens,
)
from rtt.parts.library import get_part
from rtt.parts.suite import train_suite
from rtt.sim.task import default_task_v2
from rtt.store.runs import read_table

PART = "part-0123456789ab"


@pytest.fixture(scope="module")
def block() -> Any:
    return get_part("block")


@pytest.fixture(scope="module")
def task() -> Any:
    return default_task_v2()


# --------------------------------------------------------------------------- fakes


class ToyPresets(ToyTwoPad):
    """The toy pad gripper at several grip forces (catalog group)."""

    def __init__(self, forces: Sequence[float] = (25.0,)) -> None:
        self.forces = tuple(forces)

    def presets(self) -> tuple[EffectorSpec, ...]:
        return tuple(
            EffectorSpec(
                family=EffectorFamily.CATALOG_GRIPPER,
                model="toy-two-pad",
                display_name=f"Two-pad test gripper, {f:g} N",
                primitive="pinch",
                params={"force_n": f},
                origin="catalog",
            )
            for f in self.forces
        )


class DesignedPads(ToyTwoPad):
    """A designable toy: the pad gripper's grip force is the recipe's only parameter."""

    family: ClassVar[EffectorFamily] = EffectorFamily.DESIGNED

    def presets(self) -> tuple[EffectorSpec, ...]:
        return ()

    def recipe_kinds(self) -> tuple[str, ...]:
        return ("pads",)

    def spec_from_recipe(self, recipe: MorphologyRecipe) -> EffectorSpec:
        f = float(recipe.params["force_n"])
        return EffectorSpec(
            EffectorFamily.DESIGNED,
            "toy-pads",
            f"Designed pads, {f:g} N",
            "pinch",
            {"force_n": f},
            recipe=recipe,
            origin=recipe.origin,
        )


class FakeDesigner:
    """Proposals without a model: round r proposes forces 10 r + i; the random arm draws forces."""

    def __init__(self) -> None:
        self.histories: list[list[Any]] = []

    def random_recipes(self, part: Any, n: int, seed: int) -> list[MorphologyRecipe]:
        rng = np.random.default_rng(seed)
        forces = rng.choice(np.arange(5, 60), size=n, replace=False)
        return [
            MorphologyRecipe(EffectorFamily.DESIGNED, "pads", {"force_n": int(f)}, origin="random")
            for f in forces
        ]

    def propose(self, part: Any, history: Sequence[Any], n: int) -> DesignBatch:
        self.histories.append(list(history))
        r = len(self.histories)
        recipes = tuple(
            MorphologyRecipe(
                EffectorFamily.DESIGNED,
                "pads",
                {"force_n": 10 * r + i},
                hypothesis=f"round {r} idea {i}",
                origin="nemotron",
            )
            for i in range(n)
        )
        return DesignBatch(
            recipes, ({"slot": "design_5", "reasons": ["too narrow"]},), calls=1, cost_usd=0.004
        )

    def spec_for(self, recipe: MorphologyRecipe) -> EffectorSpec:
        return DesignedPads().spec_from_recipe(recipe)


def fake_result(
    job: EffectorJob, held: bool, registered: bool, *, mode: str | None = None
) -> EffectorEpisodeResult:
    spec = job.spec
    sha = hashlib.sha256(f"{spec.effector_id}|{job.condition_id}|{job.seed}".encode()).hexdigest()
    failure = None if registered else (mode or ("not_registered" if held else "missed_grasp"))
    return EffectorEpisodeResult(
        spec=spec,
        part_id=job.part.name,
        seed=job.seed,
        success=registered,
        held=held,
        engaged=registered,
        failure_mode=failure,
        registration_flags=(),
        flags=(),
        max_lift=0.15 if held else 0.0,
        hold_time=2.0 if held else 0.0,
        bad_qacc=1 if failure == "physics_invalid" else 0,
        penetration_mm=0.2,
        traj_sha=sha,
        n_steps=1000,
        wall_time_s=0.0,
        metrics={},
        reference=job.reference.source if job.reference else "analytic",
        reg_pos=(0.0, 0.0, -0.06),
        reg_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        condition_id=job.condition_id,
        effector_mass_kg=0.05,
    )


def fake_certificate(
    spec: EffectorSpec,
    *,
    held_score: float = 1.0,
    registered_score: float = 1.0,
    held_tol: Mapping[str, float | None] | None = None,
    reg_tol: Mapping[str, float | None] | None = None,
    nominal_held: bool = True,
    nominal_registered: bool = True,
    nominal_mode: str | None = None,
    modes: Sequence[str] = (),
    part_id: str = "block",
) -> EffectorCertificate:
    held_tol = dict(held_tol or {"x": 3.0, "y": 3.0})
    reg_tol = dict(reg_tol or {"x": 3.0, "y": 3.0})
    tests = [
        EffectorCertificateTest(
            "nominal",
            "placement",
            {"axis": None, "error_mm": 0.0},
            nominal_held,
            nominal_registered,
            nominal_registered,
            nominal_mode,
            (),
            0.0,
            0.15,
            2.0,
            0.1,
            0,
            (),
            "0" * 64,
        )
    ]
    for i in range(17):
        mode = modes[i] if i < len(modes) else None
        tests.append(
            EffectorCertificateTest(
                f"t{i}",
                "placement" if i < 12 else "shake" if i < 15 else "push",
                {"axis": "x", "error_mm": 1.0},
                mode is None,
                mode is None,
                mode is None,
                mode,
                (),
                0.1,
                0.15,
                2.0,
                0.1,
                0,
                (),
                f"{i:064x}",
            )
        )
    cats = {"placement": 1.0, "shake": 1.0, "push": 1.0}
    return EffectorCertificate(
        spec=spec,
        part_id=part_id,
        task="task_v2",
        task_digest="0" * 12,
        reference="analytic",
        tests=tuple(tests),
        held_scores=cats,
        registered_scores=cats,
        held_score=held_score,
        registered_score=registered_score,
        held_tolerance_mm=held_tol,  # type: ignore[arg-type]
        registered_tolerance_mm=reg_tol,  # type: ignore[arg-type]
        effector_mass_kg=0.05,
    )


class FakePhysics:
    """An executor that answers episodes and certificates from functions of the spec (no MuJoCo)."""

    def __init__(
        self,
        outcome: Callable[[EffectorJob], tuple[bool, bool]] | None = None,
        certificate: Callable[[EffectorSpec], EffectorCertificate] | None = None,
    ) -> None:
        self.outcome = outcome or (lambda job: (True, True))
        self.certificate = certificate or (lambda spec: fake_certificate(spec))
        self.jobs: list[EffectorJob] = []
        self.certified: list[EffectorSpec] = []

    def map(self, fn: Callable[[Any], Any], items: Sequence[Any]) -> list[Any]:
        if fn is run_effector_job:
            self.jobs += list(items)
            return [fake_result(j, *self.outcome(j)) for j in items]
        if fn is _certify_job:
            self.certified += [spec for spec, _, _ in items]
            return [self.certificate(spec) for spec, _, _ in items]
        raise AssertionError(f"unexpected function {fn}")


def force_of(job_or_spec: Any) -> float:
    spec = getattr(job_or_spec, "spec", job_or_spec)
    return float(spec.params.get("force_n", 0.0))


# --------------------------------------------------------------------------- conditions


def test_screen_is_the_parts_four_train_v2_placements_with_two_seeds(block: Any) -> None:
    conds = training_conditions(block)
    suite = [c for c in train_suite() if c.part.name == "block"]
    assert [c.label for c in conds] == ["c0", "c1", "c2", "c3"]
    assert [(c.condition_id, c.seeds) for c in conds] == [(c.condition_id, tuple(c.seeds)) for c in suite]
    assert all(len(c.seeds) == 2 for c in conds) and sum(len(c.seeds) for c in conds) == 8
    up = upload_conditions(block, suite="upload")
    assert [c.label for c in up] == ["c0", "c1", "c2", "c3"] and up == upload_conditions(
        block, suite="upload"
    )
    assert all(len(c.seeds) == 2 for c in up) and {c.placement for c in up} == {c.placement for c in conds}
    with pytest.raises(ValueError, match="not a train_v2 part"):
        training_conditions(dataclasses.replace(block, name="other"))


# --------------------------------------------------------------------------- fairness


def test_every_family_shares_the_wrist_solver_part_and_phases(block: Any, task: Any) -> None:
    """The jaw's frozen scene and a native family's scene compile to the same wrist servos, solver
    options, part contact parameters and phase timings (from the compiled models)."""
    jaw = Candidate(jaw_spec("stock"), "preset")
    _build_check(jaw, block, task)
    toys = [Candidate(s, "preset") for s in ToyPresets((10.0, 40.0)).presets()]
    with registry.override_family(ToyPresets((10.0, 40.0))):
        for c in toys:
            _build_check(c, block, task)
    for c in (jaw, *toys):
        assert c.status == "pending" and c.fairness is not None, c.reason
    assert all(c.fairness == jaw.fairness for c in toys)
    sig = jaw.fairness
    assert sig["wrist"]["palm_z"]["forcerange"] == [-150.0, 150.0] and sig["wrist"]["palm_x"][
        "forcerange"
    ] == [-100.0, 100.0]
    assert sig["wrist"]["palm_yaw"]["forcerange"] == [-15.0, 15.0] and sig["wrist"]["part"]["priority"] == [1]
    assert sig["wrist"]["option"]["timestep"] == 0.001 and sig["wrist"]["palm"]["mass"] == 0.4
    assert sig["phases"]["lift_m"] == pytest.approx(0.15) and sig["phases"]["t_hold"] == pytest.approx(4.1)
    assert jaw.n_actuators == 1 and toys[0].n_actuators == 2 and jaw.actuation == {"grip_force_n": 50.0}


def test_identical_conditions_and_seeds_for_every_candidate(block: Any) -> None:
    fake = FakePhysics()
    with registry.override_family(ToyPresets((10.0, 20.0, 40.0))):
        res = run_tournament(block, families=["parallel_jaw", "catalog_gripper"], executor=fake)
    want = [(c.condition_id, s) for c in training_conditions(block) for s in c.seeds]
    screen = [j for j in fake.jobs if not j.condition_id.startswith("certificate/")]
    by_spec: dict[str, list[tuple[str, int]]] = {}
    for j in screen:
        by_spec.setdefault(j.spec.effector_id, []).append((j.condition_id, j.seed))
    assert len(by_spec) == 6 and all(v == want for v in by_spec.values())
    assert len({id(j.task) for j in screen}) == 1  # one task object: the same wrist, phases and scorer
    refs = [j for j in fake.jobs if j.condition_id == "certificate/nominal"]
    assert {j.spec.family for j in refs} == {EffectorFamily.CATALOG_GRIPPER}  # the jaw is analytic
    fair = res.fairness()
    assert fair["identical"] and fair["problems"] == [] and fair["candidates_checked"] == 6
    assert fair["conditions"] == [c.condition_id for c in training_conditions(block)]


# --------------------------------------------------------------------------- budgets


def test_each_group_is_capped_at_96_screen_episodes_and_2_certificates(block: Any) -> None:
    forces = tuple(float(f) for f in range(5, 45, 2))  # 20 presets
    fake = FakePhysics(outcome=lambda job: (True, force_of(job) > 20))
    with registry.override_family(ToyPresets(forces)):
        res = run_tournament(block, groups=["catalog"], families=["catalog_gripper"], executor=fake, seed=3)
        again = run_tournament(
            block, groups=["catalog"], families=["catalog_gripper"], executor=FakePhysics(), seed=3
        )
        other = run_tournament(
            block, groups=["catalog"], families=["catalog_gripper"], executor=FakePhysics(), seed=4
        )
    st = res.groups[EntrantGroup.CATALOG]
    budget = st.budget()
    assert budget["candidates"] == 20 and budget["screened"] == MAX_SCREEN_CANDIDATES
    assert (
        budget["screen_episodes"] == MAX_SCREEN_EPISODES == 96
        and budget["certificates"] == CERTIFICATES_PER_GROUP
    )
    assert sum(c.status == "not_sampled" for c in st.candidates) == 8
    assert all("more than 12" in (c.reason or "") for c in st.candidates if c.status == "not_sampled")
    sampled = {c.effector_id for c in st.screened()}
    assert sampled == {c.effector_id for c in again.groups[EntrantGroup.CATALOG].screened()}  # fixed seed
    assert sampled != {c.effector_id for c in other.groups[EntrantGroup.CATALOG].screened()}
    assert len(fake.certified) == 2 and all(force_of(s) > 20 for s in fake.certified)
    for g, s in res.groups.items():
        assert s.budget()["screen_episodes"] <= 96 and s.budget()["certificates"] <= 2, g
    assert res.recommendation["budget"]["catalog"]["screen_episodes"] == 96


def test_finalist_ties_break_on_held_rate_actuators_mass_then_id() -> None:
    def cand(force: float, held: int, reg: int, n_act: int, mass: float) -> Candidate:
        spec = ToyPresets((force,)).presets()[0]
        job = EffectorJob(spec, get_part("block"), None, 0)  # type: ignore[arg-type]
        screen = [fake_result(job, i < held, i < reg) for i in range(8)]
        return Candidate(spec, "preset", status="screened", n_actuators=n_act, mass_kg=mass, screen=screen)

    a = cand(1, held=8, reg=4, n_act=2, mass=0.05)
    b = cand(2, held=6, reg=4, n_act=1, mass=0.01)  # same registered rate, lower held rate
    c = cand(3, held=8, reg=4, n_act=1, mass=0.09)  # ties a on rates, fewer actuators
    d = cand(4, held=8, reg=4, n_act=1, mass=0.02)  # ties c, lighter
    e = cand(5, held=1, reg=5, n_act=9, mass=1.0)  # best registered rate wins outright
    order = sorted([a, b, c, d, e], key=lambda x: _finalist_key(x, Requirement.REGISTERED))
    assert order == [e, d, c, a, b]
    held_order = sorted([a, b, c, d, e], key=lambda x: _finalist_key(x, Requirement.HELD))
    assert held_order[:3] == [d, c, a]


# --------------------------------------------------------------------------- build check


def test_unbuildable_candidates_are_listed_with_their_reason_and_score_zero(block: Any) -> None:
    class Wide(ToyPresets):
        def build(self, ctx: Any) -> Any:
            if float(ctx.effector.params["force_n"]) > 30:
                from rtt.effectors import InvalidEffector

                raise InvalidEffector("the opening is narrower than the part")
            return super().build(ctx)

    with registry.override_family(Wide((10.0, 40.0))):
        res = run_tournament(block, groups=["catalog"], families=["catalog_gripper"], executor=FakePhysics())
    bad = next(c for c in res.candidates() if force_of(c.spec) == 40.0)
    assert bad.status == "invalid_build" and "narrower" in (bad.reason or "") and not bad.screen
    row = next(r for r in res.recommendation["ranking"] if r["effector_id"] == bad.effector_id)
    assert row["eligible"] is False and "cannot be built" in row["reason"] and row["score"] is None
    assert res.groups[EntrantGroup.CATALOG].budget()["screen_episodes"] == 8


# --------------------------------------------------------------------------- designed groups


def test_designed_group_runs_two_rounds_of_four_and_random_draws_eight(block: Any) -> None:
    designer = FakeDesigner()
    fake = FakePhysics(outcome=lambda job: (force_of(job) >= 20, force_of(job) >= 25))
    with registry.override_family(DesignedPads()):
        res = run_tournament(
            block,
            groups=["designed", "random"],
            families=["designed"],
            design=True,
            designer=designer,
            executor=fake,
        )
    d, r = res.groups[EntrantGroup.DESIGNED], res.groups[EntrantGroup.RANDOM]
    assert [c.source for c in d.candidates] == ["nemotron-1"] * 4 + ["nemotron-2"] * 4
    assert len(r.candidates) == 8 and all(c.spec.origin == "random" for c in r.candidates)
    assert d.budget()["screen_episodes"] == 64 and r.budget()["screen_episodes"] == 64
    assert d.budget()["certificates"] == 2 and r.budget()["certificates"] == 2
    assert designer.histories[0] == [] and len(designer.histories[1]) == 4  # round 2 sees round 1's physics
    shown = {h.recipe.recipe_id: h for h in designer.histories[1]}
    for c in d.candidates[:4]:
        assert c.spec.recipe is not None
        assert shown[c.spec.recipe.recipe_id].held_rate == c.screen_rates()["held"]
    assert res.llm == {"calls": 2, "usd": 0.008}
    assert [x["round"] for x in res.design_rounds] == [1, 2] and res.design_rounds[0]["rejections"]
    assert res.tournament_doc()["groups"]["designed"]["status"] == "run"


def test_the_random_control_is_matched_to_the_nemotron_designs_screened(block: Any) -> None:
    """Same count as Nemotron's screened designs, the same spaces, drawn after the rounds."""
    from rtt.effectors.tournament import RANDOM_RECIPES
    from rtt.effectors.types import InvalidEffector

    class Picky(DesignedPads):
        def build(self, ctx: Any) -> Any:
            if int(ctx.effector.params["force_n"]) % 2:
                raise InvalidEffector("odd forces do not build here")
            return super().build(ctx)

    class Matched(FakeDesigner):
        def __init__(self) -> None:
            super().__init__()
            self.draws: list[tuple[int, tuple[str, ...]]] = []

        def random_recipes(  # type: ignore[override]
            self, part: Any, n: int, seed: int, *, kinds: Sequence[str] | None = None
        ) -> list[MorphologyRecipe]:
            self.draws.append((n, tuple(kinds or ())))
            return [
                MorphologyRecipe(EffectorFamily.DESIGNED, "pads", {"force_n": 31 + i}, origin="random")
                for i in range(n)
            ]

        def propose(self, part: Any, history: Sequence[Any], n: int) -> DesignBatch:
            batch = super().propose(part, history, n)
            keep = batch.recipes if len(self.histories) == 1 else batch.recipes[:1]
            return dataclasses.replace(batch, recipes=keep, kinds=("pads",))

    designer = Matched()
    with registry.override_family(Picky()):
        res = run_tournament(
            block,
            groups=["designed", "random"],
            families=["designed"],
            design=True,
            designer=designer,
            executor=FakePhysics(),
        )
    d, r = res.groups[EntrantGroup.DESIGNED], res.groups[EntrantGroup.RANDOM]
    # Nemotron: 10, 11, 12, 13 then 20; the odd ones do not build, so 3 are screened
    assert [force_of(c.spec) for c in d.screened()] == [10.0, 12.0, 20.0]
    # the control grows its draw (a prefix-stable sequence) until exactly 3 build: 32, 34, 36
    assert [force_of(c.spec) for c in r.screened()] == [32.0, 34.0, 36.0]
    assert [c.status for c in r.candidates].count("invalid_build") == 3
    assert designer.draws == [(3, ("pads",)), (5, ("pads",)), (6, ("pads",))]
    assert d.budget()["screen_episodes"] == r.budget()["screen_episodes"] == 24
    assert r.note == "matched to the 3 Nemotron designs screened (pads)"
    assert res.tournament_doc()["groups"]["random"]["note"] == r.note
    # without the Nemotron rounds the control keeps its fixed draw, screened with the presets
    plain = Matched()
    with registry.override_family(DesignedPads()):
        alone = run_tournament(
            block, groups=["random"], families=["designed"], designer=plain, executor=FakePhysics()
        )
    ra = alone.groups[EntrantGroup.RANDOM]
    assert len(ra.screened()) == RANDOM_RECIPES and ra.note is None and plain.draws == [(RANDOM_RECIPES, ())]


def test_a_failed_proposal_round_keeps_the_earlier_designs(block: Any) -> None:
    class Flaky(FakeDesigner):
        def propose(self, part: Any, history: Sequence[Any], n: int) -> DesignBatch:
            if self.histories:
                raise RuntimeError("the spend cap refused the call")
            return super().propose(part, history, n)

    with registry.override_family(DesignedPads()):
        res = run_tournament(
            block,
            groups=["designed"],
            families=["designed"],
            design=True,
            designer=Flaky(),
            executor=FakePhysics(),
        )
    d = res.groups[EntrantGroup.DESIGNED]
    assert [c.source for c in d.candidates] == ["nemotron-1"] * 4 and d.budget()["certificates"] == 2
    assert res.design_rounds[1]["error"].startswith("RuntimeError: the spend cap")
    assert "round 2 did not complete" in (d.reason or "") and res.recommendation["winner"] is not None


def test_designed_and_random_groups_are_skipped_without_the_designer(block: Any) -> None:
    res = run_tournament(
        block, groups=["designed", "random"], families=["parallel_jaw"], designer=None, executor=FakePhysics()
    )
    groups = res.recommendation["groups"]
    assert groups["designed"]["status"] == "empty" and "track B3" in groups["designed"]["reason"]
    assert groups["random"]["status"] == "empty" and res.recommendation["winner"] is None
    assert res.recommendation["no_winner"]["reason"] == "no candidate was run"


# --------------------------------------------------------------------------- the recommendation


def _cand(
    family: str,
    name: str,
    *,
    n_act: int = 2,
    mass: float = 0.1,
    screen: tuple[int, int] = (8, 8),
    cert: dict[str, Any] | None = None,
    modes: Sequence[str] = (),
) -> Candidate:
    spec = EffectorSpec(family, name.lower().replace(" ", "-"), name, "pinch", {}, origin="preset")
    job = EffectorJob(spec, get_part("block"), None, 0)  # type: ignore[arg-type]
    eps = [
        fake_result(job, i < screen[0], i < screen[1], mode=modes[i] if i < len(modes) else None)
        for i in range(8)
    ]
    certificate = fake_certificate(spec, **cert) if cert is not None else None
    return Candidate(
        spec,
        "preset",
        status="screened",
        n_actuators=n_act,
        mass_kg=mass,
        screen=eps,
        certificate=certificate,
    )


def test_ranking_is_score_then_tolerance_then_screen() -> None:
    a = _cand("catalog_hand", "Hand A", cert={"registered_score": 0.80})
    b = _cand("catalog_hand", "Hand B", cert={"registered_score": 0.95, "reg_tol": {"x": 1.0, "y": 3.0}})
    c = _cand("catalog_hand", "Hand C", cert={"registered_score": 0.95, "reg_tol": {"x": 2.0, "y": 2.0}})
    d = _cand(
        "catalog_hand",
        "Hand D",
        screen=(8, 7),
        cert={"registered_score": 0.95, "reg_tol": {"x": 2.0, "y": 2.0}},
    )
    block = recommend_for([a, b, c, d], Requirement.REGISTERED)
    assert [r["display_name"] for r in block["ranking"]] == ["Hand C", "Hand D", "Hand B", "Hand A"]
    assert block["winner"]["effector"]["display_name"] == "Hand C" and block["runner_up_rigid"] is None
    assert "screen success" in block["winner"]["why"]


def test_a_full_tie_is_explained_by_simplicity_not_by_screen_success() -> None:
    """Equal score, tolerance and screen: the first on the base order wins because it is simpler."""
    cert = {"registered_score": 1.0, "reg_tol": {"x": 3.0, "y": 3.0}}
    other = _cand("catalog_hand", "Hand", n_act=4, cert=cert)
    simple = next(
        c
        for c in (_cand("catalog_gripper", f"Gripper {i}", n_act=1, cert=cert) for i in range(64))
        if c.effector_id < other.effector_id
    )
    why = recommend_for([other, simple], Requirement.REGISTERED)["winner"]["why"]
    assert why.startswith("Ties with Hand on registered certificate score (1.00)")
    assert "is simpler: 1 actuator against 4" in why and "higher screen" not in why
    twin = next(
        c
        for c in (_cand("catalog_gripper", f"Twin {i}", n_act=1, cert=cert) for i in range(64))
        if c.effector_id > simple.effector_id
    )
    why = recommend_for([twin, simple], Requirement.REGISTERED)["winner"]["why"]
    assert why.endswith("the order falls to the effector id.")


def test_simplicity_rule_prefers_fewer_actuators_then_mass_then_fidelity() -> None:
    hand = _cand("catalog_hand", "Hand", n_act=16, mass=0.5, cert={"registered_score": 0.97})
    jaw = _cand("parallel_jaw", "Jaw", n_act=1, mass=0.03, cert={"registered_score": 0.93})
    block = recommend_for([hand, jaw], Requirement.REGISTERED)
    assert block["winner"]["effector"]["display_name"] == "Jaw"  # within 0.05, same tolerance, 1 actuator
    assert "simpler: 1 actuator against 16" in block["winner"]["why"]
    far = _cand("parallel_jaw", "Far jaw", n_act=1, cert={"registered_score": 0.90})
    assert recommend_for([hand, far], Requirement.REGISTERED)["winner"]["effector"]["display_name"] == "Hand"
    tol = _cand(
        "parallel_jaw", "Tight jaw", n_act=1, cert={"registered_score": 0.96, "reg_tol": {"x": 2.0, "y": 3.0}}
    )
    assert recommend_for([hand, tol], Requirement.REGISTERED)["winner"]["effector"]["display_name"] == "Hand"
    light = _cand("catalog_hand", "Light hand", n_act=16, mass=0.2, cert={"registered_score": 0.95})
    assert (
        recommend_for([hand, light], Requirement.REGISTERED)["winner"]["effector"]["display_name"]
        == "Light hand"
    )
    rigid = _cand("catalog_gripper", "Gripper", n_act=1, mass=0.1, cert={"registered_score": 0.95})
    cup = _cand("suction", "Cup", n_act=1, mass=0.1, cert={"registered_score": 0.96})
    assert (
        recommend_for([cup, rigid], Requirement.REGISTERED)["winner"]["effector"]["display_name"] == "Gripper"
    )


def test_an_approximation_winner_always_carries_a_rigid_runner_up() -> None:
    cup = _cand("suction", "Cup", n_act=1, cert={"registered_score": 1.0})
    hand = _cand("catalog_hand", "Hand", n_act=16, cert={"registered_score": 0.70})
    block = recommend_for([cup, hand], Requirement.REGISTERED)
    assert (
        block["winner"]["effector"]["display_name"] == "Cup"
        and block["winner"]["caveat"] == APPROXIMATION_CAVEAT
    )
    assert (
        block["runner_up_rigid"]["effector"]["display_name"] == "Hand"
        and block["runner_up_rigid"]["basis"] == "eligible"
    )
    # the rigid candidate is not eligible: it is still named, with its basis
    weak = _cand("catalog_hand", "Weak hand", cert={"nominal_registered": False, "registered_score": 0.1})
    block = recommend_for([cup, weak], Requirement.REGISTERED)
    assert block["runner_up_rigid"]["effector"]["display_name"] == "Weak hand"
    assert block["runner_up_rigid"]["basis"] == "certified, not eligible"
    screened = _cand("parallel_jaw", "Jaw", screen=(4, 2))
    block = recommend_for([cup, screened], Requirement.REGISTERED)
    assert block["runner_up_rigid"]["basis"] == "screened only"
    block = recommend_for([cup], Requirement.REGISTERED)
    assert block["runner_up_rigid"] == {"effector": None, "basis": "no core or rigid candidate was run"}


def test_no_winner_lists_the_closest_candidates_and_never_forces_a_pick() -> None:
    a = _cand(
        "catalog_hand", "Hand", screen=(6, 0), cert={"nominal_registered": False, "registered_score": 0.3}
    )
    b = _cand(
        "parallel_jaw", "Jaw", screen=(8, 1), modes=("", "physics_invalid"), cert={"registered_score": 0.9}
    )
    c = _cand("catalog_gripper", "Gripper", screen=(2, 0))
    block = recommend_for([a, b, c], Requirement.REGISTERED)
    assert block["winner"] is None and block["runner_up_rigid"] is None
    closest = block["no_winner"]["closest"]
    assert [x["display_name"] for x in closest] == ["Jaw", "Hand", "Gripper"]
    assert (
        "physics_invalid" in closest[0]["reason"]
        and "nominal grasp is not registered" in closest[1]["reason"]
    )
    assert closest[2]["reason"].startswith("not certified") and closest[1]["failure_modes"]
    # held is a different question: the hand's nominal grasp holds
    assert recommend_for([a, b, c], Requirement.HELD)["winner"]["effector"]["display_name"] == "Hand"


def test_eligibility_excludes_an_unreliable_nominal() -> None:
    a = _cand("catalog_hand", "Hand", cert={"nominal_mode": "physics_unreliable", "nominal_registered": True})
    block = recommend_for([a], Requirement.REGISTERED)
    assert block["winner"] is None and "physics_unreliable" in block["ranking"][0]["reason"]


# --------------------------------------------------------------------------- end to end, physics


def test_toy_tournament_end_to_end_outputs_and_rerun(tmp_path: Path, block: Any) -> None:
    from rtt.effectors.__main__ import main

    out = tmp_path / "runs"
    with registry.override_family(ToyTwoPad()):
        args = ["tournament", "--part", "block", "--groups", "catalog", "--families", "catalog_gripper"]
        assert main([*args, "--out", str(out)]) == 0
        first = json.loads((out / "block" / "tournament.json").read_text())
        assert main([*args, "--out", str(out), "--compare"]) == 0
    tour = json.loads((out / "block" / "tournament.json").read_text())
    rec = json.loads((out / "block" / "recommendation.json").read_text())
    assert tour["digest"] == first["digest"] and tour["label"] == EFF_LABEL == PRODUCT_MODE_LABEL
    assert tour["fairness"]["identical"] and tour["physics"]["bad_qacc"] == 0
    for key in (
        "contract", "label", "part_id", "requirement", "winner", "runner_up_rigid", "alternatives", "ranking",
        "by_group", "budget", "conditions", "provenance",
    ):  # fmt: skip
        assert key in rec, key
    assert rec["requirement"] == "registered" and set(rec["by_requirement"]) == {"held", "registered"}
    assert rec["conditions"]["certificate_tests"] == 18 and len(rec["conditions"]["condition_ids"]) == 4
    assert rec["budget"]["catalog"] == {
        **rec["budget"]["catalog"],
        "screen_episodes": 8,
        "reference_episodes": 1,
        "certificates": 1,
        "certificate_episodes": 19,
    }
    rows = read_table(out / "block" / "records.parquet")
    assert {r["stage"] for r in rows} == {"reference", "screen", "certificate"} and len(rows) == 1 + 8 + 18
    assert all(r["label"] == PRODUCT_MODE_LABEL for r in rows)
    screen_sha = sorted(r["traj_sha"] for r in rows if r["stage"] == "screen")
    assert len(set(screen_sha)) == 8 and all(len(s) == 64 for s in screen_sha)
    certs = list((out / "block" / "certificates").glob("*.json"))
    assert len(certs) == 1 and json.loads(certs[0].read_text())["label"] == PRODUCT_MODE_LABEL
    cand = tour["candidates"][0]
    assert cand["reference"] == "nominal_episode" and cand["reference_pose"] is not None
    assert spec_from_dict(cand["spec"]).effector_id == cand["effector_id"]
    assert not find_brand_tokens(json.dumps(rec)) and not find_brand_tokens(json.dumps(tour))


def test_contract_errors_are_reported_not_scored(block: Any) -> None:
    class Broken(ToyTwoPad):
        def build(self, ctx: Any) -> Any:
            ctx.mount.add_body(name="stray")
            return super().build(ctx)

    with registry.override_family(Broken()):
        res = run_tournament(block, groups=["catalog"], families=["catalog_gripper"], executor=FakePhysics())
    cand = res.candidates()[0]
    assert cand.status == "contract_error" and "prefix" in (cand.reason or "") and not cand.screen
    assert res.recommendation["winner"] is None


# --------------------------------------------------------------------------- agent tool


def test_recommend_effector_schema() -> None:
    assert TOOL_NAMES[-1] == "cost_ledger" and len(TOOL_NAMES) == 8  # the default session keeps eight tools
    spec = EFFECTOR_TOOL_SPECS[0]
    assert spec.name == "recommend_effector" and spec.may_spend
    assert set(spec.parameters["properties"]) == set(spec.args_model.model_fields)
    assert spec.parameters["additionalProperties"] is False and spec.parameters["required"] == ["part_id"]
    assert [t["function"]["name"] for t in openai_tools(effectors=True)][-1] == "recommend_effector"
    assert len(openai_tools()) == 8 and len(openai_tools(effectors=True)) == 9
    ok = validate_args(
        "recommend_effector", {"part_id": PART, "groups": ["reference", "suction"], "design": True}
    )
    assert ok.model_dump() == {
        "part_id": PART,
        "requirement": "registered",
        "groups": ["reference", "suction"],
        "design": True,
    }
    for bad, problem in (
        ({"part_id": "../x"}, "part_id"),
        ({"part_id": PART, "requirement": "grasped"}, "requirement"),
        ({"part_id": PART, "groups": []}, "groups"),
        ({"part_id": PART, "groups": ["catalog", "catalog"]}, "unique"),
        ({"part_id": PART, "groups": ["gecko"]}, "groups"),
        ({"part_id": PART, "design": "yes"}, "design"),
        ({"part_id": PART, "extra": 1}, "extra"),
    ):
        with pytest.raises(ArgsError) as info:
            validate_args("recommend_effector", bad)
        assert problem in str(info.value)
    assert "recommend_effector" in system_prompt(AgentCaps(), ["local"], effectors=True)
    assert "recommend_effector" not in system_prompt(AgentCaps(), ["local"])


def _doc(winner_family: str = "parallel_jaw", score: float = 0.9) -> dict[str, Any]:
    cand = _cand(winner_family, "Winner", n_act=1, cert={"registered_score": score})
    block = recommend_for([cand], Requirement.REGISTERED)
    rec = {
        "requirement": "registered",
        "winner": block["winner"],
        "runner_up_rigid": block["runner_up_rigid"],
        "no_winner": block.get("no_winner"),
        "alternatives": {
            "held": {"effector_id": cand.effector_id, "display_name": "Winner", "fidelity": "core"}
        },
        "ranking": block["ranking"],
        "groups": {"reference": {"status": "run", "reason": None, "finalists": [cand.effector_id]}},
    }
    return {"recommendation": rec, "episodes": 44, "llm_usd": 0.0}


class Gate:
    """A tournament job that waits until released (to hold jobs in flight)."""

    def __init__(self, doc: dict[str, Any], llm_usd: float = 0.0) -> None:
        self.event = threading.Event()
        self.doc = {**doc, "llm_usd": llm_usd}
        self.requests: list[et.EffectorRequest] = []

    def __call__(self, request: et.EffectorRequest) -> dict[str, Any]:
        self.requests.append(request)
        assert self.event.wait(10.0)
        return self.doc


def _toolbox(
    tmp_path: Path,
    run: Any,
    *,
    unlocked: bool = True,
    caps: AgentCaps | None = None,
    session_usd: float = 0.5,
) -> Toolbox:
    backend = FakePartBackend(tmp_path / "parts")
    caps = caps or AgentCaps(session_usd=session_usd, poll_wait_s=5.0)
    guard = SpendGuard(caps.session_usd, DailySpendBook(None, caps.day_usd), "sess-000000000001")
    return Toolbox(
        session_id="sess-000000000001",
        part_id=PART,
        backend=backend,
        proposer=None,  # type: ignore[arg-type]
        runners={"local": LocalJobRunner(backend.evaluate)},
        store=AgentStore(tmp_path / "agent"),
        caps=caps,
        guard=guard,
        unlocked=unlocked,
        recorder=SessionRecorder("sess-000000000001", None),
        effectors=et.EffectorJobRunner(run),
    )


def test_recommend_effector_runs_a_job_and_fetches_a_code_verdict(tmp_path: Path) -> None:
    gate = Gate(_doc())
    box = _toolbox(tmp_path, gate)
    started = box.call("recommend_effector", {"part_id": PART, "groups": ["reference"]})
    assert started.ok, started.error
    job = started.result
    assert (
        job["job_id"].startswith("local-") and job["label"] == PRODUCT_MODE_LABEL and job["reserved_usd"] == 0
    )
    gate.event.set()
    polled = box.call("poll_job", {"job_id": job["job_id"]})
    assert polled.ok and polled.result["state"] == "COMPLETED"
    got = box.call("fetch_results", {"job_id": job["job_id"]})
    assert got.ok, got.error
    res = got.result
    assert res["kind"] == "effector_recommendation" and res["label"] == PRODUCT_MODE_LABEL
    assert res["verdict"] == "recommended" and res["recommendation"]["display_name"] == "Winner"
    assert res["certificate_verdict"] == "pass" and res["recommendation"]["certificate_score_pct"] == 90.0
    assert gate.requests[0] == et.EffectorRequest(PART, "registered", ("reference",), False, 0.0)
    verdicts = [v for v in box.verdicts() if v.get("kind") == "effector_recommendation"]
    assert verdicts == [
        {
            "kind": "effector_recommendation",
            "job_id": job["job_id"],
            "part_id": PART,
            "requirement": "registered",
            "recommendation": "recommended",
            "effector_id": res["recommendation"]["effector_id"],
            "display_name": "Winner",
            "fidelity": "core",
            "certificate_score_pct": 90.0,
            "certificate_verdict": "pass",
            "label": PRODUCT_MODE_LABEL,
        }
    ]
    assert box.calls_used == 3


def test_recommend_effector_certificate_verdict_follows_the_stress_test_rule() -> None:
    low = _doc(score=CERT_PASS_SCORE - 0.1)
    assert (
        et.compact_result("local-0123456789ab", PART, low, CERT_PASS_SCORE)["certificate_verdict"] == "fail"
    )
    cup = _doc(winner_family="suction")
    out = et.compact_result("local-0123456789ab", PART, cup, CERT_PASS_SCORE)
    assert out["recommendation"]["caveat"] == APPROXIMATION_CAVEAT and out["runner_up_note"]
    none = {
        "recommendation": {
            **recommend_for([], Requirement.REGISTERED),
            "requirement": "registered",
            "groups": {},
        }
    }
    assert et.compact_result("local-0123456789ab", PART, none, CERT_PASS_SCORE)["verdict"] == "no_winner"


def test_recommend_effector_design_needs_the_passcode_and_reserves_spend(tmp_path: Path) -> None:
    locked = _toolbox(tmp_path / "a", Gate(_doc()), unlocked=False)
    refused = locked.call("recommend_effector", {"part_id": PART, "design": True})
    assert refused.refused and refused.code == "passcode_required" and locked.calls_used == 1
    assert locked.call("recommend_effector", {"part_id": PART}).ok  # no Nemotron: free, no passcode
    gate = Gate(_doc(), llm_usd=0.006)
    box = _toolbox(tmp_path / "b", gate)
    started = box.call("recommend_effector", {"part_id": PART, "design": True})
    assert started.ok and started.result["reserved_usd"] == et.DESIGN_RESERVE_USD
    assert box.guard.reserved_usd == pytest.approx(et.DESIGN_RESERVE_USD)
    assert gate.requests == [] or gate.requests[0].budget_usd == et.DESIGN_RESERVE_USD
    gate.event.set()
    assert box.call("poll_job", {"job_id": started.result["job_id"]}).result["state"] == "COMPLETED"
    assert box.guard.reserved_usd == 0.0 and box.guard.snapshot()["by_kind_usd"] == {"llm": 0.006}
    poor = _toolbox(tmp_path / "c", Gate(_doc()), session_usd=0.01)
    over = poor.call("recommend_effector", {"part_id": PART, "design": True})
    assert over.refused and over.code == "spend_cap" and poor.guard.reserved_usd == 0.0


def test_recommend_effector_design_spend_stays_under_the_session_design_cap(tmp_path: Path) -> None:
    """At most DESIGN_SESSION_USD of Nemotron design spend per session: in flight plus charged."""
    assert (et.DESIGN_RESERVE_USD, et.DESIGN_SESSION_USD) == (0.05, 0.10)
    gate = Gate(_doc(), llm_usd=0.03)
    gate.event.set()
    box = _toolbox(tmp_path, gate)
    for _ in range(2):
        started = box.call("recommend_effector", {"part_id": PART, "design": True})
        assert started.ok and started.result["reserved_usd"] == et.DESIGN_RESERVE_USD
        polled = box.call("poll_job", {"job_id": started.result["job_id"]})
        assert polled.result["state"] == "COMPLETED"
    assert box.guard.snapshot()["by_kind_usd"] == {"llm": pytest.approx(0.06)}
    over = box.call("recommend_effector", {"part_id": PART, "design": True})  # 0.06 + 0.05 > 0.10
    assert over.refused and over.code == "spend_cap" and "design cap" in (over.error or "")
    assert box.guard.reserved_usd == 0.0
    free = box.call("recommend_effector", {"part_id": PART})  # without Nemotron it stays free
    assert free.ok and free.result["reserved_usd"] == 0
    assert box.call("poll_job", {"job_id": free.result["job_id"]}).result["state"] == "COMPLETED"
    assert [r.budget_usd for r in gate.requests] == [et.DESIGN_RESERVE_USD, et.DESIGN_RESERVE_USD, 0.0]


def test_recommend_effector_without_a_live_designer_reserves_nothing(tmp_path: Path) -> None:
    gate = Gate(_doc(), llm_usd=0.0)
    gate.event.set()
    box = _toolbox(tmp_path, gate)
    box.effectors = et.EffectorJobRunner(gate, designs=False)
    started = box.call("recommend_effector", {"part_id": PART, "design": True})
    assert started.ok and started.result["design"] is False and started.result["reserved_usd"] == 0
    assert "no live designer" in started.result["design_note"] and box.guard.reserved_usd == 0.0
    assert box.call("poll_job", {"job_id": started.result["job_id"]}).result["state"] == "COMPLETED"
    assert gate.requests[0].design is False and gate.requests[0].budget_usd == 0.0


def test_recommend_effector_shares_the_two_job_cap_and_the_tool_budget(tmp_path: Path) -> None:
    gate = Gate(_doc())
    box = _toolbox(tmp_path, gate, caps=AgentCaps(max_tool_calls=4, poll_wait_s=5.0))
    assert box.call("recommend_effector", {"part_id": PART}).ok
    assert box.call("recommend_effector", {"part_id": PART, "requirement": "held"}).ok
    third = box.call("recommend_effector", {"part_id": PART})
    assert third.refused and third.code == "too_many_jobs"
    assert box.call("cost_ledger", {}).ok
    past = box.call("recommend_effector", {"part_id": PART})
    assert past.refused and past.code == "tool_budget_exhausted" and box.calls_used == 4
    gate.event.set()
    cancelled = box.close()
    assert isinstance(cancelled, list)


def test_recommend_effector_is_refused_when_the_layer_is_off(tmp_path: Path) -> None:
    box = _toolbox(tmp_path, Gate(_doc()))
    box.effectors = None
    out = box.call("recommend_effector", {"part_id": PART})
    assert not out.ok and out.code == "invalid_args" and "not enabled" in (out.error or "")


class EffectorTransport:
    """A scripted agent that asks for an effector recommendation, polls, fetches, answers."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def list_models(self, timeout_s: float) -> Any:
        return _response({"role": "assistant", "content": ""}, "x", 1, 1)

    def chat_completions(self, payload: Mapping[str, Any], timeout_s: float) -> Any:
        self.payloads.append(dict(payload))
        model = str(payload.get("model"))
        if "tools" not in payload:
            return _response({"role": "assistant", "content": "Summary."}, model, 50, 10)
        names: dict[str, str] = {}
        results = []
        for m in payload["messages"]:
            if m.get("role") == "assistant":
                for c in m.get("tool_calls") or []:
                    names[c["id"]] = c["function"]["name"]
            elif m.get("role") == "tool":
                results.append((names[m["tool_call_id"]], json.loads(m["content"])))
        n = len(results)

        def call(name: str, args: dict[str, Any]) -> Any:
            fn = {"name": name, "arguments": json.dumps(args)}
            return _response(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": f"c{n}", "type": "function", "function": fn}],
                },
                model,
                100,
                20,
            )

        if not results:
            return call("recommend_effector", {"part_id": PART, "groups": ["reference"]})
        name, last = results[-1]
        if name in ("recommend_effector", "poll_job") and last["result"]["state"] != "COMPLETED":
            return call("poll_job", {"job_id": last["result"]["job_id"]})
        if name in ("recommend_effector", "poll_job"):
            return call("fetch_results", {"job_id": last["result"]["job_id"]})
        rec = last["result"]["recommendation"]
        text = f"Recommended: {rec['display_name']} ({last['result']['verdict']}, certificate {last['result']['certificate_verdict']})."
        return _response({"role": "assistant", "content": text}, model, 100, 20)


def test_a_session_records_the_effector_verdict_and_the_label(tmp_path: Path) -> None:
    gate = Gate(_doc())
    gate.event.set()
    box = _toolbox(tmp_path, gate)
    transport = EffectorTransport()
    guard = box.guard
    ledger = GuardedLedger(guard)
    recorder = SessionRecorder("sess-000000000001", box.store.session_path("sess-000000000001"))
    box.recorder = recorder
    engineer = GraspEngineer(
        AgentLLM.create(ledger, transport=transport), box, recorder, caps=box.caps, offline=True
    )
    result = engineer.run()
    assert result.status == "completed", result.reason
    assert result.answer.startswith("Recommended: Winner (recommended, certificate pass)")
    doc = load_session(recorder.path)
    assert doc["label"] == PRODUCT_MODE_LABEL
    called = [e["name"] for e in doc["events"] if e["type"] == "tool.call"]
    assert called[0] == "recommend_effector" and called[-1] == "fetch_results"
    rows = [v for v in doc["verdicts"] if v.get("kind") == "effector_recommendation"]
    assert (
        len(rows) == 1
        and rows[0]["recommendation"] == "recommended"
        and rows[0]["label"] == PRODUCT_MODE_LABEL
    )
    agent = [p for p in transport.payloads if p.get("tool_choice") == "auto"]
    assert agent and all(len(p["tools"]) == 9 for p in agent)
    assert doc["events"][0]["effectors"] is True


def test_resolve_upload_screens_an_uploaded_part_like_a_training_part(tmp_path: Path) -> None:
    import trimesh

    from rtt.agent.backends import SimPartBackend

    stl = tmp_path / "box.stl"
    trimesh.creation.box(extents=(30.0, 24.0, 40.0)).export(stl)
    backend = SimPartBackend(tmp_path / "parts", workers=1, decomposition="hull")
    backend.intake(stl, part_id=PART, name="box")
    part, conds, suite = et.resolve_upload(backend, PART)
    assert part.name == PART and suite == "upload" and len(conds) == 4
    assert [round(1000 * s, 1) for s in sorted(part.size)] == [24.0, 30.0, 40.0]
    assert conds == et.resolve_upload(backend, PART)[1]  # seeds from the geometry: identical every time
    with pytest.raises(RuntimeError, match="simulator backend"):
        et.resolve_upload(FakePartBackend(tmp_path / "fake"), PART)
