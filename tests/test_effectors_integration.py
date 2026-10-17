"""Integration of the effector tracks (product mode, not pre-registered).

What only holds once B1 to B4 are merged together:

* the founder's catalog decisions: the catalog group fits the tournament's 12 candidates
  without sampling (the rigid-pad gripper B is vendored but not an entrant), and the rigid-pad
  gripper A keeps its upstream gains with a card note every page prints;
* track B3's designer behind track B4's adapter: recipes, structured rejections, the batch's
  cost, the physics feedback with its episode count, and the groups of designed and random specs;
* the API service offers ``recommend_effector`` to every session whose backend runs the simulator.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from rtt.effectors.catalog.models import CATALOG_MODELS, LOW_GRIP_NOTE
from rtt.effectors.lanes import _effector_brief
from rtt.effectors.recommend import card_note, recommend_for
from rtt.effectors.registry import get_family, presets_of
from rtt.effectors.scene import build_effector_scene
from rtt.effectors.tournament import (
    MAX_SCREEN_CANDIDATES,
    Candidate,
    DesignFeedback,
    default_designer,
)
from rtt.effectors.types import EffectorFamily, EntrantGroup, Requirement
from rtt.parts.library import get_part
from rtt.sim.task import default_task_v2

RIGID_A, RIGID_B = "gripper-2f-rigid-a", "gripper-2f-rigid-b"


# --------------------------------------------------------------------------- catalog decisions


def test_the_catalog_group_fits_the_budget_without_sampling() -> None:
    specs = presets_of([EffectorFamily.CATALOG_HAND, EffectorFamily.CATALOG_GRIPPER])
    assert {s.group for s in specs} == {EntrantGroup.CATALOG}
    assert len(specs) == MAX_SCREEN_CANDIDATES == 12
    models = {s.model for s in specs}
    assert RIGID_A in models and RIGID_B not in models
    # every hand stays in, both synergies, at upstream gains (no gain changes were made)
    hands = [s for s in specs if s.family is EffectorFamily.CATALOG_HAND]
    assert len(hands) == 6 and {s.primitive for s in hands} == {"power", "pinch"}
    # B stays vendored and buildable: it is only not an entrant
    assert not CATALOG_MODELS[RIGID_B].in_presets
    fam = get_family(EffectorFamily.CATALOG_GRIPPER)
    from rtt.effectors.catalog.family import vendored_spec

    cm = CATALOG_MODELS[RIGID_B]
    spec_b = vendored_spec(EffectorFamily.CATALOG_GRIPPER, cm, cm.synergies[0])
    scene = build_effector_scene(fam, spec_b, get_part("block"), default_task_v2())
    assert scene.build.actuators


def test_the_rigid_pad_gripper_a_carries_its_low_grip_note_everywhere() -> None:
    spec = next(s for s in get_family("catalog_gripper").presets() if s.model == RIGID_A)
    assert card_note(spec.family.value, spec.model) == LOW_GRIP_NOTE
    assert "Low grip at published settings" in LOW_GRIP_NOTE and "4.1 N" in LOW_GRIP_NOTE
    others = [s for s in presets_of(["catalog_hand", "catalog_gripper", "suction"]) if s.model != RIGID_A]
    assert all(card_note(s.family.value, s.model) is None for s in others)
    assert card_note("no_such_family", "x") is None
    # the build's notes (records), the recommendation's rows (agent, JSON) and the lane brief (web)
    scene = build_effector_scene(get_family("catalog_gripper"), spec, get_part("block"), default_task_v2())
    assert LOW_GRIP_NOTE in scene.build.notes
    assert scene.build.actuation["linear_force_n_max"] == 4.1  # the upstream value, unmodified
    block = recommend_for([Candidate(spec, "preset")], Requirement.REGISTERED)
    assert block["ranking"][0]["note"] == LOW_GRIP_NOTE
    assert _effector_brief(spec.to_dict())["note"] == LOW_GRIP_NOTE


# --------------------------------------------------------------------------- B3 behind B4's adapter


class _Scripted:
    """A fake Token Factory transport: each request gets the next scripted tool call."""

    def __init__(self, scripts: list[Mapping[str, Any]]) -> None:
        self.scripts = list(scripts)
        self.payloads: list[dict[str, Any]] = []

    def chat_completions(self, payload: Mapping[str, Any], timeout_s: float) -> Any:
        from rtt.llm.providers import TransportResponse

        self.payloads.append(json.loads(json.dumps(payload)))
        args = self.scripts.pop(0)
        call = {
            "id": "call_0",
            "type": "function",
            "function": {"name": "submit_morphologies", "arguments": json.dumps(args)},
        }
        body = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 6000, "completion_tokens": 1500},
        }
        return TransportResponse(200, {}, body)

    def list_models(self, timeout_s: float) -> Any:
        raise NotImplementedError


def _design(recipe: Any, hypothesis: str = "it should hold") -> dict[str, Any]:
    from rtt.effectors.morphology import space_label

    return {
        "hypothesis": hypothesis,
        "kind": space_label((recipe.family, recipe.kind)),
        "params": dict(recipe.params),
    }


def test_the_b3_designer_matches_the_tournament_adapter(tmp_path: Path) -> None:
    from rtt.effectors.designed import reference_recipes
    from rtt.effectors.morphology import design_spaces, make_designer_client
    from rtt.llm.client import TokenBucket

    part = get_part("steel_slug")
    ref = reference_recipes()
    bad = {**ref[0].params, "n_fingers": 7}
    main = {
        "analysis": "Round parts suit a chuck.",
        "designs": [
            _design(ref[2], "a chuck centres the slug"),
            {"hypothesis": "too many", "kind": "fingered_hand", "params": bad},
            _design(ref[2], "the same chuck again"),
            _design(ref[0], "three upright fingers"),
        ],
    }
    repair = {
        "analysis": "Replacing designs 2 and 3.",
        "designs": [
            _design(ref[2]),
            _design(ref[3], "three curling digits"),
            {"hypothesis": "unknown", "kind": "gecko_pad", "params": {}},
            _design(ref[0]),
        ],
    }
    fake = _Scripted([main, repair])
    client = make_designer_client(
        tmp_path / "cache", transport=fake, limiter=TokenBucket(1e12), budget_usd=0.05
    )
    designer = default_designer(client)
    assert designer is not None
    history = [DesignFeedback(ref[1], 0.75, 0.5, {"slip_during_lift": 2}, n_episodes=8)]
    batch = designer.propose(part, history, 4)
    assert [r.kind for r in batch.recipes] == ["hook_gripper", "segmented_tentacle", "fingered_hand"]
    # rejections arrive as B3's structured records, not as their repr
    assert [(r["slot"], r["stage"], r["reason"]) for r in batch.rejections] == [
        (2, "main", "invalid"),
        (3, "main", "duplicate"),
        (3, "repair", "invalid"),
    ]
    assert all(r["errors"] for r in batch.rejections if r["reason"] == "invalid")
    json.dumps([dict(r) for r in batch.rejections], allow_nan=False)
    # the cost is the batch's live calls (B3's ledger slice), and the feedback keeps its counts
    assert batch.calls == 2 and batch.cost_usd > 0
    assert math.isclose(batch.cost_usd, client.ledger.spent_usd)
    assert "held 6/8, registered 4/8" in fake.payloads[0]["messages"][-1]["content"]
    assert batch.analysis == "Round parts suit a chuck."
    # Nemotron specs are the designed group whatever builds them; the random control is random
    assert all(designer.spec_for(r).group is EntrantGroup.DESIGNED for r in batch.recipes)
    rnd = designer.random_recipes(part, 8, 20261017)
    assert len(rnd) == 8 and all(designer.spec_for(r).group is EntrantGroup.RANDOM for r in rnd)
    assert [r.recipe_id for r in rnd] == [r.recipe_id for r in designer.random_recipes(part, 8, 20261017)]
    spaces = {(k[0].value, k[1]) for k in design_spaces()}
    assert all((r.family.value, r.kind) in spaces for r in rnd)


# --------------------------------------------------------------------------- the API service


def test_the_service_offers_recommend_effector_with_the_simulator_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trimesh

    import rtt.api.service as service_module
    from rtt.agent.backends import SimPartBackend
    from rtt.agent.caps import AgentCaps, PasscodeGate
    from rtt.agent.effector_tool import EffectorJobRunner
    from rtt.agent.fakes import FakePartBackend, ScriptedAgentTransport
    from rtt.api.service import AgentService, ApiSettings

    seen: dict[str, Any] = {}

    class Engineer:
        def __init__(self, llm: Any, toolbox: Any, *args: Any, **kwargs: Any) -> None:
            seen["effectors"] = toolbox.effectors

        def run(self) -> None:
            return None

    monkeypatch.setattr(service_module, "GraspEngineer", Engineer)
    settings = ApiSettings(root=tmp_path / "agent", offline=True, caps=AgentCaps(poll_wait_s=1.0))
    backend = SimPartBackend(tmp_path / "agent" / "parts", workers=1, decomposition="hull")
    svc = AgentService(
        settings, backend=backend, gate=PasscodeGate(None), transport_factory=ScriptedAgentTransport
    )
    try:
        assert isinstance(svc.effectors, EffectorJobRunner)
        stl = trimesh.creation.box(extents=(30.0, 24.0, 40.0)).export(file_type="stl")
        part = svc.upload_part(stl, name="box")
        sid = svc.start_session(part.part_id)
        svc.wait_session(sid, timeout_s=30)
        assert seen["effectors"] is svc.effectors
    finally:
        svc.shutdown()
    fake = AgentService(
        ApiSettings(root=tmp_path / "fake", offline=True),
        backend=FakePartBackend(tmp_path / "fake" / "parts"),
        gate=PasscodeGate(None),
    )
    try:
        assert fake.effectors is None  # the offline fake cannot run tournaments
    finally:
        fake.shutdown()
