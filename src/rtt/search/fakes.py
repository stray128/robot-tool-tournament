"""Deterministic stand-ins for the simulator and the model, for tests and offline dry runs.

* :class:`FakeEvaluator` implements the :class:`~rtt.search.base.Evaluator` protocol
  over a synthetic, seeded landscape: 12 training conditions (3 families x 2 sizes x
  2 placements) x 3 controller settings x 1 jitter seed = 36 episodes per design,
  shaped like the first training suite (the calibrated one has 16 conditions). Each
  family prefers a region of parameter space, an inner face, a tip and a grasp height;
  success of every episode is a hash-seeded Bernoulli draw from that preference, so
  results depend only on the design (batch invariant, reproducible) and there is real
  headroom above the manual entrants.
  Finger mass is the real generated mass.
* :class:`ScriptedProposer` implements the LLM transport protocol and answers
  ``submit_designs`` tool calls with designs derived from the parameters it finds in
  the prompt, at configurable rates of invalid designs, near-duplicates and
  unparseable replies, so the repair, reflection and fallback paths all run.

Neither is used for reported numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, Final

import numpy as np

from rtt.design.generate import generate_finger, mass_and_inertia
from rtt.design.schema import CATEGORICAL_OPTIONS, GEOMETRY_PARAMS, PARAM_BOUNDS, FingerDesign, GripperTask
from rtt.llm.cache import canonical_json, wire_json
from rtt.llm.providers import TransportResponse
from rtt.search.base import ConditionInfo, DesignScore, EpisodeOutcome, EvaluatorInfo, derive_seed

__all__ = ["DEFAULT_FAKE_CONDITIONS", "FakeEvaluator", "ScriptedProposer"]

_CONTINUOUS: Final = tuple(PARAM_BOUNDS)
#: Sets the difficulty: manual entrants score about 25-60 %, the best of 40 random designs
#: about 70-80 %, and random designs spread with an SD of about 0.14 (like the G3 gates).
_BASE_LOGIT: Final = 1.8


def _condition(cid: str, family: str, size: str, grip_mm: float, placement: str) -> ConditionInfo:
    return ConditionInfo(cid, family, f"{family} {size}, {placement}", grip_mm)


#: 3 families x 2 sizes x 2 placements, mirroring the shape of the real training suite.
DEFAULT_FAKE_CONDITIONS: Final[tuple[ConditionInfo, ...]] = tuple(
    _condition(f"{family}_{size_id}_{place_id}", family, size, grip, placement)
    for family, sizes in (
        ("box", (("s", "40x30x60 mm", 30.0), ("l", "60x45x80 mm", 45.0))),
        ("cylinder", (("s", "radius 15 mm x 70 mm", 30.0), ("l", "radius 22 mm x 90 mm", 44.0))),
        (
            "rod",
            (("s", "radius 8 mm x 100 mm lying flat", 16.0), ("l", "radius 12 mm x 140 mm lying flat", 24.0)),
        ),
    )
    for size_id, size, grip in sizes
    for place_id, placement in (("c", "centred"), ("o", "offset 10 mm"))
)


def _unit(*parts: object) -> float:
    """Uniform [0, 1) from a hash of ``parts`` (platform independent)."""
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


class FakeEvaluator:
    """Synthetic training-suite evaluator (see module docstring).

    Args:
        landscape_seed: Selects the landscape (family optima and preferences).
        conditions: Training conditions; default :data:`DEFAULT_FAKE_CONDITIONS`.
        controller_settings: Grasp-height fractions, one per controller setting k.
        jitter_seeds: Episode seeds per condition and setting.
        gripper_task: Validator constants reported by :meth:`describe`.
    """

    def __init__(
        self,
        landscape_seed: int = 0,
        *,
        conditions: Sequence[ConditionInfo] = DEFAULT_FAKE_CONDITIONS,
        controller_settings: Sequence[float] = (0.3, 0.5, 0.7),
        jitter_seeds: Sequence[int] = (0,),
        gripper_task: GripperTask | None = None,
    ) -> None:
        self.landscape_seed = int(landscape_seed)
        self.conditions = tuple(conditions)
        self.controller_settings = tuple(float(k) for k in controller_settings)
        self.jitter_seeds = tuple(int(s) for s in jitter_seeds)
        widest = max(c.grip_width_mm for c in self.conditions)
        self.gripper_task = gripper_task or GripperTask(widest_training_object_mm=widest)
        rng = np.random.default_rng(derive_seed(self.landscape_seed, "fake-landscape"))
        self._profiles: dict[str, dict[str, Any]] = {}
        for family in dict.fromkeys(c.family for c in self.conditions):
            self._profiles[family] = {
                "center": rng.uniform(0.15, 0.85, size=len(_CONTINUOUS)),
                "weights": rng.uniform(0.5, 1.5, size=len(_CONTINUOUS)),
                "face": CATEGORICAL_OPTIONS["inner_face"][int(rng.integers(3))],
                "tip": CATEGORICAL_OPTIONS["tip_profile"][int(rng.integers(4))],
                "k": int(rng.integers(len(self.controller_settings))),
            }
        self.calls = 0
        self.designs_evaluated = 0
        self.batch_sizes: list[int] = []

    def describe(self) -> EvaluatorInfo:
        suite = {
            "conditions": [asdict(c) for c in self.conditions],
            "controller_settings": list(self.controller_settings),
            "jitter_seeds": list(self.jitter_seeds),
        }
        return EvaluatorInfo(
            executor="fake",
            conditions=self.conditions,
            controller_settings=self.controller_settings,
            seeds_per_condition=len(self.jitter_seeds),
            gripper_task=self.gripper_task,
            task_hash=hashlib.sha256(
                canonical_json({"fake_landscape": self.landscape_seed}).encode()
            ).hexdigest(),
            suite_hash=hashlib.sha256(canonical_json(suite).encode()).hexdigest(),
        )

    def _unit_params(self, design: FingerDesign) -> np.ndarray:
        values = []
        for name in _CONTINUOUS:
            lo, hi = PARAM_BOUNDS[name]
            values.append((float(getattr(design, name)) - lo) / (hi - lo))
        return np.asarray(values)

    def _episode(
        self, design: FingerDesign, design_id: str, x: np.ndarray, cond: ConditionInfo, k: int, seed: int
    ) -> EpisodeOutcome:
        prof = self._profiles[cond.family]
        d2 = float(np.average((x - prof["center"]) ** 2, weights=prof["weights"]))
        face_ok = design.inner_face == prof["face"]
        tip_ok = design.tip_profile == prof["tip"]
        k_gap = abs(k - prof["k"])
        logit = _BASE_LOGIT - 14.0 * d2 + (1.0 if face_ok else 0.0) + (0.6 if tip_ok else 0.0) - 1.3 * k_gap
        logit -= 0.5 * cond.grip_width_mm / 45.0 + (0.3 if cond.condition_id.endswith("_o") else 0.0)
        p = 1.0 / (1.0 + math.exp(-logit))
        u = _unit(self.landscape_seed, design_id, cond.condition_id, k, seed)
        success = u < p
        mode: str | None = None
        if not success:
            if k_gap >= 1 and _unit("mode", design_id, cond.condition_id, k) < 0.6:
                mode = "missed_grasp"
            elif not face_ok and cond.family != "box":
                mode = "slip_during_lift"
            elif d2 > 0.08:
                mode = "unstable_hold"
            else:
                modes = ("slip_during_lift", "ejected_on_close", "finger_env_collision")
                mode = modes[int(_unit("alt", design_id, cond.condition_id, k) * len(modes))]
        tail = _unit("tail", design_id, cond.condition_id, k, seed)
        return EpisodeOutcome(
            condition_id=cond.condition_id,
            family=cond.family,
            controller_setting=k,
            seed=seed,
            success=success,
            failure_mode=mode,
            max_lift_m=0.15 if success else round(0.1 * tail, 6),
            hold_frac=1.0 if success else round(0.9 * tail, 6),
        )

    def evaluate(self, designs: Sequence[FingerDesign]) -> list[DesignScore]:
        self.calls += 1
        self.designs_evaluated += len(designs)
        self.batch_sizes.append(len(designs))
        out = []
        for design in designs:
            design_id = design.design_id()
            x = self._unit_params(design)
            episodes = [
                self._episode(design, design_id, x, cond, k, seed)
                for cond in self.conditions
                for k in range(len(self.controller_settings))
                for seed in self.jitter_seeds
            ]
            mass_g = mass_and_inertia(generate_finger(design)).mass_kg * 1000.0
            out.append(DesignScore.from_episodes(design_id, mass_g, episodes, len(self.controller_settings)))
        return out


# --------------------------------------------------------------------------- scripted model

_PARAMS_RE = re.compile(r'\{"finger_length_mm":[^{}]*\}')
_LABEL_RE = re.compile(r"\b(d\d{2})\b")
_FOLLOW_UP_MARKERS = ("# Correction needed", "# Too similar")
_STRAIGHT: Final[dict[str, Any]] = {
    "finger_length_mm": 80.0,
    "finger_thickness_mm": 10.0,
    "finger_width_mm": 20.0,
    "bend_angle_deg": 0.0,
    "bend_start_frac": 0.5,
    "inner_face": "flat",
    "inner_face_depth_mm": 0.0,
    "tip_profile": "blunt",
    "tip_size_mm": 0.0,
    "taper_ratio": 1.0,
}


class ScriptedProposer:
    """An offline ``Transport`` that answers forced ``submit_designs`` calls (see module docstring).

    Answers are a pure function of the request payload and ``seed``, so a scripted
    arm is reproducible and a recorded cache replays it exactly.

    Args:
        seed: Varies the answers.
        invalid_rate: Probability that a main-call slot breaks a validator rule.
        similar_rate: Probability that a main-call slot copies a design from the prompt.
        garbage_rate: Probability that a main call answers in prose with no tool call.
    """

    def __init__(
        self,
        seed: int = 0,
        *,
        invalid_rate: float = 0.15,
        similar_rate: float = 0.1,
        garbage_rate: float = 0.0,
    ) -> None:
        self.seed = seed
        self.invalid_rate = invalid_rate
        self.similar_rate = similar_rate
        self.garbage_rate = garbage_rate
        self.calls = 0
        self.payloads: list[dict[str, Any]] = []

    def list_models(self, timeout_s: float) -> TransportResponse:
        return TransportResponse(200, {}, {"data": []})

    def chat_completions(self, payload: Mapping[str, Any], timeout_s: float) -> TransportResponse:
        self.calls += 1
        self.payloads.append(json.loads(json.dumps(payload)))
        text = "\n".join(str(m.get("content") or "") for m in payload["messages"])
        rng = random.Random(f"{self.seed}:{hashlib.sha256(wire_json(payload).encode()).hexdigest()}")
        follow_up = any(marker in text for marker in _FOLLOW_UP_MARKERS)
        usage = {
            "prompt_tokens": len(wire_json(payload)) // 3,
            "completion_tokens": 0,
            "completion_tokens_details": {"reasoning_tokens": 40},
        }
        if not follow_up and rng.random() < self.garbage_rate:
            usage["completion_tokens"] = 60
            message = {"role": "assistant", "content": "I would rather describe the designs in words."}
            return self._response(message, "stop", usage)
        function = payload["tools"][0]["function"]
        properties = function["parameters"]["properties"]
        seen = [json.loads(m) for m in _PARAMS_RE.findall(text)]
        labels = sorted(set(_LABEL_RE.findall(text)))
        args: dict[str, Any] = {"analysis": "Scripted analysis: keep what works, vary one idea per slot."}
        for slot, schema in properties.items():
            if slot == "analysis":
                continue
            args[slot] = self._slot(schema, seen, labels, rng, clean=follow_up)
        arguments = json.dumps(args)
        usage["completion_tokens"] = len(arguments) // 3 + 40
        message = {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Scripted reasoning.",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": function["name"], "arguments": arguments},
                }
            ],
        }
        return self._response(message, "tool_calls", usage)

    def _response(self, message: dict[str, Any], finish: str, usage: dict[str, Any]) -> TransportResponse:
        body = {
            "id": f"scripted-{self.calls}",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {**usage, "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"]},
        }
        return TransportResponse(200, {"x-ratelimit-remaining-tokens": "1000000"}, body)

    def _slot(
        self,
        schema: Mapping[str, Any],
        seen: Sequence[Mapping[str, Any]],
        labels: Sequence[str],
        rng: random.Random,
        *,
        clean: bool,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        props = schema.get("properties", {})
        if "parents" in props:
            n = int(props["parents"].get("minItems", 1))
            out["parents"] = rng.sample(list(labels), n) if len(labels) >= n else list(labels)[:n]
        out["hypothesis"] = "Scripted variation of a promising design."
        if seen and not clean and rng.random() < self.similar_rate:
            params = dict(rng.choice(list(seen)))
        else:
            base = dict(rng.choice(list(seen))) if seen and rng.random() < 0.7 else dict(_STRAIGHT)
            params = self._mutate(base, rng)
        if not clean and rng.random() < self.invalid_rate:
            params["inner_face"], params["inner_face_depth_mm"] = "flat", 3.0
        out.update({k: params[k] for k in GEOMETRY_PARAMS})
        return out

    @staticmethod
    def _mutate(base: dict[str, Any], rng: random.Random) -> dict[str, Any]:
        params = dict(base)
        for name in rng.sample(list(_CONTINUOUS), 3):
            lo, hi = PARAM_BOUNDS[name]
            step = rng.uniform(0.15, 0.35) * (hi - lo) * rng.choice((-1.0, 1.0))
            params[name] = round(min(hi, max(lo, float(params[name]) + step)), 2)
        if rng.random() < 0.4:
            params["inner_face"] = rng.choice(CATEGORICAL_OPTIONS["inner_face"])
        if rng.random() < 0.4:
            params["tip_profile"] = rng.choice(CATEGORICAL_OPTIONS["tip_profile"])
        if params["inner_face"] == "flat":
            params["inner_face_depth_mm"] = 0.0
        else:
            cap = 0.45 * float(params["finger_thickness_mm"])
            if params["inner_face"] == "v_groove":
                cap = min(cap, 0.3 * float(params["finger_width_mm"]))
            params["inner_face_depth_mm"] = round(min(float(params["inner_face_depth_mm"]) or 2.0, cap), 2)
        if params["tip_profile"] == "hook":
            params["tip_size_mm"] = round(
                min(float(params["tip_size_mm"]), float(params["finger_thickness_mm"])), 2
            )
        return params
