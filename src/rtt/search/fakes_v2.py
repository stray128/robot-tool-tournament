"""Deterministic v2 stand-ins for the simulator, the model and the entrants (tests and dry runs only).

* :class:`FakeEvaluatorV2` implements the :class:`~rtt.search.base.Evaluator` protocol for
  ``freeform-v2`` recipes over a synthetic, seeded landscape shaped like the v2 training
  suite: 6 parts x 4 placements x 2 jitter seeds = 48 episodes per design. Each episode's
  outcome follows the spec's rules of thumb (friction rule, capture rule, round parts that
  self-centre in an imprint, flat pads that never centre, imprints that jam on prismatic
  parts), plus a hash-seeded Bernoulli draw, so ``jam_on_close`` / ``off_centre`` feedback is
  meaningful and results depend only on the design. Masses, capture reports and piece counts
  are synthetic. It also supplies a fake ``geometry_hash_fn`` so ``duplicate_geometry`` can be
  exercised (a chamfer smaller than 0.5 mm builds the same jaw as a blunt tip).
* :class:`ScriptedProposerV2` is an offline LLM transport that answers v2 ``submit_designs``
  calls with recipes derived from the ones it finds in the prompt, at configurable rates of
  invalid recipes, duplicates and unparseable replies.
* :func:`stub_entrants` is the v2 round-0 entrant set (stock blade, flat pad, configurator
  imprint, swept imprint) as :class:`~rtt.search.space_v2.BaselineEntrant` objects, until
  track G's ``configs/entrants_v2.yaml`` loader lands.

The part sizes and masses here approximate spec section 8; the real parts come from track O.
Nothing here is used for reported numbers.
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

from rtt.llm.cache import canonical_json, wire_json
from rtt.llm.providers import TransportResponse
from rtt.search.base import (
    ConditionInfo,
    ConditionV2,
    DesignScore,
    EpisodeOutcome,
    EvaluatorInfo,
    PartInfo,
    PlacementInfo,
    V2Info,
)
from rtt.search.space import V2, get_space
from rtt.search.space_v2 import (
    INACTIVE,
    NUMERIC_BOUNDS,
    NUMERIC_PARAMS,
    PAD_OPTIONS,
    RECIPE_KIND,
    TIP_OPTIONS,
    BaselineEntrant,
    V2Context,
)

__all__ = [
    "DEFAULT_FAKE_PARTS_V2",
    "DEFAULT_PLACEMENTS_V2",
    "FakeEvaluatorV2",
    "ScriptedProposerV2",
    "stub_entrants",
]

#: The v2 training parts (spec section 8) with track O's sizes, masses and features
#: (``rtt.parts.library`` at the time of writing): x across the jaw, y closing axis, z up.
DEFAULT_FAKE_PARTS_V2: Final[tuple[PartInfo, ...]] = (
    PartInfo(
        "pipe_tee",
        "pipe_tee",
        "1/2 inch slip pipe tee lying flat, branch toward the left jaw",
        (60.0, 44.2, 28.5),
        "white PVC",
        39.9,
        1,
        (
            "socket 'branch_socket' facing the left jaw, radius 10.8 mm, 16.5 mm deep",
            "socket 'run_socket_pos' opening toward +x, radius 10.8 mm, 16.5 mm deep",
            "socket 'run_socket_neg' opening toward -x, radius 10.8 mm, 16.5 mm deep",
        ),
    ),
    PartInfo(
        "block",
        "block",
        "block with sharp edges, no chamfer",
        (30.0, 40.0, 60.0),
        "anodised aluminium",
        194.4,
        2,
    ),
    PartInfo(
        "hex_standoff",
        "hex_standoff",
        "hex standoff standing upright, 17 mm across flats",
        (19.6, 17.0, 45.0),
        "zinc-plated steel",
        82.9,
        6,
        ("socket 'tapped_hole' opening up, radius 4.2 mm, 12 mm deep",),
    ),
    PartInfo(
        "flanged_bushing",
        "flanged_bushing",
        "flanged bushing standing on its flange, body diameter 20 mm",
        (36.0, 36.0, 35.0),
        "brass",
        90.9,
        0,
        (
            "socket 'bore' opening up, radius 6 mm, 35 mm deep",
            "flange 'flange' at the bottom, radius 18 mm, 5 mm thick",
        ),
    ),
    PartInfo(
        "l_bracket",
        "l_bracket",
        "L-bracket lying flat, 4 mm wall",
        (30.0, 40.0, 40.0),
        "stamped steel",
        71.6,
        1,
    ),
    PartInfo(
        "steel_slug",
        "slug",
        "ground steel slug standing upright",
        (40.0, 40.0, 50.0),
        "ground steel",
        494.8,
        0,
    ),
)
#: The v2 training placements (spec section 8).
DEFAULT_PLACEMENTS_V2: Final[tuple[PlacementInfo, ...]] = (
    PlacementInfo("c0", 0.0, 0.0),
    PlacementInfo("c1", 2.0, 0.0),
    PlacementInfo("c2", -3.0, 0.0),
    PlacementInfo("c3", 1.0, 6.0),
)
_ROUND: Final = frozenset({"steel_slug", "flanged_bushing"})
#: Numbers that shape the synthetic lift-and-hold landscape (each part prefers a region of them).
_HOLD_DIMS: Final[tuple[str, ...]] = (
    "grip_height_frac",
    "pad_height_mm",
    "finger_width_mm",
    "wall_draft_deg",
)
_MU: Final = 0.3


def _unit(*parts: object) -> float:
    """Uniform [0, 1) from a hash of ``parts`` (platform independent)."""
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def stub_entrants() -> list[BaselineEntrant]:
    """The four v2 round-0 entrants (spec section 9), as stub baselines until track G lands."""
    space = get_space(V2)
    imprint = space.recipe(
        pad="conform",
        tip="blunt",
        grip_height_frac=0.5,
        pad_height_mm=20.0,
        finger_width_mm=40.0,
        pad_depth_mm=10.0,
        wall_draft_deg=0.0,
        clearance_mm=0.4,
    )
    flat = space.recipe(
        pad="flat", tip="blunt", grip_height_frac=0.5, pad_height_mm=20.0, finger_width_mm=20.0
    )
    return [
        BaselineEntrant("stock", "stock_blade", None, {}, "The gripper's stock blade, as shipped."),
        BaselineEntrant("flat", RECIPE_KIND, flat, {}, "A plain flat pad at mid height."),
        BaselineEntrant(
            "imprint_configurator",
            "imprint_configurator",
            imprint,
            {"depth_rule": "40% half-extent, >= 4 mm"},
            "What a fingertip configurator does: an imprint of the part with 0.4 mm clearance and straight walls.",
        ),
        BaselineEntrant(
            "imprint_swept",
            "imprint_swept",
            imprint,
            {"sweep_mm": 2.0},
            "The configurator imprint swept by +-2 mm to tolerate placement error.",
        ),
    ]


class FakeEvaluatorV2:
    """Synthetic v2 training-suite evaluator (see module docstring).

    Args:
        landscape_seed: Varies the noise of the landscape.
        parts: Training parts (default :data:`DEFAULT_FAKE_PARTS_V2`).
        placements: Placements (default :data:`DEFAULT_PLACEMENTS_V2`).
        jitter_seeds: Episode seeds per condition.
    """

    def __init__(
        self,
        landscape_seed: int = 0,
        *,
        parts: Sequence[PartInfo] = DEFAULT_FAKE_PARTS_V2,
        placements: Sequence[PlacementInfo] = DEFAULT_PLACEMENTS_V2,
        jitter_seeds: Sequence[int] = (0, 1),
    ) -> None:
        self.landscape_seed = int(landscape_seed)
        self.parts = tuple(parts)
        self.placements = tuple(placements)
        self.jitter_seeds = tuple(int(s) for s in jitter_seeds)
        self.space = get_space(V2)
        self.conditions = tuple(
            ConditionV2(f"{p.part_id}_{pl.placement_id}", p.part_id, pl.placement_id)
            for p in self.parts
            for pl in self.placements
        )
        self.context = V2Context(
            part_ids=tuple(p.part_id for p in self.parts),
            geometry_hash_fn=self.geometry_hash,
            label="fake-v2",
        )
        rng = random.Random(f"fake-v2-landscape:{self.landscape_seed}")
        self._profiles = {
            p.part_id: {
                "centre": [rng.uniform(0.2, 0.8) for _ in _HOLD_DIMS],
                "weights": [rng.uniform(0.5, 1.5) for _ in _HOLD_DIMS],
            }
            for p in self.parts
        }
        self.calls = 0
        self.designs_evaluated = 0
        self.batch_sizes: list[int] = []

    # ------------------------------------------------------------------ evaluator protocol

    def describe(self) -> EvaluatorInfo:
        placements = {p.placement_id: p for p in self.placements}
        conditions = []
        for c in self.conditions:
            part = next(p for p in self.parts if p.part_id == c.part_id)
            pl = placements[c.placement_id]
            where = (
                "centred" if not pl.dx_mm and not pl.dyaw_deg else f"{pl.dx_mm:+g} mm, {pl.dyaw_deg:g} deg"
            )
            conditions.append(
                ConditionInfo(c.condition_id, c.part_id, f"{part.part_id}, {where}", part.width_across_mm)
            )
        suite = {
            "parts": [asdict(p) for p in self.parts],
            "placements": [asdict(p) for p in self.placements],
            "jitter_seeds": list(self.jitter_seeds),
        }
        return EvaluatorInfo(
            executor="fake-v2",
            conditions=tuple(conditions),
            controller_settings=(),
            seeds_per_condition=len(self.jitter_seeds),
            gripper_task=self.context,
            task_hash=hashlib.sha256(
                canonical_json({"fake_v2_landscape": self.landscape_seed}).encode()
            ).hexdigest(),
            suite_hash=hashlib.sha256(canonical_json(suite).encode()).hexdigest(),
            grip_force_n=50.0,
            friction=_MU,
            jitter_pos_mm=0.5,
            jitter_yaw_deg=1.0,
            jitter_mu_frac=0.10,
            v2=V2Info(parts=self.parts, placements=self.placements, conditions=self.conditions),
        )

    def evaluate(self, designs: Sequence[Any]) -> list[DesignScore]:
        self.calls += 1
        self.designs_evaluated += len(designs)
        self.batch_sizes.append(len(designs))
        placement_of = {c.condition_id: c.placement_id for c in self.conditions}
        out = []
        for design in designs:
            design_id = self.space.design_id(design)
            episodes = [
                self._episode(design, design_id, c, seed)
                for c in self.conditions
                for seed in self.jitter_seeds
            ]
            capture = {p.part_id: self._capture(design, p) for p in self.parts}
            out.append(
                DesignScore.from_episodes_v2(
                    design_id,
                    self._mass_g(design),
                    episodes,
                    placements=placement_of,
                    capture=capture,
                    max_pieces=self._pieces(design),
                )
            )
        return out

    # ------------------------------------------------------------------ the landscape

    def geometry_hash(self, design: Any, part_id: str) -> str:
        """Fake per-part geometry hash: a chamfer under 0.5 mm builds the same jaw as a blunt tip."""
        point = self.space.buildable(design)
        if self.space.is_baseline(point):
            key: Any = self.space.dump(point)
        else:
            key = self.space.params(point)
            if key["tip"] == "chamfer" and key["tip_size_mm"] < 0.5:
                key = {**key, "tip": "blunt", "tip_size_mm": 0.0}
        return hashlib.sha256(canonical_json({"part": part_id, "jaw": key}).encode()).hexdigest()

    def _recipe(self, design: Any) -> tuple[Mapping[str, Any] | None, str]:
        """(recipe params or None, kind) of what is built: a recipe, or a baseline's kind."""
        point = self.space.buildable(design)
        if self.space.is_baseline(point):
            inner = self.space.point(point)
            return (self.space.params(inner) if inner is not None else None), str(point.kind)
        return self.space.params(point), "recipe"

    def _capture(self, design: Any, part: PartInfo) -> tuple[float, float]:
        r, kind = self._recipe(design)
        if r is None:
            return (5.0, 0.0)  # the stock blade: never jams, never centres
        sweep = 2.0 if kind == "imprint_swept" else 0.0
        draft = math.radians(r["wall_draft_deg"])
        if r["pad"] == "flat":
            return (5.0, 0.0)
        reach = r["pad_depth_mm"] * math.tan(min(draft, math.radians(80.0)))
        chamfer = r["tip_size_mm"] if r["tip"] == "chamfer" else 0.0
        capture = r["clearance_mm"] + reach + chamfer + sweep
        if part.part_id in _ROUND:
            capture += 0.6 * r["pad_depth_mm"]  # a round outline slides into a pocket
        friction_ok = r["wall_draft_deg"] <= 90.0 - math.degrees(math.atan(_MU))
        no_jam = capture if friction_ok or r["pad"] == "conform" else min(capture, r["clearance_mm"])
        centring = capture - sweep if friction_ok else 0.0
        return (round(min(no_jam, 5.0), 1), round(max(0.0, min(centring, 5.0)), 1))

    def _mass_g(self, design: Any) -> float:
        r, _ = self._recipe(design)
        if r is None:
            return 24.0
        return round(
            8.0 + 0.45 * r["finger_width_mm"] + 0.35 * r["pad_height_mm"] + 0.4 * r["pad_depth_mm"], 3
        )

    def _pieces(self, design: Any) -> int:
        r, _ = self._recipe(design)
        if r is None:
            return 9
        base = {"flat": 3, "conform": 10, "v_funnel": 6}[r["pad"]]
        return base + (4 if r["tip"] == "hook" else 0)

    def _episode(self, design: Any, design_id: str, cond: ConditionV2, seed: int) -> EpisodeOutcome:
        r, _kind = self._recipe(design)
        part = next(p for p in self.parts if p.part_id == cond.part_id)
        place = next(p for p in self.placements if p.placement_id == cond.placement_id)
        jitter = 0.5 * (2.0 * _unit("jit", self.landscape_seed, cond.condition_id, seed) - 1.0)
        error = abs(place.dx_mm + jitter) + 0.15 * abs(place.dyaw_deg)
        u = _unit(self.landscape_seed, design_id, cond.condition_id, seed)
        tail = _unit("tail", design_id, cond.condition_id, seed)
        # lift and hold, whatever the registration
        if r is None:  # the stock blade
            lift_logit = 1.4 - part.mass_g / 400.0
        else:
            prof = self._profiles[part.part_id]
            x = [
                (float(r[n]) - NUMERIC_BOUNDS[n][0]) / (NUMERIC_BOUNDS[n][1] - NUMERIC_BOUNDS[n][0])
                for n in _HOLD_DIMS
            ]
            d2 = sum(
                w * (a - c) ** 2 for a, c, w in zip(x, prof["centre"], prof["weights"], strict=True)
            ) / sum(prof["weights"])
            lift_logit = (
                2.4
                - 14.0 * d2
                - part.mass_g / 400.0
                + (
                    1.0
                    if r["tip"] == "hook"
                    and part.part_id == "flanged_bushing"
                    and r["grip_height_frac"] < 0.45
                    else 0.0
                )
                + (0.3 if r["pad"] != "flat" else 0.0)
                - (
                    1.2 if r["wall_draft_deg"] > 55.0 and part.size_mm[2] > 35.0 else 0.0
                )  # tall parts tip out
            )
        lifted = u < 1.0 / (1.0 + math.exp(-lift_logit))
        no_jam, centring = self._capture(design, part)
        jam = r is not None and r["pad"] != "flat" and part.part_id not in _ROUND and error > no_jam + 0.05
        if r is not None and r["pad"] == "v_funnel" and r["wall_draft_deg"] > 68.0 and error > 0.75:
            jam = True  # past the friction rule: the part sticks on the flank
        centred = not jam and (error <= 0.75 or error <= centring)
        stock_push = r is None and error > 0.75 and tail < 0.7  # knocked out by the push
        off = not jam and not centred
        registered = (
            lifted and centred and not stock_push and _unit("reg", design_id, cond.condition_id, seed) < 0.95
        )
        if registered:
            mode = None
        elif stock_push:
            mode = "ejected_on_close"
        elif not lifted:
            mode = "slip_during_lift" if tail < 0.6 else "unstable_hold"
        else:
            mode = "not_registered"
        seat_short = (error - no_jam) if jam else 0.1 * tail
        return EpisodeOutcome(
            condition_id=cond.condition_id,
            family=part.part_id,
            controller_setting=0,
            seed=seed,
            success=registered,
            failure_mode=mode,
            max_lift_m=0.15 if lifted else round(0.1 * tail, 6),
            hold_frac=1.0 if lifted else round(0.9 * tail, 6),
            lift_success=lifted and not stock_push,
            registered=registered,
            jam_on_close=bool(jam),
            off_centre=bool(off),
            seat_short_mm=round(seat_short, 3),
            reg_dx_mm=round(0.0 if centred else error, 3),
            reg_dy_mm=0.0,
            reg_dyaw_deg=round(abs(place.dyaw_deg) if not centred else 0.2, 3),
            reg_tilt_deg=round(1.0 * tail, 3),
            part_id=part.part_id,
            placement=place.placement_id,
        )


# --------------------------------------------------------------------------- scripted model

_RECIPE_RE = re.compile(r'\{"pad": "[a-z_]+", "tip": "[a-z]+"[^{}]*\}')
_LABEL_RE = re.compile(r"\b(d\d{2})\b")
_FOLLOW_UP_MARKERS: Final = ("# Correction needed", "# Too similar")
_DEFAULT: Final[dict[str, Any]] = {
    "pad": "conform",
    "tip": "blunt",
    "grip_height_frac": 0.5,
    "pad_height_mm": 18.0,
    "finger_width_mm": 26.0,
    "pad_depth_mm": 4.0,
    "wall_draft_deg": 30.0,
    "clearance_mm": 0.5,
    "tip_size_mm": 0.0,
}


class ScriptedProposerV2:
    """An offline ``Transport`` that answers forced v2 ``submit_designs`` calls (see module docstring).

    Answers are a pure function of the request payload and ``seed``, so a scripted arm is
    reproducible and a recorded cache replays it exactly.

    Args:
        seed: Varies the answers.
        invalid_rate: Probability that a main-call slot breaks a strategy rule.
        similar_rate: Probability that a main-call slot repeats a recipe from the prompt
            (sometimes with an inactive number added, so only canonicalisation shows it).
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
            message = {"role": "assistant", "content": "I would rather describe the jaws in words."}
            return self._response(message, "stop", usage)
        function = payload["tools"][0]["function"]
        properties = function["parameters"]["properties"]
        seen = [json.loads(m) for m in _RECIPE_RE.findall(text)]
        labels = sorted(set(_LABEL_RE.findall(text)))
        args: dict[str, Any] = {"analysis": "Scripted analysis: widen the capture range where parts jam."}
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
            "id": f"scripted-v2-{self.calls}",
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
        if "operator" in props:
            out["operator"] = rng.choice(["mutate", "crossover", "explore"])
        if "parents" in props:
            n = int(props["parents"].get("minItems", 1)) or 1
            out["parents"] = rng.sample(list(labels), n) if len(labels) >= n else list(labels)[:n]
        out["hypothesis"] = "Scripted variation of a promising recipe."
        if "changes" in props:
            out["changes"] = "Scripted change: about 120 points."
        if "same_cell_distances" in props:
            out["same_cell_distances"] = "untried cell"
        recipes = [s for s in seen if "baseline" not in s]
        if recipes and not clean and rng.random() < self.similar_rate:
            params = dict(rng.choice(recipes))
            if params.get("pad") == "flat" and rng.random() < 0.5:
                params["pad_depth_mm"] = 3.0  # inactive: canonicalisation drops it
        else:
            base = dict(rng.choice(recipes)) if recipes and rng.random() < 0.7 else dict(_DEFAULT)
            params = self._mutate(base, rng)
        if not clean and rng.random() < self.invalid_rate:
            params["pad"], params["wall_draft_deg"], params["pad_depth_mm"] = "v_funnel", 5.0, 6.0
        out["pad"], out["tip"] = params["pad"], params["tip"]
        out["params"] = {k: params[k] for k in NUMERIC_PARAMS if k in params}
        return out

    @staticmethod
    def _mutate(base: dict[str, Any], rng: random.Random) -> dict[str, Any]:
        params = {**_DEFAULT, **base}
        if rng.random() < 0.4:
            params["pad"] = rng.choice(PAD_OPTIONS)
        if rng.random() < 0.4:
            params["tip"] = rng.choice(TIP_OPTIONS)
        active = [n for n in NUMERIC_PARAMS if not _inactive(n, params)]
        for name in rng.sample(active, min(3, len(active))):
            lo, hi = NUMERIC_BOUNDS[name]
            step = rng.uniform(0.2, 0.4) * (hi - lo) * rng.choice((-1.0, 1.0))
            params[name] = round(min(hi, max(lo, float(params[name]) + step)), 2)
        if params["pad"] == "v_funnel":
            params["wall_draft_deg"] = min(70.0, max(15.0, float(params["wall_draft_deg"])))
            params["pad_depth_mm"] = max(3.0, float(params["pad_depth_mm"]))
        if params["pad"] == "conform":
            params["pad_depth_mm"] = max(1.0, float(params["pad_depth_mm"]))
        if params["tip"] == "hook":
            params["tip_size_mm"] = max(2.0, float(params["tip_size_mm"]))
        elif params["tip"] == "chamfer":
            params["tip_size_mm"] = max(1.0, float(params["tip_size_mm"]))
        return {k: v for k, v in params.items() if k in ("pad", "tip") or not _inactive(k, params)}


def _inactive(name: str, params: Mapping[str, Any]) -> bool:
    rule = INACTIVE.get(name)
    return rule is not None and params.get(rule[0]) in rule[1]
