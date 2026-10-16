"""Arms C and Z on the ``freeform-v2`` design space: prompts, renderers and the forced tool.

The proposal machinery (islands asked in turn, reserves, one repair and one reflection per
island and round, ``fallback_random``, the call cap, failure records, replay) is the v1
machinery of :mod:`rtt.search.nemotron`, unchanged; only what a design is and what the
prompt says differ. Templates live in ``prompts/v2/`` and are hashed separately from v1.

**Tool.** ``submit_designs`` keeps the v1 slots (``mutate_1``, ``mutate_2``, ``crossover``,
``explore``, then ``reserve_1`` ... ``reserve_4``: 4 slots + 4 reserves, the near-copy fix
of ``docs/PROPOSER.md``). Each slot gives ``hypothesis``, ``changes`` and
``same_cell_distances`` first, then ``pad``, ``tip`` and ``params``. Only the parameters
that are active for the chosen pad and tip are required; inactive ones are ignored and the
proposal logs that they were (``canon_notes``).

**Prompt additions over v1** (spec section 12): the training parts table (family, size,
features, material, mass, yaw symmetry), the placement conditions and the registration
tolerances; per elite, the per-part and per-placement success vectors, the ``jam_on_close``
and ``off_centre`` counts, the capture report per part, the mass and the piece count. The
**engineering notes** block (friction rule, capture rule, stroke cap, print rules) is the same
text for arms C and Z, rendered from ``prompts/v2/engineering_notes.md`` with the task's
numbers and hashed into the manifest. It is guidance, never a validator.

**Screen.** Every proposal goes through the v2 screen: ``duplicate_design`` (same canonical
recipe), ``duplicate_geometry`` (same jaws on every training part), ``too_similar`` (below the
rule-set threshold) and, for ``explore``, ``not_novel``. All four are answered by the
reflection call.

**Guard.** Besides held-out vocabulary, no v2 prompt may name the unseen held-out part
families (pipe elbows, spur gears); model text quoted back is scrubbed of them first.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from rtt.llm.cache import canonical_json
from rtt.search import prompts
from rtt.search.base import Evaluated, EvaluatorInfo, Rejection, V2Info
from rtt.search.diversity import (
    BASELINE_CELL,
    EXPLORE_MIN_DISTANCE_V2,
    MIN_DISTANCE_V2,
    POINTS_PER_RANGE,
    UNREACHABLE_POINTS,
    cell_name,
    cell_of,
    distance_points,
    neighbours,
)
from rtt.search.nemotron import (
    DIVERSITY_REASONS,
    PROPOSER_SLOTS,
    NemotronProposer,
    ProposerConfig,
    _Claim,
    _failure_counts,
    _json_safe,
    _num,
    _origin_note,
    _pct,
    render_target,
    scrub_model_text,
    submit_designs_tool,
)
from rtt.search.space import V2, DesignSpace, get_space
from rtt.search.space_v2 import POINT_COLUMNS

__all__ = [
    "DIVERSITY_REASONS_V2",
    "FORBIDDEN_PROMPT_TEXT_V2",
    "NemotronProposerV2",
    "prompt_spec_hashes_v2",
    "render_distance_rule_v2",
    "render_elites_v2",
    "render_failures_v2",
    "render_notes",
    "render_operators_v2",
    "render_parts",
    "render_placements",
    "render_points_rule_v2",
    "render_rejections_v2",
    "render_rules_v2",
    "render_task_v2",
    "render_tested_v2",
    "require_v2",
]

#: Held-out vocabulary, the secrets folder, the unseen held-out part families and the held-out
#: procedural part ids (``hp2-<index>-<family>``, track O).
FORBIDDEN_PROMPT_TEXT_V2: Final = re.compile(
    r"held[\s_-]?out|secrets?[/\\]|\belbows?\b|\bspur\b|\bgears?\b|\bhp2-\d", re.IGNORECASE
)
#: v2 rejection reasons the reflection call answers.
DIVERSITY_REASONS_V2: Final[tuple[str, ...]] = (*DIVERSITY_REASONS, "duplicate_design", "duplicate_geometry")


def require_v2(info: EvaluatorInfo) -> V2Info:
    if info.v2 is None:
        raise ValueError("this is a v1 evaluator: the v2 arms need EvaluatorInfo.v2 (parts and placements)")
    return info.v2


def _space() -> DesignSpace:
    return get_space(V2)


def _mm(value: float) -> str:
    return _num(value)


def _num3(value: float) -> str:
    """Up to 3 decimals (thresholds such as 0.081 must not print as 0.08)."""
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


# --------------------------------------------------------------------------- task, parts, placements


def render_task_v2(info: EvaluatorInfo) -> str:
    """The task, the controller, the registered-grasp rule and the physics (never anything held out)."""
    v2 = require_v2(info)
    reg = v2.registration
    lift_cm = _num(info.success_lift_m * 100.0)
    return "\n".join(
        [
            (
                "Pick each training part up from a table and register it: close the jaws so that the part seats "
                f"and centres between them, then lift it at least {lift_cm} cm and hold it for "
                f"{_num(info.success_hold_s)} s, all within {_num(info.max_time_s)} s."
            ),
            (
                f"- Gripper: a parallel gripper with two coupled jaws, each sliding 0-{_num(v2.stroke_mm)} mm toward "
                f"the other ({_num(v2.opening_mm)} mm apart fully open), {_num(v2.grip_force_per_jaw_n)} N per jaw. "
                "Your recipe is compiled into a separate left and right jaw for every part, shaped to that part, "
                f"and both are 3D-printed in PLA ({_num(info.finger_density_kg_m3)} kg/m^3)."
            ),
            (
                "- Controller (the same for every design): approach from above, lower the jaws until the bottom of "
                "the pad band is grip_height_frac x part height - pad_height_mm / 2 above the table (at least "
                f"{_num(v2.min_tip_height_mm)} mm), close, settle for {_num(reg.settle_s)} s, lift, hold. There is no "
                "other grasp-height setting: the grip height is part of your design."
            ),
            (
                f"- Registered grasp (the score): {_num(reg.settle_s)} s after closing, before the lift, the part "
                f"must sit at most {_num(reg.seat_short_mm)} mm short of its seat, within {_num(reg.dx_mm)} mm of "
                f"centre across the jaw and {_num(reg.dy_mm)} mm along the closing axis, within "
                f"{_num(reg.yaw_deg)} deg of its yaw (modulo the part's symmetry) and tilted at most "
                f"{_num(reg.tilt_deg)} deg; it must then be lifted and held, and be tilted at most "
                f"{_num(reg.tilt_after_hold_deg)} deg after the hold. A grasp that lifts and holds but misses the "
                'registration counts as "lift" only. A miss is flagged jam_on_close when the part stopped more '
                f"than {_num(reg.seat_short_mm)} mm short of its seat, and off_centre when it seated outside the "
                "centring tolerances."
            ),
            (
                f"- Physics: rigid parts, friction mu {_num(info.friction)}; every episode jitters the placement by "
                f"up to {_num(info.jitter_pos_mm)} mm and {_num(info.jitter_yaw_deg)} deg and the friction by up to "
                f"{_pct(info.jitter_mu_frac)}, with the same jitter for every design."
            ),
        ]
    )


def render_parts(v2: V2Info) -> str:
    """Markdown table of the training parts."""
    rows = [
        "Sizes are x across the jaw, y along the closing axis (the left jaw closes from +y), z up.",
        "",
        "| part | what it is | size x / y / z (mm) | features | material | mass | yaw symmetry |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in v2.parts:
        size = " / ".join(_mm(v) for v in p.size_mm)
        features = "; ".join(p.features) or "none"
        sym = (
            "any yaw"
            if p.yaw_symmetry == 0
            else ("none" if p.yaw_symmetry == 1 else f"{p.yaw_symmetry}-fold")
        )
        rows.append(
            f"| {p.part_id} | {p.description} | {size} | {features} | {p.material} | {_num(p.mass_g)} g | {sym} |"
        )
    return "\n".join(rows)


def _placement_text(dx_mm: float, dyaw_deg: float) -> str:
    if dx_mm == 0.0 and dyaw_deg == 0.0:
        return "centred"
    parts = []
    if dx_mm:
        parts.append(f"{_num(abs(dx_mm))} mm off centre across the jaw ({'+' if dx_mm > 0 else '-'}x)")
    if dyaw_deg:
        parts.append(f"turned {_num(dyaw_deg)} deg")
    return " and ".join(parts)


def render_placements(v2: V2Info, info: EvaluatorInfo) -> str:
    """The placement conditions every part is tested at."""
    lines = [
        (
            f"Every part is tested at {len(v2.placements)} placements, each with {info.seeds_per_condition} "
            f"jitter seeds ({info.episodes_per_design} episodes per design, the same for every design):"
        )
    ]
    for p in v2.placements:
        lines.append(f"- {p.placement_id}: {p.label or _placement_text(p.dx_mm, p.dyaw_deg)}")
    return "\n".join(lines)


def render_rules_v2(v2: V2Info) -> str:
    """The validators' rules (spec section 13) with this task's constants."""
    lo, hi = v2.q_seat_range_mm
    return "\n".join(
        [
            (
                "- Strategies: a v_funnel needs wall_draft_deg between 15 and 70 and pad_depth_mm >= 3; a conform "
                "pad needs pad_depth_mm >= 1; a hook needs tip_size_mm >= 2. Inactive numbers are ignored (see the "
                "parameter table)."
            ),
            (
                "- On every training part: the pad band must cover part of the part's outline; fully open, each jaw "
                f"must clear the part by at least {_num(v2.open_clearance_mm)} mm beyond the "
                f"{_num(v2.error_envelope_mm)} mm error envelope; at the seat the two pads stay at least "
                f"{_num(v2.min_pad_gap_mm)} mm apart; the jaws seat {_num(lo)}-{_num(hi)} mm into the stroke; the "
                f"jaw is at most {_num(v2.max_length_mm)} mm long with the palm at least "
                f"{_num(v2.palm_clearance_mm)} mm above the part; the jaw is at most {_num(v2.max_finger_width_mm)} mm "
                f"wide and at least {_num(v2.min_table_clearance_mm)} mm above the table."
            ),
            (
                "- Seat: the pad never cuts into the part and keeps its clearance from it (only a flat pad may touch "
                "it), and the two jaws never intersect."
            ),
            (
                f"- Build: at most {v2.max_pieces_per_jaw} convex pieces per jaw, and each printed jaw is one closed "
                "solid."
            ),
            (
                f"- Printability: walls at least {_num(v2.min_wall_mm)} mm and features at least "
                f"{_num(v2.min_feature_mm)} mm thick, at most {_num(v2.max_jaw_mass_g)} g of PLA per jaw on the "
                f"heaviest part, body slopes of at most {_num(v2.max_body_slope_deg)} deg."
            ),
        ]
    )


def notes_values(info: EvaluatorInfo) -> dict[str, str]:
    """The numbers the engineering notes are rendered with (all from the task, none tuned)."""
    v2 = require_v2(info)
    mu = info.friction
    max_draft = math.floor(90.0 - math.degrees(math.atan(mu)) - v2.friction_margin_deg + 1e-9)
    max_error = max((abs(p.dx_mm) for p in v2.placements), default=0.0)
    margin = v2.open_clearance_mm + v2.error_envelope_mm
    caps = []
    for part in v2.parts:
        cap = (v2.opening_mm - part.width_across_mm) / 2.0 - margin
        caps.append(f"{part.part_id} {_num(math.floor(cap * 10.0) / 10.0)} mm")
    return {
        "mu": _num(mu),
        "margin_deg": _num(v2.friction_margin_deg),
        "max_draft_deg": str(max_draft),
        "max_error_mm": _num(max_error),
        "jitter_mm": _num(info.jitter_pos_mm),
        "error_band_mm": _num(max_error + info.jitter_pos_mm),
        "opening_mm": _num(v2.opening_mm),
        "open_clear_mm": _num(v2.open_clearance_mm),
        "envelope_mm": _num(v2.error_envelope_mm),
        "open_margin_mm": _num(margin),
        "stroke_caps": ", ".join(caps) or "no training parts",
        "min_wall_mm": _num(v2.min_wall_mm),
        "min_feature_mm": _num(v2.min_feature_mm),
        "max_mass_g": _num(v2.max_jaw_mass_g),
        "max_pieces": str(v2.max_pieces_per_jaw),
        "max_slope_deg": _num(v2.max_body_slope_deg),
    }


def render_notes(info: EvaluatorInfo) -> str:
    """The engineering-notes block shared by arms C and Z (hashed into the manifest)."""
    return prompts.render("v2/engineering_notes.md", **notes_values(info))


# --------------------------------------------------------------------------- distance rule and operators


def _need(min_distance: float, space: DesignSpace) -> int:
    return math.ceil(min_distance * POINTS_PER_RANGE * space.n_terms - 1e-9)


def _points(distance: float, space: DesignSpace) -> int:
    return UNREACHABLE_POINTS if math.isinf(distance) else round(distance * POINTS_PER_RANGE * space.n_terms)


def render_points_rule_v2(min_distance: float, space: DesignSpace | None = None) -> str:
    """The v2 duplicate and diversity rule in points and cells (arm C's prompt)."""
    space = space or _space()
    need = _need(min_distance, space)
    n_numbers = len(space.numeric_params)
    cells = (
        "Designs in different cells are always at least 100 points apart, so only the tested designs in your "
        "design's own cell can be too close, and a design in a cell that no tested design uses always passes."
        if need <= 100
        else (
            f"A different cell is worth only 100 points, so a design in another cell must still differ by "
            f"{need - 100} points or more in its numbers."
        )
    )
    return (
        "Positions are given in points: each number's range runs from 0 points (its minimum) to 100 points (its "
        'maximum), so 1 point is 1% of the range and the "10% of range" column above is 10 points. Inactive '
        "numbers sit at their fixed values (see the parameter table) and count like any other number. A design's "
        f"cell is its pad + tip pair. The distance between two designs in points is the sum over the {n_numbers} "
        "numbers of their difference in points, plus 100 if the pad differs and 100 if the tip differs.\n\n"
        f"Every new design must be at least {need} points from every tested design and from every other new design "
        "this round, or it is rejected without being simulated and its slot goes to a reserve or a random design. "
        f"{cells} Within a cell the numbers must differ by at least {need} points in total from each tested design "
        f"there; aim for {need + 20} or more, because the listed positions are rounded. Two designs that are the "
        "same once inactive numbers are set to their fixed values and every number is rounded to its grid (0.1 mm, "
        "0.5 deg, 0.01) are duplicates, and so are two designs that build exactly the same jaws on every training "
        "part: both are rejected. Tweaking one or two numbers of a tested design by 10-30 points is always rejected."
    )


def render_distance_rule_v2(min_distance: float, space: DesignSpace | None = None) -> str:
    """The same rule in range fractions (arm Z's prompt, which has no points table)."""
    space = space or _space()
    n = space.n_terms
    total = min_distance * n
    return (
        "Every new design must differ from every design already submitted in this run, and from your other new "
        f"designs, by a normalised distance of at least {_num3(min_distance)}, or it is rejected without being "
        f"simulated. The distance is the sum over the {n} parameters of |change| / range (a changed pad or tip "
        f"counts 1; inactive numbers count at their fixed values), divided by {n}, so the changes must add up to "
        f"at least {_num3(total)}: switching the pad or the tip is enough on its own; otherwise numbers must move by "
        f"{_pct(total)} of a range in total, for example three numbers by about {_pct(total / 3)} of their range "
        'each (the "10% of range" column gives the step sizes). Designs that are identical once inactive numbers are '
        "ignored and values are rounded to their grid are duplicates and are always rejected."
    )


def render_operators_v2(
    untried: Sequence[str],
    min_distance: float,
    explore_min_distance: float,
    space: DesignSpace,
    *,
    claims: bool = False,
    reserves: int = 0,
) -> str:
    """What each slot must do, with the point budgets of the distance rule (v2 wording)."""
    need = _need(min_distance, space)
    far = _need(explore_min_distance, space) if explore_min_distance > 0 else 0
    cells = ", ".join(untried)
    if untried and far:
        explore_rule = (
            f"Put it in a cell that no tested design uses ({cells}); if you use a cell that already has tested "
            f"designs, it must be at least {far} points from every one of them. An explore that misses this is "
            "rejected like a near-copy."
        )
    elif far:
        explore_rule = (
            f"Every cell already has a tested design, so it must be at least {far} points from every tested design "
            "in its cell. An explore that misses this is rejected like a near-copy."
        )
    elif untried:
        explore_rule = f"Prefer a cell that no tested design uses ({cells})."
    else:
        explore_rule = "Keep it far from every tested design in its cell."
    claims_text = (
        ' and from the designs other teams already submitted this round (rows "t<team>.<slot>" in the table)'
        if claims
        else ""
    )
    lines = [
        (
            f"Call submit_designs with exactly 4 new designs, one per slot, plus {reserves} reserves:"
            if reserves
            else "Call submit_designs with exactly 4 new designs, one per slot:"
        ),
        (
            '- mutate_1 and mutate_2: each starts from ONE tested design (put its label, e.g. "d05", in parents) and '
            "makes one bold change aimed at that design's main failure (for example the parts and placements where it "
            "jams on close or ends off centre), worth 100-250 points in total: for example 2-4 active numbers moved by "
            "25-60 points each, or a switch of pad or tip (100 points) together with the numbers that switch needs. "
            "Give the two mutations different parents or clearly different changes. Mutations usually fail by "
            "landing on an earlier variation of the same parent, so check every tested design in the cell you end "
            "up in."
        ),
        (
            f"- crossover: combines TWO tested designs (both labels in parents) that are at least {2 * need} points "
            "apart. Take each parameter whole from one parent or the other; do not average numbers, because an "
            f"average lands close to both parents. The child must differ from each parent by at least {need} points."
        ),
        f"- explore: tests an idea that no tested design covers. {explore_rule}",
    ]
    if reserves:
        lines.append(
            f"- reserve_1 to reserve_{reserves}: {reserves} more designs, each with its own operator (mutate, "
            "crossover or explore) and parents, following the same rules. Whenever one of the 4 designs above is "
            "rejected (it breaks a validator rule, duplicates a design or is too close to a tested design), the next "
            "reserve that passes takes its place, so make the reserves as good as the first four and different "
            "from them."
        )
    lines += [
        "",
        "In each slot, before the pad, tip and params:",
        (
            '- "changes": every parameter you change from the parent (explore: from the nearest tested design in its '
            "cell) as name old -> new with the difference in points, then the total, e.g. "
            '"wall_draft_deg 0 -> 30 deg (+43), tip blunt -> chamfer (+100), tip_size_mm 0 -> 3 mm (+38): 181 points '
            'from d06".'
        ),
        (
            '- "same_cell_distances": go through the cell\'s rows in the table above and give the distance in points '
            "from the new design to each tested design in the same cell (at least the 3 nearest), nearest first, "
            f'e.g. "d25 95, d06 150, d11 210", or "untried cell". If any is below {need}, change the design before '
            "you write its parameters."
        ),
        (
            f"Every new design must also be at least {need} points from your other new designs this round"
            f"{claims_text}."
        ),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- run history


def _score_text(item: Evaluated) -> str:
    s = item.score
    lift = s.lift_success if s.lift_success is not None else s.success
    return f"{_pct(s.success)} (lift {_pct(lift)})"


def render_tested_v2(
    items: Sequence[Evaluated],
    island: int | None = None,
    claims: Sequence[_Claim] = (),
    space: DesignSpace | None = None,
) -> str:
    """Every tested design of the run grouped by cell, positions in points, plus the untried cells."""
    space = space or _space()
    legend = (
        "Scores are the registered-grasp share (lift-only share in brackets). Positions are in points (0-100): "
        + ", ".join(f"{short} = {name}" for name, short in POINT_COLUMNS.items())
        + ". Cells run from the lowest best score to the highest; inside a cell, rows run from the lowest score "
        "to the highest."
    )
    if claims:
        legend += (
            ' Rows labelled "t<team>.<slot>" were submitted this round by another team and are not simulated '
            "yet; they count as tested for the distance rule."
        )
    groups: dict[tuple[str, str], list[tuple[float, int, str]]] = {}
    for item in items:
        line = f"  {item.label} {_score_text(item)} | {space.points_row(item.design)} | {_origin_note(item, island)}"
        groups.setdefault(cell_of(item.design, space), []).append((item.score.success, item.index, line))
    for order, claim in enumerate(claims):
        line = f"  {claim.label} new | {space.points_row(claim.design)} | {claim.note}"
        groups.setdefault(cell_of(claim.design, space), []).append((math.inf, len(items) + order + 1, line))
    lines = [legend, ""]

    def cell_key(cell: tuple[str, str]) -> tuple[int, float, str]:
        scores = [s for s, _, _ in groups[cell] if s != math.inf]
        return (int(cell == BASELINE_CELL), max(scores) if scores else math.inf, cell_name(cell))

    for cell in sorted(groups, key=cell_key):
        rows = sorted(groups[cell], key=lambda t: (t[0], t[1]))
        scores = [s for s, _, _ in rows if s != math.inf]
        if cell == BASELINE_CELL:
            head = f"reference entrants that are not recipes ({len(scores)} tested, never too close to a recipe):"
        else:
            head = f"{cell_name(cell)} ({len(scores)} tested"
            head += f", best {_pct(max(scores))}):" if scores else "):"
        lines.append(head)
        lines.extend(line for _, _, line in rows)
    untried = [cell_name(c) for c in space.cells() if c not in groups]
    lines.append("")
    lines.append(
        "Cells with no tested design yet: "
        + (", ".join(untried) if untried else "none, every cell has at least one tested design")
        + "."
    )
    return "\n".join(lines)


def _vector(values: Mapping[str, float], order: Sequence[str]) -> str:
    return ", ".join(f"{k} {_pct(values.get(k, 0.0))}" for k in order)


def _nearest_line(item: Evaluated, history: Sequence[Evaluated], k: int, space: DesignSpace) -> str:
    others = [o for o in history if o.index != item.index and space.cell(o.design) is not None]
    near = neighbours(item.design, [o.design for o in others], [o.label for o in others], k, space=space)
    return ", ".join(f"{n.label} {_points(n.distance, space)}" for n in near) or "none"


def render_elites_v2(
    ranked_pool: Sequence[Evaluated],
    info: EvaluatorInfo,
    n: int,
    *,
    history: Sequence[Evaluated] | None = None,
    near_k: int = 0,
    space: DesignSpace | None = None,
    scrub: Callable[[str], str] | None = None,
) -> str:
    """Top ``n`` of an already ranked pool (best first), printed worst first, with the v2 feedback."""
    space = space or _space()
    scrub = scrub or (lambda text: scrub_model_text(text, FORBIDDEN_PROMPT_TEXT_V2))
    v2 = require_v2(info)
    parts = [p.part_id for p in v2.parts]
    places = [p.placement_id for p in v2.placements]
    lines: list[str] = []
    for item in reversed(list(ranked_pool[:n])):
        s = item.score
        lift = s.lift_success if s.lift_success is not None else s.success
        pieces = f", up to {s.max_pieces} pieces per jaw" if s.max_pieces is not None else ""
        lines.append(
            f"{item.label} [{item.proposal.origin}, round {item.round}] score {_pct(s.success)}, lift {_pct(lift)} | "
            f"per part: {_vector(s.per_family, parts)} | per placement: {_vector(s.per_placement or {}, places)} | "
            f"jam_on_close {s.jam_on_close or 0}, off_centre {s.off_centre or 0} | failures: "
            f"{_failure_counts(s.failure_modes)} | mass {s.mass_g:.1f} g{pieces}"
        )
        lines.append(f"  {space.params_json(item.design)}")
        if s.capture:
            capture = ", ".join(
                f"{pid} {_num(s.capture[pid][0])}/{_num(s.capture[pid][1])}"
                for pid in parts
                if pid in s.capture
            )
            lines.append(f"  capture no-jam/centring (mm): {capture}")
        if history is not None and near_k > 0:
            if space.cell(item.design) is None:
                lines.append("  not a recipe: a fixed reference entrant, outside the distance rule")
            else:
                lines.append(
                    f"  cell {cell_name(cell_of(item.design, space))}; nearest (points): "
                    f"{_nearest_line(item, history, near_k, space)}"
                )
        idea = item.proposal.hypothesis or getattr(space, "rationale", lambda d: None)(item.design)
        if idea:
            lines.append(f"  idea: {scrub(' '.join(idea.split())[:200])}")
    return "\n".join(lines) if lines else "None yet."


def render_pairs_v2(ranked_pool: Sequence[Evaluated], n: int, need: int, space: DesignSpace) -> str:
    """Pairs among the top ``n`` recipes at least ``need`` points apart (crossover candidates)."""
    top = [item for item in ranked_pool[:n] if space.cell(item.design) is not None]
    pairs: list[tuple[float, int, str]] = []
    for i, a in enumerate(top):
        for b in top[i + 1 :]:
            d = distance_points(a.design, b.design, space)
            if d >= need:
                pairs.append((-(a.score.success + b.score.success), -d, f"{a.label} + {b.label} ({d})"))
    if not pairs:
        return (
            f"Crossover candidates: no two of your top {len(top)} recipes are {need} points apart; take a second "
            "parent from another cell in the table."
        )
    pairs.sort()
    return (
        f"Crossover candidates (pairs of your top {len(top)} recipes at least {need} points apart, best pairs "
        f"first): " + ", ".join(text for _, _, text in pairs[:6]) + "."
    )


def render_failures_v2(ranked_pool: Sequence[Evaluated], info: EvaluatorInfo, n: int) -> str:
    """Up to ``n`` failed (design, condition) pairs that say the most (distinct part and failure first)."""
    v2 = require_v2(info)
    labels = {c.condition_id: c.label for c in info.conditions}
    candidates: list[tuple[Evaluated, str]] = []
    for item in ranked_pool:
        failed = sorted(
            (cid for cid, rate in item.score.per_condition.items() if rate < 1.0),
            key=lambda cid: (item.score.per_condition[cid], cid),
        )
        candidates.extend((item, cid) for cid in failed)
    chosen: list[tuple[Evaluated, str]] = []
    seen: set[tuple[str, str]] = set()
    for item, cid in candidates:
        modes = item.score.condition_failures.get(cid, {})
        key = (v2.part_of(cid) or cid, next(iter(sorted(modes, key=lambda m: -modes[m])), "unknown"))
        if key not in seen:
            seen.add(key)
            chosen.append((item, cid))
        if len(chosen) == n:
            break
    for pair in candidates:
        if len(chosen) >= n:
            break
        if pair not in chosen:
            chosen.append(pair)
    lines = []
    for item, cid in chosen:
        eps = [e for e in item.score.episodes if e.condition_id == cid]
        registered = sum(1 for e in eps if e.success)
        lifted = sum(1 for e in eps if (e.lift_success if e.lift_success is not None else e.success))
        jams = sum(1 for e in eps if e.jam_on_close)
        off = sum(1 for e in eps if e.off_centre)
        modes = _failure_counts(item.score.condition_failures.get(cid, {}))
        flags = f" (jam_on_close {jams}, off_centre {off})" if jams or off else ""
        winner = next(
            (o for o in ranked_pool if o is not item and o.score.per_condition.get(cid, 0.0) >= 1.0), None
        )
        tail = (
            f" {winner.label} registered it every time."
            if winner
            else " No tested design registered it every time yet."
        )
        lines.append(
            f"- {item.label} (score {_pct(item.score.success)}) on {cid} ({labels.get(cid, cid)}): "
            f"{registered}/{len(eps)} registered, {lifted}/{len(eps)} lifted; {modes}{flags}.{tail}"
        )
    return (
        "\n".join(lines) if lines else "None: every tested design registered every part at every placement."
    )


def _raw_params_json_v2(params: Mapping[str, Any] | None, scrub: Callable[[str], str]) -> str:
    if not params:
        return "(no parameters)"
    try:
        text = json.dumps(dict(params), separators=(", ", ": "))
    except (TypeError, ValueError):
        text = repr(dict(params))[:400]
    return scrub(text)


def render_rejections_v2(rejections: Sequence[Rejection], n: int, scrub: Callable[[str], str]) -> str:
    if not rejections:
        return "None."
    lines = []
    for r in rejections[:n]:
        errors = scrub("; ".join(e[:240] for e in r.errors[:3]))
        where = f"{r.slot} ({r.stage} call)" if r.slot else f"{r.stage} call"
        lines.append(f"- {where}, {r.reason}: {_raw_params_json_v2(r.params, scrub)} -> {errors}")
    if len(rejections) > n:
        lines.append(f"- ... and {len(rejections) - n} more.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- arm C v2


class NemotronProposerV2(NemotronProposer):
    """Arm C on the v2 design space (same machinery and knobs as v1 arm C; v2 prompt and tool)."""

    arm = "nemotron"
    origin = "nemotron"
    repair_template = "v2/repair.md"
    diversity_template = "v2/diversity.md"
    diversity_reasons = DIVERSITY_REASONS_V2
    forbidden = FORBIDDEN_PROMPT_TEXT_V2

    def build_messages(self, round_index: int, island: int) -> list[dict[str, str]]:
        cfg = self.config
        info = self.ctx.info
        v2 = require_v2(info)
        space = self.ctx.space
        ranked = self.ranked_pool(island)
        if not ranked:
            raise RuntimeError("the island pool is empty: round 0 was not told to the strategy")
        history = list(self.ctx.history)
        claims = self.round_claims(round_index, island)
        used = {cell_of(item.design, space) for item in history} | {cell_of(c.design, space) for c in claims}
        untried = [cell_name(c) for c in space.cells() if c not in used]
        last_round = [r for r in self.ctx.rejections if r.round == round_index - 1 and r.island == island]
        elites = render_elites_v2(
            ranked,
            info,
            cfg.elites_shown,
            history=history,
            near_k=cfg.neighbours_shown,
            space=space,
            scrub=self._scrub,
        )
        pairs = render_pairs_v2(ranked, cfg.crossover_pool, 2 * self.kit.min_points(cfg.min_distance), space)
        system = prompts.render(
            "v2/proposer_system.md", n_islands=cfg.islands, island=island + 1, budget=self.ctx.budget.budget
        )
        user = prompts.render(
            "v2/proposer_user.md",
            task=render_task_v2(info),
            parts=render_parts(v2),
            placements=render_placements(v2, info),
            parameters=space.describe(),
            rules=render_rules_v2(v2),
            notes=render_notes(info),
            distance_rule=render_points_rule_v2(cfg.min_distance, space),
            n_tested=len(history),
            tested=render_tested_v2(history, island, claims, space),
            n_elites=min(cfg.elites_shown, len(ranked)),
            n_pool=len(ranked),
            n_episodes=info.episodes_per_design,
            n_parts=len(v2.parts),
            n_placements=len(v2.placements),
            n_seeds=info.seeds_per_condition,
            elites=f"{elites}\n\n{pairs}",
            failures=render_failures_v2(ranked[: cfg.elites_shown], info, cfg.failures_shown),
            rejections=render_rejections_v2(last_round, cfg.rejections_shown, self._scrub),
            target=render_target(ranked[0].score.success, cfg.target_factor),
            round=round_index,
            last_round=self.ctx.budget.n_rounds - 1,
            operators=render_operators_v2(
                untried,
                cfg.min_distance,
                cfg.explore_min_distance,
                space,
                claims=bool(claims),
                reserves=self.n_reserves,
            ),
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- hashes


def prompt_spec_hashes_v2(info: EvaluatorInfo | None = None) -> dict[str, str]:
    """Hashes of everything that shapes a v2 prompt besides the run's own data (for PREREG).

    With ``info``, also the rendered engineering-notes block, task and parts table (they carry
    the task's numbers); the templates, tools, parameter table and points rule are always in.
    """
    from rtt.search.zeroshot import ZEROSHOT_SLOTS

    def h(obj: object) -> str:
        return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()

    space = _space()
    out = {
        **{f"template:{k}": v for k, v in prompts.template_hashes(prompts.TEMPLATE_NAMES_V2).items()},
        "tool:proposer": h(
            submit_designs_tool(
                PROPOSER_SLOTS,
                explain=NemotronProposerV2.explain_changes,
                reserves=ProposerConfig().reserves,
                space=space,
            )
        ),
        "tool:zeroshot": h(submit_designs_tool(ZEROSHOT_SLOTS, space=space)),
        "parameter_table": h(space.describe()),
        "points_rule": h(render_points_rule_v2(MIN_DISTANCE_V2, space)),
        "distance_rule": h(render_distance_rule_v2(MIN_DISTANCE_V2, space)),
        "explore_min_distance": h(EXPLORE_MIN_DISTANCE_V2),
        "guard": h(FORBIDDEN_PROMPT_TEXT_V2.pattern),
    }
    if info is not None and info.v2 is not None:
        out["engineering_notes"] = h(render_notes(info))
        out["task"] = h(render_task_v2(info))
        out["parts"] = h(render_parts(info.v2))
        out["rules"] = h(render_rules_v2(info.v2))
    return out


def json_safe(value: Any) -> Any:
    """Re-export of :func:`rtt.search.nemotron._json_safe` for the v2 fakes."""
    return _json_safe(value)
