"""Arm C exploration aids: the all-designs table in points, island claims, explore novelty and reflection detail."""

from __future__ import annotations

import re
from typing import Any

from rtt.design.schema import FingerDesign
from rtt.search.base import Proposal, RunContext
from rtt.search.diversity import distance_points, nearest
from rtt.search.fakes import FakeEvaluator
from rtt.search.nemotron import (
    NemotronProposer,
    ProposerConfig,
    min_points,
    points_row,
    prompt_spec_hashes,
    render_operators,
    render_pairs,
    submit_designs_tool,
)
from test_search_nemotron import (
    PROPOSER_SLOTS,
    ReplyTransport,
    client_for,
    diverse_designs,
    round0_context,
    submission,
)


def with_(design: FingerDesign, **changes: Any) -> FingerDesign:
    return FingerDesign.model_validate({**design.geometry_params(), **changes})


def strategy_after_round1(ctx: RunContext, items: list[Any]) -> NemotronProposer:
    """A strategy told round 0 and a round 1 whose designs alternate between the two islands."""
    strategy = NemotronProposer(ctx, client_for(ReplyTransport()))
    strategy.tell(items)
    designs = diverse_designs(ctx, 8, seed=31)
    ev = FakeEvaluator(0)
    ctx.budget.charge(1, 8)
    new = [
        ctx.history.add(1, Proposal(d, "nemotron", spec.operator, i // 4, spec.name, ("d01",)), s)
        for i, (d, s, spec) in enumerate(zip(designs, ev.evaluate(designs), PROPOSER_SLOTS * 2, strict=True))
    ]
    strategy.tell(new)
    return strategy


def test_points_helpers() -> None:
    assert min_points(0.08) == 80 and min_points(0.2) == 200
    _, items = round0_context()
    straight = items[0].design
    assert points_row(straight) == "len 50 thick 50 width 50 bend 25 bend_at 43 depth 0 tip_size 0 taper 100"


def test_table_lists_every_tested_design_of_the_run_by_cell() -> None:
    ctx, items = round0_context()
    strategy = strategy_after_round1(ctx, items)
    user = strategy.build_messages(2, 0)[1]["content"]
    table = user.split("# All 16 designs tested in this run, by cell")[1].split("# Your team's best designs")[
        0
    ]
    rows = re.findall(r"^  (d\d\d) (\d+)% \| (.+?) \| (.+)$", table, re.MULTILINE)
    assert sorted(label for label, *_ in rows) == [f"d{i:02d}" for i in range(1, 17)]  # both islands
    by_label = {label: (int(score), pts, note) for label, score, pts, note in rows}
    for item in ctx.history:
        score, pts, note = by_label[item.label]
        assert score == round(100 * item.score.success) and pts == points_row(item.design)
        if item.round == 1:
            team = "your team" if item.proposal.island == 0 else "team 2"
            assert note == f"{item.proposal.operator} of d01, round 1, {team}"
    # Each cell heading counts its rows; cells run from the lowest best score to the highest.
    heads = re.findall(r"^([a-z_]+ \+ [a-z]+) \((\d+) tested, best (\d+)%\):$", table, re.MULTILINE)
    assert sum(int(n) for _, n, _ in heads) == 16
    assert [int(b) for *_, b in heads] == sorted(int(b) for *_, b in heads)
    used = {name for name, *_ in heads}
    untried = table.split("Cells with no tested design yet: ")[1].strip().rstrip(".")
    names = set() if untried.startswith("none") else set(untried.split(", "))
    assert used | names == {
        f"{f} + {t}"
        for f in ("flat", "concave_arc", "v_groove")
        for t in ("blunt", "rounded", "chamfer", "hook")
    }
    assert not used & names
    assert "Put it in a cell that no tested design uses" in user or "Every cell already has" in user


def test_second_island_sees_the_designs_the_first_island_claimed() -> None:
    ctx, items = round0_context()
    designs = diverse_designs(ctx, 8)
    transport = ReplyTransport(submission(designs[:4]), submission(designs[4:]))
    strategy = NemotronProposer(ctx, client_for(transport))
    strategy.tell(items)
    strategy.ask(1, 8)
    first, second = transport.user_text(0), transport.user_text(1)
    assert "t1." not in first and "other teams already submitted" not in first
    for spec, design in zip(PROPOSER_SLOTS, designs[:4], strict=True):
        assert f"  t1.{spec.name} new | {points_row(design)} | {spec.operator}, this round, team 1" in second
    assert 'rows "t<team>.<slot>"' in second and "other teams already submitted this round" in second
    assert "design team 2 of 2" in transport.payloads[1]["messages"][0]["content"]


def test_copy_of_another_islands_new_design_is_rejected_and_explained() -> None:
    ctx, items = round0_context()
    designs = diverse_designs(ctx, 9)
    # Island 1's mutate_1 repeats island 0's mutate_1.
    copy = submission([designs[0], designs[5], designs[6], designs[7]])
    transport = ReplyTransport(submission(designs[:4]), copy, submission([designs[8], *designs[5:8]]))
    strategy = NemotronProposer(ctx, client_for(transport))
    strategy.tell(items)
    proposals = strategy.ask(1, 8)
    assert [(r.reason, r.island, r.slot) for r in ctx.rejections] == [("too_similar", 1, "mutate_1")]
    assert "another team's new mutate_1 this round" in ctx.rejections[0].errors[0]
    reflection = transport.user_text(2)
    assert "- mutate_1 (parent d01): 0 points from another team's new mutate_1 this round" in reflection
    assert proposals[4].via == "reflection" and proposals[4].design.design_id() == designs[8].design_id()
    assert ctx.budget.fallbacks == 0


def crowded_explore(ctx: RunContext) -> FingerDesign:
    """A valid flat + blunt design 90 points from the straight entrant d01: passes 0.08, fails the explore rule."""
    straight = ctx.history[0].design
    design = with_(straight, finger_length_mm=44.0, finger_width_mm=11.0)  # -45 and -45 points
    refs = ctx.history.designs()
    assert distance_points(design, straight) == 90
    assert nearest(design, refs).distance >= 0.08
    return design


def test_explore_in_a_crowded_cell_is_rejected_as_not_novel_and_reflected() -> None:
    ctx, items = round0_context()
    designs = diverse_designs(ctx, 9)
    crowded = crowded_explore(ctx)
    first = submission([*designs[:3], crowded])
    first["explore"]["same_cell_distances"] = "d01: 300 points"
    transport = ReplyTransport(first, submission([*designs[:3], designs[8]]), submission(designs[4:8]))
    strategy = NemotronProposer(ctx, client_for(transport))
    strategy.tell(items)
    proposals = strategy.ask(1, 8)
    assert [(r.reason, r.slot) for r in ctx.rejections] == [("not_novel", "explore")]
    assert ctx.rejections[0].errors == (
        (
            "explore in the tried cell flat + blunt: distance 0.090 to d01 (needs an untried cell or >= 0.2 "
            "within the cell)"
        ),
        "model's same_cell_distances: d01: 300 points",
    )
    reflection = transport.user_text(1)
    assert (
        "or is an explore in a cell that is already tested and less than 200 points from a design there"
        in reflection
    )
    assert "- explore: its cell flat + blunt already holds d01" in reflection
    assert (
        "the nearest, d01, is 90 points away, and an explore needs an untried cell or 200 points"
        in reflection
    )
    assert 'You wrote same_cell_distances "d01: 300 points".' in reflection
    assert f"    yours: flat + blunt | {points_row(crowded)}" in reflection
    assert "differs by: len 45, width 45" in reflection
    assert proposals[3].via == "reflection" and ctx.budget.fallbacks == 0


def test_explore_rule_can_be_turned_off() -> None:
    ctx, items = round0_context()
    designs = diverse_designs(ctx, 8)
    transport = ReplyTransport(submission([*designs[:3], crowded_explore(ctx)]), submission(designs[4:8]))
    strategy = NemotronProposer(ctx, client_for(transport), ProposerConfig(explore_min_distance=0.0))
    strategy.tell(items)
    proposals = strategy.ask(1, 8)
    assert ctx.rejections == [] and {p.origin for p in proposals} == {"nemotron"}
    assert "Prefer a cell that no tested design uses" in transport.user_text(0)


def test_mutations_are_not_held_to_the_explore_rule() -> None:
    ctx, items = round0_context()
    designs = diverse_designs(ctx, 8)
    transport = ReplyTransport(submission([crowded_explore(ctx), *designs[1:4]]), submission(designs[4:8]))
    strategy = NemotronProposer(ctx, client_for(transport))
    strategy.tell(items)
    strategy.ask(1, 8)
    assert ctx.rejections == []


def test_render_pairs_and_operators() -> None:
    ctx, _ = round0_context()
    ranked = ctx.history.ranked()
    text = render_pairs(ranked, 6, 160)
    pairs = re.findall(r"(d\d\d) \+ (d\d\d) \((\d+)\)", text)
    assert pairs and all(int(d) >= 160 for *_, d in pairs)
    assert render_pairs(ranked, 6, 10_000).startswith("Crossover candidates: no two of your top 6 designs")
    none_left = render_operators([], 0.08, 0.2)
    assert "Every cell already has a tested design, so it must be at least 200 points" in none_left
    assert "at least 160 points apart" in none_left and "do not average numbers" in none_left
    some = render_operators(["v_groove + blunt"], 0.08, 0.2, claims=True)
    assert "(v_groove + blunt)" in some and "other teams already submitted this round" in some


def test_config_and_prompt_hashes_record_the_new_knobs() -> None:
    cfg = ProposerConfig().to_json()
    assert cfg["explore_min_distance"] == 0.2 and cfg["neighbours_shown"] == 2 and cfg["crossover_pool"] == 6
    hashes = prompt_spec_hashes()
    assert "points_rule" in hashes
    assert hashes["tool:proposer"] != hashes["tool:zeroshot"]
    assert submit_designs_tool(PROPOSER_SLOTS) != submit_designs_tool(PROPOSER_SLOTS, explain=True)


def reserve_args(design: FingerDesign, operator: str, parents: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "operator": operator,
        "parents": list(parents),
        "hypothesis": f"{operator} reserve",
        **design.geometry_params(),
    }


def test_reserves_fill_rejected_slots_in_order_before_any_follow_up_call() -> None:
    ctx, items = round0_context()
    designs = diverse_designs(ctx, 11)
    first = submission(designs[:4])
    first["mutate_2"]["inner_face"], first["mutate_2"]["inner_face_depth_mm"] = "flat", 3.0  # invalid
    first["explore"] = {**first["explore"], **items[1].design.geometry_params()}  # copy of d02
    bad = reserve_args(designs[4], "mutate", ("d01",))
    bad["inner_face"], bad["inner_face_depth_mm"] = "flat", 2.0
    first["reserve_1"] = bad  # invalid
    first["reserve_2"] = reserve_args(items[0].design, "mutate", ("d01",))  # copy of d01
    first["reserve_3"] = reserve_args(designs[5], "crossover", ("d01", "d02"))
    first["reserve_4"] = reserve_args(designs[10], "explore")
    transport = ReplyTransport(first, submission(designs[6:10]))
    strategy = NemotronProposer(ctx, client_for(transport))
    strategy.tell(items)
    proposals = strategy.ask(1, 8)
    # No repair or reflection: the reserves covered both holes.
    assert [(c.kind, c.island) for c in ctx.llm_calls] == [("main", 0), ("main", 1)]
    assert [(r.reason, r.slot) for r in ctx.rejections] == [
        ("invalid", "mutate_2"),
        ("too_similar", "explore"),
        ("invalid", "reserve_1"),
        ("too_similar", "reserve_2"),
    ]
    assert [(p.slot, p.operator, p.via) for p in proposals[:4]] == [
        ("mutate_1", "mutate", "main"),
        ("reserve_3", "crossover", "main"),
        ("crossover", "crossover", "main"),
        ("reserve_4", "explore", "main"),
    ]
    assert proposals[1].parents == ("d01", "d02") and proposals[1].hypothesis == "crossover reserve"
    assert ctx.budget.fallbacks == 0 and {p.origin for p in proposals} == {"nemotron"}
    # Island 1 sees the reserves that island 0 used as claims.
    assert "t1.reserve_3 new |" in transport.user_text(1)


def test_unused_reserves_are_dropped_and_leftover_holes_still_get_a_repair() -> None:
    ctx, items = round0_context()
    designs = diverse_designs(ctx, 11)
    first = submission(designs[:4])
    for name in ("mutate_1", "mutate_2"):
        first[name]["inner_face"], first[name]["inner_face_depth_mm"] = "flat", 3.0
    first["reserve_1"] = reserve_args(designs[4], "mutate", ("d03",))
    repair = submission([designs[0], designs[8], designs[2], designs[3]])
    repair["reserve_1"] = reserve_args(designs[9], "explore")  # never needed: the repair fixed mutate_2
    transport = ReplyTransport(first, repair, submission([*designs[5:8], designs[10]]))
    strategy = NemotronProposer(ctx, client_for(transport))
    strategy.tell(items)
    proposals = strategy.ask(1, 8)
    assert [c.kind for c in ctx.llm_calls] == ["main", "repair", "main"]
    text = transport.user_text(1)
    assert (
        "- mutate_2: inner_face='flat' requires inner_face_depth_mm=0" in text and "- mutate_1:" not in text
    )
    assert "Fill reserve_1 to reserve_4 with fresh designs" in text
    assert [(p.slot, p.via) for p in proposals[:2]] == [("reserve_1", "main"), ("mutate_2", "repair")]
    assert [r.slot for r in ctx.rejections] == ["mutate_1", "mutate_2"]


def test_reserves_are_off_for_zero_shot_and_configurable() -> None:
    ctx, _ = round0_context(arm="zeroshot")
    from rtt.search.zeroshot import ZeroShotProposer

    zs = ZeroShotProposer(ctx, client_for(ReplyTransport()), ProposerConfig(reserves=4))
    assert zs.n_reserves == 0 and "reserve_1" not in zs.tool["function"]["parameters"]["properties"]
    ctx, _ = round0_context()
    c = NemotronProposer(ctx, client_for(ReplyTransport()), ProposerConfig(reserves=2))
    assert list(c.tool["function"]["parameters"]["properties"])[-2:] == ["reserve_1", "reserve_2"]


def test_reserves_copied_into_a_follow_up_call_are_not_tried_or_logged_again() -> None:
    ctx, items = round0_context()
    designs = diverse_designs(ctx, 9)
    first = submission(designs[:4])
    first["mutate_1"]["inner_face"], first["mutate_1"]["inner_face_depth_mm"] = "flat", 3.0
    copy_of_d01 = reserve_args(items[0].design, "mutate", ("d01",))
    first["reserve_1"] = copy_of_d01
    repair = submission([designs[8], *designs[1:4]])
    repair["reserve_1"] = dict(copy_of_d01)  # the model copies its earlier reserve unchanged
    transport = ReplyTransport(first, repair, submission(designs[4:8]))
    strategy = NemotronProposer(ctx, client_for(transport))
    strategy.tell(items)
    proposals = strategy.ask(1, 8)
    assert [c.kind for c in ctx.llm_calls] == ["main", "repair", "main"]
    assert [(r.reason, r.slot, r.stage) for r in ctx.rejections] == [
        ("invalid", "mutate_1", "main"),
        ("too_similar", "reserve_1", "main"),
    ]
    assert proposals[0].via == "repair" and ctx.budget.fallbacks == 0


def test_leftover_repair_reserves_fill_near_copy_holes_before_a_reflection_call() -> None:
    """A repair call's unused reserves go to every hole first, so no reflection call is spent."""
    ctx, items = round0_context()
    designs = diverse_designs(ctx, 10)
    first = submission(designs[:4])
    first["mutate_1"]["inner_face"], first["mutate_1"]["inner_face_depth_mm"] = "flat", 3.0  # invalid
    first["mutate_2"] = {**first["mutate_2"], **items[1].design.geometry_params()}  # copy of d02
    repair = submission([designs[8], *designs[1:4]])
    repair["reserve_1"] = reserve_args(designs[9], "mutate", ("d02",))
    transport = ReplyTransport(first, repair, submission(designs[4:8]))
    strategy = NemotronProposer(ctx, client_for(transport))
    strategy.tell(items)
    proposals = strategy.ask(1, 8)
    assert [c.kind for c in ctx.llm_calls] == ["main", "repair", "main"]
    assert [(p.slot, p.via) for p in proposals[:2]] == [("mutate_1", "repair"), ("reserve_1", "repair")]
    assert proposals[1].design.design_id() == designs[9].design_id()
    assert [(r.reason, r.slot) for r in ctx.rejections] == [
        ("invalid", "mutate_1"),
        ("too_similar", "mutate_2"),
    ]
    assert ctx.budget.fallbacks == 0
