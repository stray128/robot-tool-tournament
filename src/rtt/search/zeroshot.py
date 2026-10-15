"""Arm Z: zero-shot Nemotron, which never sees a simulation result.

Z separates the model's prior knowledge from in-context feedback. It shares
everything else with arm C (round 0, budget, forced tool call, validator repair,
diversity check, ``fallback_random``), but its prompt is built only from the task,
the parameter table, the validator rules and the *parameters* of the designs
already submitted in the run (so it can avoid repeating itself). No score,
failure mode or other simulator output is passed to the prompt builder:
:meth:`ZeroShotProposer.build_messages` reads designs only, never
:class:`~rtt.search.base.DesignScore` objects.

Each round makes one call per island, each asking for 4 designs
(``design_1`` ... ``design_4``), so a run asks for 32 designs after the shared
round 0, within the same 16-call cap.
"""

from __future__ import annotations

from typing import Final

from rtt.search import prompts
from rtt.search.nemotron import (
    LLMProposer,
    SlotSpec,
    params_json,
    render_distance_rule,
    render_parameters,
    render_rules,
    render_task,
)

__all__ = ["ZEROSHOT_SLOTS", "ZeroShotProposer"]

ZEROSHOT_SLOTS: Final[tuple[SlotSpec, ...]] = tuple(SlotSpec(f"design_{i}", "zeroshot") for i in range(1, 5))


class ZeroShotProposer(LLMProposer):
    """Arm Z: designs from the model's priors only."""

    arm = "zeroshot"
    origin = "zeroshot"
    slots = ZEROSHOT_SLOTS

    def build_messages(self, round_index: int, island: int) -> list[dict[str, str]]:
        cfg = self.config
        designs = [item.design for item in self.ctx.history]  # parameters only: no scores
        proposed = "\n".join(f"- {params_json(d)}" for d in designs) or "None yet."
        n_calls = cfg.islands * (self.ctx.budget.n_rounds - 1)
        system = prompts.render("zeroshot_system.md")
        user = prompts.render(
            "zeroshot_user.md",
            task=render_task(self.ctx.info, self.ctx.gripper_task),
            parameters=render_parameters(),
            rules=render_rules(self.ctx.gripper_task),
            proposed=proposed,
            call=(round_index - 1) * cfg.islands + island + 1,
            n_calls=n_calls,
            distance_rule=render_distance_rule(cfg.min_distance),
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
