"""The site turns the simulator's failure modes, caveats and refusals into plain words
(``web/lib/labels.ts``): it matches each caveat by its fixed wording and keeps one label per
failure mode. These tests pin both sides, so a reworded caveat or a new failure mode fails here
until the site learns it (``web/tests/labels.test.ts`` holds the same sentences).
"""

from __future__ import annotations

import re
from pathlib import Path

from rtt.studio.primitives.push import LID_STOP_DPS_MAX, lid_fall_caveat
from rtt.studio.types import STEP_FAILURE_MODES

REPO = Path(__file__).resolve().parents[1]
LABELS_TS = REPO / "web" / "lib" / "labels.ts"
PRIMITIVES = REPO / "src" / "rtt" / "studio" / "primitives"


def _ts_list(name: str) -> list[str]:
    src = LABELS_TS.read_text(encoding="utf-8")
    block = re.search(rf"export const {name} = \[(.*?)\] as const;", src, re.DOTALL)
    assert block, f"{name} not found in {LABELS_TS}"
    return re.findall(r'"([^"]+)"', block.group(1))


def test_the_site_lists_every_failure_mode_with_a_label() -> None:
    assert _ts_list("STEP_FAILURE_MODES") == list(STEP_FAILURE_MODES)
    src = LABELS_TS.read_text(encoding="utf-8")
    labels = re.search(r"STEP_FAILURE_LABELS: Record<StepFailureMode, string> = \{(.*?)\};", src, re.DOTALL)
    assert labels
    keys = re.findall(r"^\s*(\w+):", labels.group(1), re.MULTILINE)
    assert keys == list(STEP_FAILURE_MODES)


def test_the_lid_fall_caveat_reads_as_the_site_expects() -> None:
    assert lid_fall_caveat(21.3, 317.4) == (
        "The lid fell shut from 21 deg and met its stop at about 320 deg/s: the tool let go of it above its stop"
    )
    assert lid_fall_caveat(7.5, 90.0).startswith(
        "The lid fell shut from 7.5 deg and met its stop at about 90 deg/s"
    )
    src = LABELS_TS.read_text(encoding="utf-8")
    assert f"export const LID_STOP_DPS_MAX = {LID_STOP_DPS_MAX:g};" in src


def test_the_refusals_keep_the_wording_the_site_matches() -> None:
    """The fixed parts of each caveat the site's patterns read (labels.ts CAVEAT_PATTERNS)."""
    sources = {
        "_pick.py": (
            'f"The part stopped {high:.0f} mm above {where}',
            '"it was not let go, since releasing it would drop it"',
        ),
        "push.py": (
            'f"{who} {_newtons(cap)} N; this button needs at least {_newtons(need)} N to actuate, so it "',
            "f\"not pressed: the limit allows {_newtons(stiff['press_cap_n'])} N, the button actuates at \"",
            '"The step\'s force limit allows"',
            '"Without a force limit on the step, the push allows"',
        ),
        "_common.py": ("f\"Out of reach: {v.get('reach_what', 'its target')} would take the palm to x \"",),
    }
    for name, parts in sources.items():
        text = (PRIMITIVES / name).read_text(encoding="utf-8")
        for part in parts:
            assert part in text, f"{name} no longer says {part}: update web/lib/labels.ts"
    common = (PRIMITIVES / "_common.py").read_text(encoding="utf-8")
    assert (
        "f\"{v['reach_x_mm']:.0f} mm, y {v['reach_y_mm']:.0f} mm, past its +-{v['palm_limit_mm']:.0f} mm \""
        in common
    )
