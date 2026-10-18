"""Open-library entrants of the catalog families (product mode, not pre-registered).

Each :class:`~rtt.effectors.catalog.models.CatalogModel` here mounts one item of the open
effector library (``assets/effectors/library/<key>/``, :mod:`rtt.library`) on the shared wrist,
through the same catalog families and loaders as the vendored catalog models. The library's
vendoring script already wrote each item's effector as one MJCF (hull meshes, pinned inertials,
no upstream positioning joints), so every entry here has ``root = None`` and no dropped joints.

They are **not** default presets (``in_presets = False``): the founder's decision of 2026-10-17
keeps the per-part tournament's catalog group at exactly 12 candidates. They are buildable
entrants that :func:`rtt.library.entrants.library_entrants` hands to the Studio's candidate
search, and that any tournament can be given explicitly.

Synergies are the plainest honest choice: the upstream actuator driven from its open to its
closed limit (grippers), every flexion actuator to a fixed fraction of its range (hands), no
control at all (passive tools). One grip-height fraction per model and primitive; none was
tuned on any part. Gains and force ranges are the upstream model's.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from rtt.effectors.catalog.models import CatalogModel, Synergy

__all__ = ["LIBRARY_MODELS", "UNTUNED_NOTE"]

#: Printed on every card of a library hand: its synergy was not fitted to the training parts.
UNTUNED_NOTE: Final = "Library entrant: a plain synergy, not tuned on any part"


def _gripper(
    key: str,
    display_name: str,
    *,
    down: tuple[float, float, float],
    forward: tuple[float, float, float],
    actuator: str,
    open_ctrl: float,
    closed_ctrl: float,
    sides: tuple[tuple[str, ...], tuple[str, ...]] | None,
    centre: tuple[str, ...] = (),
    grip: float = 0.3,
    note: str = "",
) -> CatalogModel:
    return CatalogModel(
        key=key,
        kind="gripper",
        display_name=display_name,
        root=None,
        down=down,
        forward=forward,
        synergies=(
            Synergy(
                "pinch",
                open={actuator: open_ctrl},
                closed={actuator: closed_ctrl},
                grip_height_frac=grip,
                centre_bodies=centre,
            ),
        ),
        finger_sides=sides,
        card_note=note,
        in_presets=False,
    )


def _hand(
    key: str,
    display_name: str,
    *,
    down: tuple[float, float, float],
    forward: tuple[float, float, float],
    power: tuple[dict[str, float], dict[str, float]],
    pinch: tuple[dict[str, float], dict[str, float]] | None = None,
    centre: tuple[str, ...] = (),
    pinch_centre: tuple[str, ...] = (),
) -> CatalogModel:
    syn = [
        Synergy(
            "power",
            open=power[0],
            closed=power[1],
            grip_height_frac=0.3,
            centre_bodies=centre,
            note="every finger flexes",
        )
    ]
    if pinch is not None:
        syn.append(
            Synergy(
                "pinch",
                open=pinch[0],
                closed=pinch[1],
                grip_height_frac=0.2,
                centre_bodies=pinch_centre,
                note="index against thumb",
            )
        )
    return CatalogModel(
        key=key,
        kind="hand",
        display_name=display_name,
        root=None,
        down=down,
        forward=forward,
        synergies=tuple(syn),
        card_note=UNTUNED_NOTE,
        in_presets=False,
    )


def _tool(key: str, display_name: str, *, actuators: tuple[str, ...] = ()) -> CatalogModel:
    """A rigid proxy written in the mount frame (tool axis -z, top on the mount plane)."""
    still = {a: 0.0 for a in actuators}
    return CatalogModel(
        key=key,
        kind="tool",
        display_name=display_name,
        root=None,
        down=(0.0, 0.0, -1.0),
        forward=(0.0, 1.0, 0.0),
        synergies=(
            Synergy(
                "hold", open=still, closed=still, grip_height_frac=0.5, note="a passive tool: nothing closes"
            ),
        ),
        card_note="A tool, not a grasp: it cannot hold a part on its own",
        in_presets=False,
    )


_ORCA_T = "right_t-"
_MODELS: tuple[CatalogModel, ...] = (
    _gripper(
        "gripper-2f-force-48",
        "Force-controlled parallel gripper, 48 mm stroke",
        down=(0.0, 0.0, -1.0),
        forward=(0.0, 1.0, 0.0),
        actuator="actuator_jaw",
        open_ctrl=0.0,
        closed_ctrl=-0.0235,
        sides=(("jaw1",), ("jaw2",)),
    ),
    _gripper(
        "gripper-2f-adaptive-85",
        "Two-finger adaptive linkage gripper, 85 mm",
        down=(0.0, 0.0, 1.0),
        forward=(1.0, 0.0, 0.0),
        actuator="fingers_actuator",
        open_ctrl=0.0,
        closed_ctrl=255.0,
        sides=(("left_pad", "left_follower"), ("right_pad", "right_follower")),
        centre=("left_pad", "right_pad"),
    ),
    _gripper(
        "gripper-2f-cobot-hand-80",
        "Two-finger parallel cobot hand, 80 mm",
        down=(0.0, 0.0, 1.0),
        forward=(0.0, 1.0, 0.0),
        actuator="actuator8",
        open_ctrl=255.0,
        closed_ctrl=0.0,
        sides=(("left_finger",), ("right_finger",)),
    ),
    _gripper(
        "gripper-2f-teleop",
        "Two-finger parallel gripper, teleop-arm fingers",
        down=(1.0, 0.0, 0.0),
        forward=(0.0, 1.0, 0.0),
        actuator="left/gripper",
        open_ctrl=0.037,
        closed_ctrl=0.002,
        sides=None,  # the finger hulls include carriages that span the centre: no opening check
    ),
    _gripper(
        "gripper-angular-jaw-hobby",
        "Angular-jaw gripper of an open hobby arm",
        down=(0.0, -1.0, 0.0),
        forward=(0.0, 0.0, 1.0),
        actuator="Jaw",
        open_ctrl=1.75,
        closed_ctrl=-0.174,
        sides=None,  # the jaw hulls overlap along the closing axis: no opening check
    ),
    _gripper(
        "gripper-2f-linkage-cobot",
        "Two-finger linkage gripper of a cobot arm",
        down=(0.0, 0.0, 1.0),
        forward=(0.0, 1.0, 0.0),
        actuator="fingers_actuator",
        open_ctrl=0.0,
        closed_ctrl=255.0,
        sides=(("left_finger",), ("right_finger",)),
    ),
    _gripper(
        "gripper-2f-air-mini",
        "Air-actuated mini parallel gripper",
        down=(0.0, 0.0, 1.0),
        forward=(0.0, 1.0, 0.0),
        actuator="gripper",
        open_ctrl=-10.0,
        closed_ctrl=10.0,
        sides=(("gripper_left_finger",), ("gripper_right_finger",)),
    ),
    _gripper(
        "gripper-2f-hobby-parallel-84",
        "Rack-and-pinion parallel gripper for an open hobby arm",
        down=(0.0, -1.0, 0.0),
        forward=(1.0, 0.0, 0.0),
        actuator="clamp",
        open_ctrl=0.0,
        closed_ctrl=0.037,
        sides=None,  # the clamp hulls interleave along the closing axis: no opening check
    ),
    _gripper(
        "gripper-2f-servo-110",
        "Industrial servo parallel gripper, 110 mm stroke",
        down=(0.0, 1.0, 0.0),
        forward=(1.0, 0.0, 0.0),
        actuator="fingers",
        open_ctrl=-0.055,
        closed_ctrl=0.0,
        sides=(("left_finger",), ("right_finger",)),
    ),
    _hand(
        "hand-3f-12dof-industrial",
        "Three-finger 12-joint industrial hand",
        down=(0.0, 0.0, 1.0),
        forward=(0.0, 1.0, 0.0),
        power=(
            {"F?/J1_actuator": -0.6, "F?/J2_actuator": 0.0, "F?/J3_actuator": 0.0},
            {"F?/J1_actuator": 0.6, "F?/J2_actuator": 0.9, "F?/J3_actuator": 0.6},
        ),
        centre=("F0/finger_distal", "F1/finger_distal", "F2/finger_distal"),
    ),
    _hand(
        "hand-5f-tendon-16dof",
        "Tendon-driven five-finger hand, 16 joints",
        down=(0.0, 0.0, -1.0),
        forward=(0.0, 1.0, 0.0),
        power=(
            {"*": 0.0},
            {
                "right_?-mcp_actuator": 1.2,
                "right_?-pip_actuator": 1.2,
                _ORCA_T + "abd_actuator": 0.8,
                _ORCA_T + "mcp_actuator": 0.9,
                _ORCA_T + "pip_actuator": 0.9,
            },
        ),
        pinch=(
            {"*": 0.0},
            {
                "right_i-mcp_actuator": 1.0,
                "right_i-pip_actuator": 0.8,
                _ORCA_T + "abd_actuator": 0.8,
                _ORCA_T + "mcp_actuator": 0.7,
                _ORCA_T + "pip_actuator": 0.7,
            },
        ),
        centre=(
            "right_I-FingerTipAssembly_ec49c16c",
            "right_M-FingerTipAssembly_34afb748",
            "right_T-DP_b7429e50",
        ),
        pinch_centre=("right_I-FingerTipAssembly_ec49c16c", "right_T-DP_b7429e50"),
    ),
    _hand(
        "hand-4f-servo-13act",
        "Servo-driven four-finger hand, 13 actuators",
        down=(0.0, 0.0, -1.0),
        forward=(0.0, 1.0, 0.0),
        power=(
            {"*": 0.0},
            {
                "*_mcp_pitch_*": -1.2,
                "*_pip_joint_*": -1.0,
                "thumb_cmc_roll_*": 1.2,
                "thumb_mcp_*": -0.6,
                "thumb_dip_*": -0.6,
            },
        ),
        pinch=(
            {"*": 0.0},
            {
                "index_mcp_pitch_*": -1.0,
                "index_pip_*": -0.8,
                "thumb_cmc_roll_*": 1.2,
                "thumb_mcp_*": -0.5,
                "thumb_dip_*": -0.5,
            },
        ),
        centre=("index_dip_link", "middle_dip_link", "thumb_dip"),
        pinch_centre=("index_dip_link", "thumb_dip"),
    ),
    _tool("tool-hook-drawer", "Drawer-pull hook tool"),
    _tool("tool-button-pusher", "Button-pusher tool"),
    _tool("tool-marker-holder", "Spring-loaded marker holder tool"),
    _tool(
        "tool-screwdriver-servo", "Servo screwdriver attachment for a hobby arm", actuators=("spindle_speed",)
    ),
)

LIBRARY_MODELS: Final[Mapping[str, CatalogModel]] = MappingProxyType({m.key: m for m in _MODELS})
