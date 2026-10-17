"""The catalog table: which vendored models exist, how each is mounted and how each grasps.

Each :class:`CatalogModel` names a vendored directory (a generic key) and says how its MJCF
becomes an effector on the shared wrist:

* ``root``: the upstream body whose subtree is the effector (the arm links above it are not
  vendored); ``drop_joints``: upstream wrist or positioning joints removed with their
  actuators, because the shared 4-DoF wrist replaces them;
* ``down`` and ``forward``: two axes of the root body frame. The mount turns the model so
  ``down`` points at the table (a hand's palm normal, a gripper's approach axis: palm-down) and
  ``forward`` along the wrist's +y, the parts' closing axis. The model is then raised until its
  highest collision point in the open pose sits on the mount plane;
* ``synergies``: the grasp primitives, each an open and a closed control vector over the
  upstream actuators (glob patterns on upstream actuator names; unlisted actuators keep
  ``rest``) and one grip-height fraction: one value per model and primitive, chosen on the c0
  condition of all six training parts together, never per part. A hand synergy may also list
  ``via`` waypoints (the controls then follow open -> waypoints -> closed, evenly spaced on the
  close ramp: :class:`rtt.effectors.catalog.family.PathSynergy`).

How the hand synergies were made (scratch tooling, not shipped): a kinematic solve on the
mounted model puts the finger tips and the thumb tip at one height on either side of the
grasp centre, with the distal pads near vertical, at a few openings down to contact; the
closed controls extrapolate past contact so the position servos squeeze; open width,
overdrive and grip height were then picked by the number of the six training parts held at
c0. The controls are stored below as plain numbers (radians, or metres of tendon).

Control values are clipped to each actuator's own ``ctrlrange``. Gains, force ranges and joint
limits are the upstream model's, unmodified.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal

__all__ = ["CATALOG_MODELS", "CatalogModel", "Synergy", "catalog_model"]


@dataclass(frozen=True)
class Synergy:
    """One scripted grasp of a model: open and closed controls and its grip height."""

    name: str
    open: Mapping[str, float]
    closed: Mapping[str, float]
    grip_height_frac: float
    tip_offset: float = 0.0
    rest: float = 0.0  # control of an actuator no pattern names (clipped to its range)
    centre_bodies: tuple[str, ...] = ()  # upstream bodies whose mean sets the grasp centre (all: empty)
    via: tuple[Mapping[str, float], ...] = ()  # waypoints between open and closed (evenly spaced in closure)
    note: str = ""


@dataclass(frozen=True)
class CatalogModel:
    """One vendored model as the catalog families use it (module docstring)."""

    key: str
    kind: Literal["hand", "gripper"]
    display_name: str
    root: str | None
    down: tuple[float, float, float]
    forward: tuple[float, float, float]
    synergies: tuple[Synergy, ...]
    drop_joints: tuple[str, ...] = ()
    mount_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)  # m, in the mount frame, after levelling
    min_bodies: int = 2  # effector bodies on the part for engagement (ContactCriterion)
    finger_sides: tuple[tuple[str, ...], tuple[str, ...]] | None = None  # grippers: +y and -y finger bodies
    notes: tuple[str, ...] = field(default=())

    def synergy(self, name: str) -> Synergy:
        for s in self.synergies:
            if s.name == name:
                return s
        raise KeyError(f"{self.key} has no {name!r} primitive")


# 16-joint four-finger hand. Root frame: fingers along +z, palm face +x, index at +y.
_HAND_4F = CatalogModel(
    key="hand-4f-16dof",
    kind="hand",
    display_name="Four-finger 16-joint hand",
    root="palm",
    down=(1.0, 0.0, 0.0),
    forward=(0.0, 0.0, 1.0),
    synergies=(
        Synergy(
            "power",
            open={
                "ffa1": 0.662,
                "ffa2": 0.741,
                "ffa3": 0.688,
                "mfa1": 0.669,
                "mfa2": 0.77,
                "mfa3": 0.704,
                "rfa1": 0.662,
                "rfa2": 0.741,
                "rfa3": 0.688,
                "tha0": 1.214,
                "tha1": 0.801,
                "tha2": 0.817,
                "tha3": 0.788,
            },
            closed={
                "ffa1": 1.041,
                "ffa2": 1.21,
                "ffa3": 0.922,
                "mfa1": 1.097,
                "mfa2": 1.217,
                "mfa3": 0.853,
                "rfa1": 1.095,
                "rfa2": 1.225,
                "rfa3": 0.838,
                "tha0": 1.396,
                "tha1": 0.313,
                "tha2": 1.33,
                "tha3": 0.001,
            },
            grip_height_frac=0.35,
            centre_bodies=("ff_tip", "mf_tip", "rf_tip", "th_tip", "th_tip", "th_tip"),
            note="three fingers against the thumb",
        ),
        Synergy(
            "pinch",
            open={
                "ffa1": 1.3057,
                "ffa2": 0.1725,
                "ffa3": 0.0943,
                "tha0": 1.006,
                "tha1": 0.8206,
                "tha2": 0.7945,
                "tha3": -0.162,
            },
            closed={
                "ffa1": 2.0969,
                "ffa2": 0.5131,
                "ffa3": -0.7411,
                "tha0": 1.7738,
                "tha1": -0.9107,
                "tha2": 0.8821,
                "tha3": -0.162,
            },
            via=(
                {
                    "ffa1": 1.3403,
                    "ffa2": 0.4237,
                    "ffa3": -0.1914,
                    "tha0": 1.1671,
                    "tha1": 0.4693,
                    "tha2": 0.717,
                    "tha3": -0.162,
                },
                {
                    "ffa1": 1.4306,
                    "ffa2": 0.4469,
                    "ffa3": -0.227,
                    "tha0": 1.2619,
                    "tha1": 0.2352,
                    "tha2": 0.7571,
                    "tha3": -0.162,
                },
                {
                    "ffa1": 1.61,
                    "ffa2": 0.3035,
                    "ffa3": -0.227,
                    "tha0": 1.3013,
                    "tha1": 0.1547,
                    "tha2": 0.8282,
                    "tha3": -0.162,
                },
            ),
            grip_height_frac=0.15,
            centre_bodies=("ff_tip", "th_tip"),
            note="index tip against the thumb tip, waypoints keep the tips level",
        ),
    ),
)

# Five-finger hand with coupled distal joints (attached from the palm, wrist joint dropped).
# Root frame: fingers +z, palm face -y.
_HAND_5F_18 = CatalogModel(
    key="hand-5f-18act",
    kind="hand",
    display_name="Five-finger 18-actuator hand",
    root="rh_palm",
    drop_joints=("rh_WRJ1",),
    down=(0.0, -1.0, 0.0),
    forward=(0.0, 0.0, -1.0),
    synergies=(
        Synergy(
            "power",
            open={
                "rh_A_FFJ3": 1.5708,
                "rh_A_FFJ0": 1.9294,
                "rh_A_MFJ3": 1.5708,
                "rh_A_MFJ0": 1.9306,
                "rh_A_RFJ3": 1.5708,
                "rh_A_RFJ0": 1.9294,
                "rh_A_LFJ3": 1.5708,
                "rh_A_LFJ0": 1.9272,
                "rh_A_THJ5": -0.1732,
                "rh_A_THJ4": 1.2217,
                "rh_A_THJ3": 0.2094,
                "rh_A_THJ0": 1.3962,
                "rh_A_THJ1": 1.5708,
            },
            closed={
                "rh_A_FFJ3": 1.5586,
                "rh_A_FFJ0": 2.4017,
                "rh_A_MFJ3": 1.5708,
                "rh_A_MFJ0": 2.363,
                "rh_A_RFJ3": 1.5586,
                "rh_A_RFJ0": 2.4017,
                "rh_A_LFJ3": 1.3413,
                "rh_A_LFJ0": 2.7397,
                "rh_A_THJ5": 0.5758,
                "rh_A_THJ4": 1.2217,
                "rh_A_THJ3": 0.2094,
                "rh_A_THJ0": 1.3962,
                "rh_A_THJ1": 1.5708,
            },
            via=(
                {
                    "rh_A_FFJ3": 1.5708,
                    "rh_A_FFJ0": 2.2022,
                    "rh_A_MFJ3": 1.5708,
                    "rh_A_MFJ0": 2.1952,
                    "rh_A_RFJ3": 1.5708,
                    "rh_A_RFJ0": 2.2022,
                    "rh_A_LFJ3": 1.5708,
                    "rh_A_LFJ0": 2.2192,
                    "rh_A_THJ5": 0.4676,
                    "rh_A_THJ4": 1.2217,
                    "rh_A_THJ3": 0.2094,
                    "rh_A_THJ0": 0.978,
                    "rh_A_THJ1": 1.5708,
                },
                {
                    "rh_A_FFJ3": 1.5708,
                    "rh_A_FFJ0": 2.2328,
                    "rh_A_MFJ3": 1.5708,
                    "rh_A_MFJ0": 2.2226,
                    "rh_A_RFJ3": 1.5708,
                    "rh_A_RFJ0": 2.2328,
                    "rh_A_LFJ3": 1.5708,
                    "rh_A_LFJ0": 2.2578,
                    "rh_A_THJ5": 0.3184,
                    "rh_A_THJ4": 1.2217,
                    "rh_A_THJ3": 0.2094,
                    "rh_A_THJ0": 1.3962,
                    "rh_A_THJ1": 1.5708,
                },
                {
                    "rh_A_FFJ3": 1.564,
                    "rh_A_FFJ0": 2.1918,
                    "rh_A_MFJ3": 1.5708,
                    "rh_A_MFJ0": 2.1708,
                    "rh_A_RFJ3": 1.564,
                    "rh_A_RFJ0": 2.1918,
                    "rh_A_LFJ3": 1.4433,
                    "rh_A_LFJ0": 2.3786,
                    "rh_A_THJ5": 0.2429,
                    "rh_A_THJ4": 1.2217,
                    "rh_A_THJ3": 0.2094,
                    "rh_A_THJ0": 1.3962,
                    "rh_A_THJ1": 1.5708,
                },
            ),
            grip_height_frac=0.15,
            centre_bodies=(
                "rh_ffdistal",
                "rh_mfdistal",
                "rh_rfdistal",
                "rh_lfdistal",
                "rh_thdistal",
                "rh_thdistal",
                "rh_thdistal",
                "rh_thdistal",
            ),
            note="four fingers against the thumb, waypoints keep the tips level",
        ),
        Synergy(
            "pinch",
            open={
                "rh_A_FFJ3": 1.5708,
                "rh_A_FFJ0": 1.8254,
                "rh_A_THJ5": -0.6701,
                "rh_A_THJ4": 1.2217,
                "rh_A_THJ3": 0.2094,
                "rh_A_THJ0": 1.3962,
                "rh_A_THJ1": 1.5708,
            },
            closed={
                "rh_A_FFJ3": 1.5708,
                "rh_A_FFJ0": 1.4734,
                "rh_A_THJ5": 0.1715,
                "rh_A_THJ4": 1.2217,
                "rh_A_THJ3": 0.2094,
                "rh_A_THJ0": 1.3962,
                "rh_A_THJ1": 1.5708,
            },
            via=(
                {
                    "rh_A_FFJ3": 1.5708,
                    "rh_A_FFJ0": 1.871,
                    "rh_A_THJ5": -0.1535,
                    "rh_A_THJ4": 0.9914,
                    "rh_A_THJ3": 0.2094,
                    "rh_A_THJ0": 1.3962,
                    "rh_A_THJ1": 1.5708,
                },
                {
                    "rh_A_FFJ3": 1.5708,
                    "rh_A_FFJ0": 1.7824,
                    "rh_A_THJ5": -0.2452,
                    "rh_A_THJ4": 1.1043,
                    "rh_A_THJ3": 0.2094,
                    "rh_A_THJ0": 1.3962,
                    "rh_A_THJ1": 1.5708,
                },
                {
                    "rh_A_FFJ3": 1.5708,
                    "rh_A_FFJ0": 1.69,
                    "rh_A_THJ5": -0.3464,
                    "rh_A_THJ4": 1.2217,
                    "rh_A_THJ3": 0.2094,
                    "rh_A_THJ0": 1.3962,
                    "rh_A_THJ1": 1.5708,
                },
            ),
            grip_height_frac=0.35,
            centre_bodies=("rh_ffdistal", "rh_thdistal"),
            note="index tip against the thumb tip, waypoints keep the tips level",
        ),
    ),
)

# 22-joint five-finger hand. Root frame: fingers +z, palm face +x, thumb at +y.
_HAND_5F_22 = CatalogModel(
    key="hand-5f-22dof",
    kind="hand",
    display_name="Five-finger 22-joint hand",
    root="right_hand_C_MC",
    down=(1.0, 0.0, 0.0),
    forward=(0.0, 0.0, 1.0),
    synergies=(
        Synergy(
            "power",
            open={
                "right_index_MCP_FE": 1.6347,
                "right_index_PIP": 1.4753,
                "right_index_DIP": 0.0512,
                "right_middle_MCP_FE": 1.5638,
                "right_middle_PIP": 1.2765,
                "right_middle_DIP": 0.2286,
                "right_ring_MCP_FE": 1.7832,
                "right_ring_PIP": 1.4416,
                "right_ring_DIP": 0.0514,
                "right_pinky_MCP_FE": 1.4863,
                "right_pinky_PIP": 1.3635,
                "right_pinky_DIP": 0.2448,
                "right_thumb_CMC_FE": 1.2842,
                "right_thumb_CMC_AA": -0.1715,
                "right_thumb_MCP_FE": 1.2493,
                "right_thumb_MCP_AA": -0.35,
                "right_thumb_IP": 1.9031,
            },
            closed={
                "right_index_MCP_FE": 1.8385,
                "right_index_PIP": 1.3279,
                "right_index_DIP": 0.1819,
                "right_middle_MCP_FE": 1.9505,
                "right_middle_PIP": 1.5572,
                "right_middle_DIP": -0.2657,
                "right_ring_MCP_FE": 1.6132,
                "right_ring_PIP": 1.7484,
                "right_ring_DIP": -0.1113,
                "right_pinky_MCP_FE": 1.708,
                "right_pinky_PIP": 1.0888,
                "right_pinky_DIP": 0.3012,
                "right_thumb_CMC_FE": 1.2465,
                "right_thumb_CMC_AA": -0.2278,
                "right_thumb_MCP_FE": 1.5327,
                "right_thumb_MCP_AA": -0.7663,
                "right_thumb_IP": 1.5304,
            },
            grip_height_frac=0.283,
            centre_bodies=(
                "right_index_DP",
                "right_middle_DP",
                "right_ring_DP",
                "right_pinky_DP",
                "right_thumb_DP",
                "right_thumb_DP",
                "right_thumb_DP",
                "right_thumb_DP",
            ),
            note="four fingers against the thumb",
        ),
        Synergy(
            "pinch",
            open={
                "right_index_MCP_FE": 1.57,
                "right_index_PIP": 1.455,
                "right_index_DIP": 0.1257,
                "right_thumb_CMC_FE": 0.564,
                "right_thumb_CMC_AA": -0.2004,
                "right_thumb_MCP_FE": 1.3711,
                "right_thumb_MCP_AA": -0.5847,
                "right_thumb_IP": 1.7586,
            },
            closed={
                "right_index_MCP_FE": 1.5943,
                "right_index_PIP": 1.1354,
                "right_index_DIP": -0.0291,
                "right_thumb_CMC_FE": 0.8531,
                "right_thumb_CMC_AA": -0.1372,
                "right_thumb_MCP_FE": 1.5671,
                "right_thumb_MCP_AA": -0.1656,
                "right_thumb_IP": 1.9131,
            },
            grip_height_frac=0.1,
            centre_bodies=("right_index_DP", "right_thumb_DP"),
            note="index tip against the thumb tip",
        ),
    ),
)

# Fin-ray parallel gripper (a hand-held capture gripper: its six positioning joints are dropped).
_GRIPPER_FINRAY = CatalogModel(
    key="gripper-2f-finray",
    kind="gripper",
    display_name="Two-finger parallel gripper, fin-ray pads",
    root=None,
    drop_joints=tuple(f"gripper_joint_{a}" for a in ("x", "y", "z", "rx", "ry", "rz")),
    down=(0.0, 0.0, 1.0),
    forward=(1.0, 0.0, 0.0),
    synergies=(
        Synergy(
            "pinch", open={"fingers_actuator": 0.0}, closed={"fingers_actuator": 0.05}, grip_height_frac=0.5
        ),
    ),
    finger_sides=(("right_finger_holder",), ("left_finger_holder",)),
)

# Rigid-pad grippers: the last arm link and both fingers; the arm's wrist roll joint is dropped.
_GRIPPER_RIGID_A = CatalogModel(
    key="gripper-2f-rigid-a",
    kind="gripper",
    display_name="Two-finger parallel gripper, rigid pads A",
    root="link_6",
    drop_joints=("joint6",),
    down=(0.0, 0.0, 1.0),
    forward=(1.0, 0.0, 0.0),
    synergies=(
        Synergy(
            "pinch",
            open={"gripper": 0.041},
            closed={"gripper": 0.0},
            grip_height_frac=0.3,
            centre_bodies=("lf_down", "rf_down"),
        ),
    ),
    finger_sides=(("link_right_finger", "rf_rot", "rf_down"), ("link_left_finger", "lf_rot", "lf_down")),
)
_GRIPPER_RIGID_B = CatalogModel(
    key="gripper-2f-rigid-b",
    kind="gripper",
    display_name="Two-finger parallel gripper, rigid pads B",
    root="link6",
    drop_joints=("joint6",),
    down=(0.0, 0.0, 1.0),
    forward=(0.0, 1.0, 0.0),
    synergies=(
        Synergy(
            "pinch",
            open={"gripper": 0.035},
            closed={"gripper": 0.0},
            grip_height_frac=0.3,
            centre_bodies=("link7", "link8"),
        ),
    ),
    finger_sides=(("link7",), ("link8",)),
)

CATALOG_MODELS: Final[Mapping[str, CatalogModel]] = MappingProxyType(
    {
        m.key: m
        for m in (_HAND_4F, _HAND_5F_18, _HAND_5F_22, _GRIPPER_FINRAY, _GRIPPER_RIGID_A, _GRIPPER_RIGID_B)
    }
)


def catalog_model(key: str) -> CatalogModel:
    try:
        return CATALOG_MODELS[key]
    except KeyError:
        raise KeyError(f"no catalog model {key!r}; known: {sorted(CATALOG_MODELS)}") from None
