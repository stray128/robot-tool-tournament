"""Materials of the procedural parts: density (sets the mass) and the PBR look.

Densities are handbook values for the named material [assumed: not weighed]. Colours are
sRGB in [0, 1]; viewers convert to linear before lighting. A material never changes
friction: scored physics uses the task's friction for every part.
"""

from __future__ import annotations

from types import MappingProxyType

from rtt.parts.types import Material

PVC_WHITE = Material("pvc_white", 1400.0, (0.93, 0.93, 0.90), metallic=0.0, roughness=0.45)
ANODISED_ALUMINIUM = Material("anodised_aluminium", 2700.0, (0.62, 0.64, 0.68), metallic=0.85, roughness=0.42)
ZINC_PLATED_STEEL = Material("zinc_plated_steel", 7850.0, (0.78, 0.79, 0.80), metallic=1.0, roughness=0.30)
BRASS = Material("brass", 8500.0, (0.84, 0.69, 0.38), metallic=1.0, roughness=0.28)
STAMPED_STEEL = Material("stamped_steel", 7850.0, (0.55, 0.56, 0.58), metallic=0.9, roughness=0.50)
GROUND_STEEL = Material("ground_steel", 7850.0, (0.70, 0.71, 0.72), metallic=1.0, roughness=0.22)
NYLON_BLACK = Material("nylon_black", 1150.0, (0.10, 0.10, 0.11), metallic=0.0, roughness=0.55)
#: Scanned objects: the task's default object density (configs/task_v1.yaml), in clay grey.
SCAN_CLAY = Material("scan_clay", 1200.0, (0.74, 0.72, 0.69), metallic=0.0, roughness=0.75)

MATERIALS: MappingProxyType[str, Material] = MappingProxyType(
    {
        m.name: m
        for m in (
            PVC_WHITE,
            ANODISED_ALUMINIUM,
            ZINC_PLATED_STEEL,
            BRASS,
            STAMPED_STEEL,
            GROUND_STEEL,
            NYLON_BLACK,
            SCAN_CLAY,
        )
    }
)


def material(name: str) -> Material:
    """The named material; raises ``KeyError`` listing the known names."""
    try:
        return MATERIALS[name]
    except KeyError:
        raise KeyError(f"unknown material {name!r}; known: {sorted(MATERIALS)}") from None
