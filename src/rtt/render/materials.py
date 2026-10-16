"""The studio palette: one material description for MuJoCo, glTF and Blender.

Colours are authored as display (sRGB) values in [0, 1], the way a designer picks them.
Each consumer converts once:

* glTF ``baseColorFactor`` is linear, so :meth:`Look.gltf` converts sRGB to linear;
* MuJoCo's classic renderer lights display values directly, so :meth:`Look.mujoco` keeps
  sRGB and maps metallic/roughness onto its specular/shininess terms (it also fills the
  ``metallic``/``roughness`` fields newer MuJoCo renderers read).

Part looks come from :class:`rtt.parts.types.Material` (:func:`look_from_material`); the
presets in :data:`PART_LOOKS` cover v1 primitive objects and the hero fixtures. Looks never
touch physics: friction is task-level and mass comes from the part's own density.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from rtt.parts.types import Material

__all__ = [
    "CARRIAGE",
    "FINGER_BASELINE",
    "FINGER_DESIGNED",
    "FRAME",
    "LOOKS",
    "PART_LOOKS",
    "STOCK_BLADE",
    "XRAY_ALPHA",
    "Look",
    "linear_to_srgb",
    "look_from_material",
    "srgb_to_linear",
]

XRAY_ALPHA: Final = 0.35  # part opacity in the x-ray view (spec section 10)


def srgb_to_linear(c: float) -> float:
    """IEC 61966-2-1 transfer function, display value -> linear light."""
    c = float(c)
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    c = float(c)
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1.0 / 2.4) - 0.055


@dataclass(frozen=True)
class Look:
    """A PBR-ish surface: sRGB base colour, opacity, metallic and roughness factors."""

    name: str
    base_color: tuple[float, float, float]
    alpha: float = 1.0
    metallic: float = 0.0
    roughness: float = 0.5

    def __post_init__(self) -> None:
        if len(self.base_color) != 3 or not all(0.0 <= c <= 1.0 for c in self.base_color):
            raise ValueError(f"{self.name}: base_color must be 3 sRGB values in [0, 1]")
        for field_name in ("alpha", "metallic", "roughness"):
            if not 0.0 <= getattr(self, field_name) <= 1.0:
                raise ValueError(f"{self.name}: {field_name} must be in [0, 1]")
        object.__setattr__(self, "base_color", tuple(float(c) for c in self.base_color))

    @property
    def rgba(self) -> tuple[float, float, float, float]:
        """Display-space RGBA (what MuJoCo's ``rgba`` takes)."""
        return (*self.base_color, self.alpha)

    def gltf(self) -> dict[str, Any]:
        """A glTF 2.0 ``pbrMetallicRoughness`` block (linear base colour)."""
        lin = [round(srgb_to_linear(c), 6) for c in self.base_color]
        return {
            "baseColorFactor": [*lin, round(self.alpha, 6)],
            "metallicFactor": round(self.metallic, 6),
            "roughnessFactor": round(self.roughness, 6),
        }

    def mujoco(self) -> dict[str, Any]:
        """Keyword arguments for ``MjSpec.add_material`` (classic renderer terms)."""
        gloss = 1.0 - self.roughness
        return {
            "rgba": list(self.rgba),
            "specular": round(0.12 + 0.22 * gloss + 0.18 * self.metallic * gloss, 4),
            "shininess": round(0.05 + 0.6 * gloss * gloss, 4),
            "reflectance": 0.0,  # MuJoCo reflects only on planes; meshes stay matte
            "metallic": self.metallic,
            "roughness": self.roughness,
        }

    def with_alpha(self, alpha: float) -> Look:
        return replace(self, alpha=alpha)


# --------------------------------------------------------------------------- gripper and fingers

FRAME: Final = Look("frame", (0.115, 0.118, 0.125), roughness=0.62)  # PET-CF, matte black
CARRIAGE: Final = Look("carriage", (0.16, 0.165, 0.175), roughness=0.62)
STOCK_BLADE: Final = Look("stock_blade", (0.16, 0.165, 0.175), roughness=0.62)
FINGER_DESIGNED: Final = Look("finger_designed", (0.96, 0.50, 0.12), roughness=0.5)  # orange PLA
FINGER_BASELINE: Final = Look("finger_baseline", (0.55, 0.58, 0.62), roughness=0.45)  # imprint, grey PLA
COLLISION: Final = Look("collision", (0.20, 0.80, 0.30), alpha=0.45, roughness=0.9)  # overlay toggle

# --------------------------------------------------------------------------- parts (render fallbacks)

PART_LOOKS: Final = {
    "pvc_white": Look("pvc_white", (0.93, 0.93, 0.91), roughness=0.42),
    "aluminium_anodised": Look("aluminium_anodised", (0.70, 0.72, 0.76), metallic=0.65, roughness=0.38),
    "steel_zinc": Look("steel_zinc", (0.76, 0.78, 0.81), metallic=0.85, roughness=0.32),
    "brass": Look("brass", (0.80, 0.64, 0.33), metallic=1.0, roughness=0.3),
    "steel_stamped": Look("steel_stamped", (0.52, 0.53, 0.56), metallic=0.8, roughness=0.5),
    "steel_ground": Look("steel_ground", (0.64, 0.65, 0.67), metallic=0.95, roughness=0.26),
    "plastic_blue": Look("plastic_blue", (0.18, 0.47, 0.86), roughness=0.45),  # the v1 object colour
    "clay": Look("clay", (0.78, 0.76, 0.73), roughness=0.8),  # untextured scans in the gallery
}

LOOKS: Final = {
    look.name: look
    for look in (
        FRAME,
        CARRIAGE,
        STOCK_BLADE,
        FINGER_DESIGNED,
        FINGER_BASELINE,
        COLLISION,
        *PART_LOOKS.values(),
    )
}


def look_from_material(material: Material) -> Look:
    """The render look of a part's :class:`~rtt.parts.types.Material`."""
    return Look(material.name, material.base_color, metallic=material.metallic, roughness=material.roughness)
