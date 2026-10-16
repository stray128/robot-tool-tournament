"""The x-ray look: chosen bodies drawn translucent so the jaws show through the part.

Only material and geom colours of the render model change (never the scored model). Use it
as a context manager around rendering::

    with xray(scene, ("object",)):
        frame = renderer.render()
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import numpy as np

from rtt.render.materials import XRAY_ALPHA
from rtt.render.studio import RenderScene

__all__ = ["XRAY_ALPHA", "set_alpha", "xray"]


def _targets(scene: RenderScene, bodies: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    empty = np.zeros(0, dtype=int)
    mats = np.unique(np.concatenate([empty, *(scene.body_materials.get(b, empty) for b in bodies)]))
    geoms = np.concatenate([empty, *(scene.body_geoms.get(b, empty) for b in bodies)])
    return mats[mats >= 0].astype(int), geoms.astype(int)


def set_alpha(scene: RenderScene, bodies: Sequence[str], alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Set the opacity of ``bodies``' visual geoms; returns their previous (material, geom) alphas."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    m = scene.model
    mats, geoms = _targets(scene, bodies)
    old = (m.mat_rgba[mats, 3].copy(), m.geom_rgba[geoms, 3].copy())
    m.mat_rgba[mats, 3] = alpha
    m.geom_rgba[geoms, 3] = alpha
    return old


@contextmanager
def xray(
    scene: RenderScene, bodies: Sequence[str] = ("object",), alpha: float = XRAY_ALPHA
) -> Iterator[None]:
    """Render ``bodies`` at ``alpha`` (default 0.35) inside the block, then restore them."""
    mats, geoms = _targets(scene, bodies)
    old_m, old_g = set_alpha(scene, bodies, alpha)
    try:
        yield
    finally:
        scene.model.mat_rgba[mats, 3] = old_m
        scene.model.geom_rgba[geoms, 3] = old_g
