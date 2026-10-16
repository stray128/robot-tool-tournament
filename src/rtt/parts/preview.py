"""Clay thumbnails of parts (MuJoCo offscreen): top, front, side and an oblique view.

For visual checks, the held-out gallery (untextured, so no label can show) and docs. Uses
``MUJOCO_GL`` (``egl`` by default when unset); a machine without an offscreen GL context
raises from :func:`render_views`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

from rtt.parts.types import PartSpec

#: (azimuth, elevation) in degrees: top (looking down -z), from -y... front, side, oblique
DEFAULT_VIEWS: tuple[tuple[float, float], ...] = ((90.0, -89.0), (90.0, 0.0), (0.0, 0.0), (135.0, -30.0))


def render_views(
    part: PartSpec,
    *,
    size: int = 300,
    views: Sequence[tuple[float, float]] = DEFAULT_VIEWS,
    rgba: Sequence[float] | None = None,
    axes: bool = True,
) -> np.ndarray:
    """An ``(size, size * len(views), 3)`` uint8 strip of the part's union mesh in its
    material colour (or ``rgba``), with x/y/z axes in red/green/blue when ``axes``."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from rtt.parts.library import visual_mesh

    mesh = visual_mesh(part)
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = max(size, 64)
    spec.visual.global_.offheight = max(size, 64)
    spec.add_mesh(name="part", uservert=mesh.vertices.ravel().tolist(), userface=mesh.faces.ravel().tolist())
    color = list(rgba) if rgba is not None else [*part.material.base_color, 1.0]
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="part", rgba=color)
    extent = float(max(part.size))
    if axes:
        for d, col in (((1, 0, 0), (1, 0, 0, 1)), ((0, 1, 0), (0, 1, 0, 1)), ((0, 0, 1), (0, 0, 1, 1))):
            tip = [0.6 * extent * c for c in d]
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[0, 0, 0, *tip],
                size=[0.004 * extent],
                rgba=list(col),
                contype=0,
                conaffinity=0,
            )
    spec.worldbody.add_light(pos=[0.3, -0.3, 0.5], dir=[-0.3, 0.3, -0.5], diffuse=[0.8, 0.8, 0.8])
    spec.worldbody.add_light(pos=[-0.3, 0.3, 0.4], dir=[0.3, -0.3, -0.4], diffuse=[0.4, 0.4, 0.4])
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, size, size)
    try:
        tiles = []
        for azimuth, elevation in views:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = 0.0
            cam.distance = 2.8 * extent
            cam.azimuth = azimuth
            cam.elevation = elevation
            renderer.update_scene(data, cam)
            tiles.append(renderer.render())
    finally:
        renderer.close()
    return np.concatenate(tiles, axis=1)


def contact_sheet(parts: Sequence[PartSpec], **kwargs: object) -> np.ndarray:
    """One :func:`render_views` strip per part, stacked."""
    return np.concatenate([render_views(p, **kwargs) for p in parts], axis=0)  # type: ignore[arg-type]


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m rtt.parts.preview OUT.png``: a contact sheet of the training parts."""
    import argparse

    import imageio.v2 as imageio

    from rtt.parts.library import training_parts

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("out", help="PNG to write")
    ap.add_argument("--size", type=int, default=300)
    args = ap.parse_args(argv)
    imageio.imwrite(args.out, contact_sheet(training_parts(), size=args.size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
