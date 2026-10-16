"""Render from the replay: poses in, pixels out. Never re-simulates.

For every frame, each replay body's recorded world pose is written into its mocap body in
the studio render model (:func:`rtt.render.studio.build_render_scene`), ``mj_forward``
places the visuals, and MuJoCo's offscreen renderer draws the frame. Nothing is integrated,
so a render cannot change an outcome, and :func:`pose_residual` proves that the rendered
bodies sit exactly at the replay's poses (gate GV-R: within 1e-6).

Every still and clip carries a burned-in caption built from the replay's provenance
(:func:`caption_text`): "simulation replay · MuJoCo · run <id> · design <id>", plus
"product mode" for product-mode episodes and "rendered offline" for offline renderers.

MuJoCo picks its OpenGL backend when it is first imported. The public render functions run
in-process when ``MUJOCO_GL`` is ``egl`` or ``osmesa`` and otherwise in a fresh subprocess
(:func:`rtt.sim.render.call_with_gl`), like the v1 MP4 renderer.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Final

import numpy as np

from rtt.render.glb import AssetStore
from rtt.render.studio import VIEWS, RenderScene, StudioConfig, View
from rtt.sim.replay import Replay

__all__ = [
    "CAPTION_SEPARATOR",
    "apply_frame",
    "burn_caption",
    "caption_text",
    "iter_frames",
    "pose_residual",
    "render_stills",
    "render_video",
]

CAPTION_SEPARATOR: Final = " · "  # a middle dot between caption parts


# --------------------------------------------------------------------------- poses


def apply_frame(scene: RenderScene, data: Any, frame: np.ndarray) -> None:
    """Pose every replay body from one ``(n_bodies, 7)`` frame (xyzw quaternions) and run
    ``mj_forward`` on the render model."""
    import mujoco

    f = np.asarray(frame, dtype=np.float64)
    if f.shape != (len(scene.body_names), 7):
        raise ValueError(f"frame has shape {f.shape}, the scene has {len(scene.body_names)} bodies")
    data.mocap_pos[scene.mocap_ids] = f[:, :3]
    data.mocap_quat[scene.mocap_ids] = f[:, [6, 3, 4, 5]]  # xyzw -> MuJoCo wxyz
    mujoco.mj_forward(scene.model, data)


def pose_residual(scene: RenderScene, frames: np.ndarray) -> dict[str, float]:
    """Largest difference between the rendered body poses and the replay's, over all frames.

    Positions are compared component-wise (m); quaternions component-wise after aligning the
    sign (q and -q are the same rotation). MuJoCo normalises mocap quaternions, so the
    quaternion residual includes the float32 rounding of the stored unit quaternions.
    """
    import mujoco

    data = mujoco.MjData(scene.model)
    pos_err = quat_err = 0.0
    for frame in np.asarray(frames):
        apply_frame(scene, data, frame)
        pos = data.xpos[scene.body_ids]
        q = data.xquat[scene.body_ids][:, [1, 2, 3, 0]]  # back to xyzw
        ref = np.asarray(frame, dtype=np.float64)
        pos_err = max(pos_err, float(np.abs(pos - ref[:, :3]).max()))
        sign = np.where(np.sum(q * ref[:, 3:], axis=1, keepdims=True) < 0, -1.0, 1.0)
        quat_err = max(quat_err, float(np.abs(sign * q - ref[:, 3:]).max()))
    return {"pos": pos_err, "quat": quat_err, "frames": len(frames)}


def body_track(replay: Replay, body: str = "object") -> np.ndarray:
    """``(n_frames, 3)`` world positions of one replay body."""
    names = [b["name"] for b in replay.header["bodies"]]
    return np.asarray(replay.frames[:, names.index(body), :3], dtype=np.float64)


# --------------------------------------------------------------------------- captions


def caption_text(header: Mapping[str, Any], *, offline: bool = False) -> str:
    """The burned-in caption for a replay (v2 provenance; v1 replays say "v1")."""
    prov = header.get("provenance") or {}
    meta = header.get("meta") or {}
    parts = ["simulation replay", "MuJoCo"]
    run = prov.get("run_id") or meta.get("run_id")
    if run:
        parts.append(f"run {run}")
    design = meta.get("design_id")
    if design:
        parts.append(f"design {str(design)[:12]}")
    if prov.get("product_mode"):
        parts.append("product mode")
    if offline:
        parts.append("rendered offline")
    return CAPTION_SEPARATOR.join(parts)


def burn_caption(img: np.ndarray, text: str, *, readout: str | None = None) -> np.ndarray:
    """Draw ``text`` bottom-left (and ``readout`` top-left, larger) on an RGB frame."""
    from PIL import Image, ImageDraw, ImageFont

    h = img.shape[0]
    im = Image.fromarray(np.ascontiguousarray(img)).convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w = img.shape[1]
    # 1/42 of the height, shrunk to fit narrow panels (a glyph is about 0.52 em wide)
    small = ImageFont.load_default(size=max(8, min(round(h / 42), int(0.9 * w / (0.52 * max(1, len(text)))))))
    pad = max(4, round(h / 90))
    x0, y1 = pad * 2, h - pad * 2
    box = draw.textbbox((0, 0), text, font=small)
    tw, th = box[2] - box[0], box[3] - box[1]
    draw.rounded_rectangle(
        (x0 - pad, y1 - th - 2 * pad, x0 + tw + pad, y1), radius=pad, fill=(20, 22, 26, 150)
    )
    draw.text((x0, y1 - th - pad - box[1]), text, font=small, fill=(240, 240, 240, 255))
    if readout:
        big = ImageFont.load_default(size=max(14, round(h / 22)))
        rb = draw.textbbox((0, 0), readout, font=big)
        rw, rh = rb[2] - rb[0], rb[3] - rb[1]
        draw.rounded_rectangle(
            (x0 - pad, 2 * pad, x0 + rw + pad, 2 * pad + rh + 2 * pad), radius=pad, fill=(20, 22, 26, 160)
        )
        draw.text((x0, 3 * pad - rb[1]), readout, font=big, fill=(255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(im, layer).convert("RGB"))


# --------------------------------------------------------------------------- rendering


def _target(replay: Replay, i: int, target: str | Sequence[float], follow: float) -> np.ndarray:
    if not isinstance(target, str):
        return np.asarray(target, dtype=np.float64)
    track = body_track(replay, "object") if "object" in [b["name"] for b in replay.header["bodies"]] else None
    if target == "object" and track is not None:
        return track[i]
    if target == "rest" and track is not None:
        out = track[0].copy()
        out[2] += follow * (track[i][2] - track[0][2])  # ease the framing up with the lift
        return out
    if target in ("object", "rest"):
        return np.zeros(3)
    raise ValueError(f"unknown camera target {target!r}")


def _resolve_view(view: str | View) -> View:
    return VIEWS[view] if isinstance(view, str) else view


def _stills_impl(
    replay: Replay,
    assets: AssetStore | None,
    shots: Mapping[str, Mapping[str, Any]],
    out_dir: str,
    cfg: StudioConfig | None,
    caption: bool,
    looks: Mapping[str, Any] | None,
    align: bool = True,
) -> dict[str, str]:
    import imageio.v2 as imageio
    import mujoco

    from rtt.render.studio import align_to_palm, build_render_scene, free_camera, make_renderer, render_flags
    from rtt.render.xray import xray

    cfg = cfg or StudioConfig()
    if align:
        cfg = align_to_palm(cfg, replay.header, replay.frames)
    scene = build_render_scene(replay.header, assets, cfg, looks=looks)
    data = mujoco.MjData(scene.model)
    r, opt = make_renderer(scene)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    try:
        for label, shot in shots.items():
            i = int(shot.get("frame", replay.n_frames - 1)) % replay.n_frames
            apply_frame(scene, data, replay.frames[i])
            cam = free_camera(
                scene,
                _resolve_view(shot.get("view", "close")),
                _target(replay, i, shot.get("target", "object"), float(shot.get("follow", 0.0))),
            )
            with (
                xray(scene, tuple(shot.get("xray_bodies", ("object",))), float(shot.get("xray_alpha", 0.35)))
                if shot.get("xray")
                else nullcontext()
            ):
                r.update_scene(data, camera=cam, scene_option=opt)
                render_flags(r)
                img = r.render()
            if caption:
                img = burn_caption(img, caption_text(replay.header), readout=shot.get("readout"))
            path = out / f"{label}.png"
            imageio.imwrite(path, img)
            written[label] = str(path)
    finally:
        r.close()
    return written


def _video_impl(
    replay: Replay,
    assets: AssetStore | None,
    path: str,
    view: str | View,
    start: int,
    stop: int | None,
    target: str,
    follow: float,
    cfg: StudioConfig | None,
    caption: bool,
    quality: int,
    looks: Mapping[str, Any] | None,
    xray_on: bool,
    align: bool = True,
) -> str:
    import imageio.v2 as imageio
    import mujoco

    from rtt.render.studio import align_to_palm, build_render_scene, free_camera, make_renderer, render_flags
    from rtt.render.xray import set_alpha

    cfg = cfg or StudioConfig()
    if align:
        cfg = align_to_palm(cfg, replay.header, replay.frames)
    scene = build_render_scene(replay.header, assets, cfg, looks=looks)
    if xray_on:
        set_alpha(scene, ("object",), 0.35)
    data = mujoco.MjData(scene.model)
    r, opt = make_renderer(scene)
    text = caption_text(replay.header)
    fps = int(replay.header["fps"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    v = _resolve_view(view)
    try:
        with imageio.get_writer(
            path,
            format="FFMPEG",  # type: ignore[arg-type]
            mode="I",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=quality,
            macro_block_size=16,
        ) as writer:
            for i in range(start, replay.n_frames if stop is None else min(stop, replay.n_frames)):
                apply_frame(scene, data, replay.frames[i])
                cam = free_camera(scene, v, _target(replay, i, target, follow))
                r.update_scene(data, camera=cam, scene_option=opt)
                render_flags(r)
                img = r.render()
                writer.append_data(burn_caption(img, text) if caption else img)
    finally:
        r.close()
    return path


def iter_frames(
    replay: Replay,
    assets: AssetStore | None,
    plan: Sequence[Mapping[str, Any]],
    *,
    cfg: StudioConfig | None = None,
    caption: bool = True,
    looks: Mapping[str, Any] | None = None,
    align: bool = True,
) -> Iterator[np.ndarray]:
    """Render a frame plan in this process (``MUJOCO_GL`` must be ``egl`` or ``osmesa``).

    Each plan entry is a shot dict as for :func:`render_stills` (``frame``, ``view``,
    ``target``, ``follow``, ``xray``, ``readout``). Used by the video tools to compose
    sequences (turntables, triptychs) without passing frames between processes.
    """
    import os

    import mujoco

    from rtt.render.studio import align_to_palm, build_render_scene, free_camera, make_renderer, render_flags
    from rtt.render.xray import xray
    from rtt.sim.render import BACKENDS

    if os.environ.get("MUJOCO_GL", "").lower() not in BACKENDS:
        raise RuntimeError(
            "iter_frames renders in-process: set MUJOCO_GL=egl (or osmesa) before importing mujoco"
        )
    cfg = cfg or StudioConfig()
    if align:
        cfg = align_to_palm(cfg, replay.header, replay.frames)
    scene = build_render_scene(replay.header, assets, cfg, looks=looks)
    data = mujoco.MjData(scene.model)
    r, opt = make_renderer(scene)
    text = caption_text(replay.header)
    try:
        for shot in plan:
            i = int(shot.get("frame", 0)) % replay.n_frames
            apply_frame(scene, data, replay.frames[i])
            view = _resolve_view(shot.get("view", "close"))
            cam = free_camera(
                scene, view, _target(replay, i, shot.get("target", "object"), float(shot.get("follow", 0.0)))
            )
            ctx = xray(scene) if shot.get("xray") else nullcontext()
            with ctx:
                r.update_scene(data, camera=cam, scene_option=opt)
                render_flags(r)
                img = r.render()
            yield burn_caption(img, text, readout=shot.get("readout")) if caption else img
    finally:
        r.close()


def render_stills(
    replay: Replay,
    assets: AssetStore | None,
    shots: Mapping[str, Mapping[str, Any]],
    out_dir: str | Path,
    *,
    cfg: StudioConfig | None = None,
    caption: bool = True,
    looks: Mapping[str, Any] | None = None,
    align: bool = True,
) -> dict[str, Path]:
    """Render named stills; each shot is ``{"frame": i, "view": name, "target": "object" |
    "rest" | xyz, "follow": f, "xray": bool, "readout": str}``. Returns label -> PNG path.

    With ``align`` (default) the light rig and the views turn with the replay's palm yaw."""
    from rtt.sim.render import call_with_gl

    written = call_with_gl(
        "rtt.render.playback:_stills_impl",
        replay,
        assets,
        dict(shots),
        str(out_dir),
        cfg,
        caption,
        looks,
        align,
    )
    return {k: Path(v) for k, v in written.items()}


def render_video(
    replay: Replay,
    assets: AssetStore | None,
    path: str | Path,
    *,
    view: str | View = "wide",
    start: int = 0,
    stop: int | None = None,
    target: str = "rest",
    follow: float = 0.7,
    cfg: StudioConfig | None = None,
    caption: bool = True,
    quality: int = 7,
    looks: Mapping[str, Any] | None = None,
    xray: bool = False,
    align: bool = True,
) -> Path:
    """Render frames ``start:stop`` of a replay to an H.264 MP4 with the caption burned in."""
    from rtt.sim.render import call_with_gl

    return Path(
        call_with_gl(
            "rtt.render.playback:_video_impl",
            replay,
            assets,
            str(path),
            view,
            start,
            stop,
            target,
            follow,
            cfg,
            caption,
            quality,
            looks,
            xray,
            align,
        )
    )
