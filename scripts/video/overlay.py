#!/usr/bin/env python
"""Burn captions and readouts into clips, and compose the 10-second hook from shots.yaml.

Everything is rendered from recorded replays (``rtt.render.playback``); nothing is staged.
Every frame keeps the replay's caption ("simulation replay · MuJoCo · run … · design …").

Usage:
    MUJOCO_GL=egl python scripts/video/overlay.py burn STEM IN.mp4 OUT.mp4 \\
        [--start N] [--channel offset_mm --format "{value:.2f} mm"] [--offline]
    MUJOCO_GL=egl python scripts/video/overlay.py hook scripts/video/shots.yaml \\
        --replays REPLAYS.json --out hook.mp4

``burn`` adds the caption (and an optional readout from the replay's ``.ovl.bin``) to a clip
whose frame k shows replay frame ``start + k`` (for example a Blender render; ``--offline``
adds "rendered offline"). ``hook`` renders the shot list; ``REPLAYS.json`` maps each replay
key in it to ``{"stem": ..., "assets": ...}``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from rtt.render.glb import AssetStore
from rtt.render.playback import burn_caption, caption_text, iter_frames
from rtt.render.studio import StudioConfig, View
from rtt.sim.replay import Replay, read_replay

TINTS = {"red": (214, 69, 65), "amber": (232, 160, 32), "green": (60, 170, 90)}
CARD_RGB = (204, 204, 207)


def readout_text(replay: Replay, channel: str, fmt: str, i: int, start: int = 0) -> str | None:
    """The formatted readout of ``channel`` at replay frame ``i`` (``None`` without overlays)."""
    ovl = replay.header.get("overlays")
    if replay.overlays is None or not ovl or channel not in ovl["channels"]:
        return None
    col = ovl["channels"].index(channel)
    i = min(max(i, 0), replay.n_frames - 1)
    return fmt.format(value=float(replay.overlays[i, col]), start=float(replay.overlays[start, col]))


def frame_indices(replay: Replay, span: Any, n: int) -> np.ndarray:
    """``n`` replay frame indices covering ``span`` ("all" or [start_s, stop_s])."""
    fps = float(replay.header["fps"])
    if span in (None, "all"):
        a, b = 0, replay.n_frames - 1
    else:
        a, b = round(float(span[0]) * fps), round(float(span[1]) * fps)
    return np.clip(np.round(np.linspace(a, b, n)).astype(int), 0, replay.n_frames - 1)


def _view(spec: Any) -> str | View:
    if isinstance(spec, str):
        return spec
    return View(tuple(spec["offset"]), spec["distance"], spec["azimuth"], spec["elevation"], spec["fovy"])


def _turn(view: str | View, deg: float) -> View:
    from rtt.render.studio import VIEWS

    v = VIEWS[view] if isinstance(view, str) else view
    return View(v.offset, v.distance, v.azimuth + deg, v.elevation, v.fovy)


def _label(img: np.ndarray, text: str | None, tint: str | None) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    if tint:
        w, h = im.size
        draw.rectangle((0, 0, w - 1, h - 1), outline=TINTS[tint], width=max(3, h // 120))
    if text:
        font = ImageFont.load_default(size=max(14, im.size[1] // 24))
        draw.text(
            (im.size[0] // 2, im.size[1] // 16),
            text,
            font=font,
            fill=TINTS.get(tint or "", (30, 30, 30)),
            anchor="mt",
        )
    return np.asarray(im)


def _card(text: str, size: Sequence[int]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    im = Image.new("RGB", tuple(size), CARD_RGB)
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=max(18, size[1] // 20))
    draw.text((size[0] // 2, size[1] // 2), text, font=font, fill=(25, 27, 30), anchor="mm")
    return np.asarray(im)


class Library:
    """Replays and their asset stores by key (``REPLAYS.json``)."""

    def __init__(self, mapping: Mapping[str, Mapping[str, str]], base: Path) -> None:
        self._map, self._base, self._cache = mapping, base, {}

    def get(self, key: str) -> tuple[Replay, AssetStore]:
        if key not in self._cache:
            if key not in self._map:
                raise KeyError(f"replay key {key!r} is not in the replays file")
            entry = self._map[key]
            self._cache[key] = (
                read_replay(self._base / entry["stem"]),
                AssetStore(self._base / entry["assets"]),
            )
        return self._cache[key]


def render_shot(shot: Mapping[str, Any], lib: Library, fps: int, size: Sequence[int]) -> list[np.ndarray]:
    """All output frames of one shot."""
    n = max(1, round(float(shot["seconds"]) * fps))
    w, h = int(size[0]), int(size[1])
    layout = shot.get("layout", "single")
    if layout == "card":
        card = _card(shot["text"], (w, h))
        return [card] * n
    if layout == "single":
        replay, store = lib.get(shot["replay"])
        idx = frame_indices(replay, shot.get("frames"), n)
        ro = shot.get("readout")
        plan = [
            {
                "frame": int(i),
                "view": _view(shot.get("view", "close")),
                "xray": bool(shot.get("xray")),
                "readout": readout_text(replay, ro["channel"], ro["format"], int(i), int(idx[0]))
                if ro
                else None,
            }
            for i in idx
        ]
        return list(iter_frames(replay, store, plan, cfg=StudioConfig(width=w, height=h)))
    if layout == "triptych":
        columns = []
        pw = (w // 3) // 16 * 16
        for panel in shot["panels"]:
            replay, store = lib.get(panel["replay"])
            idx = frame_indices(replay, shot.get("frames"), n)
            ro = panel.get("readout")
            plan = [
                {
                    "frame": int(i),
                    "view": _view(panel.get("view", shot.get("view", "close"))),
                    "readout": readout_text(replay, ro["channel"], ro["format"], int(i), int(idx[0]))
                    if ro
                    else None,
                }
                for i in idx
            ]
            frames = iter_frames(replay, store, plan, cfg=StudioConfig(width=pw, height=h))
            columns.append([_label(f, panel.get("label"), panel.get("tint")) for f in frames])
        out = [np.concatenate(row, axis=1) for row in zip(*columns, strict=True)]
        pad = w - out[0].shape[1]
        return [np.pad(f, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)), constant_values=200) for f in out]
    if layout == "sequence":
        panels = shot["panels"]
        per = max(1, n // len(panels))
        out = []
        for panel in panels:
            replay, store = lib.get(panel["replay"])
            at = int(panel.get("frame", int(0.7 * (replay.n_frames - 1))))
            spin = float(shot.get("turntable", 0.0))
            plan = [
                {"frame": at, "view": _turn(_view(shot.get("view", "close")), spin * k / max(1, per - 1))}
                for k in range(per)
            ]
            out += list(iter_frames(replay, store, plan, cfg=StudioConfig(width=w, height=h)))
        return out
    raise ValueError(f"unknown layout {layout!r}")


def cmd_hook(args: argparse.Namespace) -> int:
    import imageio.v2 as imageio
    import yaml

    spec = yaml.safe_load(Path(args.shots).read_text(encoding="utf-8"))
    lib = Library(json.loads(Path(args.replays).read_text(encoding="utf-8")), Path(args.replays).parent)
    fps, size = int(spec.get("fps", 30)), spec.get("size", [1280, 720])
    with imageio.get_writer(
        args.out,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=7,
        macro_block_size=16,
    ) as writer:  # type: ignore[arg-type]
        for shot in spec["shots"]:
            for frame in render_shot(shot, lib, fps, size):
                writer.append_data(frame)
    print(f"overlay: wrote {args.out}")
    return 0


def cmd_burn(args: argparse.Namespace) -> int:
    import imageio.v2 as imageio

    replay = read_replay(args.stem)
    text = caption_text(replay.header, offline=args.offline)
    reader = imageio.get_reader(args.inp)
    fps = reader.get_meta_data().get("fps", replay.header["fps"])
    with imageio.get_writer(
        args.out,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=7,
        macro_block_size=16,
    ) as writer:  # type: ignore[arg-type]
        for k, frame in enumerate(reader):
            i = args.start + k
            ro = readout_text(replay, args.channel, args.format, i, args.start) if args.channel else None
            writer.append_data(burn_caption(np.asarray(frame), text, readout=ro))
    print(f"overlay: wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("burn", help="burn the caption (and a readout) into a clip")
    b.add_argument("stem")
    b.add_argument("inp")
    b.add_argument("out")
    b.add_argument("--start", type=int, default=0)
    b.add_argument("--channel")
    b.add_argument("--format", default="{value:.2f}")
    b.add_argument("--offline", action="store_true")
    b.set_defaults(fn=cmd_burn)
    h = sub.add_parser("hook", help="compose the hook from a shot list")
    h.add_argument("shots")
    h.add_argument("--replays", required=True)
    h.add_argument("--out", required=True)
    h.set_defaults(fn=cmd_hook)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
