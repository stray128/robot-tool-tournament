#!/usr/bin/env python
"""Run a few v2 episodes end to end and write them as web-ready replay v2 files and stills.

The whole v2 path, with no stand-in anywhere:

    recipe or entrant (configs/entrants_v2.yaml)
      -> rtt.design.freeform.build_pair_full (exact convex jaw pieces, print meshes)
      -> rtt.sim.episode.run_episode_v2 (the scored scene, registered-grasp score, overlays)
      -> rtt.pipeline.replays.lift_freeform_to_v2 (replay v2: gripper CAD, each jaw its own GLB)
      -> rtt.sim.replay.write_replay / read_replay (schema-checked header, pose .bin, .ovl.bin)
      -> rtt.render.playback.render_stills (MuJoCo studio, from the replay, never re-simulated)

Every episode is a ``train_v2`` condition with its own jitter seed (the first of the two the
suite gives it), so each clip is an episode the scored suite runs: nothing is staged.

``OUT_DIR`` becomes a v2 bout directory: ``<stem>.json``, ``<stem>.bin``, ``<stem>.ovl.bin``,
``assets/<id>.glb`` and ``bout.json`` (each lane's label and scored outcome). Publish it on the
site as a bout of v2 lanes (GLBs meshopt-compressed) and check it in a browser::

    python scripts/v2_bout.py --out OUT_DIR --stills OUT_DIR/stills
    rtt export-web OUT_DIR --replay-version 2
    (cd web && pnpm build && node scripts/smoke_v2.mjs --v2-bout v2-bout)

``--spec FILE`` replaces the default episodes: a JSON list of objects with ``stem``,
``design`` (an entrant name, or a recipe as a JSON object), ``part``, ``placement`` (a
``train_v2`` label) and optionally ``label`` (the lane's title) and ``still`` (``{"label", "view", "frame": "grasp" | "hold",
"xray"}``), or an object ``{"title", "summary", "episodes": [...]}`` that also names the bout.
A JSON summary (outcomes, registration, pose residuals, payload sizes) is printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[1]

#: The three shots of the look sign-off (GV-F), all at train_v2 placement c1 (the part 2 mm
#: off across the jaw): the drafted wrap centring the flanged bushing (the one part that
#: qualified in the hero-screen rehearsal of 2026-10-16), the stock blade on the same
#: bushing, and the configurator imprint on the block, the clearest jam case. ``design`` is
#: an entrant name from configs/entrants_v2.yaml or a recipe (a JSON object of FingerRecipe
#: fields).
EPISODES: tuple[dict[str, Any], ...] = (
    {
        "stem": "designed_bushing_c1",
        "design": "ref_wrap",
        "part": "flanged_bushing",
        "placement": "c1",
        "still": {"label": "hero_close_designed_bushing", "view": "close", "frame": "hold"},
    },
    {
        "stem": "imprint_block_c1",
        "design": "imprint_configurator",
        "part": "block",
        "placement": "c1",
        "still": {"label": "hero_xray_imprint_block", "view": "xray", "frame": "grasp", "xray": True},
    },
    {
        "stem": "stock_bushing_c1",
        "design": "stock",
        "part": "flanged_bushing",
        "placement": "c1",
        "still": {"label": "hero_wide_stock_bushing", "view": "wide", "frame": "hold"},
    },
)


TITLE = "Look check: three v2 jaws with the part 2 mm off centre"
SUMMARY = (
    "Three recorded episodes of the v2 training suite (train_v2) at placement c1, the part 2 mm off centre "
    "across the jaw, each the condition's first jitter seed: the drafted wrap on the flanged bushing, the stock "
    "blade on the same bushing and the configurator imprint on the block. They were picked to show the look "
    "(gate GV-F); single episodes, not a score. Replayed from the recordings, never re-simulated."
)


def run(
    out: Path,
    episodes: Sequence[dict[str, Any]],
    run_id: str,
    *,
    title: str | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    """Record the episodes as a v2 bout (``rtt.pipeline.replays.record_bout_v2``: replays, GLB
    store and ``bout.json``) and check each one renders at its recorded poses."""
    from rtt.pipeline.replays import record_bout_v2
    from rtt.render.glb import AssetStore
    from rtt.render.playback import pose_residual
    from rtt.render.studio import build_render_scene
    from rtt.sim.replay import read_replay

    doc = record_bout_v2(out, episodes, run_id=run_id, title=title, summary=summary, repo=REPO)
    store = AssetStore(out / "assets")
    report: dict[str, Any] = {
        "code_version": doc["code_version"],
        "task_digest": doc["task"]["digest"],
        "episodes": {},
    }
    for lane in doc["lanes"]:
        stem = lane["stem"]
        back = read_replay(out / stem)
        scene = build_render_scene(back.header, store)
        payload = sum(
            (out / name).stat().st_size
            for name in (f"{stem}.json", f"{stem}.bin", f"{stem}.ovl.bin")
            if (out / name).is_file()
        )
        payload += sum(store.entries[a]["bytes"] for a in back.header["assets"])
        report["episodes"][stem] = {
            "label": lane["label"],
            "design": lane["design"],
            "design_id": lane["design_id"],
            "part": lane["part_id"],
            "condition": lane["condition_id"],
            "seed": lane["seed"],
            "registered_success": lane["success"],
            "lift_success": lane["lift_success"],
            "failure_mode": lane["failure_mode"],
            "registration": None
            if lane["registration"] is None
            else {
                "registered": lane["registered"],
                "flags": lane["registration_flags"],
                **lane["registration"],
            },
            "q_seat_mm": lane["q_seat_mm"],
            "frames": back.n_frames,
            "sim_s": lane["wall_time_s"],
            "pose_residual": pose_residual(scene, back.frames),
            "payload_bytes": payload,  # raw GLBs; the site serves them meshopt-compressed
            "assets": sorted(back.header["assets"]),
        }
    return report


def stills(out: Path, still_dir: Path, episodes: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Each episode's ``still``: at the end of closing (``grasp``) or lifted and held
    (``hold``), frames from the task's phase times."""
    from rtt.render.glb import AssetStore
    from rtt.render.playback import render_stills
    from rtt.render.studio import View
    from rtt.sim.replay import read_replay
    from rtt.sim.task import default_task_v2

    c = default_task_v2().controller
    fps = 30
    t_lift = c.settle_time + c.descend_time + c.close_time + c.squeeze_time
    frames = {
        "grasp": round(t_lift * fps) - 2,  # jaws closed and squeezing, part still on the mat
        "hold": round((t_lift + c.lift_time) * fps) + 27,  # lifted and held
    }
    # a little above the jaws, so the frame, both carriages and the floor shadow are in shot
    views: dict[str, Any] = {"close": View((0.0, 0.0, 0.040), 0.32, 228.0, -14.0, 30.0)}
    store = AssetStore(out / "assets")
    written: dict[str, str] = {}
    for ep in episodes:
        still = ep.get("still")
        if not still:
            continue
        rp = read_replay(out / ep["stem"])
        shot: dict[str, Any] = {
            "view": views.get(still["view"], still["view"]),
            "frame": min(frames[still.get("frame", "hold")], rp.n_frames - 1),
        }
        if still["view"] == "wide":
            shot.update(target="rest", follow=0.7)
        if still.get("xray"):
            shot["xray"] = True
        written.update(
            {k: str(v) for k, v in render_stills(rp, store, {still["label"]: shot}, still_dir).items()}
        )
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="replay v2 fixture directory")
    ap.add_argument("--stills", type=Path, default=None, help="render the GV-F stills into this directory")
    ap.add_argument("--run-id", default="v2-bout")
    ap.add_argument(
        "--spec", type=Path, default=None, help="JSON list of episodes (see the module docstring)"
    )
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.spec is None:
        episodes, title, text = EPISODES, TITLE, SUMMARY
    else:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        if isinstance(spec, dict):
            episodes, title, text = tuple(spec["episodes"]), spec.get("title"), spec.get("summary")
        else:
            episodes, title, text = tuple(spec), None, None
    summary = run(args.out, episodes, args.run_id, title=title, summary=text)
    if args.stills is not None:
        t0 = time.perf_counter()
        summary["stills"] = stills(args.out, args.stills, episodes)
        summary["still_render_s"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
