#!/usr/bin/env python
"""Render the look sign-off set (gate GV-F): hero stills and a short clip in the studio look.

Every image is rendered from a real replay (never re-simulated): v1 episodes are run with
recording on, lifted to replay v2 (open-source gripper CAD, each jaw's own asset), written to
disk, read back and rendered. The stock entrant is drawn with the stock blade's CAD over its
v1 box approximation (the boxes remain the collision overlay); designed v1 entrants are drawn
as their fused boxes in orange PLA.

Usage:
    python scripts/video/look_demo.py --out OUT_DIR [--docs docs/look]

``OUT_DIR`` receives the replays and assets; the PNGs and the MP4 go to ``--docs`` (default
``docs/look``). A JSON summary (outcomes, pose residuals, payload sizes, timings) is printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from rtt.design.generate import generate_finger
from rtt.design.schema import load_entrants
from rtt.pipeline.export_web import task_summary
from rtt.pipeline.replays import lift_to_v2
from rtt.render import materials as mat
from rtt.render.glb import AssetStore
from rtt.render.playback import pose_residual, render_stills, render_video
from rtt.render.studio import View, build_render_scene
from rtt.sim import objects as objs
from rtt.sim.episode import run_episode
from rtt.sim.replay import code_version, read_replay, write_replay
from rtt.sim.task import default_task

REPO = Path(__file__).resolve().parents[2]

# (name, entrant, object, placement, controller setting, seed, object look)
EPISODES = (
    (
        "stock_slug",
        "stock",
        objs.ObjectSpec("cylinder", (0.020, 0.050), density=7850.0),
        0.0,
        1,
        0,
        "steel_ground",
    ),
    (
        "stock_block",
        "stock",
        objs.ObjectSpec("box", (0.040, 0.030, 0.060), density=2700.0),
        0.0,
        1,
        1,
        "aluminium_anodised",
    ),
    (
        "concave_peg",
        "concave",
        objs.ObjectSpec("rod", (0.007, 0.140), density=2700.0),
        0.0,
        0,
        2,
        "steel_zinc",
    ),
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--docs", type=Path, default=REPO / "docs" / "look")
    ap.add_argument("--run-id", default="look-demo")
    args = ap.parse_args(argv)
    entrants = {e.name: e for e in load_entrants(REPO / "configs" / "entrants.yaml")}
    store = AssetStore(args.out / "assets")
    code = code_version(REPO)
    summary: dict[str, object] = {"code_version": code, "episodes": {}}
    replays = {}
    for name, entrant, obj, err_x, setting, seed, look in EPISODES:
        design = entrants[entrant]
        t0 = time.perf_counter()
        res = run_episode(generate_finger(design), obj, objs.Placement(err_x=err_x), setting, seed)
        v1 = res.replay
        v1.header["meta"]["design_id"] = design.design_id()
        v2 = lift_to_v2(
            v1,
            store,
            run_id=args.run_id,
            code_version=code,
            stock_blade=entrant == "stock",
            object_look=mat.PART_LOOKS[look],
            part_id=name.split("_", 1)[1],
            episode=name,
        )
        write_replay(v2, args.out / "replays" / name)
        replays[name] = read_replay(args.out / "replays" / name)
        scene = build_render_scene(replays[name].header, store)
        summary["episodes"][name] = {  # type: ignore[index]
            "entrant": entrant,
            "object": obj.label,
            "success": res.success,
            "failure_mode": res.failure_mode,
            "frames": v2.n_frames,
            "sim_s": round(time.perf_counter() - t0, 2),
            "pose_residual": pose_residual(scene, replays[name].frames),
        }
    store.write_manifest()

    # views are relative to the gripper: playback turns them (and the lights) with the palm
    wide, xray = "wide", "xray"
    phases = {p["name"]: p for p in task_summary(default_task())["phases"]}
    grasp = round(phases["lift"]["start_s"] * 30) - 2  # jaws closed and squeezing, part on the mat
    close = View((0.0, 0.0, 0.030), 0.24, 215.0, -14.0, 30.0)
    t0 = time.perf_counter()
    stills = {}
    slug, block, peg = replays["stock_slug"], replays["stock_block"], replays["concave_peg"]
    stills.update(
        render_stills(
            slug,
            store,
            {
                "hero_wide_stock_slug": {
                    "frame": slug.n_frames - 8,
                    "view": wide,
                    "target": "rest",
                    "follow": 0.7,
                }
            },
            args.docs,
        )
    )
    stills.update(
        render_stills(
            block, store, {"hero_xray_stock_block": {"frame": grasp, "view": xray, "xray": True}}, args.docs
        )
    )
    stills.update(
        render_stills(peg, store, {"hero_close_concave_peg": {"frame": grasp, "view": close}}, args.docs)
    )
    still_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    clip = render_video(
        slug, store, args.docs / "grasp_wide_stock_slug.mp4", view=wide, target="rest", follow=0.7, quality=6
    )
    clip_s = time.perf_counter() - t0
    files = [*stills.values(), clip]
    summary.update(
        stills={k: str(v.relative_to(REPO)) if v.is_relative_to(REPO) else str(v) for k, v in stills.items()},
        clip=str(clip.relative_to(REPO)) if clip.is_relative_to(REPO) else str(clip),
        still_render_s=round(still_s, 1),
        clip_render_s=round(clip_s, 1),
        docs_bytes=sum(f.stat().st_size for f in files),
        assets={e["label"]: e["bytes"] for e in store.entries.values()},
    )
    for name, rp in replays.items():
        stem = args.out / "replays" / name
        payload = stem.with_suffix(".json").stat().st_size + stem.with_suffix(".bin").stat().st_size
        payload += sum(store.entries[a]["bytes"] for a in rp.header["assets"])
        summary["episodes"][name]["payload_bytes"] = payload  # type: ignore[index]
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
