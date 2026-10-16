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

``OUT_DIR`` becomes a replay v2 fixture the web app can play: ``<stem>.json``, ``<stem>.bin``,
``<stem>.ovl.bin`` and ``assets/<id>.glb``. For example::

    python scripts/v2_bout.py --out OUT_DIR --stills OUT_DIR/stills
    node web/scripts/smoke_v2.mjs --fixture OUT_DIR --stem designed_bushing_c1 --shots OUT_DIR/web

``--spec FILE`` replaces the default episodes: a JSON list of objects with ``stem``,
``design`` (an entrant name, or a recipe as a JSON object), ``part``, ``placement`` (a
``train_v2`` label) and optionally ``still`` (``{"label", "view", "frame": "grasp" | "hold",
"xray"}``). A JSON summary (outcomes, registration, pose residuals, payload sizes) is printed.
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


def _trimesh(mesh: Any) -> Any:
    import trimesh

    return trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)


def _design(spec: Any, entrants: dict[str, Any]) -> tuple[str, Any]:
    from rtt.design.freeform import FingerRecipe

    if isinstance(spec, str):
        return spec, entrants[spec]
    recipe = FingerRecipe.model_validate(spec)
    return recipe.name or "recipe", recipe


def run(out: Path, episodes: Sequence[dict[str, Any]], run_id: str) -> dict[str, Any]:
    from rtt.design.freeform import build_pair, load_entrants_v2
    from rtt.design.freeform.generate import build_pair_full
    from rtt.parts.suite import train_suite
    from rtt.pipeline.replays import lift_freeform_to_v2
    from rtt.render.glb import AssetStore
    from rtt.render.playback import pose_residual
    from rtt.render.studio import build_render_scene
    from rtt.sim.episode import run_episode_v2
    from rtt.sim.replay import code_version, read_replay, write_replay
    from rtt.sim.task import default_task_v2, task_digest

    task = default_task_v2()
    entrants = {e.name: e for e in load_entrants_v2()}
    suite = train_suite()
    conds = {(c.part.name, c.placement_label): c for c in suite}
    store = AssetStore(out / "assets")
    code = code_version(REPO)
    summary: dict[str, Any] = {"code_version": code, "task_digest": task_digest(task), "episodes": {}}
    for ep in episodes:
        stem, part_name, label = ep["stem"], ep["part"], ep["placement"]
        design_name, design = _design(ep["design"], entrants)
        cond = conds[(part_name, label)]
        seed = int(cond.seeds[0])
        t0 = time.perf_counter()
        if getattr(design, "kind", None) == "stock":
            pair, meshes = build_pair(design, cond.part, task), None
        else:
            built = build_pair_full(design, cond.part, task)
            pair, meshes = built.pair, (_trimesh(built.left_mesh), _trimesh(built.right_mesh))
        res = run_episode_v2(pair, cond.part, cond.placement, seed, record=True, task=task)
        v2 = lift_freeform_to_v2(
            res.replay,
            store,
            pair,
            part=cond.part,
            overlays=res.overlays,
            jaw_meshes=meshes,
            run_id=run_id,
            code_version=code,
            episode=stem,
            suite=suite.name,
            condition=cond.condition_id,
            seed=seed,
        )
        write_replay(v2, out / stem)
        back = read_replay(out / stem)
        scene = build_render_scene(back.header, store)
        reg = res.registration
        payload = (out / f"{stem}.json").stat().st_size + (out / f"{stem}.bin").stat().st_size
        ovl = out / f"{stem}.ovl.bin"
        payload += ovl.stat().st_size if ovl.is_file() else 0
        payload += sum(store.entries[a]["bytes"] for a in back.header["assets"])
        summary["episodes"][stem] = {
            "design": design_name,
            "design_id": pair.design_id,
            "part": part_name,
            "condition": cond.condition_id,
            "seed": seed,
            "registered_success": bool(res.success),
            "lift_success": bool(res.lift_success),
            "failure_mode": res.failure_mode,
            "registration": None
            if reg is None
            else {
                "registered": bool(reg.registered),
                "flags": list(reg.sub_flags),
                "seat_short_mm": reg.seat_short_mm,
                "dx_mm": reg.dx_mm,
                "dy_mm": reg.dy_mm,
            },
            "q_seat_mm": round(1000.0 * pair.q_seat, 2),
            "frames": back.n_frames,
            "sim_s": round(time.perf_counter() - t0, 2),
            "pose_residual": pose_residual(scene, back.frames),
            "payload_bytes": payload,
            "assets": sorted(back.header["assets"]),
        }
    store.write_manifest()
    return summary


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
    episodes = EPISODES if args.spec is None else tuple(json.loads(args.spec.read_text(encoding="utf-8")))
    summary = run(args.out, episodes, args.run_id)
    if args.stills is not None:
        t0 = time.perf_counter()
        summary["stills"] = stills(args.out, args.stills, episodes)
        summary["still_render_s"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
