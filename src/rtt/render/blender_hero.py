"""Offline Blender hero renders from a replay v2 and its GLB assets (runs inside Blender).

Usage (Blender 4.5, not part of CI)::

    blender -b -P src/rtt/render/blender_hero.py -- REPLAY_STEM ASSETS_DIR OUT_PREFIX \\
        [--frame N] [--engine eevee|cycles] [--samples 16] [--anim A:B] [--view wide|close]

``REPLAY_STEM`` is the replay path without extension (``<stem>.json`` + ``<stem>.bin``);
``ASSETS_DIR`` holds the GLBs, uncompressed ones preferred from ``ASSETS_DIR/raw/`` (Blender's
importer has no meshopt decoder). Replay poses are MuJoCo world poses (Z up, metres), which is
also Blender's world frame. The GLBs hold Z-up body-frame data while glTF importers assume Y
up, so each import sits under an Rx(-90 deg) correction inside a per-body empty that carries
the replay pose. Visual scales are positive by contract, so nothing is mirrored here.

Outputs PNGs (``OUT_PREFIX_<view>.png`` or ``OUT_PREFIX_anim_####.png``) and
``OUT_PREFIX.json`` with the caption to burn in, which says "rendered offline"
(``scripts/video/overlay.py`` burns it). The studio (cyclorama, 10 mm grid mat, soft area
key, fill and rim, AgX view transform) follows the render track's prototype.

The helpers above :func:`main` need only the standard library and numpy, so they can be
tested outside Blender; ``bpy`` is imported inside :func:`main`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

CAPTION_SEPARATOR = " · "


def load_replay(stem: str | Path) -> tuple[dict[str, Any], np.ndarray]:
    """The header and the ``(frames, bodies, 7)`` poses of a replay v2."""
    stem = Path(stem)
    header = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    if header.get("format") != "rtt-replay" or header.get("version") != 2:
        raise ValueError(f"{stem}: Blender hero renders need a replay v2 header")
    poses = np.frombuffer((stem.parent / header["bin"]).read_bytes(), dtype="<f4")
    return header, poses.reshape(header["n_frames"], header["n_bodies"], 7)


def asset_path(assets_dir: str | Path, aid: str) -> Path:
    """The GLB to import for asset ``aid``: the uncompressed copy if there is one."""
    root = Path(assets_dir)
    for p in (root / "raw" / f"{aid}.glb", root / f"{aid}.glb"):
        if p.is_file():
            return p
    raise FileNotFoundError(f"asset {aid} not found in {root}")


def blender_quat(q_xyzw: Any) -> tuple[float, float, float, float]:
    """Replay quaternion (x, y, z, w) -> Blender's (w, x, y, z)."""
    x, y, z, w = (float(c) for c in q_xyzw)
    return (w, x, y, z)


AUTHORED_PALM_YAW_DEG = 180.0  # cameras and lights below are authored for this palm yaw


def rig_turn_deg(header: dict[str, Any], poses: np.ndarray, frame: int = 0) -> float:
    """How far to turn the authored cameras and lights so they keep their place relative to
    the gripper (the same rule as rtt.render.studio.align_to_palm)."""
    names = [b["name"] for b in header["bodies"]]
    if "palm" not in names:
        return 0.0
    x, y, z, w = (float(c) for c in poses[frame, names.index("palm"), 3:])
    yaw = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return (yaw - AUTHORED_PALM_YAW_DEG + 180.0) % 360.0 - 180.0


def turn_xy(point: Any, deg: float, centre: Any = (0.0, 0.0)) -> tuple[float, float, float]:
    """``point`` turned about the vertical axis through ``centre`` (x, y) by ``deg``."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    x, y = float(point[0]) - float(centre[0]), float(point[1]) - float(centre[1])
    return (c * x - s * y + float(centre[0]), s * x + c * y + float(centre[1]), float(point[2]))


def caption(header: dict[str, Any]) -> str:
    """The burned-in caption of an offline render (same rule as rtt.render.playback)."""
    prov, meta = header.get("provenance") or {}, header.get("meta") or {}
    parts = ["simulation replay", "MuJoCo"]
    if prov.get("run_id"):
        parts.append(f"run {prov['run_id']}")
    if meta.get("design_id"):
        parts.append(f"design {str(meta['design_id'])[:12]}")
    if prov.get("product_mode"):
        parts.append("product mode")
    parts.append("rendered offline")
    return CAPTION_SEPARATOR.join(parts)


def _args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="blender_hero")
    ap.add_argument("stem")
    ap.add_argument("assets")
    ap.add_argument("out")
    ap.add_argument("--frame", type=int, default=-1)
    ap.add_argument("--engine", choices=("eevee", "cycles"), default="eevee")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--anim", default="")
    ap.add_argument("--view", choices=("wide", "close", "both"), default="both")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:  # pragma: no cover - runs inside Blender only
    import bmesh
    import bpy
    from mathutils import Euler, Quaternion, Vector

    a = _args(argv)
    header, poses = load_replay(a.stem)
    frame = a.frame % header["n_frames"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    def link(obj: Any) -> Any:
        bpy.context.collection.objects.link(obj)
        return obj

    def import_visual(vis: dict[str, Any], parent: Any) -> None:
        holder = link(bpy.data.objects.new(f"vis_{vis['asset']}", None))
        holder.parent = parent
        holder.location = Vector(vis["pos"])
        holder.rotation_mode = "QUATERNION"
        holder.rotation_quaternion = Quaternion(blender_quat(vis["quat_xyzw"]))
        holder.scale = Vector(vis["scale"])
        fix = link(bpy.data.objects.new(f"zup_{vis['asset']}", None))
        fix.rotation_euler = Euler((-math.pi / 2.0, 0.0, 0.0))  # undo the importer's Y-up -> Z-up turn
        fix.parent = holder
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(asset_path(a.assets, vis["asset"])))
        for obj in set(bpy.data.objects) - before:
            if obj.parent is None:
                obj.parent = fix

    bodies = []
    for i, body in enumerate(header["bodies"]):
        e = link(bpy.data.objects.new(body["name"], None))
        e.rotation_mode = "QUATERNION"
        p = poses[frame, i]
        e.location = Vector(p[:3].tolist())
        e.rotation_quaternion = Quaternion(blender_quat(p[3:]))
        for vis in body["visuals"]:
            import_visual(vis, e)
        bodies.append(e)
    for i, vis in enumerate(header.get("static_visuals") or []):
        import_visual(vis, link(bpy.data.objects.new(f"static{i}", None)))

    # studio: a cyclorama (floor curving into a back wall) and a 10 mm grid work mat
    bpy.ops.mesh.primitive_plane_add(size=1)
    cyc = bpy.context.active_object
    bm = bmesh.new()
    prof = [(1.4, 0.0)]
    radius, height = 0.5, 1.6
    for k in range(40):
        t = (k / 39) * (math.pi / 2)
        prof.append((-0.6 - radius * math.sin(t), radius - radius * math.cos(t)))
    prof.append((-0.6 - radius, height))
    rows = [[bm.verts.new((x, y, z)) for (y, z) in prof] for x in (-2.0, 2.0)]
    for j in range(len(prof) - 1):
        bm.faces.new((rows[0][j], rows[1][j], rows[1][j + 1], rows[0][j + 1]))
    bm.to_mesh(cyc.data)
    bm.free()
    for poly in cyc.data.polygons:
        poly.use_smooth = True
    cyc.rotation_euler = Euler((0.0, 0.0, math.radians(-51.0 + rig_turn_deg(header, poses, frame))))
    m = bpy.data.materials.new("cyc")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.62, 0.62, 0.63, 1.0)
    m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.85
    cyc.data.materials.append(m)

    bpy.ops.mesh.primitive_plane_add(size=1, location=(0.02, 0.03, 0.0004))
    work = bpy.context.active_object
    work.scale = (0.68, 0.52, 1.0)
    gm = bpy.data.materials.new("mat")
    gm.use_nodes = True
    nt = gm.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc, mp, br = (nt.nodes.new(n) for n in ("ShaderNodeTexCoord", "ShaderNodeMapping", "ShaderNodeTexBrick"))
    mp.inputs["Scale"].default_value = (68.0, 52.0, 1.0)
    for key, val in (
        ("Color1", (0.64, 0.67, 0.70, 1)),
        ("Color2", (0.64, 0.67, 0.70, 1)),
        ("Mortar", (0.45, 0.48, 0.52, 1)),
    ):
        br.inputs[key].default_value = val
    br.inputs["Scale"].default_value = 1.0
    br.inputs["Mortar Size"].default_value = 0.04
    br.inputs["Brick Width"].default_value = 1.0
    br.inputs["Row Height"].default_value = 1.0
    br.offset = 0.0
    br.squash = 1.0
    nt.links.new(tc.outputs["UV"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], br.inputs["Vector"])
    nt.links.new(br.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.7
    work.data.materials.append(gm)

    names = [b["name"] for b in header["bodies"]]
    obj_i = names.index("object") if "object" in names else len(names) - 1
    rest = poses[0, obj_i, :3].astype(float)
    now = poses[frame, obj_i, :3].astype(float)
    turn = rig_turn_deg(header, poses, frame)

    def placed(p: tuple) -> Vector:  # authored around the origin, turned with the palm, moved to the part
        x, y, z = turn_xy(p, turn)
        return Vector((x + rest[0], y + rest[1], z))

    def area(name: str, loc: tuple, size: float, power: float, color: tuple = (1, 1, 1)) -> None:
        light = bpy.data.lights.new(name, "AREA")
        light.size, light.energy, light.color = size, power, color
        obj = link(bpy.data.objects.new(name, light))
        obj.location = placed(loc)
        aim = Vector((rest[0], rest[1], 0.08))
        obj.rotation_euler = (aim - obj.location).to_track_quat("-Z", "Y").to_euler()

    area("key", (0.55, -0.55, 0.85), 0.6, 14.0, (1.0, 0.97, 0.92))
    area("fill", (0.5, 0.75, 0.35), 1.0, 5.0, (0.9, 0.95, 1.0))
    area("rim", (-0.45, -0.25, 0.45), 0.4, 9.0)
    world = bpy.data.worlds.new("w")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.55, 0.56, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.25

    lift = float(now[2] - rest[2])

    def camera(name: str, loc: tuple, look: tuple, lens: float, fstop: float | None = None) -> Any:
        cam = bpy.data.cameras.new(name)
        cam.lens, cam.sensor_width = lens, 36.0
        eye, at = placed(loc), placed(look)
        if fstop:
            cam.dof.use_dof = True
            cam.dof.focus_distance = (eye - at).length
            cam.dof.aperture_fstop = fstop
        obj = link(bpy.data.objects.new(name, cam))
        obj.location = eye
        obj.rotation_euler = (at - eye).to_track_quat("-Z", "Y").to_euler()
        return obj

    follow = 0.7 * lift  # the wide framing eases up with the lift, as in the MuJoCo studio
    views = {
        "wide": camera("wide", (0.42, 0.34, 0.33 + follow), (0.0, 0.0, 0.085 + follow), 50.0),
        "close": camera("close", (0.24, 0.13, now[2] + 0.035), (0.0, 0.006, now[2] + 0.012), 55.0, 16.0),
    }
    scene.render.engine = "CYCLES" if a.engine == "cycles" else "BLENDER_EEVEE_NEXT"
    if a.engine == "cycles":
        scene.cycles.samples = a.samples
        scene.cycles.use_denoising = True
    else:
        scene.eevee.taa_render_samples = a.samples
    scene.render.resolution_x, scene.render.resolution_y = a.width, a.height
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{a.out}.json").write_text(
        json.dumps({"caption": caption(header), "frame": frame}), encoding="utf-8"
    )
    if a.anim:
        start, stop = (int(x) for x in a.anim.split(":"))
        for f in range(start, stop + 1):
            for i, e in enumerate(bodies):
                p = poses[f, i]
                e.location = Vector(p[:3].tolist())
                e.rotation_quaternion = Quaternion(blender_quat(p[3:]))
                e.keyframe_insert("location", frame=f)
                e.keyframe_insert("rotation_quaternion", frame=f)
        scene.frame_start, scene.frame_end = start, stop
        scene.render.fps = int(header["fps"])
        scene.camera = views["wide"]
        scene.render.filepath = f"{a.out}_anim_"
        bpy.ops.render.render(animation=True)
        return 0
    for name, cam in views.items():
        if a.view not in (name, "both"):
            continue
        scene.camera = cam
        scene.render.filepath = f"{a.out}_{name}.png"
        bpy.ops.render.render(write_still=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    raise SystemExit(main(argv))
