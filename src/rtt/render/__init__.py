"""Rendering, replay assets and print-pack visuals (track R of the v2 visual upgrade).

* :mod:`rtt.render.gripper`: the open-source gripper's vendored meshes, cut into frame,
  carriages and stock blades, plus the stock blade's committed collision hulls.
* :mod:`rtt.render.materials`: the studio palette (sRGB authoring, linear glTF factors).
* :mod:`rtt.render.glb`: deterministic GLB writer and reader, content-addressed asset store.
* :mod:`rtt.render.studio`: the MuJoCo studio render model built from a replay header.
* :mod:`rtt.render.playback`: render from the replay (poses in, pixels out; never re-simulates).
* :mod:`rtt.render.xray`: the x-ray look (part drawn translucent).
* :mod:`rtt.render.blender_hero`: offline Blender hero script (runs inside Blender only).

Importing this package imports nothing heavy; each module pulls in what it needs.
"""
