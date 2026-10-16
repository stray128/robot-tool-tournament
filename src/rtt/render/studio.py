"""MuJoCo studio look: the render model built from a replay header.

Render from the replay, never re-simulate (spec section 10). :func:`build_render_scene`
turns a replay header into a separate MuJoCo model in which every replay body is a **mocap
body** carrying only visual geoms (``contype = conaffinity = 0``, mass 0):

* version 2 bodies draw their ``visuals``: GLB assets read from the asset store, placed at
  the visual's pose and (positive) scale, CAD subdivided to <= 2.5 mm edges with normals
  split at 28 degree creases so MuJoCo's per-vertex lighting reads as machined;
* version 1 bodies (and the v2 collision overlay, when asked for) draw their primitive
  ``geoms``;
* the scored scene's table and floor are replaced by a seamless studio floor, a 10 mm grid
  work mat and a gradient sky that the fog blends into.

The model has no joints, so the scored model is untouched and cannot be affected: playback
(:mod:`rtt.render.playback`) only writes mocap poses and calls ``mj_forward``.

Lighting follows the render track's studio: a soft shadowed key, a cool fill, a rim spot and
a low ambient headlight; 8192 px shadow map; 8x MSAA; 1280 x 720 by default.
"""

from __future__ import annotations

import functools
import io
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final

import mujoco
import numpy as np

from rtt.render import materials as mat
from rtt.render.glb import AssetStore, GlbMesh, read_glb
from rtt.render.materials import Look

__all__ = [
    "GROUP_COLLISION",
    "GROUP_VISUAL",
    "VIEWS",
    "RenderScene",
    "StudioConfig",
    "View",
    "align_to_palm",
    "build_render_scene",
    "free_camera",
    "make_renderer",
]

GROUP_VISUAL: Final = 2  # rendered by default
GROUP_COLLISION: Final = 3  # the collision overlay: hidden unless the option turns group 3 on
AssetSource = Callable[[str], bytes]


@dataclass(frozen=True)
class StudioConfig:
    """Studio lighting, environment and output settings (the render track's tuned values)."""

    width: int = 1280
    height: int = 720
    offsamples: int = 8  # MSAA
    shadowsize: int = 8192
    floor_rgb: tuple[float, float, float] = (0.66, 0.66, 0.67)
    mat_rgb: tuple[float, float, float] = (0.70, 0.72, 0.74)
    mat_size: tuple[float, float] | None = (0.45, 0.35)  # half extents of the work mat (m)
    mat_offset: tuple[float, float] = (0.0, 0.0)
    grid_contrast: float = 0.86
    grid_repeat: float = 50.0  # checker repeats per metre (2 squares each): a 10 mm grid
    sky_top: tuple[float, float, float] = (0.90, 0.90, 0.91)
    sky_bottom: tuple[float, float, float] = (0.80, 0.80, 0.81)
    key_dir: tuple[float, float, float] = (-0.75, -0.55, -1.2)  # from the cameras' right, high
    key: float = 0.58
    fill: float = 0.34
    rim: float = 0.55
    ambient: float = 0.32
    soft: float = 0.12  # key light bulb radius (soft shadow edge)
    shadowclip: float = 1.6
    fogstart: float = 2.0
    fogend: float = 7.0
    max_edge: float = 0.0025  # subdivide CAD triangles to this edge length (m)
    crease_deg: float = 28.0
    collision_overlay: bool = False  # also draw the v1 geoms of v2 bodies (group 3)
    rig_yaw_deg: float = 0.0  # turn the whole light rig about z (playback aligns it with the palm)


@dataclass(frozen=True)
class View:
    """A free camera relative to a target point (MuJoCo ``mjvCamera`` conventions, degrees)."""

    offset: tuple[float, float, float]  # lookat = target + offset (m)
    distance: float
    azimuth: float
    elevation: float
    fovy: float


# Views and the light rig are authored for a palm yawed by AUTHORED_PALM_YAW_DEG (the frame's
# camera plate then faces away from the cameras); playback turns both with the replay's palm.
AUTHORED_PALM_YAW_DEG: Final = 180.0
VIEWS: Final = {
    "wide": View((0.0, 0.0, 0.070), 0.36, 215.0, -18.0, 36.0),
    "close": View((0.0, 0.0, 0.012), 0.21, 212.0, -8.0, 30.0),
    "side": View((0.0, 0.0, 0.015), 0.30, 180.0, -4.0, 30.0),
    "detail": View((0.0, -0.024, -0.020), 0.16, 238.0, -3.0, 32.0),
    "xray": View((0.0, 0.0, 0.004), 0.26, 205.0, -6.0, 30.0),  # used with the x-ray look
}


@dataclass(frozen=True, eq=False)
class RenderScene:
    """A compiled render model and how replay bodies map onto it."""

    model: mujoco.MjModel
    body_names: tuple[str, ...]
    body_ids: np.ndarray  # MuJoCo body id per replay body
    mocap_ids: np.ndarray  # mocap index per replay body
    body_geoms: dict[str, np.ndarray]  # replay body name -> its visual geom ids
    body_materials: dict[str, np.ndarray]  # replay body name -> material ids its geoms use
    cfg: StudioConfig
    info: dict[str, Any] = field(default_factory=dict)

    def index(self, name: str) -> int:
        return self.body_names.index(name)


# --------------------------------------------------------------------------- meshes


@functools.lru_cache(maxsize=64)
def _prepared(key: tuple[Any, ...], v: bytes, f: bytes, nv: int, max_edge: float, crease_deg: float) -> tuple:
    """Subdivide and crease-shade one mesh for MuJoCo's per-vertex lighting (cached)."""
    import trimesh

    verts = np.frombuffer(v, dtype=np.float64).reshape(nv, 3)
    faces = np.frombuffer(f, dtype=np.int64).reshape(-1, 3)
    m = trimesh.Trimesh(verts, faces, process=True)
    m.merge_vertices()
    if max_edge > 0:
        sv, sf = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=max_edge, max_iter=12)
        m = trimesh.Trimesh(sv, sf, process=True)
    m = trimesh.graph.smooth_shade(m, angle=np.radians(crease_deg), facet_minarea=None)
    return np.asarray(m.vertices), np.asarray(m.faces), np.asarray(m.vertex_normals)


def _prepare(mesh: GlbMesh, scale: np.ndarray, cfg: StudioConfig, textured: bool) -> tuple:
    verts = mesh.vertices * scale
    if textured:  # keep the uv layout: no subdivision, the file's own normals
        normals = mesh.normals if mesh.normals is not None else None
        return verts, mesh.faces, normals
    return _prepared(
        ("mesh", mesh.name),
        np.ascontiguousarray(verts).tobytes(),
        np.ascontiguousarray(mesh.faces).tobytes(),
        len(verts),
        cfg.max_edge,
        cfg.crease_deg,
    )


def _quat_wxyz(q_xyzw: Sequence[float]) -> list[float]:
    x, y, z, w = (float(c) for c in q_xyzw)
    return [w, x, y, z]


# --------------------------------------------------------------------------- scene


class _Builder:
    def __init__(self, cfg: StudioConfig, source: AssetSource | None) -> None:
        self.cfg = cfg
        self.source = source
        self.spec = mujoco.MjSpec()
        self.spec.modelname = "rtt_studio"
        self.spec.compiler.degree = False
        self.n_mesh = 0
        self.materials: dict[tuple[str, str], str] = {}
        self.body_geoms: dict[str, list[str]] = {}
        self.body_mats: dict[str, set[str]] = {}
        self._docs: dict[str, list[GlbMesh]] = {}

    def material(self, body: str, look: Look, texture: bytes | None = None) -> str:
        key = (
            body,
            f"{look.name}|{look.rgba}|{look.metallic}|{look.roughness}|{id(texture) if texture else ''}",
        )
        if key in self.materials:
            return self.materials[key]
        name = f"m{len(self.materials):03d}_{body}_{look.name}"
        kw = look.mujoco()
        m = self.spec.add_material(
            name=name,
            rgba=kw["rgba"],
            specular=kw["specular"],
            shininess=kw["shininess"],
            reflectance=kw["reflectance"],
        )
        m.metallic, m.roughness = kw["metallic"], kw["roughness"]
        if texture is not None:
            from PIL import Image

            img = np.asarray(Image.open(io.BytesIO(texture)).convert("RGB"))
            tex = self.spec.add_texture(name=f"{name}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D)
            tex.width, tex.height, tex.nchannel = int(img.shape[1]), int(img.shape[0]), 3
            tex.data = np.ascontiguousarray(img).tobytes()
            m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
            m.rgba = [1.0, 1.0, 1.0, look.alpha]
        self.materials[key] = name
        return name

    def glb(self, aid: str) -> list[GlbMesh]:
        if aid not in self._docs:
            if self.source is None:
                raise ValueError(f"replay draws asset {aid} but no asset source was given")
            self._docs[aid] = read_glb(self.source(aid)).meshes
        return self._docs[aid]

    def add_visual(self, body: mujoco.MjsBody, bname: str, vis: Mapping[str, Any]) -> None:
        scale = np.asarray(vis.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64)
        if np.any(scale <= 0):
            raise ValueError("visual scales must be positive (a right jaw is its own asset)")
        for mesh in self.glb(vis["asset"]):
            textured = mesh.texture is not None and mesh.uv is not None
            verts, faces, normals = _prepare(mesh, scale, self.cfg, textured)
            name = f"mesh{self.n_mesh:03d}"
            self.n_mesh += 1
            kw: dict[str, Any] = {"uservert": verts.ravel().tolist(), "userface": faces.ravel().tolist()}
            if normals is not None:
                kw.update(
                    usernormal=np.asarray(normals).ravel().tolist(), userfacenormal=faces.ravel().tolist()
                )
            if textured:
                tc = np.stack([mesh.uv[:, 0], 1.0 - mesh.uv[:, 1]], axis=1)  # MuJoCo flips v
                kw.update(usertexcoord=tc.ravel().tolist(), userfacetexcoord=faces.ravel().tolist())
            self.spec.add_mesh(name=name, **kw)
            g = body.add_geom(
                name=f"{bname}_{name}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name,
                pos=list(vis.get("pos", (0.0, 0.0, 0.0))),
                quat=_quat_wxyz(vis.get("quat_xyzw", (0.0, 0.0, 0.0, 1.0))),
                group=GROUP_VISUAL,
                material=self.material(bname, mesh.look, mesh.texture if textured else None),
            )
            self._visual_only(g)
            self.body_geoms.setdefault(bname, []).append(g.name)
            self.body_mats.setdefault(bname, set()).add(g.material)

    def add_geom(
        self, body: mujoco.MjsBody, bname: str, entry: Mapping[str, Any], look: Look, group: int, i: int
    ) -> None:
        kind = entry["type"]
        kw: dict[str, Any] = {
            "name": f"{bname}_g{i:02d}_{group}",
            "pos": list(entry["pos"]),
            "quat": _quat_wxyz(entry["quat_xyzw"]),
            "group": group,
            "material": self.material(bname, look),
        }
        if kind == "box":
            kw.update(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(entry["half_size"]))
        elif kind == "cylinder":
            kw.update(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[entry["radius"], entry["half_height"], 0.0])
        elif kind == "capsule":
            kw.update(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[entry["radius"], entry["half_length"], 0.0])
        elif kind == "mesh":
            name = f"mesh{self.n_mesh:03d}"
            self.n_mesh += 1
            self.spec.add_mesh(name=name, uservert=list(entry["vertices"]), userface=list(entry["faces"]))
            kw.update(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name)
        elif kind == "plane":
            kw.update(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[*entry["half_size"], 0.05])
        else:
            raise ValueError(f"unknown geom type {kind!r}")
        g = body.add_geom(**kw)
        self._visual_only(g)
        if group == GROUP_VISUAL:
            self.body_geoms.setdefault(bname, []).append(g.name)
            self.body_mats.setdefault(bname, set()).add(g.material)

    @staticmethod
    def _visual_only(g: mujoco.MjsGeom) -> None:
        g.contype = 0
        g.conaffinity = 0
        g.mass = 0.0
        g.density = 0.0

    def environment(self) -> None:
        cfg, spec = self.cfg, self.spec
        spec.add_texture(
            name="sky",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=list(cfg.sky_top),
            rgb2=list(cfg.sky_bottom),
            width=800,
            height=800,
        )
        spec.add_texture(
            name="grid",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            mark=mujoco.mjtMark.mjMARK_EDGE,
            rgb1=list(cfg.mat_rgb),
            rgb2=list(cfg.mat_rgb),
            markrgb=(np.array(cfg.mat_rgb) * cfg.grid_contrast).tolist(),
            width=512,
            height=512,
        )
        grid = spec.add_material(
            name="work_mat",
            texrepeat=[cfg.grid_repeat, cfg.grid_repeat],
            texuniform=True,
            specular=0.12,
            shininess=0.08,
        )
        grid.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
        spec.add_material(
            name="studio_floor", rgba=[*cfg.floor_rgb, 1.0], reflectance=0.0, specular=0.05, shininess=0.02
        )
        w = spec.worldbody
        floor = w.add_geom(
            name="studio_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[30.0, 30.0, 0.05],
            pos=[0.0, 0.0, -0.0005],
            material="studio_floor",
            group=GROUP_VISUAL,
        )
        self._visual_only(floor)
        if cfg.mat_size:
            work = w.add_geom(
                name="work_mat",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[cfg.mat_size[0], cfg.mat_size[1], 0.0015],
                pos=[cfg.mat_offset[0], cfg.mat_offset[1], -0.0012],
                material="work_mat",
                group=GROUP_VISUAL,
            )
            self._visual_only(work)
        c, s_ = np.cos(np.radians(cfg.rig_yaw_deg)), np.sin(np.radians(cfg.rig_yaw_deg))
        turn = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])

        def rig(v: Sequence[float]) -> list[float]:
            return (turn @ np.asarray(v, dtype=np.float64)).tolist()

        key_dir = np.array(cfg.key_dir, dtype=np.float64)
        key_dir /= np.linalg.norm(key_dir)
        directional = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        w.add_light(
            name="key",
            type=directional,
            pos=rig(-key_dir * 2.0),
            dir=rig(key_dir),
            castshadow=True,
            diffuse=[cfg.key * c for c in (1.0, 0.97, 0.92)],
            specular=[0.35] * 3,
            bulbradius=cfg.soft,
        )
        w.add_light(
            name="fill",
            type=directional,
            pos=rig([-1.5, -0.9, 1.0]),
            dir=rig([0.8, 0.45, -0.55]),
            castshadow=False,
            diffuse=[cfg.fill * c for c in (0.92, 0.96, 1.0)],
            specular=[0.03] * 3,
        )
        w.add_light(
            name="rim",
            pos=rig([-0.55, -0.45, 0.45]),
            dir=rig([0.55, 0.45, -0.3]),
            castshadow=False,
            diffuse=[cfg.rim] * 3,
            specular=[0.9] * 3,
            cutoff=45,
            exponent=1,
        )
        vis = spec.visual
        vis.headlight.ambient = [cfg.ambient] * 3
        vis.headlight.diffuse = [0.06] * 3
        vis.headlight.specular = [0.0] * 3
        vis.quality.shadowsize = cfg.shadowsize
        vis.quality.offsamples = cfg.offsamples
        vis.map.shadowclip = cfg.shadowclip
        vis.map.shadowscale = 0.6
        vis.map.fogstart = cfg.fogstart
        vis.map.fogend = cfg.fogend
        vis.map.znear = 0.002
        vis.map.zfar = 50.0
        vis.rgba.fog = [*cfg.sky_bottom, 1.0]
        vis.rgba.haze = [*cfg.sky_bottom, 1.0]
        vis.global_.offwidth = max(cfg.width, 1920)
        vis.global_.offheight = max(cfg.height, 1080)
        spec.stat.extent = 0.5  # fog and shadow distances are multiples of this (m)
        spec.stat.center = [0.0, 0.0, 0.1]


DEFAULT_V1_LOOKS: Final = {
    "palm": mat.FRAME,
    "finger_left": mat.FINGER_DESIGNED,
    "finger_right": mat.FINGER_DESIGNED,
    "object": mat.PART_LOOKS["plastic_blue"],
}


def build_render_scene(
    header: Mapping[str, Any],
    assets: AssetStore | AssetSource | None = None,
    cfg: StudioConfig | None = None,
    *,
    looks: Mapping[str, Look] | None = None,
    static_visuals: bool = True,
) -> RenderScene:
    """Compile the studio render model for a replay header (version 1 or 2).

    ``assets`` resolves asset ids to uncompressed GLB bytes: an :class:`AssetStore` or any
    callable. ``looks`` overrides the per-body look of v1 geoms (defaults: dark palm,
    orange fingers, blue object).
    """
    cfg = cfg or StudioConfig()
    source: AssetSource | None
    if isinstance(assets, AssetStore):
        source = assets.raw_bytes
    else:
        source = assets
    b = _Builder(cfg, source)
    b.environment()
    v1_looks = {**DEFAULT_V1_LOOKS, **dict(looks or {})}
    names = []
    for entry in header["bodies"]:
        name = entry["name"]
        names.append(name)
        body = b.spec.worldbody.add_body(name=name, mocap=True)
        visuals = entry.get("visuals") or []
        for vis in visuals:
            b.add_visual(body, name, vis)
        geoms = entry.get("geoms") or []
        if not visuals:
            look = v1_looks.get(name, mat.PART_LOOKS["clay"])
            for i, g in enumerate(geoms):
                b.add_geom(body, name, g, look, GROUP_VISUAL, i)
        elif cfg.collision_overlay:
            for i, g in enumerate(geoms):
                b.add_geom(body, name, g, mat.COLLISION, GROUP_COLLISION, i)
        if not b.body_geoms.get(name):  # an empty body still needs a placeholder for xpos
            b.body_geoms.setdefault(name, [])
    if static_visuals:
        for i, vis in enumerate(header.get("static_visuals") or []):
            prop = b.spec.worldbody.add_body(
                name=f"static{i:02d}", pos=list(vis["pos"]), quat=_quat_wxyz(vis["quat_xyzw"])
            )
            b.add_visual(prop, f"static{i:02d}", {**vis, "pos": [0, 0, 0], "quat_xyzw": [0, 0, 0, 1]})
    model = b.spec.compile()
    body_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names])
    mocap_ids = np.array([int(model.body_mocapid[i]) for i in body_ids])

    def gid(n: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)

    def mid(n: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, n)

    return RenderScene(
        model=model,
        body_names=tuple(names),
        body_ids=body_ids,
        mocap_ids=mocap_ids,
        body_geoms={k: np.array([gid(n) for n in v], dtype=int) for k, v in b.body_geoms.items()},
        body_materials={k: np.array(sorted(mid(n) for n in v), dtype=int) for k, v in b.body_mats.items()},
        cfg=cfg,
        info={"ngeom": model.ngeom, "nmesh": model.nmesh, "nmeshvert": model.nmeshvert, "nmat": model.nmat},
    )


# --------------------------------------------------------------------------- cameras and renderer


def turned(view: View, yaw_deg: float) -> View:
    """``view`` turned about the world z axis by ``yaw_deg`` (azimuth and offset)."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    ox, oy, oz = view.offset
    return View(
        (c * ox - s * oy, s * ox + c * oy, oz),
        view.distance,
        view.azimuth + yaw_deg,
        view.elevation,
        view.fovy,
    )


def free_camera(scene: RenderScene, view: View | str, target: Sequence[float]) -> mujoco.MjvCamera:
    """A free camera for ``view`` aimed at ``target`` + the view's offset; sets the model's fovy.

    The view is turned by the scene's ``rig_yaw_deg``, so cameras and lights stay together."""
    v = turned(VIEWS[view] if isinstance(view, str) else view, scene.cfg.rig_yaw_deg)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray(target, dtype=np.float64) + np.asarray(v.offset)
    cam.distance, cam.azimuth, cam.elevation = v.distance, v.azimuth, v.elevation
    scene.model.vis.global_.fovy = v.fovy
    return cam


def align_to_palm(cfg: StudioConfig, header: Mapping[str, Any], frames: np.ndarray) -> StudioConfig:
    """``cfg`` with the rig (and so every view) turned to the replay's palm yaw at frame 0.

    Views are authored for a palm at ``AUTHORED_PALM_YAW_DEG``; a replay whose palm faces
    another way is framed the same way relative to the gripper."""
    names = [b["name"] for b in header["bodies"]]
    if "palm" not in names:
        return cfg
    x, y, z, w = (float(c) for c in np.asarray(frames[0][names.index("palm")][3:], dtype=np.float64))
    yaw = np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    delta = (yaw - AUTHORED_PALM_YAW_DEG + 180.0) % 360.0 - 180.0
    return replace(cfg, rig_yaw_deg=cfg.rig_yaw_deg + float(delta))


def make_renderer(scene: RenderScene) -> tuple[mujoco.Renderer, mujoco.MjvOption]:
    """An offscreen renderer at the studio size, plus the scene option (groups 0-2 shown;
    group 3, the collision overlay, follows ``cfg.collision_overlay``)."""
    r = mujoco.Renderer(scene.model, height=scene.cfg.height, width=scene.cfg.width)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = opt.geomgroup[1] = opt.geomgroup[GROUP_VISUAL] = 1
    opt.geomgroup[GROUP_COLLISION] = 1 if scene.cfg.collision_overlay else 0
    return r, opt


def render_flags(r: mujoco.Renderer) -> None:
    f = r.scene.flags
    f[mujoco.mjtRndFlag.mjRND_SHADOW] = True
    f[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
    f[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
    f[mujoco.mjtRndFlag.mjRND_FOG] = True
    f[mujoco.mjtRndFlag.mjRND_HAZE] = False
