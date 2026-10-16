"""Frozen v2 contracts: freeform jaw types, part spec, replay v2 header schema, asset manifest."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from rtt.design.freeform import (
    CARRIAGE_INNER_Y,
    DESIGN_SPACE,
    ROOT_PLANE_PALM_Z,
    CaptureReport,
    FingerPair,
    JawGeometry,
    build_pair,
)
from rtt.parts import REGISTER_AXES, Feature, Material, PartSpec
from rtt.sim.objects import MeshObject
from rtt.sim.task import load_task
from schema_lite import Validator, unsupported_keywords

REPO = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO / "src" / "rtt" / "sim" / "replay_v2.schema.json"
MANIFEST_DOC = REPO / "docs" / "ASSET_MANIFEST.md"
SHA_A = "a" * 64
SHA_B = "0123456789abcdef" * 4


def _box_corners(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _jaw(y_sign: float = 1.0) -> JawGeometry:
    pad = _box_corners((-0.01, 0.0004, -0.07), (0.01, 0.0034, -0.05))
    body = _box_corners((-0.01, 0.0034, -0.07), (0.01, 0.0134, 0.0))
    pieces = [p * np.array([1.0, y_sign, 1.0]) for p in (pad, body)]
    return JawGeometry(
        pieces=pieces,
        mass=0.04,
        com=[0.0, y_sign * 0.008, -0.03],
        inertia6=[2e-5, 2e-5, 1e-6, 0.0, 0.0, 0.0],
        print_mesh_sha=SHA_A,
    )


def _capture() -> CaptureReport:
    return CaptureReport(
        no_jam_mm=3.0, centring_mm=2.4, mu=0.3, by_error=((0.0, "centred"), (3.0, "uncentred"))
    )


def _pair(**overrides: Any) -> FingerPair:
    kwargs: dict[str, Any] = {
        "left": _jaw(1.0),
        "right": _jaw(-1.0),
        "tip_height": 0.006,
        "finger_length": 0.07,
        "q_seat": 0.021,
        "design_id": SHA_A,
        "part_id": "pipe_tee",
        "geometry_hash": SHA_B,
        "capture": _capture(),
        "build_s": 0.1,
        "auto_repairs": ["clearance_mm reset to 0.2 (inactive for v_funnel)"],
    }
    kwargs.update(overrides)
    return FingerPair(**kwargs)


# ----------------------------------------------------------------------------- freeform types


def test_jaw_geometry_freezes_its_arrays() -> None:
    jaw = _jaw()
    assert isinstance(jaw.pieces, tuple) and jaw.n_pieces == 2
    for arr in (*jaw.pieces, jaw.com, jaw.inertia6):
        assert arr.dtype == np.float64 and not arr.flags.writeable
    assert jaw.com.shape == (3,) and jaw.inertia6.shape == (6,)
    np.testing.assert_allclose(jaw.bounds, [[-0.01, 0.0004, -0.07], [0.01, 0.0134, 0.0]])
    with pytest.raises(dataclasses.FrozenInstanceError):
        jaw.mass = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "field,value",
    [
        ("pieces", []),
        ("pieces", [np.zeros((3, 3))]),
        ("pieces", [np.zeros((8, 2))]),
        ("pieces", [np.full((8, 3), np.nan)]),
        ("pieces", np.zeros((8, 3))),
        ("mass", 0.0),
        ("mass", float("inf")),
        ("mass", True),
        ("com", [0.0, 0.0]),
        ("inertia6", [0.0, 1e-5, 1e-5, 0.0, 0.0, 0.0]),
        ("inertia6", [1e-5] * 3),
        ("print_mesh_sha", "A" * 64),
        ("print_mesh_sha", "abc"),
    ],
)
def test_jaw_geometry_rejects_malformed_data(field: str, value: Any) -> None:
    good = _jaw()
    kwargs = {f.name: getattr(good, f.name) for f in dataclasses.fields(JawGeometry)}
    kwargs[field] = value
    with pytest.raises(ValueError):
        JawGeometry(**kwargs)


def test_finger_pair_constructs_and_normalises() -> None:
    pair = _pair(tip_height=np.float64(0.006), q_seat=np.float32(0.021))
    assert type(pair.tip_height) is float and type(pair.q_seat) is float
    assert pair.auto_repairs == ("clearance_mm reset to 0.2 (inactive for v_funnel)",)
    assert pair.boxes is None
    assert pair.left is not pair.right
    assert pair.capture.by_error[0] == (0.0, "centred")
    assert _pair(auto_repairs=()).auto_repairs == ()


@pytest.mark.parametrize(
    "overrides",
    [
        {"design_id": "not-a-sha"},
        {"geometry_hash": SHA_A[:-1]},
        {"part_id": ""},
        {"finger_length": 0.0},
        {"q_seat": -0.001},
        {"tip_height": float("nan")},
        {"build_s": -1.0},
        {"auto_repairs": "one string"},
    ],
)
def test_finger_pair_rejects_malformed_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _pair(**overrides)


def test_finger_pair_rejects_wrong_types() -> None:
    with pytest.raises(TypeError):
        _pair(left=object())
    with pytest.raises(TypeError):
        _pair(capture={"no_jam_mm": 1.0})


def test_capture_report_validates_outcomes() -> None:
    with pytest.raises(ValueError):
        CaptureReport(no_jam_mm=1.0, centring_mm=1.0, mu=0.3, by_error=((1.0, "stuck"),))
    with pytest.raises(ValueError):
        CaptureReport(no_jam_mm=-1.0, centring_mm=1.0, mu=0.3)
    with pytest.raises(ValueError):
        CaptureReport(no_jam_mm=1.0, centring_mm=1.0, mu=0.0)
    report = CaptureReport(no_jam_mm=1, centring_mm=0, mu=0.3, by_error=[(1, "jammed")])
    assert report.by_error == ((1.0, "jammed"),) and type(report.no_jam_mm) is float


def test_build_pair_keeps_the_frozen_signature() -> None:
    sig = inspect.signature(build_pair)
    assert list(sig.parameters) == ["design", "part", "task"]
    assert sig.return_annotation == "FingerPair"
    with pytest.raises(TypeError):
        build_pair(None, None, None)  # type: ignore[arg-type]


def test_jaw_frame_constants_match_the_v1_scene() -> None:
    task = load_task()
    assert DESIGN_SPACE == "freeform-v2"
    assert ROOT_PLANE_PALM_Z == pytest.approx(-task.gripper.palm_half_size[2])
    assert CARRIAGE_INNER_Y == pytest.approx(0.0004)
    # at full closure the jaw origin is on the palm centre plane
    assert task.gripper.open_half_gap - task.gripper.stroke == pytest.approx(0.0)


def test_contract_modules_import_only_numpy() -> None:
    code = (
        "import sys; import rtt.design.freeform, rtt.parts; "
        "heavy = {'mujoco', 'trimesh', 'manifold3d', 'scipy', 'pydantic', 'coacd'}; "
        "print(sorted(heavy & set(sys.modules)))"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "[]"


# ----------------------------------------------------------------------------- part spec

STEEL = Material("zinc_steel", density=7850.0, base_color=(0.78, 0.79, 0.8), metallic=1.0, roughness=0.35)


def _box_part(size=(0.03, 0.04, 0.06), **overrides: Any) -> PartSpec:
    sx, sy, sz = size
    corners = _box_corners((-sx / 2, -sy / 2, -sz / 2), (sx / 2, sy / 2, sz / 2))
    hull = ConvexHull(corners)
    volume = sx * sy * sz
    unit_inertia = (
        volume * (sy * sy + sz * sz) / 12,
        volume * (sx * sx + sz * sz) / 12,
        volume * (sx * sx + sy * sy) / 12,
        0.0,
        0.0,
        0.0,
    )
    kwargs: dict[str, Any] = {
        "name": "block_30x40x60",
        "family": "block",
        "pieces": [corners],
        "faces": [hull.simplices],
        "volume": volume,
        "com": (0.0, 0.0, 0.0),
        "unit_inertia": unit_inertia,
        "material": STEEL,
        "yaw_symmetry": 2,
        "decomposition": "exact",
    }
    kwargs.update(overrides)
    return PartSpec(**kwargs)


def test_part_spec_is_a_mesh_object_with_extras() -> None:
    part = _box_part()
    assert part.part_id == part.name == "block_30x40x60"
    assert part.parts is part.pieces
    assert part.register_axes == REGISTER_AXES and part.features == ()
    assert part.density == 7850.0
    assert part.mass == pytest.approx(0.03 * 0.04 * 0.06 * 7850.0)
    assert part.size == pytest.approx((0.03, 0.04, 0.06))
    assert not part.pieces[0].flags.writeable
    mesh = part.to_mesh_object()
    assert isinstance(mesh, MeshObject)
    assert mesh.geometry_sha256 == part.geometry_sha256
    assert mesh.size == pytest.approx(part.size)
    assert mesh.density == part.density and mesh.decomposition == "exact"


@pytest.mark.parametrize(
    "symmetry,error,residual",
    [(6, 59.0, 1.0), (6, 30.0, 30.0), (6, -31.0, 29.0), (2, 179.0, 1.0), (1, 359.0, 1.0), (1, -2.0, 2.0)],
)
def test_yaw_residual_is_taken_modulo_the_symmetry(symmetry: int, error: float, residual: float) -> None:
    assert _box_part(yaw_symmetry=symmetry).yaw_residual_deg(error) == pytest.approx(residual)


def test_axisymmetric_parts_do_not_register_yaw() -> None:
    with pytest.raises(ValueError):
        _box_part(yaw_symmetry=0)
    slug = _box_part(yaw_symmetry=0, register_axes=("x", "y", "tilt"))
    assert slug.yaw_residual_deg(123.0) == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"register_axes": ()},
        {"register_axes": ("x", "x", "tilt")},
        {"register_axes": ("x", "roll")},
        {"yaw_symmetry": -1},
        {"yaw_symmetry": 1.5},
        {"name": "has space"},
        {"family": ""},
        {"volume": 0.0},
        {"com": (0.0, 0.0)},
        {"visual_asset": "ABC"},
        {"pose_probability": 1.5},
        {"faces": []},
        {"faces": [np.array([[0, 1, 99]])]},
        {"pieces": [np.zeros((3, 3))], "faces": [np.array([[0, 1, 2]])]},
    ],
)
def test_part_spec_rejects_malformed_data(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _box_part(**overrides)


def test_part_spec_rejects_metre_mistakes() -> None:
    with pytest.raises(ValueError):
        _box_part(size=(30.0, 40.0, 60.0))


def test_features_declare_sockets_facing_a_jaw() -> None:
    branch = Feature(
        "socket",
        "branch_socket",
        origin=(0.0, 0.0215, 0.0),
        axis=(0.0, 1.0, 0.0),
        radius=0.01075,
        depth=0.017,
    )
    back = Feature(
        "socket", "back_bore", origin=(0.0, -0.02, 0.0), axis=(0.0, -1.0, 0.0), radius=0.005, depth=0.01
    )
    flange = Feature(
        "flange", "flange", origin=(0.0, 0.0, 0.0125), axis=(0.0, 0.0, 1.0), radius=0.018, depth=0.005
    )
    assert branch.facing() == "left" and back.facing() == "right" and flange.facing() is None
    part = _box_part(features=[branch, flange], visual_asset="0123456789abcdef")
    assert part.features == (branch, flange) and part.sockets == (branch,)
    assert hash(branch) == hash(Feature(**dataclasses.asdict(branch)))
    with pytest.raises(ValueError):
        _box_part(features=[branch, branch])
    with pytest.raises(ValueError):
        Feature("socket", "s", origin=(0, 0, 0), axis=(0.0, 2.0, 0.0), radius=0.01, depth=0.01)
    with pytest.raises(ValueError):
        Feature("socket", "s", origin=(0, 0, 0), axis=(0.0, 1.0, 0.0), radius=0.0, depth=0.01)
    with pytest.raises(ValueError):
        Feature("groove", "s", origin=(0, 0, 0), axis=(0.0, 1.0, 0.0), radius=0.01, depth=0.01)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"density": 0.0},
        {"base_color": (1.2, 0.0, 0.0)},
        {"base_color": (1.0, 0.0)},
        {"metallic": 1.5},
        {"roughness": -0.1},
        {"name": ""},
    ],
)
def test_material_rejects_bad_values(kwargs: dict[str, Any]) -> None:
    base: dict[str, Any] = {"name": "pvc_white", "density": 1380.0, "base_color": (0.93, 0.93, 0.9)}
    base.update(kwargs)
    with pytest.raises(ValueError):
        Material(**base)


# ----------------------------------------------------------------------------- replay v2 schema


def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _asset(content: bytes, role: str, license: str, attribution: str, changes: list[str]) -> tuple[str, dict]:
    sha = hashlib.sha256(content).hexdigest()
    aid = sha[:16]
    return aid, {
        "uri": f"assets/{aid}.glb",
        "sha256": sha,
        "bytes": len(content),
        "role": role,
        "license": license,
        "attribution": attribution,
        "changes": changes,
    }


def _visual(aid: str) -> dict[str, Any]:
    return {"asset": aid, "pos": [0.0, 0.0, 0.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0]}


def sample_header() -> dict[str, Any]:
    ours = "Robot Tool Tournament (Apache-2.0)"
    upstream = "Example Author, Example Gripper, https://example.org/gripper, CC BY 4.0"
    frame_id, frame = _asset(
        b"glTF frame", "gripper_frame", "CC-BY-4.0", upstream, ["cut", "re-encoded as GLB"]
    )
    left_id, left = _asset(b"glTF left jaw", "finger_left", "Apache-2.0", ours, [])
    right_id, right = _asset(b"glTF right jaw", "finger_right", "Apache-2.0", ours, [])
    part_id, part = _asset(b"glTF tee", "object", "Apache-2.0", ours, [])
    frame["source"] = {
        "kind": "upstream",
        "url": "https://example.org/gripper",
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "path": "simulation/meshes/frame.stl",
        "sha256": SHA_B,
    }
    left["source"] = {"kind": "generated", "generator": "rtt.render.glb"}
    return {
        "format": "rtt-replay",
        "version": 2,
        "fps": 30,
        "dtype": "float32",
        "byte_order": "little",
        "layout": ["frames", "bodies", "pose"],
        "pose_fields": ["px", "py", "pz", "qx", "qy", "qz", "qw"],
        "up_axis": "z",
        "units": "m",
        "n_frames": 184,
        "n_bodies": 4,
        "bin": "bout-001.bin",
        "assets": {frame_id: frame, left_id: left, right_id: right, part_id: part},
        "bodies": [
            {"name": "palm", "visuals": [_visual(frame_id)]},
            {"name": "finger_left", "visuals": [_visual(left_id)]},
            {
                "name": "finger_right",
                "visuals": [_visual(right_id)],
                "geoms": [
                    {
                        "name": "pad0",
                        "type": "box",
                        "half_size": [0.01, 0.0015, 0.01],
                        "pos": [0.0, -0.002, -0.06],
                        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "rgba": [0.9, 0.5, 0.1, 1.0],
                    }
                ],
            },
            {"name": "object", "visuals": [_visual(part_id)]},
        ],
        "static_visuals": [_visual(part_id)],
        "meta": {
            "design_space": DESIGN_SPACE,
            "design_id": SHA_A,
            "geometry_id": SHA_B,
            "part_id": "pipe_tee",
            "metrics": {"success": True},
        },
        "provenance": {
            "source": "simulation",
            "engine": "mujoco",
            "engine_version": "3.15.0",
            "run_id": "pilot-a1",
            "code_version": "f60f954+dirty",
            "product_mode": False,
            "suite": "train_v2",
            "condition": "c1",
            "seed": 0,
            "created_utc": "2026-10-17T10:00:00Z",
        },
        "overlays": {
            "bin": "bout-001.ovl.bin",
            "dtype": "float16",
            "byte_order": "little",
            "layout": ["frames", "channels"],
            "channels": ["offset_mm", "seat_short_mm", "hold_s", "n_contacts"],
        },
    }


def cross_field_errors(header: dict[str, Any]) -> list[str]:
    """The header rules the schema's $comment lists that JSON Schema cannot express."""
    errs = []
    if header["n_bodies"] != len(header["bodies"]):
        errs.append("n_bodies != len(bodies)")
    for aid, entry in header["assets"].items():
        if entry["uri"] != f"assets/{aid}.glb" or not entry["sha256"].startswith(aid):
            errs.append(f"asset {aid} is not content addressed")
    visuals = [v for b in header["bodies"] for v in b["visuals"]] + header.get("static_visuals", [])
    errs += [f"unknown asset {v['asset']}" for v in visuals if v["asset"] not in header["assets"]]
    return errs


def test_schema_is_draft_2020_12_within_the_supported_subset() -> None:
    schema = _schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert unsupported_keywords(schema) == []
    assert schema["properties"]["version"] == {"const": 2}


def test_sample_header_validates() -> None:
    header = sample_header()
    assert Validator(_schema()).errors(header) == []
    assert cross_field_errors(header) == []


def _mutations() -> list[tuple[str, Any]]:
    def drop(key):
        return lambda h: h.pop(key)

    def first_asset(h):
        return next(iter(h["assets"].values()))

    def upstream_asset(h):
        return next(a for a in h["assets"].values() if a["license"] != "Apache-2.0")

    return [
        ("v1 header", lambda h: h.update(version=1)),
        ("no provenance", drop("provenance")),
        ("no meta", drop("meta")),
        ("unknown top-level key", lambda h: h.update(qpos_bin="x.qpos.bin")),
        ("bin path with a directory", lambda h: h.update(bin="../x.bin")),
        ("layout changed", lambda h: h.update(layout=["bodies", "frames", "pose"])),
        ("float fps", lambda h: h.update(fps=29.97)),
        ("asset id not hex", lambda h: h["assets"].update({"Gripper": first_asset(h)})),
        ("asset uri elsewhere", lambda h: first_asset(h).update(uri="data/assets/x.glb")),
        ("asset sha uppercase", lambda h: first_asset(h).update(sha256="A" * 64)),
        ("asset role unknown", lambda h: first_asset(h).update(role="finger")),
        ("asset extra key", lambda h: first_asset(h).update(local_path="/tmp/x.glb")),
        ("third-party asset without changes", lambda h: upstream_asset(h).update(changes=[])),
        ("upstream source without sha", lambda h: upstream_asset(h)["source"].pop("sha256")),
        (
            "generated source without generator",
            lambda h: h["assets"][h["bodies"][1]["visuals"][0]["asset"]]["source"].pop("generator"),
        ),
        ("mirrored visual", lambda h: h["bodies"][2]["visuals"][0].update(scale=[1.0, -1.0, 1.0])),
        ("visual without scale", lambda h: h["bodies"][1]["visuals"][0].pop("scale")),
        ("quat with 3 terms", lambda h: h["bodies"][1]["visuals"][0].update(quat_xyzw=[0.0, 0.0, 1.0])),
        ("body with nothing to draw", lambda h: h["bodies"][0].update(visuals=[])),
        ("meta without part_id", lambda h: h["meta"].pop("part_id")),
        ("geometry_id not a sha", lambda h: h["meta"].update(geometry_id="abc")),
        ("provenance from another engine", lambda h: h["provenance"].update(engine="isaac")),
        ("provenance product_mode missing", lambda h: h["provenance"].pop("product_mode")),
        ("provenance bad commit", lambda h: h["provenance"].update(code_version="main")),
        ("overlay channel unknown", lambda h: h["overlays"]["channels"].append("force_n")),
        ("overlay channel repeated", lambda h: h["overlays"]["channels"].append("hold_s")),
        ("overlay float32", lambda h: h["overlays"].update(dtype="float32")),
    ]


@pytest.mark.parametrize("name,mutate", _mutations(), ids=[m[0] for m in _mutations()])
def test_schema_rejects_bad_headers(name: str, mutate: Any) -> None:
    header = copy.deepcopy(sample_header())
    mutate(header)
    assert Validator(_schema()).errors(header), name


def test_geometry_id_may_be_null_for_v1_designs() -> None:
    header = sample_header()
    header["meta"].update(design_space="v1", design_id="stock", geometry_id=None)
    assert Validator(_schema()).errors(header) == []


def test_schema_cross_check_with_jsonschema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator
    validator.check_schema(_schema())
    validator(_schema()).validate(sample_header())


# ----------------------------------------------------------------------------- asset manifest


def _manifest_example() -> dict[str, Any]:
    blocks = re.findall(r"```json\n(.*?)```", MANIFEST_DOC.read_text(encoding="utf-8"), flags=re.DOTALL)
    return json.loads(blocks[-1])


def test_asset_manifest_example_matches_the_schema_entry() -> None:
    manifest = _manifest_example()
    assert manifest["format"] == "rtt-asset-manifest" and manifest["version"] == 1
    schema = _schema()
    validator = Validator(schema)
    assert manifest["assets"]
    for aid, entry in manifest["assets"].items():
        assert validator.errors(aid, schema["$defs"]["asset_id"]) == []
        assert validator.errors(entry, schema["$defs"]["asset"]) == [], aid
        assert entry["uri"] == f"assets/{aid}.glb" and entry["sha256"].startswith(aid)
