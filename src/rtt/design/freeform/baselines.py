"""Baselines and entrants for the v2 design space (``configs/entrants_v2.yaml``).

Most entrants are plain points of the recipe space (``flat``, the reference entrants
``ref_wrap`` and ``ref_hook``); their design id is the recipe's own, so a search arm that
proposes the same recipe is recognised as the same design. Two are wrappers that sit
outside the search space:

* ``imprint_configurator``: a conform pad with straight walls (draft 0) and 0.4 mm
  clearance whose depth is set per part, the way a commercial custom-jaw configurator cuts an
  imprint: ``max(4 mm, min(40% of the part's half-extent along the closing axis in the grip
  band, stroke cap))``;
* ``imprint_swept``: the same with a +-2 mm sweep of the part along x (an
  uncertainty-integrating imprint), which the search space does not offer.

``stock`` is the stock gripper's own clamp blade. Its collision hulls are a fixed asset
produced by the render track (convex decomposition of the real blade, committed with its
sha256). Until that file exists, or when it cannot be read, the stock entrant falls back to
the v1 box approximation (``configs/entrants.yaml`` entry ``stock``), carried in
``FingerPair.boxes`` so the scene can use box colliders.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rtt.design.freeform.schema import FingerRecipe

REPO = Path(__file__).resolve().parents[4]
DEFAULT_ENTRANTS_PATH = REPO / "configs" / "entrants_v2.yaml"
STOCK_GRIP_FRAC = 0.5
STOCK_PAD_HEIGHT = 0.022  # m; the stock blade's tip pad

BaselineKind = Literal["recipe", "stock"]
DepthRule = Literal["fixed", "configurator"]


@dataclass(frozen=True)
class Baseline:
    """An entrant: a recipe (possibly wrapped) or the stock blade."""

    name: str
    kind: BaselineKind = "recipe"
    recipe: FingerRecipe | None = None
    sweep_mm: float = 0.0
    depth_rule: DepthRule = "fixed"
    round0: bool = True
    reference: bool = False
    hulls: str | None = None  # stock only: committed blade hulls (.npz), relative to the repo
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("recipe", "stock"):
            raise ValueError(f"baseline kind must be 'recipe' or 'stock', got {self.kind!r}")
        if self.kind == "recipe" and not isinstance(self.recipe, FingerRecipe):
            raise ValueError(f"baseline {self.name!r} needs a recipe")
        if self.kind == "stock" and self.recipe is not None:
            raise ValueError("the stock baseline has no recipe")
        if self.depth_rule not in ("fixed", "configurator"):
            raise ValueError(f"unknown depth rule {self.depth_rule!r}")
        if self.sweep_mm < 0 or not math.isfinite(self.sweep_mm):
            raise ValueError("sweep_mm must be finite and >= 0")
        if (self.sweep_mm or self.depth_rule != "fixed") and (
            self.recipe is None or self.recipe.pad != "conform"
        ):
            raise ValueError("sweep and the configurator depth apply to conform pads only")

    @property
    def is_point(self) -> bool:
        """True when the entrant is a plain point of the recipe space."""
        return self.kind == "recipe" and self.sweep_mm == 0.0 and self.depth_rule == "fixed"

    def canonical_json(self) -> str:
        from rtt.design.freeform.canon import canonical_dict

        payload: dict[str, Any] = {"kind": self.kind}
        if self.kind == "recipe":
            assert self.recipe is not None
            if self.is_point:
                from rtt.design.freeform.canon import canonical_json

                return canonical_json(self.recipe)
            d = canonical_dict(self.recipe)
            if self.depth_rule == "configurator":
                d.pop("pad_depth_mm")
            payload.update(recipe=d, sweep_mm=round(self.sweep_mm, 1), depth_rule=self.depth_rule)
        else:
            payload.update(name="stock", hulls=self.hulls or "")
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def id_payload(self) -> str:
        return ("recipe:" if self.is_point else "baseline:") + self.canonical_json()

    def pair_spec(self) -> tuple[Any, tuple[str, ...]]:
        from dataclasses import replace

        from rtt.design.freeform.generate import spec_from_recipe

        if self.kind != "recipe":
            raise TypeError("the stock baseline is not built from a recipe")
        assert self.recipe is not None
        spec, repairs = spec_from_recipe(self.recipe)
        spec = replace(spec, sweep=self.sweep_mm / 1000.0, depth_rule=self.depth_rule)
        return spec, repairs


def baseline_from_dict(entry: Mapping[str, Any]) -> Baseline:
    data = dict(entry)
    name = str(data.pop("name"))
    kind = data.pop("kind", "recipe")
    recipe_raw = data.pop("recipe", None)
    recipe = FingerRecipe(name=name, **recipe_raw) if recipe_raw is not None else None
    allowed = {"sweep_mm", "depth_rule", "round0", "reference", "hulls", "rationale"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"entrant {name!r}: unknown keys {sorted(unknown)}")
    return Baseline(name=name, kind=kind, recipe=recipe, **data)


def load_entrants_v2(path: str | Path | None = None) -> list[Baseline]:
    """Every entrant in ``configs/entrants_v2.yaml`` (round-0 entrants and references)."""
    import yaml

    p = Path(path) if path is not None else DEFAULT_ENTRANTS_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    entries = raw.get("entrants", [])
    out = [baseline_from_dict(e) for e in entries]
    names = [b.name for b in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate entrant names in {p}: {names}")
    return out


# ----------------------------------------------------------------------------- stock


def _stock_hulls(baseline: Baseline) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    """(left, right) blade hulls in their jaw frames, or ``None`` when the asset is absent.

    The hull file (written once by the render track) holds ``left_NN`` and ``right_NN``
    vertex arrays in the jaw body frames of :mod:`rtt.design.freeform.types`, plus a ``meta``
    record; a file with left hulls only gets the mirror image for the right jaw.
    """
    if not baseline.hulls:
        return None
    path = REPO / baseline.hulls
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        keys = sorted(k for k in data.files if k != "meta")
        left = [np.asarray(data[k], dtype=np.float64) for k in keys if k.startswith("left")]
        right = [np.asarray(data[k], dtype=np.float64) for k in keys if k.startswith("right")]
        if not left:
            left = [np.asarray(data[k], dtype=np.float64) for k in keys if k.startswith("hull")]
    if not left:
        return None
    if not right:
        right = [p * np.array([1.0, -1.0, 1.0]) for p in left]
    return left, right


def _stock_jaw_geometry(baseline: Baseline) -> tuple[Any, Any] | None:
    """The render track's stock jaws (blade mesh mass, inertia and print hash), if available."""
    if not baseline.hulls:
        return None
    try:
        from rtt.render.gripper import stock_jaws
    except ImportError:
        return None
    try:
        return stock_jaws(REPO / baseline.hulls)
    except Exception:  # the vendored blade meshes are absent: use the hull union instead
        return None


def _v1_stock_pieces() -> tuple[list[np.ndarray], tuple[Any, ...]]:
    """The v1 box approximation of the stock finger as 8-corner pieces (left jaw frame)."""
    from rtt.design.generate import box_corners, generate_finger, is_parked
    from rtt.design.schema import load_entrants

    v1 = {d.name: d for d in load_entrants(REPO / "configs" / "entrants.yaml")}["stock"]
    geom = generate_finger(v1)
    pieces = [
        np.round(np.asarray(box_corners(b), dtype=np.float64), 7) for b in geom.boxes if not is_parked(b)
    ]
    return pieces, tuple(geom.boxes)


def stock_seat(
    part_g: Any, jaw_g: Any, xs: np.ndarray, tip: float, reach: float, height: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """(Y0 contribution, part outline, blade front) for one blade against one side of a part.

    Row by row over the heights both reach, the blade (largest ``-y`` of its mirror image =
    its innermost surface) is compared with the part's outline; the coupled jaws stop where
    the first row touches: ``Y0 = max(part - front)``.
    """
    from rtt.design.freeform.outline import band_rows, cell_outline_rows

    root = tip + reach
    lo, hi = tip, min(height, root)
    rows = np.unique(
        np.concatenate([band_rows(part_g, lo, hi), band_rows(jaw_g, lo - root, hi - root) + root])
    )
    p_rows = cell_outline_rows(part_g, xs, rows)
    f_rows = -cell_outline_rows(jaw_g, xs, rows - root)
    diff = np.where(np.isfinite(p_rows) & np.isfinite(f_rows), p_rows - f_rows, -np.inf)
    return float(diff.max()), p_rows.max(axis=1), f_rows.min(axis=1)


def build_stock_pair(baseline: Baseline, part: Any, task: Any) -> Any:
    """The stock blade (or its box approximation) as a :class:`FingerPair` for ``part``."""
    import time
    from types import SimpleNamespace

    import manifold3d as m3d

    from rtt.design.freeform.export import inertia6, mass_properties, stl_sha256
    from rtt.design.freeform.hashing import design_id, geometry_hash
    from rtt.design.freeform.outline import PartGeometry, part_geometry
    from rtt.design.freeform.predict import capture_report
    from rtt.design.freeform.types import PLA_DENSITY, FingerPair, JawGeometry
    from rtt.design.freeform.validate import InvalidDesign

    t0 = time.perf_counter()
    hulls = _stock_hulls(baseline)
    boxes = None
    notes = []
    if hulls is None:
        left, boxes = _v1_stock_pieces()
        right = [p * np.array([1.0, -1.0, 1.0]) for p in left]
        notes.append("stock blade hulls not found: using the v1 box approximation")
    else:
        left, right = hulls
    part_id = str(getattr(part, "part_id", None) or part.name)
    g, gm = part_geometry(part)
    h = float(g.height)
    c = task.controller
    reach = float(-min(p[:, 2].min() for p in left))
    tip = max(
        float(c.min_tip_clearance), STOCK_GRIP_FRAC * h - STOCK_PAD_HEIGHT / 2.0, h + c.palm_clearance - reach
    )
    tip = float(np.round(tip, 7))
    ohg = float(task.gripper.open_half_gap)
    mirror = np.array([1.0, -1.0, 1.0])
    # in each jaw's build convention its innermost surface is the largest y of its mirror image
    jaw_left = PartGeometry.from_pieces(tuple(p * mirror for p in left), 0.0)
    jaw_right = PartGeometry.from_pieces(tuple(np.asarray(p) for p in right), 0.0)
    width = 2.0 * max(float(np.abs(p[:, 0]).max()) for p in left)
    n = round(width / 1e-4)
    xs = np.round(-width / 2.0 + 1e-4 * np.arange(n + 1), 7)
    sides = []
    contacts = []
    for part_g, jaw_g in ((g, jaw_left), (gm, jaw_right)):
        contact, p, front = stock_seat(part_g, jaw_g, xs, tip, reach, h)
        if not math.isfinite(contact):
            raise InvalidDesign("outline_empty", "the stock blade does not reach the part", part_id)
        contacts.append(contact)
        sides.append((p, front))
    # The stock jaws are not seat-aligned: on a part that is not symmetric across the closing
    # axis the first jaw to touch pushes it until the other touches too. q_seat is that
    # closure (both jaws touching); the part ends up off by half the difference.
    y0 = float(np.round(0.5 * (contacts[0] + contacts[1]), 7))
    if abs(contacts[0] - contacts[1]) > 1e-4:  # more than 0.05 mm off-centre
        notes.append(
            f"stock jaws touch {abs(contacts[0] - contacts[1]) * 500.0:.2f} mm off-centre on {part_id}: "
            "the part is pushed until both touch"
        )
    q_seat = float(np.round(ohg - y0, 7))
    plan = SimpleNamespace(
        left=_StockJaw(xs, sides[0][0], xs, np.where(np.isfinite(sides[0][1]), sides[0][1], 1.0) + y0),
        right=_StockJaw(xs, sides[1][0], xs, np.where(np.isfinite(sides[1][1]), sides[1][1], 1.0) + y0),
    )
    capture = capture_report(plan, mu=float(task.physics.friction))
    jaws = []
    official = _stock_jaw_geometry(baseline) if hulls is not None else None
    if official is not None:
        jaws = list(official)
        notes.append("stock blade: hull pieces with the blade mesh's mass properties")
    for pieces in () if official is not None else (left, right):
        hull_m = [m3d.Manifold.hull_points(p * 1000.0) for p in pieces]
        solid = m3d.Manifold.batch_boolean(hull_m, m3d.OpType.Add).to_mesh64()
        verts = np.asarray(solid.vert_properties)[:, :3] / 1000.0
        faces = np.asarray(solid.tri_verts, dtype=np.int64)
        mass, com, inertia = mass_properties(verts, faces, PLA_DENSITY)
        jaws.append(
            JawGeometry(
                pieces=tuple(pieces),
                mass=mass,
                com=com,
                inertia6=inertia6(inertia),
                print_mesh_sha=stl_sha256(verts, faces),
            )
        )
    return FingerPair(
        left=jaws[0],
        right=jaws[1],
        tip_height=tip,
        finger_length=reach,
        q_seat=q_seat,
        design_id=design_id(baseline),
        part_id=part_id,
        geometry_hash=geometry_hash(left, right, tip, q_seat),
        capture=capture,
        build_s=time.perf_counter() - t0,
        auto_repairs=tuple(notes),
        boxes=boxes,
    )


@dataclass(frozen=True, eq=False)
class _StockJaw:
    xs: np.ndarray
    p: np.ndarray
    xk: np.ndarray
    yk: np.ndarray


def build_entrant(baseline: Baseline, part: Any, task: Any) -> Any:
    """Any entrant against one part: the stock path or the generator."""
    if baseline.kind == "stock":
        return build_stock_pair(baseline, part, task)
    from rtt.design.freeform.generate import build_pair

    return build_pair(baseline, part, task)


def round0_entrants(entrants: Sequence[Baseline]) -> list[Baseline]:
    return [b for b in entrants if b.round0 and not b.reference]


def reference_entrants(entrants: Sequence[Baseline]) -> list[Baseline]:
    return [b for b in entrants if b.reference]
