"""Content ids for v2 designs and their compiled geometry.

* ``design_id`` = sha256 of the generator version and the canonical recipe (or baseline).
* ``geometry_hash`` = sha256 of the quantised convex pieces of both jaws of one
  (design, part) build, plus the two scalars that place them (tip height and seat closure).
  The pieces are built analytically, so the hash does not depend on manifold3d.
* ``geometry_id`` = sha256 over the training parts' geometry hashes. A later design with
  the same ``geometry_id`` is a ``duplicate_geometry`` in every arm.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

#: Bump whenever generated geometry changes for an unchanged recipe.
GENERATOR_VERSION = "rtt-freeform-gen/1"

#: Vertices are rounded to this many decimals of a metre (1e-7 m) before hashing.
VERTEX_DECIMALS = 7


def _recipe_json(design: object) -> str:
    from rtt.design.freeform.schema import FingerRecipe

    if isinstance(design, FingerRecipe):
        from rtt.design.freeform.canon import canonical_json

        return "recipe:" + canonical_json(design)
    method = getattr(design, "id_payload", None)
    if callable(method):
        return str(method())
    raise TypeError(f"cannot compute a design id for {type(design).__name__}")


def design_id(design: object) -> str:
    """sha256 (hex) of the generator version and the canonical recipe or baseline."""
    payload = f"{GENERATOR_VERSION}\n{_recipe_json(design)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def quantized_vertices(vertices: np.ndarray) -> np.ndarray:
    """Vertices as int64 multiples of 1e-7 m (what the geometry hash sees)."""
    return np.rint(np.asarray(vertices, dtype=np.float64) * 10**VERTEX_DECIMALS).astype("<i8")


def _update_pieces(h: hashlib._Hash, pieces: Sequence[np.ndarray]) -> None:
    h.update(np.int64(len(pieces)).tobytes())
    for p in pieces:
        q = quantized_vertices(p)
        h.update(np.asarray(q.shape, dtype="<i8").tobytes())
        h.update(np.ascontiguousarray(q).tobytes())


def geometry_hash(
    left: Sequence[np.ndarray], right: Sequence[np.ndarray], tip_height: float, q_seat: float
) -> str:
    """sha256 (hex) of both jaws' quantised pieces, the tip height and the seat closure."""
    h = hashlib.sha256(b"rtt-freeform-geometry/1\n")
    _update_pieces(h, left)
    _update_pieces(h, right)
    h.update(quantized_vertices(np.array([tip_height, q_seat])).tobytes())
    return h.hexdigest()


def geometry_id(hashes: Mapping[str, str] | Iterable[tuple[str, str]]) -> str:
    """sha256 (hex) over ``{part_id: geometry_hash}``, in part-id order."""
    items = sorted(dict(hashes).items())
    if not items:
        raise ValueError("geometry_id needs at least one part")
    h = hashlib.sha256(b"rtt-freeform-geometry-id/1\n")
    for part_id, value in items:
        h.update(f"{part_id}={value}\n".encode())
    return h.hexdigest()
