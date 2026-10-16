"""Bring Your Part: upload a mesh, search fingers for it, certify them, print them.

* :mod:`rtt.part.intake` - :func:`intake` validates an STL/OBJ/PLY upload into a
  :class:`PartSpec` (units, repair, mass, stable poses, convex pieces, grip candidates).
* :mod:`rtt.part.conditions` - :func:`conditions_for_part`: the part's training suite.
* :mod:`rtt.part.search` - :func:`run_part_search` (the existing runner on that suite),
  :func:`certify_part` (the existing certificate per pose) and :func:`run_part` (all of it).
* :mod:`rtt.part.mount` - a printable jaw: rod-carriage block fused with the finger.
* :mod:`rtt.part.pack` - :func:`make_print_pack`: STLs, certificate HTML, manifest, zip.
* :mod:`rtt.part.examples` - generic example parts (CC0) in ``assets/parts``.

Command line: ``python -m rtt.part run PART.stl --out DIR`` and ``python -m rtt.part pack DIR``.
"""

from rtt.part.conditions import conditions_for_part
from rtt.part.intake import GraspCandidate, PartError, PartPose, PartSpec, intake, load_part, save_part

__all__ = [
    "GraspCandidate",
    "PartError",
    "PartPose",
    "PartSpec",
    "certify_part",
    "conditions_for_part",
    "intake",
    "load_part",
    "make_print_pack",
    "run_part",
    "run_part_search",
    "save_part",
]


def __getattr__(name: str) -> object:
    if name in ("certify_part", "run_part", "run_part_search"):
        from rtt.part import search

        return getattr(search, name)
    if name == "make_print_pack":
        from rtt.part.pack import make_print_pack

        return make_print_pack
    raise AttributeError(f"module 'rtt.part' has no attribute {name!r}")
