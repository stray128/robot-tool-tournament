"""Uploaded parts for the Studio: ``upload:<id>`` refs, resolved through the existing part intake.

An uploaded STL goes through :func:`rtt.part.intake.intake` (units, repair, mass, resting
poses, convex pieces), the same intake "Bring your part" uses, and is saved with
:func:`rtt.part.intake.save_part` as ``<root>/<id>/part/``. That is also the layout the API's
part backend writes (``<agent root>/parts/<part_id>/part/``), so one registry class serves the
command line (ids ``stl-<sha256[:12]>``, content addressed) and the API (ids ``part-<hex>``).

* :meth:`UploadRegistry.add_stl` runs the intake (or reuses a stored one) and returns an
  :class:`UploadInfo`: what the Task Engineer is told about the part.
* :meth:`UploadRegistry.resolve` gives the :class:`rtt.parts.types.PartSpec` the ``part``
  template builds: the most likely resting pose, its convex pieces, family ``upload``.
* :func:`install_upload_resolver` registers the registries with
  :func:`rtt.studio.templates.part.register_part_resolver` for the ``upload`` scheme.

The intake keeps its own limits: a part must fit the 79 mm parallel-gripper opening in some
resting pose and weigh at most 2 kg (:mod:`rtt.part.intake`), or it is refused with a message.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from rtt.studio.types import get_template

if TYPE_CHECKING:
    from rtt.parts.types import PartSpec

__all__ = ["UPLOAD_ID", "UploadInfo", "UploadRegistry", "install_upload_resolver", "installed_roots"]

#: Ids a registry hands out or accepts (the ``upload:`` ref's key, see ``rtt.studio.types``).
UPLOAD_ID: Final = re.compile(r"^(stl-[0-9a-f]{12}|part-[0-9a-f]{12})$")


@dataclass(frozen=True)
class UploadInfo:
    """What the Task Engineer and the intake page know about an uploaded part (no paths)."""

    upload_id: str
    name: str
    size_mm: tuple[float, float, float]  # the most likely resting pose: x, y, z (z up)
    mass_g: float
    material: str
    n_pieces: int
    notes: tuple[str, ...] = ()

    @property
    def ref(self) -> str:
        return f"upload:{self.upload_id}"

    def part_params(self) -> dict[str, Any]:
        """``part`` template params for this upload (sizes and mass clamped into the template's bounds)."""
        t = get_template("part")
        bounds = {p.name: p for p in t.params}

        def clamp(name: str, value: float) -> float:
            p = bounds[name]
            assert p.low is not None and p.high is not None
            return round(min(max(float(value), p.low), p.high), 1)

        x, y, z = self.size_mm
        return {
            "shape": "mesh",
            "size_x_mm": clamp("size_x_mm", x),
            "size_y_mm": clamp("size_y_mm", y),
            "size_z_mm": clamp("size_z_mm", z),
            "mass_g": clamp("mass_g", self.mass_g),
        }

    def prompt_line(self) -> str:
        p = self.part_params()
        return (
            f"- {self.ref}: {self.name}, {p['size_x_mm']:g} x {p['size_y_mm']:g} x {p['size_z_mm']:g} mm "
            f"(x, y, z as it rests on the table), {p['mass_g']:g} g of {self.material.replace('_', ' ')}. "
            f"Model it as one part with params {json.dumps(p)} and ref {self.ref}."
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "upload_id": self.upload_id,
            "ref": self.ref,
            "name": self.name,
            "size_mm": list(self.size_mm),
            "mass_g": self.mass_g,
            "material": self.material,
            "n_pieces": self.n_pieces,
            "notes": list(self.notes),
        }


def _check_id(upload_id: str) -> str:
    if not isinstance(upload_id, str) or not UPLOAD_ID.match(upload_id):
        raise ValueError(f"malformed upload id {upload_id!r}")
    return upload_id


class UploadRegistry:
    """Uploaded parts under ``root`` (``<root>/<id>/part/`` as written by ``save_part``)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()
        self._specs: dict[str, PartSpec] = {}

    def __repr__(self) -> str:
        return f"UploadRegistry(root={str(self.root)!r})"

    def part_dir(self, upload_id: str) -> Path:
        return self.root / _check_id(upload_id) / "part"

    def __contains__(self, upload_id: object) -> bool:
        return (
            isinstance(upload_id, str)
            and UPLOAD_ID.match(upload_id) is not None
            and (self.part_dir(upload_id) / "part.json").is_file()
        )

    def add_stl(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        units: Literal["auto", "mm", "cm", "m", "in"] = "mm",
        density: float | None = None,
        decomposition: Literal["auto", "coacd", "hull"] = "auto",
    ) -> UploadInfo:
        """Run the part intake on an STL (or reuse the stored result for the same bytes)."""
        from rtt.part.intake import intake, save_part

        source = Path(path)
        upload_id = "stl-" + hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        if upload_id in self:
            return self.info(upload_id)
        spec = intake(
            source,
            units=units,
            density=density,
            name=name or source.stem,
            decomposition=decomposition,
            cache_dir=self.root / upload_id / "cache",
        )
        save_part(spec, self.part_dir(upload_id))
        return self.info(upload_id)

    def info(self, upload_id: str) -> UploadInfo:
        """What the prompt says about the upload (read from ``part.json`` only, no meshes)."""
        path = self.part_dir(upload_id) / "part.json"
        if not path.is_file():
            raise KeyError(upload_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        pose = data["poses"][0]
        extents = tuple(round(float(v), 1) for v in pose["extents_mm"])
        n_pieces = int(pose.get("n_pieces") or 0)
        return UploadInfo(
            upload_id=upload_id,
            name=str(data.get("name") or upload_id),
            size_mm=extents,  # type: ignore[arg-type]
            mass_g=round(float(data["mass_g"]), 1),
            material=str(data.get("material") or "plastic"),
            n_pieces=n_pieces,
            notes=tuple(str(w) for w in data.get("warnings") or ()),
        )

    def resolve(self, upload_id: str) -> PartSpec:
        """The part the ``part`` template builds for ``upload:<upload_id>``."""
        import dataclasses

        from rtt.part.intake import load_part
        from rtt.parts.library import part_from_mesh_object

        with self._lock:
            cached = self._specs.get(upload_id)
        if cached is not None:
            return cached
        directory = self.part_dir(upload_id)
        if not (directory / "part.json").is_file():
            raise KeyError(upload_id)
        held = load_part(directory)
        part = dataclasses.replace(
            part_from_mesh_object(held.primary.obj("narrow"), family="upload"), name=upload_id
        )
        with self._lock:
            self._specs[upload_id] = part
        return part


_ROOTS: list[UploadRegistry] = []
_ROOTS_LOCK = threading.Lock()


def installed_roots() -> tuple[UploadRegistry, ...]:
    with _ROOTS_LOCK:
        return tuple(_ROOTS)


def _resolve(upload_id: str) -> PartSpec:
    from rtt.studio.templates import TemplateUnavailable

    for registry in installed_roots():
        try:
            if upload_id in registry:
                return registry.resolve(upload_id)
        except ValueError:
            break
    raise TemplateUnavailable(f"no uploaded part {upload_id!r} in the Studio's upload folders")


def install_upload_resolver(*registries: UploadRegistry) -> None:
    """Resolve ``upload:<id>`` refs from ``registries`` (added to any installed before)."""
    from rtt.studio.templates.part import register_part_resolver

    with _ROOTS_LOCK:
        for r in registries:
            if all(r.root.resolve() != x.root.resolve() for x in _ROOTS):
                _ROOTS.append(r)
    register_part_resolver("upload", _resolve)
