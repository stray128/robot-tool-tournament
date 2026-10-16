"""What the Grasp Engineer's tools call: part intake, physics evaluation, certificate, print pack.

The agent never judges a grasp itself. Every number it reports comes from a
:class:`PartBackend`, which runs MuJoCo (CPU, the executor of record):

* :class:`SimPartBackend` is built directly on the simulator: the STL is loaded with
  :func:`rtt.sim.mesh.load_mesh_object` (stable pose, convex decomposition), tested
  under :func:`part_suite` (the part centred, ``seeds`` jittered episodes per grasp
  height), scored by :class:`rtt.search.sim_adapter.SimEvaluator` (the same scoring rule
  as every search arm), certified by :func:`rtt.sim.certify.certify`, and exported with
  :func:`rtt.design.export.export_stl`.
* :class:`RttPartBackend` is a thin adapter over the "Bring Your Part" package
  (``rtt.part``: ``intake``, ``conditions_for_part``, ``certify_part``,
  ``make_print_pack``): several stable poses and grip candidates per part, the
  per-pose certificate, and the full print pack with a printable jaw.

:func:`default_part_backend` picks ``rtt.part`` when it is importable and complete, else
the direct backend; ``RTT_AGENT_PART_BACKEND=sim|part`` forces one.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
import os
import shutil
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Literal, Protocol, runtime_checkable

from rtt.design.schema import FingerDesign, GripperTask
from rtt.search.base import DesignScore
from rtt.store.manifest import write_json_atomic

__all__ = [
    "PART_FUNCTIONS",
    "PartBackend",
    "PartInfo",
    "PartNotFoundError",
    "RttPartBackend",
    "SimPartBackend",
    "default_part_backend",
    "part_suite",
    "to_jsonable",
]

Units = Literal["mm", "m"]
#: Functions the adapter needs from ``rtt.part``.
PART_FUNCTIONS: Final = ("intake", "save_part", "conditions_for_part", "certify_part", "make_print_pack")
#: Clearance the gripper needs beyond the part's width at full open (mm).
OPEN_MARGIN_MM: Final = 5.0


class PartNotFoundError(KeyError):
    """No part with this ID was uploaded."""


@dataclass(frozen=True)
class PartInfo:
    """What the agent (and the UI) may know about an uploaded part. No local paths."""

    part_id: str
    name: str
    source_sha256: str
    size_mm: tuple[float, float, float]
    """Bounding box in the grasp frame: x along the finger width, y across the jaw, z up."""
    grip_width_mm: float
    height_mm: float
    volume_cm3: float
    mass_g: float
    density_kg_m3: float
    max_opening_mm: float
    fits_gripper: bool
    n_convex_pieces: int
    stable_pose_probability: float
    conditions: tuple[str, ...]
    episodes_per_design: int
    backend: str
    notes: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["size_mm"] = list(self.size_mm)
        data["conditions"] = list(self.conditions)
        data["notes"] = list(self.notes)
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> PartInfo:
        known = {f.name for f in dataclasses.fields(cls)}
        values = {k: v for k, v in data.items() if k in known}
        values["size_mm"] = tuple(float(v) for v in values["size_mm"])
        values["conditions"] = tuple(values.get("conditions", ()))
        values["notes"] = tuple(values.get("notes", ()))
        return cls(**values)


@runtime_checkable
class PartBackend(Protocol):
    """Physics services behind the agent's tools (all results are simulator output)."""

    name: str

    def intake(
        self, stl_path: Path, *, part_id: str, name: str, units: Units = "mm", density: float | None = None
    ) -> PartInfo:
        """Load, rest, decompose and register an uploaded mesh."""
        ...

    def info(self, part_id: str) -> PartInfo:
        """The registered part (``PartNotFoundError`` if unknown)."""
        ...

    def gripper_task(self, part_id: str) -> GripperTask:
        """Validator constants for designs that must open around this part."""
        ...

    def evaluate(self, part_id: str, designs: Sequence[FingerDesign]) -> list[DesignScore]:
        """Simulate every design on every condition of the part (one score per design, in order)."""
        ...

    def certify(
        self, part_id: str, design: FingerDesign, *, score: DesignScore | None = None
    ) -> dict[str, Any]:
        """Robustness certificate as JSON (``score``, ``category_scores``, ``tolerance_mm``, ``tests``).

        ``score`` is the design's simulated result on this part, if known (lets the
        certificate pick the grip the design did best on)."""
        ...

    def print_pack(
        self,
        part_id: str,
        design: FingerDesign,
        out_dir: Path,
        *,
        results: Mapping[str, Any] | None = None,
        certificate: Mapping[str, Any] | None = None,
    ) -> list[Path]:
        """Write the print pack's files into ``out_dir`` and return them."""
        ...


# --------------------------------------------------------------------------- helpers


def to_jsonable(obj: Any) -> Any:
    """Best-effort JSON view of a result object (``to_dict``/``to_json``/pydantic/dataclass)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return obj.name
    for method in ("to_dict", "to_json", "model_dump"):
        fn = getattr(obj, method, None)
        if callable(fn):
            try:
                return to_jsonable(fn())
            except TypeError:
                continue
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj) if f.repr}
    return str(obj)


def part_suite(part_id: str, obj: Any, seeds: int = 3, base_seed: int = 0) -> Any:
    """Test conditions for an uploaded part: centred on the table, ``seeds`` jittered episodes.

    Each episode still gets the task's seed jitter (position, yaw and friction), and
    every grasp height of the controller is tried, so one design costs
    ``3 x seeds`` episodes per condition.
    """
    from rtt.sim.conditions import mesh_suite
    from rtt.sim.objects import Placement

    return mesh_suite(
        part_id, [obj], {"centred": Placement()}, seeds_per_condition=seeds, base_seed=base_seed
    )


@dataclass
class _Part:
    info: PartInfo
    obj: Any
    suite: Any
    gripper_task: GripperTask
    spec: Any = None
    lock: threading.Lock = field(default_factory=threading.Lock)


# --------------------------------------------------------------------------- direct backend


class SimPartBackend:
    """Part services built directly on :mod:`rtt.sim` (see the module docstring).

    Args:
        root: Directory for per-part files (``<root>/<part_id>/``).
        task: Simulator task (default ``configs/task_v1.yaml``).
        workers: CPU worker processes per evaluation (``None``: all CPUs).
        seeds: Jittered episodes per condition and grasp height.
        decomposition: Convex decomposition for uploads (``"auto"``: CoACD if installed).
    """

    name = "sim"

    def __init__(
        self,
        root: str | Path,
        *,
        task: Any | None = None,
        workers: int | None = None,
        seeds: int = 3,
        decomposition: str = "auto",
    ) -> None:
        from rtt.sim.task import default_task

        self.root = Path(root)
        self.task = task or default_task()
        self.workers = workers
        self.seeds = seeds
        self.decomposition = decomposition
        self._parts: dict[str, _Part] = {}
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(root={str(self.root)!r}, workers={self.workers!r})"

    # -- object + conditions (overridden by the rtt.part adapter)

    def _load(
        self, stl_path: Path, part_id: str, name: str, scale: float, density: float | None
    ) -> tuple[Any, Any]:
        from rtt.sim.mesh import load_mesh_object

        obj = load_mesh_object(
            stl_path,
            scale=scale,
            name=name,
            density=density,
            decomposition=self.decomposition,  # type: ignore[arg-type]
        )
        return obj, None

    def _suite(self, part_id: str, obj: Any, spec: Any) -> Any:
        return part_suite(part_id, obj, self.seeds)

    # -- registry

    def intake(
        self, stl_path: Path, *, part_id: str, name: str, units: Units = "mm", density: float | None = None
    ) -> PartInfo:
        if units not in ("mm", "m"):
            raise ValueError(f"units must be 'mm' or 'm', got {units!r}")
        scale = 0.001 if units == "mm" else 1.0
        part_dir = self.root / part_id
        stored = part_dir / "source.stl"
        part_dir.mkdir(parents=True, exist_ok=True)
        if Path(stl_path).resolve() != stored.resolve():
            shutil.copyfile(stl_path, stored)  # kept so the part can be re-loaded after a restart
        obj, spec = self._load(stored, part_id, name, scale, density)
        suite = self._suite(part_id, obj, spec)
        gripper_task = suite.gripper_task(self.task)
        info = self._describe(part_id, name, obj, suite, gripper_task, spec)
        write_json_atomic(
            part_dir / "part.json",
            {"info": info.to_json(), "intake": {"units": units, "density": density, "name": name}},
        )
        with self._lock:
            self._parts[part_id] = _Part(info, obj, suite, gripper_task, spec)
        return info

    def _describe(
        self, part_id: str, name: str, obj: Any, suite: Any, gripper_task: GripperTask, spec: Any = None
    ) -> PartInfo:
        from rtt.search.sim_adapter import describe_condition
        from rtt.sim import objects as objs

        size = tuple(round(1000.0 * s, 2) for s in obj.size)
        density = obj.density if obj.density is not None else self.task.physics.object_density
        volume = objs.volume(obj)
        grip_mm = round(1000.0 * objs.grip_width(obj), 2)
        opening = gripper_task.max_opening_mm
        fits = grip_mm + OPEN_MARGIN_MM <= opening
        notes = []
        if not fits:
            notes.append(
                f"the part is {grip_mm:g} mm across the jaw; the gripper opens {opening:g} mm, "
                f"so it needs at least {OPEN_MARGIN_MM:g} mm of clearance it does not have"
            )
        settings = len(self.task.controller.grasp_height_fracs)
        return PartInfo(
            part_id=part_id,
            name=name,
            source_sha256=getattr(obj, "source_sha256", ""),
            size_mm=size,  # type: ignore[arg-type]
            grip_width_mm=grip_mm,
            height_mm=round(1000.0 * objs.height(obj), 2),
            volume_cm3=round(volume * 1e6, 3),
            mass_g=round(volume * density * 1000.0, 2),
            density_kg_m3=float(density),
            max_opening_mm=opening,
            fits_gripper=fits,
            n_convex_pieces=len(getattr(obj, "parts", ())) or 1,
            stable_pose_probability=round(float(getattr(obj, "pose_probability", 1.0)), 4),
            conditions=tuple(describe_condition(c) for c in suite),
            episodes_per_design=sum(len(c.seeds) for c in suite) * settings,
            backend=self.name,
            notes=tuple(notes),
        )

    def _get(self, part_id: str) -> _Part:
        with self._lock:
            part = self._parts.get(part_id)
        if part is not None:
            return part
        return self._reload(part_id)

    def _reload(self, part_id: str) -> _Part:
        """Re-run intake from the stored upload (after a server restart)."""
        part_dir = self.root / part_id
        meta_path, stl_path = part_dir / "part.json", part_dir / "source.stl"
        if not meta_path.is_file() or not stl_path.is_file():
            raise PartNotFoundError(part_id)
        meta = json.loads(meta_path.read_text(encoding="utf-8")).get("intake", {})
        self.intake(
            stl_path,
            part_id=part_id,
            name=str(meta.get("name") or part_id),
            units=meta.get("units", "mm"),
            density=meta.get("density"),
        )
        with self._lock:
            return self._parts[part_id]

    def info(self, part_id: str) -> PartInfo:
        return self._get(part_id).info

    def gripper_task(self, part_id: str) -> GripperTask:
        return self._get(part_id).gripper_task

    # -- physics

    def evaluate(self, part_id: str, designs: Sequence[FingerDesign]) -> list[DesignScore]:
        from rtt.search.sim_adapter import SimEvaluator

        part = self._get(part_id)
        evaluator = SimEvaluator(
            part.suite, task=self.task, executor="cpu", workers=self.workers, gripper_task=part.gripper_task
        )
        return evaluator.evaluate(list(designs))

    def certify(
        self, part_id: str, design: FingerDesign, *, score: DesignScore | None = None
    ) -> dict[str, Any]:
        from rtt.sim.certify import certify

        part = self._get(part_id)
        return certify(design, part.obj, task=self.task, workers=self.workers).to_dict()

    def print_pack(
        self,
        part_id: str,
        design: FingerDesign,
        out_dir: Path,
        *,
        results: Mapping[str, Any] | None = None,
        certificate: Mapping[str, Any] | None = None,
    ) -> list[Path]:
        from rtt.design.export import export_stl
        from rtt.design.generate import generate_finger

        part = self._get(part_id)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        geom = generate_finger(design)
        files = [
            export_stl(geom, out / "finger_left.stl"),
            export_stl(geom, out / "finger_right.stl", mirror=True),
            write_json_atomic(out / "design.json", {"design_id": design.design_id(), **design.model_dump()}),
            write_json_atomic(out / "part.json", part.info.to_json()),
        ]
        if results is not None:
            files.append(write_json_atomic(out / "results.json", dict(results)))
        if certificate is not None:
            files.append(write_json_atomic(out / "certificate.json", dict(certificate)))
        readme = out / "README.txt"
        readme.write_text(_readme(design, part.info, results, certificate), encoding="utf-8")
        files.append(readme)
        return files


def _readme(
    design: FingerDesign,
    part: PartInfo,
    results: Mapping[str, Any] | None,
    certificate: Mapping[str, Any] | None,
) -> str:
    lines = [
        f"Print pack: finger design {design.design_id()} for part {part.name} ({part.part_id})",
        "",
        "finger_left.stl / finger_right.stl: one finger each side, millimetres, PLA.",
        "design.json: the 10 design parameters. part.json: the part as simulated.",
        "",
        "Physics (MuJoCo, CPU): lift 100 mm and hold 2 s, every grasp height, jittered placement.",
    ]
    if results is not None:
        lines.append(f"Simulated success: {100.0 * float(results.get('success', 0.0)):.0f}%")
    if certificate is not None:
        lines.append(f"Robustness certificate score: {100.0 * float(certificate.get('score', 0.0)):.0f}%")
    lines += [
        "",
        "Generated in product mode by the Grasp Engineer agent. Not part of the pre-registered result.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- rtt.part adapter


class RttPartBackend(SimPartBackend):
    """Adapter over ``rtt.part`` ("Bring Your Part"); scoring stays :class:`SimEvaluator`.

    * intake: :func:`rtt.part.intake` (units, repair, mass, up to ``top_k`` stable poses,
      CoACD pieces, grip candidates), saved with :func:`rtt.part.save_part`;
    * conditions: :func:`rtt.part.conditions_for_part` (poses x grip candidates x
      centred/sloppy placements), ``seeds`` jittered episodes each (default 2, so a
      verdict rests on at least 4 episodes per grasp height);
    * certificate: :func:`rtt.part.certify_part` on every kept pose, at the grip
      candidate the design scored best on;
    * print pack: :func:`rtt.part.make_print_pack` (fingers, printable jaw, certificate
      HTML, manifest) from a run directory assembled here: the part, the design frozen
      with :func:`rtt.search.select.freeze_top1` from its simulated score, and the
      certificate.
    """

    name = "rtt.part"

    def __init__(
        self, root: str | Path, *, module: ModuleType | Any | None = None, seeds: int = 2, **kwargs: Any
    ) -> None:
        super().__init__(root, seeds=seeds, **kwargs)
        self.module = module if module is not None else importlib.import_module("rtt.part")
        missing = [f for f in PART_FUNCTIONS if not callable(getattr(self.module, f, None))]
        if missing:
            raise ImportError(f"rtt.part lacks {', '.join(missing)}")

    def _load(
        self, stl_path: Path, part_id: str, name: str, scale: float, density: float | None
    ) -> tuple[Any, Any]:
        part_dir = self.root / part_id
        spec = self.module.intake(
            stl_path,
            units="mm" if scale == 0.001 else "m",
            density=density,
            name=name,
            task=self.task,
            cache_dir=part_dir / "cache",
        )
        self.module.save_part(spec, part_dir / "part")
        return spec.primary.obj("narrow"), spec

    def _suite(self, part_id: str, obj: Any, spec: Any) -> Any:
        return self.module.conditions_for_part(spec, seeds_per_condition=self.seeds, name=part_id)

    def _describe(
        self, part_id: str, name: str, obj: Any, suite: Any, gripper_task: GripperTask, spec: Any = None
    ) -> PartInfo:
        info = super()._describe(part_id, name, obj, suite, gripper_task, spec)
        if spec is None:
            return info
        primary = spec.primary
        narrow = [c for c in primary.candidates if c.axis == "narrow"] or list(primary.candidates)
        notes = [*spec.warnings]
        for pose in spec.poses:
            notes.append(
                f"pose {pose.rank}: lands this way {100 * pose.probability:.0f}% of drops, "
                + " x ".join(f"{v:.0f}" for v in pose.extents_mm)
                + " mm; grasps: "
                + ", ".join(c.label for c in pose.candidates)
            )
        return dataclasses.replace(
            info,
            name=spec.name,
            source_sha256=spec.source_sha256,
            size_mm=tuple(round(float(v), 2) for v in primary.extents_mm),  # type: ignore[arg-type]
            grip_width_mm=round(float(narrow[0].width_mm), 2) if narrow else info.grip_width_mm,
            height_mm=round(float(primary.extents_mm[2]), 2),
            volume_cm3=round(float(spec.volume_cm3), 3),
            mass_g=round(float(spec.mass_g), 2),
            density_kg_m3=float(spec.density_kg_m3),
            fits_gripper=True,  # intake refuses parts with no pose that fits the opening
            n_convex_pieces=primary.n_pieces,
            stable_pose_probability=round(float(primary.probability), 4),
            notes=tuple(notes),
        )

    def certify(
        self, part_id: str, design: FingerDesign, *, score: DesignScore | None = None
    ) -> dict[str, Any]:
        part = self._get(part_id)
        cert = self.module.certify_part(
            design,
            part.spec,
            per_condition=dict(score.per_condition) if score is not None else None,
            task=self.task,
            workers=self.workers,
        )
        raw = cert.to_json()
        write_json_atomic(self.root / part_id / "certificates" / f"{design.design_id()}.json", raw)
        return certificate_summary(raw)

    def print_pack(
        self,
        part_id: str,
        design: FingerDesign,
        out_dir: Path,
        *,
        results: Mapping[str, Any] | None = None,
        certificate: Mapping[str, Any] | None = None,
    ) -> list[Path]:
        from rtt.search.base import Proposal, RunHistory
        from rtt.search.select import freeze_top1, write_frozen

        part = self._get(part_id)
        out = Path(out_dir)
        run = out / "_run"
        if results is None:
            score = self.evaluate(part_id, [design])[0]
        else:
            score = DesignScore.from_json({k: v for k, v in results.items() if k != "episodes"})
        history = RunHistory()
        history.add(0, Proposal(design, origin="grasp-engineer", hypothesis=design.rationale), score)
        write_frozen(freeze_top1("agent", 0, 0, history), run / "search" / "frozen")
        self.module.save_part(part.spec, run / "part")
        cert_path = self.root / part_id / "certificates" / f"{design.design_id()}.json"
        if not cert_path.is_file():
            self.certify(part_id, design, score=score)
        shutil.copyfile(cert_path, run / "certificate.json")
        self.module.make_print_pack(run, out_dir=out)
        pack = out / "print_pack"
        files = sorted(p for p in pack.iterdir() if p.is_file())
        if results is not None:
            files.append(write_json_atomic(pack / "results.json", dict(results)))
            _add_to_manifest(pack / "manifest.json", pack / "results.json")
        return files


def _add_to_manifest(manifest_path: Path, extra: Path) -> None:
    """List ``extra`` (added after ``rtt.part`` wrote the pack) in the pack manifest.

    The manifest promises a checksum for every file of the pack, and the product-mode
    label goes on the pack itself, not only on the added results file.
    """
    import hashlib

    from rtt.agent.recorder import PRODUCT_MODE_LABEL

    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = extra.read_bytes()
    files = manifest.setdefault("files", {})
    if isinstance(files, dict):
        files[extra.name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest["label"] = PRODUCT_MODE_LABEL
    write_json_atomic(manifest_path, manifest)


def certificate_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The agent's view of an ``rtt-part-certificate``: the shape :func:`rtt.sim.certify.certify` uses.

    Category scores and tests are pooled over poses, weighted by how often the part
    lands in each; a test name is prefixed with its pose (``pose0/nominal``); the
    tolerance per axis is the smallest over poses (``None`` if any pose failed nominal).
    """
    poses = list(raw.get("poses") or [])
    total = sum(float(p.get("probability", 0.0)) for p in poses) or 1.0
    categories: dict[str, float] = {}
    tolerance: dict[str, float | None] = {"x": None, "y": None}
    tests: list[dict[str, Any]] = []
    for i, pose in enumerate(poses):
        weight = float(pose.get("probability", 0.0)) / total
        for cat, value in dict(pose.get("category_scores") or {}).items():
            categories[cat] = categories.get(cat, 0.0) + weight * float(value)
        for t in pose.get("tests") or []:
            tests.append({**t, "name": f"pose{pose.get('pose_rank', i)}/{t.get('name')}"})
    for axis in ("x", "y"):
        values = [(p.get("tolerance_mm") or {}).get(axis) for p in poses]
        tolerance[axis] = None if not values or None in values else min(float(v) for v in values)
    return {
        "design_id": raw.get("design_id"),
        "score": float(raw.get("score") or 0.0),
        "category_scores": categories,
        "tolerance_mm": tolerance,
        "passed": f"{raw.get('passed')}/{raw.get('n_tests')}",
        "poses": [
            {"pose": p.get("pose_rank"), "probability": p.get("probability"), "score": p.get("score")}
            for p in poses
        ],
        "tests": tests,
    }


def default_part_backend(root: str | Path, **kwargs: Any) -> SimPartBackend:
    """``rtt.part`` when available and complete, else :class:`SimPartBackend`.

    ``RTT_AGENT_PART_BACKEND=sim`` or ``=part`` forces a choice (``part`` raises if the
    package is missing).
    """
    choice = os.environ.get("RTT_AGENT_PART_BACKEND", "").strip().lower()
    if choice == "sim":
        return SimPartBackend(root, **kwargs)
    try:
        return RttPartBackend(root, **kwargs)
    except ImportError:
        if choice == "part":
            raise
        return SimPartBackend(root, **kwargs)
