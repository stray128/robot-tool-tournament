"""Test conditions: the training suite and the commit-reveal held-out suite.

A :class:`Condition` is one object at one placement, with the jitter seeds its
episodes use. A :class:`Suite` is an ordered set of conditions; every candidate
design is evaluated on the same conditions and seeds (common random numbers).

**Training suite** (:func:`training_suite`, ``configs/suites/train_v1.yaml``):
3 object families (box, cylinder, rod) x 2 sizes x 2 placements = 12 conditions,
in file order. Jitter seeds are derived from the suite's public ``base_seed`` and
the condition id with SHA-256, so they are identical on every machine.

**Held-out suite** (:func:`heldout_suite`): 100 conditions nobody has seen, drawn
from a secret seed that is passed in at run time and never derived from repository
contents. Commit-reveal protocol:

1. Before any search runs, create a secret with :func:`new_secret_seed`, keep it in
   a git-ignored file (or outside the repository), and commit only
   :func:`seed_commitment` of it (a SHA-256 hex digest) to the pre-registration.
2. The held-out job receives the file (e.g. ``--inject-file``), loads it with
   :func:`read_secret_seed` and calls :func:`heldout_suite` with the committed digest,
   which refuses a secret that does not match.
3. The secret is published with the results, so anyone can regenerate the suite and
   check it against the commitment.

Each held-out condition draws, from the secret alone:

* a family (stratified: 34 / 33 / 33 over box, cylinder, rod, shuffled);
* a size: 40 % interpolate between the family's training sizes (each dimension
  independently at 15-85 % of the way, so aspect ratios are new), 30 % extrapolate
  below the smallest training size by 15-30 %, 30 % above the largest by 15-30 % (one
  factor for all dimensions). The width across the jaw is capped at the gripper's
  maximum opening minus 5 mm, and a rod stays at least 1.5 diameters long;
* a material: one of the densities the family's training conditions use, uniformly
  (a size without its own density uses the task's ``physics.object_density``);
* a new placement: x, y uniform in +-80 mm, yaw uniform over a full turn, and a pose
  error unknown to the controller of up to +-3 mm (x, y) and +-8 deg;
* an off-centre grasp, mirroring training: with the probability that one of the
  family's training conditions is an off-centre grasp (``|err_x| >= 10 mm``), and only
  for parts at least 70 mm long, the x error becomes a grasp offset of 15-60 % of the
  half-length (at least 10 mm, and at least 15 mm short of the end), either side;
* ``seeds_per_condition`` (2) jitter seeds.

:func:`mesh_suite` builds a suite from mesh objects (uploaded parts, scanned objects).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import secrets
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rtt.design.schema import GripperTask
from rtt.sim import objects as objs
from rtt.sim.task import TaskConfig, default_task, gripper_task_for

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_SUITE = REPO_ROOT / "configs" / "suites" / "train_v1.yaml"
FAMILIES: tuple[objs.ObjectKind, ...] = ("box", "cylinder", "rod")

HELDOUT_NAME = "heldout_v1"
HELDOUT_CONDITIONS = 100
HELDOUT_SEEDS_PER_CONDITION = 2
HELDOUT_SIZE_MODES: tuple[tuple[str, float], ...] = (("interp", 0.4), ("down", 0.3), ("up", 0.3))
HELDOUT_INTERP = (0.15, 0.85)  # fraction of the way between the smallest and largest training size
HELDOUT_EXTRAP = (0.15, 0.30)  # relative size change beyond the training range
HELDOUT_XY = 0.08  # m, nominal placement range (+-)
HELDOUT_ERR_XY = 0.003  # m, pose error range (+-)
HELDOUT_ERR_YAW_DEG = 8.0  # deg, pose error range (+-)
HELDOUT_WIDTH_MARGIN = 0.005  # m, the jaw-axis width stays this far below the maximum opening
HANDLE_MIN_OFFSET = 0.010  # m: a training placement with |err_x| >= this is an off-centre grasp
HELDOUT_HANDLE_FRAC = (0.15, 0.60)  # off-centre grasp offset as a fraction of the half-length
HELDOUT_HANDLE_MIN_HALF = 0.035  # m: only parts with a half-length of at least this get one
HELDOUT_HANDLE_END_MARGIN = 0.015  # m: the grasp stays at least this far from the part's end
ROD_MIN_LENGTH_DIAMETERS = 1.5
MIN_SECRET_BYTES = 32
_HELDOUT_DOMAIN = b"rtt-heldout-v1\x00"


# --------------------------------------------------------------------------- data model


@dataclass(frozen=True)
class Condition:
    """One object at one placement, plus the jitter seeds of its episodes."""

    condition_id: str
    family: str  # box / cylinder / rod / mesh
    obj: objs.AnyObject
    placement: objs.Placement
    seeds: tuple[int, ...]
    size_label: str = ""
    placement_label: str = ""

    def describe(self) -> dict[str, Any]:
        """JSON-friendly description (sizes in metres, angles in degrees)."""
        p = self.placement
        out: dict[str, Any] = {
            "condition_id": self.condition_id,
            "family": self.family,
            "object": {"kind": self.obj.kind, "size_m": [round(s, 9) for s in self.obj.size]},
            "size_label": self.size_label,
            "placement_label": self.placement_label,
            "placement": {
                "x": round(p.x, 9),
                "y": round(p.y, 9),
                "yaw_deg": round(math.degrees(p.yaw), 9),
                "err_x": round(p.err_x, 9),
                "err_y": round(p.err_y, 9),
                "err_yaw_deg": round(math.degrees(p.err_yaw), 9),
            },
            "seeds": list(self.seeds),
        }
        if isinstance(self.obj, objs.MeshObject):
            out["object"].update(
                name=self.obj.name,
                source_sha256=self.obj.source_sha256,
                pose_index=self.obj.pose_index,
                decomposition=self.obj.decomposition,
                n_parts=len(self.obj.parts),
                geometry_sha256=self.obj.geometry_sha256,
            )
        if self.obj.density is not None:
            out["object"]["density"] = self.obj.density
        return out


@dataclass(frozen=True)
class Suite:
    """An ordered set of test conditions."""

    name: str
    conditions: tuple[Condition, ...]

    def __post_init__(self) -> None:
        ids = [c.condition_id for c in self.conditions]
        if len(set(ids)) != len(ids):
            raise ValueError(f"suite {self.name!r} has duplicate condition ids")
        if not self.conditions:
            raise ValueError(f"suite {self.name!r} has no conditions")

    def __len__(self) -> int:
        return len(self.conditions)

    def __iter__(self) -> Iterator[Condition]:
        return iter(self.conditions)

    @property
    def widest_grip_width(self) -> float:
        """Widest object across the jaw (m)."""
        return max(objs.grip_width(c.obj) for c in self.conditions)

    @property
    def n_episodes_per_setting(self) -> int:
        return sum(len(c.seeds) for c in self.conditions)

    def gripper_task(self, task: TaskConfig | None = None, **overrides: Any) -> GripperTask:
        """The design validator's task for this suite: the simulator's gripper plus the
        suite's widest object (rounded up to 0.1 mm), so every valid design opens around it."""
        widest_mm = math.ceil(self.widest_grip_width * 1e4 - 1e-6) / 10.0
        return gripper_task_for(task or default_task(), widest_training_object_mm=widest_mm, **overrides)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "conditions": [c.describe() for c in self.conditions]}

    def digest(self) -> str:
        """SHA-256 (hex) of :meth:`describe`, for manifests and pre-registration."""
        payload = json.dumps(self.describe(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class SuiteSpec:
    """A parsed suite file: families with named sizes, named placements, seeding.

    ``densities`` gives a size's own material density (kg/m^3) where the file sets one
    (``None``: the task's ``physics.object_density``). ``combos`` lists the
    ``(family, size label, placement label)`` triples of the suite, in order; ``None``
    means every family x size x placement.
    """

    name: str
    base_seed: int
    jitter_seeds_per_condition: int
    families: Mapping[str, Mapping[str, tuple[float, ...]]]
    placements: Mapping[str, objs.Placement]
    densities: Mapping[str, Mapping[str, float | None]] = dataclasses.field(default_factory=dict)
    combos: tuple[tuple[str, str, str], ...] | None = None

    def size_range(self, family: str) -> tuple[np.ndarray, np.ndarray]:
        """Element-wise smallest and largest training size of ``family``."""
        sizes = np.array(list(self.families[family].values()), dtype=np.float64)
        return sizes.min(axis=0), sizes.max(axis=0)

    def density(self, family: str, size_label: str) -> float | None:
        """The size's own density, or ``None`` for the task default."""
        return self.densities.get(family, {}).get(size_label)

    def family_densities(self, family: str) -> tuple[float | None, ...]:
        """Distinct densities used by ``family``'s conditions, in file order."""
        labels = [s for f, s, _ in self.triples() if f == family]
        return tuple(dict.fromkeys(self.density(family, s) for s in labels))

    def triples(self) -> tuple[tuple[str, str, str], ...]:
        """``(family, size label, placement label)`` of every condition, in suite order."""
        if self.combos is not None:
            return self.combos
        return tuple(
            (kind, size_label, place_label)
            for kind, sizes in self.families.items()
            for size_label in sizes
            for place_label in self.placements
        )


# --------------------------------------------------------------------------- training suite


def derive_seed(*parts: object) -> int:
    """A 63-bit episode seed from SHA-256 of the parts (identical on every platform)."""
    text = "\x1f".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") & (2**63 - 1)


def _placement_from_mapping(raw: Mapping[str, Any], where: str) -> objs.Placement:
    allowed = {"x", "y", "yaw_deg", "err_x_mm", "err_y_mm", "err_yaw_deg"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown placement keys {sorted(unknown)}; allowed: {sorted(allowed)}")
    return objs.Placement(
        x=float(raw.get("x", 0.0)),
        y=float(raw.get("y", 0.0)),
        yaw=math.radians(float(raw.get("yaw_deg", 0.0))),
        err_x=float(raw.get("err_x_mm", 0.0)) / 1000.0,
        err_y=float(raw.get("err_y_mm", 0.0)) / 1000.0,
        err_yaw=math.radians(float(raw.get("err_yaw_deg", 0.0))),
    )


def _size_entry(kind: str, label: str, entry: Any, materials: Mapping[str, float], where: str) -> tuple:
    """``(size, density)`` from a size entry: a list, or ``{size: [...], material|density: ...}``."""
    density: float | None = None
    if isinstance(entry, Mapping):
        allowed = {"size", "material", "density"}
        unknown = set(entry) - allowed
        if unknown:
            raise ValueError(
                f"{where}: {kind} {label!r}: unknown keys {sorted(unknown)}; allowed {sorted(allowed)}"
            )
        if "material" in entry and "density" in entry:
            raise ValueError(f"{where}: {kind} {label!r}: give a material or a density, not both")
        if "material" in entry:
            name = str(entry["material"])
            if name not in materials:
                raise ValueError(f"{where}: {kind} {label!r}: unknown material {name!r}")
            density = materials[name]
        elif "density" in entry:
            density = float(entry["density"])
        size = entry.get("size")
    else:
        size = entry
    if size is None:
        raise ValueError(f"{where}: {kind} {label!r} has no size")
    spec = objs.ObjectSpec(kind, tuple(float(v) for v in size), density)  # validates size and density
    return spec.size, spec.density


def suite_spec_from_dict(raw: Mapping[str, Any], where: str = "suite") -> SuiteSpec:
    """Validate and parse a suite mapping (see ``configs/suites/train_v1.yaml``).

    A size is ``[...]`` or ``{size: [...], material: <name>}`` (``materials`` maps names to
    densities in kg/m^3) or ``{size: [...], density: <kg/m^3>}``. ``conditions`` (optional)
    lists ``family/size/placement`` strings; without it the suite is the full cross product.
    """
    allowed = {
        "name",
        "base_seed",
        "jitter_seeds_per_condition",
        "families",
        "placements",
        "materials",
        "conditions",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    materials: dict[str, float] = {}
    for name, value in (raw.get("materials") or {}).items():
        if float(value) <= 0:
            raise ValueError(f"{where}: material {name!r} needs a positive density")
        materials[str(name)] = float(value)
    families: dict[str, dict[str, tuple[float, ...]]] = {}
    densities: dict[str, dict[str, float | None]] = {}
    for kind, sizes in (raw.get("families") or {}).items():
        if kind not in FAMILIES:
            raise ValueError(f"{where}: unknown family {kind!r}; use one of {FAMILIES}")
        if not sizes:
            raise ValueError(f"{where}: family {kind!r} has no sizes")
        families[kind] = {}
        densities[kind] = {}
        for label, entry in sizes.items():
            size, density = _size_entry(kind, str(label), entry, materials, where)
            families[kind][str(label)] = size
            densities[kind][str(label)] = density
    placements = {
        str(label): _placement_from_mapping(p or {}, f"{where}: placement {label!r}")
        for label, p in (raw.get("placements") or {}).items()
    }
    if not families or not placements:
        raise ValueError(f"{where}: needs at least one family and one placement")
    n_seeds = int(raw.get("jitter_seeds_per_condition", 1))
    if n_seeds < 1:
        raise ValueError(f"{where}: jitter_seeds_per_condition must be >= 1")
    combos: tuple[tuple[str, str, str], ...] | None = None
    if raw.get("conditions") is not None:
        parsed = []
        for item in raw["conditions"]:
            parts = str(item).split("/")
            if len(parts) != 3:
                raise ValueError(f"{where}: condition {item!r} must be family/size/placement")
            kind, size_label, place_label = parts
            if kind not in families or size_label not in families[kind]:
                raise ValueError(f"{where}: condition {item!r}: unknown family or size")
            if place_label not in placements:
                raise ValueError(f"{where}: condition {item!r}: unknown placement")
            parsed.append((kind, size_label, place_label))
        if not parsed or len(set(parsed)) != len(parsed):
            raise ValueError(f"{where}: conditions must be a non-empty list without duplicates")
        unused = {(k, s) for k, sizes in families.items() for s in sizes} - {(k, s) for k, s, _ in parsed}
        if unused:
            raise ValueError(
                f"{where}: sizes not used by any condition: {sorted('/'.join(u) for u in unused)}"
            )
        combos = tuple(parsed)
    return SuiteSpec(
        name=str(raw.get("name", "suite")),
        base_seed=int(raw.get("base_seed", 0)),
        jitter_seeds_per_condition=n_seeds,
        families=families,
        placements=placements,
        densities=densities,
        combos=combos,
    )


def load_suite_spec(path: str | Path = DEFAULT_TRAIN_SUITE) -> SuiteSpec:
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        return suite_spec_from_dict(yaml.safe_load(fh) or {}, where=str(p))


def training_suite(spec: SuiteSpec | str | Path | None = None) -> Suite:
    """The spec's conditions (default: ``train_v1``), in file order.

    Without an explicit ``conditions`` list that is every family x size x placement.
    """
    if not isinstance(spec, SuiteSpec):
        spec = load_suite_spec(spec if spec is not None else DEFAULT_TRAIN_SUITE)
    conditions = []
    for kind, size_label, place_label in spec.triples():
        cid = f"{spec.name}/{kind}-{size_label}/{place_label}"
        seeds = tuple(derive_seed(spec.base_seed, cid, rep) for rep in range(spec.jitter_seeds_per_condition))
        conditions.append(
            Condition(
                condition_id=cid,
                family=kind,
                obj=objs.ObjectSpec(kind, spec.families[kind][size_label], spec.density(kind, size_label)),  # type: ignore[arg-type]
                placement=spec.placements[place_label],
                seeds=seeds,
                size_label=size_label,
                placement_label=place_label,
            )
        )
    return Suite(spec.name, tuple(conditions))


def mesh_suite(
    name: str,
    objects: Sequence[objs.MeshObject],
    placements: Mapping[str, objs.Placement] | None = None,
    seeds_per_condition: int = 1,
    base_seed: int = 0,
) -> Suite:
    """One condition per mesh object x placement (default: centred, no pose error)."""
    placements = placements or {"centred": objs.Placement()}
    conditions = []
    for obj in objects:
        for label, placement in placements.items():
            cid = f"{name}/{obj.name}/{label}"
            seeds = tuple(derive_seed(base_seed, cid, rep) for rep in range(seeds_per_condition))
            conditions.append(Condition(cid, "mesh", obj, placement, seeds, obj.name, label))
    return Suite(name, tuple(conditions))


# --------------------------------------------------------------------------- held-out (commit-reveal)


def new_secret_seed() -> str:
    """A fresh 256-bit secret (64 hex characters) for the held-out suite."""
    return secrets.token_hex(32)


def _secret_bytes(secret: bytes | str) -> bytes:
    raw = secret.encode() if isinstance(secret, str) else bytes(secret)
    raw = raw.strip()
    if len(raw) < MIN_SECRET_BYTES:
        raise ValueError(
            f"the held-out secret must be at least {MIN_SECRET_BYTES} characters "
            f"(use rtt.sim.conditions.new_secret_seed()); got {len(raw)}"
        )
    return raw


def seed_commitment(secret: bytes | str) -> str:
    """SHA-256 hex digest of the secret: commit this, never the secret."""
    return hashlib.sha256(_secret_bytes(secret)).hexdigest()


def _is_git_ignored(path: Path) -> bool:
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", str(path)],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def read_secret_seed(path: str | Path) -> bytes:
    """Read the held-out secret from a file (surrounding whitespace is stripped).

    A file inside this repository is accepted only if git ignores it, so the secret can
    never be committed or derived from tracked contents.
    """
    p = Path(path).expanduser().resolve()
    if p.is_relative_to(REPO_ROOT) and not ((REPO_ROOT / ".git").exists() and _is_git_ignored(p)):
        raise PermissionError(
            f"{p} is inside the repository and not git-ignored; keep the held-out secret outside "
            "the repository or in a git-ignored file"
        )
    return _secret_bytes(p.read_bytes())


def _heldout_rng(secret: bytes) -> np.random.Generator:
    entropy = int.from_bytes(hashlib.sha256(_HELDOUT_DOMAIN + secret).digest(), "big")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def _cap_size(kind: str, size: np.ndarray, max_width: float) -> tuple[float, ...]:
    """Keep the jaw-axis width under ``max_width`` and rods long enough."""
    size = size.copy()
    if kind == "box":
        size[1] = min(size[1], max_width)
    else:
        size[0] = min(size[0], max_width / 2.0)
    if kind == "rod":
        size[1] = max(size[1], ROD_MIN_LENGTH_DIAMETERS * 2.0 * size[0])
    return tuple(round(float(v), 6) for v in size)


def heldout_suite(
    secret: bytes | str,
    train: SuiteSpec | str | Path | None = None,
    *,
    n: int = HELDOUT_CONDITIONS,
    seeds_per_condition: int = HELDOUT_SEEDS_PER_CONDITION,
    commitment: str | None = None,
    task: TaskConfig | None = None,
    name: str = HELDOUT_NAME,
) -> Suite:
    """The held-out suite drawn from ``secret`` (see the module docstring).

    ``train`` gives the training size ranges (default ``train_v1``). With
    ``commitment``, a secret whose SHA-256 does not match it is refused. The result is
    a pure function of the secret, the training spec and the gripper's opening.
    """
    raw = _secret_bytes(secret)
    if commitment is not None and hashlib.sha256(raw).hexdigest() != commitment.strip().lower():
        raise ValueError("the held-out secret does not match the committed SHA-256")
    if n < 1 or seeds_per_condition < 1:
        raise ValueError("n and seeds_per_condition must be >= 1")
    if not isinstance(train, SuiteSpec):
        train = load_suite_spec(train if train is not None else DEFAULT_TRAIN_SUITE)
    task = task or default_task()
    max_width = 2.0 * task.gripper.open_half_gap - HELDOUT_WIDTH_MARGIN
    kinds = [k for k in FAMILIES if k in train.families]
    rng = _heldout_rng(raw)
    triples = train.triples()
    handle_share = {
        kind: sum(
            1 for f, _, pl in triples if f == kind and abs(train.placements[pl].err_x) >= HANDLE_MIN_OFFSET
        )
        / max(1, sum(1 for f, _, _ in triples if f == kind))
        for kind in kinds
    }

    order = rng.permutation(n)
    modes = [m for m, _ in HELDOUT_SIZE_MODES]
    probs = [p for _, p in HELDOUT_SIZE_MODES]
    conditions = []
    for i in range(n):
        kind = kinds[int(order[i]) % len(kinds)]
        lo, hi = train.size_range(kind)
        mode = str(rng.choice(modes, p=probs))
        if mode == "interp":
            size = lo + rng.uniform(*HELDOUT_INTERP, size=lo.shape) * (hi - lo)
        elif mode == "down":
            size = lo * (1.0 - rng.uniform(*HELDOUT_EXTRAP))
        else:
            size = hi * (1.0 + rng.uniform(*HELDOUT_EXTRAP))
        materials = train.family_densities(kind)
        density = materials[int(rng.integers(len(materials)))]
        obj = objs.ObjectSpec(kind, _cap_size(kind, size, max_width), density)  # type: ignore[arg-type]
        err_x = round(float(rng.uniform(-HELDOUT_ERR_XY, HELDOUT_ERR_XY)), 6)
        handle = bool(rng.random() < handle_share[kind])
        half_x = objs.half_extents(obj)[0]
        placement_label = "random"
        if handle and half_x >= HELDOUT_HANDLE_MIN_HALF:
            frac = float(rng.uniform(*HELDOUT_HANDLE_FRAC))
            offset = min(max(frac * half_x, HANDLE_MIN_OFFSET), half_x - HELDOUT_HANDLE_END_MARGIN)
            err_x = round(offset if rng.random() < 0.5 else -offset, 6)
            placement_label = "random-offcentre"
        placement = objs.Placement(
            x=round(float(rng.uniform(-HELDOUT_XY, HELDOUT_XY)), 6),
            y=round(float(rng.uniform(-HELDOUT_XY, HELDOUT_XY)), 6),
            yaw=round(float(rng.uniform(-math.pi, math.pi)), 6),
            err_x=err_x,
            err_y=round(float(rng.uniform(-HELDOUT_ERR_XY, HELDOUT_ERR_XY)), 6),
            err_yaw=round(math.radians(float(rng.uniform(-HELDOUT_ERR_YAW_DEG, HELDOUT_ERR_YAW_DEG))), 6),
        )
        seeds = tuple(int(s) for s in rng.integers(0, 2**63 - 1, size=seeds_per_condition))
        conditions.append(
            Condition(
                condition_id=f"{name}/{i:03d}-{kind}-{mode}",
                family=kind,
                obj=obj,
                placement=placement,
                seeds=seeds,
                size_label=mode,
                placement_label=placement_label,
            )
        )
    return Suite(name, tuple(conditions))


def overlapping_conditions(a: Suite, b: Suite, size_tol: float = 0.01, pos_tol: float = 1e-3) -> list[str]:
    """Ids of conditions in ``a`` that duplicate one in ``b`` (same family, every size within
    ``size_tol`` relative, nominal position within ``pos_tol`` m and yaw within 1 deg)."""
    out = []
    for ca in a:
        for cb in b:
            if ca.family != cb.family or len(ca.obj.size) != len(cb.obj.size):
                continue
            same_size = all(
                abs(sa - sb) <= size_tol * max(sa, sb)
                for sa, sb in zip(ca.obj.size, cb.obj.size, strict=True)
            )
            pa, pb = ca.placement, cb.placement
            dyaw = abs(math.remainder(pa.yaw - pb.yaw, 2 * math.pi))
            if same_size and math.hypot(pa.x - pb.x, pa.y - pb.y) <= pos_tol and dyaw <= math.radians(1.0):
                out.append(ca.condition_id)
                break
    return out


def replace_seeds(suite: Suite, seeds: Sequence[int]) -> Suite:
    """The same conditions with every condition's jitter seeds replaced by ``seeds``."""
    seeds = tuple(int(s) for s in seeds)
    if not seeds:
        raise ValueError("seeds must not be empty")
    return Suite(suite.name, tuple(dataclasses.replace(c, seeds=seeds) for c in suite.conditions))
