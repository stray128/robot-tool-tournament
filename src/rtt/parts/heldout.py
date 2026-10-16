"""The procedural held-out suite ``heldout_procedural_v2``: commit-reveal, drawn from a secret.

Same protocol as v1 (:mod:`rtt.sim.conditions`): a 256-bit secret is created before any
search, only its SHA-256 is committed, and the held-out job receives the secret at run
time. :func:`heldout_procedural_suite` is a pure function of the secret, the config file
(``configs/suites/heldout_procedural_v2.yaml``), the parts library and the task's gripper,
and refuses a secret that does not match the commitment. The secret is domain separated
(:data:`DOMAIN`) from the v1 suites, so one secret can serve several suites uncorrelated.

Each of the ``n`` conditions draws, from the secret alone:

* a slot: ``training_share`` of the conditions are new instances of the six training
  families (stratified over them), the rest are the unseen families (stratified), and the
  slots are shuffled;
* for a training family, a size mode (``near`` / ``down`` / ``up``, in the fixed shares
  ``mode_probs``, assigned to the training slots at random):

  - ``near``: one overall factor in ``near.scale``, and every *free* dimension of the family
    (:attr:`rtt.parts.library.Family.free`) times its own factor in ``near.aspect``, so
    proportions are new;
  - ``down`` / ``up``: one factor for every dimension, below / above the training size by
    the ``down`` / ``up`` range (these are outside the training dimensions);

* for an unseen family, its own draw (``pipe_elbow``: scale and free-dimension aspect, as
  ``near`` but over the whole ``scale`` range; ``spur_gear``: module, tooth count, face width,
  bore as a fraction of the pitch diameter);
* a placement: nominal x, y uniform in +-``xy_mm``, yaw uniform over a full turn; a pose error
  uniform in a disc of radius ``err_disc_mm`` (across the jaw and along the closing axis),
  a yaw error uniform in +-``err_yaw_deg`` and a height error uniform in +-``err_z_mm``;
* ``seeds_per_condition`` jitter seeds.

A drawn part must build, fit the gripper (width across the jaw at most ``max_grip_width_mm``,
height in [``min_height_mm``, ``max_height_mm``]) and weigh at most the grip cap
(``mu_min * F / (g * mass_safety_factor)``, as for the real-object suite); otherwise the
size is drawn again from the same stream (at most ``max_tries`` times), so the result stays
a pure function of the secret. The materials are the families' library materials.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rtt.parts import library as lib
from rtt.parts.suite import PartCondition, PartSuite
from rtt.parts.types import PartSpec
from rtt.sim import objects as objs
from rtt.sim.conditions import MIN_SECRET_BYTES
from rtt.sim.task import TaskConfig, default_task

DEFAULT_CONFIG = lib.REPO_ROOT / "configs" / "suites" / "heldout_procedural_v2.yaml"
DOMAIN = b"rtt-heldout-procedural-v2\x00"
SIZE_MODES = ("near", "down", "up")


@dataclass(frozen=True)
class HeldoutConfig:
    """A parsed ``heldout_procedural_v2.yaml`` (lengths in millimetres, angles in degrees)."""

    name: str = "heldout_procedural_v2"
    n_conditions: int = 100
    seeds_per_condition: int = 2
    parts_file: Path = lib.DEFAULT_PARTS_FILE
    training_share: float = 0.7
    mode_probs: tuple[float, float, float] = (0.4, 0.3, 0.3)
    near_scale: tuple[float, float] = (0.85, 1.15)
    near_aspect: tuple[float, float] = (0.9, 1.1)
    down: tuple[float, float] = (0.70, 0.85)
    up: tuple[float, float] = (1.15, 1.30)
    unseen: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    xy_mm: float = 80.0
    err_disc_mm: float = 3.0
    err_yaw_deg: float = 8.0
    err_z_mm: float = 1.5
    max_grip_width_mm: float = 79.0
    min_height_mm: float = 6.0
    max_height_mm: float = 100.0
    mass_safety_factor: float = 1.5
    max_tries: int = 64
    path: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.n_conditions < 1 or self.seeds_per_condition < 1 or self.max_tries < 1:
            raise ValueError("n_conditions, seeds_per_condition and max_tries must be >= 1")
        if not 0.0 <= self.training_share <= 1.0:
            raise ValueError("training_share must be in [0, 1]")
        if len(self.mode_probs) != 3 or abs(sum(self.mode_probs) - 1.0) > 1e-9 or min(self.mode_probs) < 0:
            raise ValueError("size mode probabilities must be 3 non-negative numbers summing to 1")
        for name, (lo, hi) in (
            ("near_scale", self.near_scale),
            ("near_aspect", self.near_aspect),
            ("down", self.down),
            ("up", self.up),
        ):
            if not 0 < lo <= hi:
                raise ValueError(f"{name} must be a range 0 < lo <= hi")
        if not (self.down[1] <= self.near_scale[0] and self.near_scale[1] <= self.up[0]):
            raise ValueError("down, near and up ranges must not overlap")
        unknown = set(self.unseen) - set(lib.HELDOUT_FAMILIES)
        if unknown or not self.unseen:
            raise ValueError(f"unseen families must be a non-empty subset of {lib.HELDOUT_FAMILIES}")

    def mass_cap_kg(self, task: TaskConfig) -> float:
        """Heaviest part the grip holds at the worst jittered friction, with a safety margin."""
        mu_min = task.physics.friction * (1.0 - task.jitter.mu_frac)
        return mu_min * task.gripper.grip_force / (9.81 * self.mass_safety_factor)

    def digest(self) -> str:
        payload = {k: v for k, v in self.__dict__.items() if k not in ("path", "parts_file")}
        payload["unseen"] = {k: dict(v) for k, v in self.unseen.items()}
        payload["parts_library"] = lib.library_digest(self.parts_file)
        return hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()


def _pair(raw: Any, where: str) -> tuple[float, float]:
    if not isinstance(raw, list | tuple) or len(raw) != 2:
        raise ValueError(f"{where} must be [lo, hi]")
    return (float(raw[0]), float(raw[1]))


def load_heldout_config(path: str | Path = DEFAULT_CONFIG) -> HeldoutConfig:
    """Load and validate a ``heldout_procedural`` suite file."""
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    allowed = {
        "name",
        "n_conditions",
        "seeds_per_condition",
        "parts_file",
        "families",
        "sizes",
        "unseen",
        "placement",
        "limits",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{p}: unknown keys {sorted(unknown)}")
    fam = raw.get("families") or {}
    sizes = raw.get("sizes") or {}
    place = raw.get("placement") or {}
    limits = raw.get("limits") or {}
    for label, section, keys in (
        ("families", fam, {"training_share"}),
        ("sizes", sizes, {"mode_probs", "near_scale", "near_aspect", "down", "up"}),
        ("placement", place, {"xy_mm", "err_disc_mm", "err_yaw_deg", "err_z_mm"}),
        (
            "limits",
            limits,
            {"max_grip_width_mm", "min_height_mm", "max_height_mm", "mass_safety_factor", "max_tries"},
        ),
    ):
        extra = set(section) - keys
        if extra:
            raise ValueError(f"{p}: unknown keys in '{label}': {sorted(extra)}")
    probs = sizes.get("mode_probs", {"near": 0.4, "down": 0.3, "up": 0.3})
    if set(probs) != set(SIZE_MODES):
        raise ValueError(f"{p}: sizes.mode_probs must give {SIZE_MODES}")
    cfg = HeldoutConfig(
        name=str(raw.get("name", "heldout_procedural_v2")),
        n_conditions=int(raw.get("n_conditions", 100)),
        seeds_per_condition=int(raw.get("seeds_per_condition", 2)),
        parts_file=(p.resolve().parent / str(raw.get("parts_file", "../parts_v2.yaml"))).resolve(),
        training_share=float(fam.get("training_share", 0.7)),
        mode_probs=tuple(float(probs[m]) for m in SIZE_MODES),  # type: ignore[arg-type]
        near_scale=_pair(sizes.get("near_scale", [0.85, 1.15]), "sizes.near_scale"),
        near_aspect=_pair(sizes.get("near_aspect", [0.9, 1.1]), "sizes.near_aspect"),
        down=_pair(sizes.get("down", [0.70, 0.85]), "sizes.down"),
        up=_pair(sizes.get("up", [1.15, 1.30]), "sizes.up"),
        unseen={str(k): dict(v or {}) for k, v in (raw.get("unseen") or {}).items()},
        xy_mm=float(place.get("xy_mm", 80.0)),
        err_disc_mm=float(place.get("err_disc_mm", 3.0)),
        err_yaw_deg=float(place.get("err_yaw_deg", 8.0)),
        err_z_mm=float(place.get("err_z_mm", 1.5)),
        max_grip_width_mm=float(limits.get("max_grip_width_mm", 79.0)),
        min_height_mm=float(limits.get("min_height_mm", 6.0)),
        max_height_mm=float(limits.get("max_height_mm", 100.0)),
        mass_safety_factor=float(limits.get("mass_safety_factor", 1.5)),
        max_tries=int(limits.get("max_tries", 64)),
        path=p,
    )
    for family, spec in cfg.unseen.items():
        _check_unseen(family, spec, str(p))
    return cfg


_UNSEEN_KEYS = {
    "pipe_elbow": {"material", "params", "scale", "aspect"},
    "spur_gear": {"material", "module", "teeth", "face_width_mm", "bore_frac", "pressure_angle_deg"},
}


def _check_unseen(family: str, spec: Mapping[str, Any], where: str) -> None:
    extra = set(spec) - _UNSEEN_KEYS[family]
    if extra:
        raise ValueError(f"{where}: unknown keys for unseen family {family}: {sorted(extra)}")


# --------------------------------------------------------------------------- draws


def _secret_bytes(secret: bytes | str) -> bytes:
    raw = secret.encode() if isinstance(secret, str) else bytes(secret)
    raw = raw.strip()
    if len(raw) < MIN_SECRET_BYTES:
        raise ValueError(
            f"the held-out secret must be at least {MIN_SECRET_BYTES} characters; got {len(raw)}"
        )
    return raw


def _rng(secret: bytes) -> np.random.Generator:
    entropy = int.from_bytes(hashlib.sha256(DOMAIN + secret).digest(), "big")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def _stratified(n: int, labels: tuple[str, ...], rng: np.random.Generator) -> list[str]:
    """``n`` labels as evenly as possible (the remainder to a random subset), unshuffled."""
    base, extra = divmod(n, len(labels))
    bonus = {int(i) for i in rng.permutation(len(labels))[:extra]}
    out: list[str] = []
    for i, label in enumerate(labels):
        out += [label] * (base + (1 if i in bonus else 0))
    return out


def _mode_slots(n: int, cfg: HeldoutConfig, rng: np.random.Generator) -> list[str]:
    """``n`` size modes in the configured shares (largest remainder), shuffled, so the share
    outside the training dimensions (down + up) is fixed, not left to chance."""
    exact = [n * p for p in cfg.mode_probs]
    counts = [math.floor(x) for x in exact]
    for i in sorted(range(3), key=lambda i: (-(exact[i] - counts[i]), i))[: n - sum(counts)]:
        counts[i] += 1
    modes = [m for m, c in zip(SIZE_MODES, counts, strict=True) for _ in range(c)]
    return [modes[int(i)] for i in rng.permutation(len(modes))]


def _round_params(params: Mapping[str, float], family: lib.Family) -> dict[str, float]:
    """Lengths to 0.01 mm (integers kept): the drawn dimensions are what the record shows."""
    return {k: (int(v) if k in family.integers else round(float(v), 2)) for k, v in params.items()}


def _scaled(entry: lib.PartEntry, factor: float, aspects: Mapping[str, float]) -> dict[str, float]:
    fam = lib.FAMILIES[entry.family]
    out = dict(entry.params)
    for k in fam.lengths():
        out[k] = entry.params[k] * factor * aspects.get(k, 1.0)
    return _round_params(out, fam)


def _pipe_repair(params: dict[str, float]) -> dict[str, float]:
    """Keep the socket floors clear of the core after an aspect change of centre_to_end."""
    floor = math.ceil((params["socket_depth"] + params["body_od"] / 2) * 100 - 1e-6) / 100
    params["centre_to_end"] = max(params["centre_to_end"], floor)
    return params


def _draw_training(
    entry: lib.PartEntry, mode: str, cfg: HeldoutConfig, rng: np.random.Generator
) -> dict[str, float]:
    fam = lib.FAMILIES[entry.family]
    if mode == "near":
        factor = float(rng.uniform(*cfg.near_scale))
        aspects = {k: float(rng.uniform(*cfg.near_aspect)) for k in fam.free}
    else:
        factor = float(rng.uniform(*(cfg.down if mode == "down" else cfg.up)))
        aspects = {}
    params = _scaled(entry, factor, aspects)
    return _pipe_repair(params) if entry.family in ("pipe_tee", "pipe_elbow") else params


def _draw_unseen(family: str, spec: Mapping[str, Any], rng: np.random.Generator) -> dict[str, float]:
    fam = lib.FAMILIES[family]
    if family == "pipe_elbow":
        base = fam.resolve(spec.get("params") or {})
        factor = float(rng.uniform(*_pair(spec.get("scale", [0.7, 1.3]), "pipe_elbow.scale")))
        aspect = _pair(spec.get("aspect", [0.9, 1.1]), "pipe_elbow.aspect")
        aspects = {k: float(rng.uniform(*aspect)) for k in fam.free}
        out = dict(base)
        for k in fam.lengths():
            out[k] = base[k] * factor * aspects.get(k, 1.0)
        return _pipe_repair(_round_params(out, fam))
    if family == "spur_gear":
        m = float(rng.uniform(*_pair(spec.get("module", [1.5, 2.0]), "spur_gear.module")))
        t_lo, t_hi = (int(v) for v in spec.get("teeth", [16, 24]))
        z = int(rng.integers(t_lo, t_hi + 1))
        width = float(rng.uniform(*_pair(spec.get("face_width_mm", [10.0, 16.0]), "spur_gear.face_width_mm")))
        bore = float(spec.get("bore_frac", 0.25)) * m * z
        return _round_params(
            {
                "module": round(m, 3),
                "teeth": z,
                "face_width": width,
                "bore_d": bore,
                "pressure_angle": float(spec.get("pressure_angle_deg", 20.0)),
            },
            fam,
        )
    raise ValueError(f"no draw rule for unseen family {family!r}")


def _fits(part: PartSpec, cfg: HeldoutConfig, mass_cap: float) -> str | None:
    _sx, sy, sz = (1000.0 * s for s in part.size)
    if sy > cfg.max_grip_width_mm:
        return f"too wide ({sy:.1f} mm)"
    if not cfg.min_height_mm <= sz <= cfg.max_height_mm:
        return f"height {sz:.1f} mm out of range"
    if part.mass > mass_cap:
        return f"too heavy ({part.mass:.3f} kg)"
    return None


@dataclass(frozen=True)
class HeldoutDraw:
    """What one slot drew: family, mode, dimensions and how many size draws it took."""

    index: int
    family: str
    mode: str
    params_mm: Mapping[str, float]
    tries: int
    material: str


def heldout_procedural_suite(
    secret: bytes | str,
    config: HeldoutConfig | str | Path | None = None,
    *,
    commitment: str | None = None,
    task: TaskConfig | None = None,
    n: int | None = None,
) -> PartSuite:
    """The procedural held-out suite drawn from ``secret`` (see the module docstring).

    With ``commitment`` (the committed SHA-256 hex digest), a secret that does not match is
    refused. ``n`` overrides the config's condition count (for smoke runs).
    """
    raw = _secret_bytes(secret)
    if commitment is not None and hashlib.sha256(raw).hexdigest() != commitment.strip().lower():
        raise ValueError("the held-out secret does not match the committed SHA-256")
    if not isinstance(config, HeldoutConfig):
        config = load_heldout_config(config if config is not None else DEFAULT_CONFIG)
    cfg = config
    task = task or default_task()
    total = cfg.n_conditions if n is None else int(n)
    if total < 1:
        raise ValueError("n must be >= 1")
    library = lib.load_parts_library(cfg.parts_file)
    train_entries = {e.family: e for e in library.entries}
    families = tuple(f for f in lib.TRAINING_FAMILIES if f in train_entries)
    unseen = tuple(f for f in lib.HELDOUT_FAMILIES if f in cfg.unseen)
    cap = cfg.mass_cap_kg(task)

    rng = _rng(raw)
    n_train = round(total * cfg.training_share)
    slots = [
        (f, m)
        for f, m in zip(_stratified(n_train, families, rng), _mode_slots(n_train, cfg, rng), strict=True)
    ]
    slots += [(f, "unseen") for f in _stratified(total - n_train, unseen, rng)]
    order = rng.permutation(total)
    conditions = []
    for i in range(total):
        family, mode = slots[int(order[i])]
        is_unseen = mode == "unseen"
        part = None
        params: dict[str, float] = {}
        mat = (cfg.unseen[family].get("material") if is_unseen else None) or (
            train_entries[family].material if not is_unseen else lib.DEFAULT_MATERIALS[family]
        )
        name = f"hp2-{i:03d}-{family}"
        reason = "no draw"
        for _ in range(cfg.max_tries):
            params = (
                _draw_unseen(family, cfg.unseen[family], rng)
                if is_unseen
                else _draw_training(train_entries[family], mode, cfg, rng)
            )
            try:
                candidate = lib.build_part(name, family, params, str(mat))
            except ValueError as exc:
                reason = str(exc)
                continue
            reason = _fits(candidate, cfg, cap) or ""
            if not reason:
                part = candidate
                break
        if part is None:
            raise ValueError(
                f"condition {i} ({family}, {mode}): no valid part in {cfg.max_tries} draws: {reason}"
            )
        r = cfg.err_disc_mm * 1e-3 * math.sqrt(float(rng.random()))
        theta = float(rng.uniform(-math.pi, math.pi))
        xy = cfg.xy_mm * 1e-3
        placement = objs.Placement(
            x=round(float(rng.uniform(-xy, xy)), 6),
            y=round(float(rng.uniform(-xy, xy)), 6),
            yaw=round(float(rng.uniform(-math.pi, math.pi)), 6),
            err_x=round(r * math.cos(theta), 6),
            err_y=round(r * math.sin(theta), 6),
            err_yaw=round(math.radians(float(rng.uniform(-cfg.err_yaw_deg, cfg.err_yaw_deg))), 6),
        )
        err_z = round(float(rng.uniform(-cfg.err_z_mm, cfg.err_z_mm)) * 1e-3, 6)
        seeds = tuple(int(s) for s in rng.integers(0, 2**63 - 1, size=cfg.seeds_per_condition))
        conditions.append(
            PartCondition(
                condition_id=f"{cfg.name}/{i:03d}-{family}-{mode}",
                part=part,
                placement=placement,
                seeds=seeds,
                placement_label="random",
                size_label=mode,
                err_z=err_z,
                params_mm=params,
            )
        )
    return PartSuite(cfg.name, tuple(conditions))


def suite_summary(suite: PartSuite, train: PartSuite | None = None) -> dict[str, Any]:
    """Counts by family and mode, and the share of training-family instances outside the
    training dimensions (any length changed by more than 15 %)."""
    from collections import Counter

    fam = Counter(c.part.family for c in suite)
    modes = Counter(c.size_label for c in suite)
    out: dict[str, Any] = {
        "n": len(suite),
        "families": dict(sorted(fam.items())),
        "modes": dict(sorted(modes.items())),
    }
    library = lib.load_parts_library()
    base = {e.family: e.params for e in library.entries}
    inst = [c for c in suite if c.part.family in base and c.params_mm is not None]
    outside = 0
    for c in inst:
        ref = base[c.part.family]
        lengths = lib.FAMILIES[c.part.family].lengths()
        if any(abs(c.params_mm[k] / ref[k] - 1.0) > 0.15 + 1e-9 for k in lengths if ref[k]):
            outside += 1
    out["training_family_instances"] = len(inst)
    out["outside_training_dims"] = outside
    return out
