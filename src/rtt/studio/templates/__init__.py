"""Physics builders of the object templates: one module per :class:`~rtt.studio.types.TemplateKind`.

A builder module ``rtt.studio.templates.<kind>`` subclasses :class:`TemplateBuilder` and calls
:func:`register_template_builder` on import; :func:`get_template_builder` imports it on first use
(as :func:`rtt.effectors.registry.get_family` does for families), so a spawned worker finds every
builder. A kind without a module is not simulated yet (:class:`TemplateUnavailable`).

S1 ships the two non-articulated builders (:mod:`.part`, :mod:`.bin`); track S2 owns the
articulated ones (hinged lid, drawer, latch, threaded joint, handwheel valve, port and plug).

Rules every builder follows (the scene checks them, :mod:`rtt.studio.scene`):

* every element it adds is named with ``ctx.prefix`` (``obj_<instance>_``) and hangs under the
  world body (never under the wrist or a tool);
* every geom it adds has ``priority = 1``, so the object's contact parameters govern each contact
  a tool makes with it (the v1 part rule), and its friction is jittered with the condition's seed;
* joints, tendons, equalities and actuators stay inside its own bodies (static anchors may use
  the world body); passive behaviour (spring, detent, torque ramp) is a joint property or one of
  its own actuators, driven by :meth:`TemplateBuilder.passive` from the object's state only;
* every handle of the template has a site (``TemplateHandles.sites``), placed where a tool
  engages it; :meth:`TemplateBuilder.state` reports the template's quantities in their units.
"""

from __future__ import annotations

import contextlib
import importlib
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Final

import numpy as np

from rtt.effectors.types import FrozenDict
from rtt.studio.types import ObjectInstance, TemplateKind, get_template

if TYPE_CHECKING:
    from pathlib import Path

    import mujoco

    from rtt.sim.task import TaskV2

__all__ = [
    "OBJ_PREFIX",
    "ResolvedHandles",
    "TemplateBuilder",
    "TemplateContext",
    "TemplateHandles",
    "TemplateUnavailable",
    "get_template_builder",
    "object_prefix",
    "override_template_builder",
    "register_template_builder",
]

#: Every element a template adds starts with ``obj_<instance name>_``.
OBJ_PREFIX: Final = "obj_"


def object_prefix(name: str) -> str:
    return f"{OBJ_PREFIX}{name}_"


class TemplateUnavailable(LookupError):
    """No physics for this template kind or reference yet: the step is ``not_simulated``."""


@dataclass(frozen=True)
class TemplateContext:
    """What a builder gets: the scene spec, the instance, its name prefix and the task."""

    spec: mujoco.MjSpec
    instance: ObjectInstance
    prefix: str
    task: TaskV2
    repo_root: Path

    @property
    def table_top_z(self) -> float:
        return float(self.task.physics.table_top_z)


@dataclass(frozen=True, eq=False)
class TemplateHandles:
    """What a builder added, by name (resolved to ids by the scene after compiling).

    * ``bodies``: its bodies, root first (empty for a static object on the world body).
    * ``geoms``: every geom it added; ``joints``: role -> joint name (``"hinge"``, ``"free"``...).
    * ``sites``: handle -> site name, one per handle of the template.
    * ``actuators``: its own passive actuators, in compile order (driven by ``passive``).
    * ``info``: builder numbers its ``state`` needs (rest height, interior box...), SI units.
    """

    instance: str
    kind: TemplateKind
    bodies: tuple[str, ...]
    geoms: tuple[str, ...]
    joints: Mapping[str, str]
    sites: Mapping[str, str]
    actuators: tuple[str, ...] = ()
    info: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", TemplateKind(self.kind))
        for name in ("bodies", "geoms", "actuators"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "joints", FrozenDict(dict(self.joints)))
        object.__setattr__(self, "sites", FrozenDict(dict(self.sites)))
        object.__setattr__(self, "info", FrozenDict({k: float(v) for k, v in self.info.items()}))
        missing = [h for h in get_template(self.kind).handles if h not in self.sites]
        if missing:
            raise ValueError(f"{self.instance}: no site for handles {missing}")
        prefix = object_prefix(self.instance)
        names = [*self.bodies, *self.geoms, *self.joints.values(), *self.sites.values(), *self.actuators]
        bad = [n for n in names if not n.startswith(prefix)]
        if bad:
            raise ValueError(f"{self.instance}: names without the {prefix!r} prefix: {bad[:5]}")


@dataclass(frozen=True, eq=False)
class ResolvedHandles:
    """:class:`TemplateHandles` resolved against a compiled model (ids and addresses)."""

    handles: TemplateHandles
    body_ids: tuple[int, ...]
    geom_ids: np.ndarray
    joint_qpos: Mapping[str, int]  # role -> qpos address
    joint_dof: Mapping[str, int]  # role -> dof address
    site_ids: Mapping[str, int]  # handle -> site id
    actuator_ids: np.ndarray

    @property
    def name(self) -> str:
        return self.handles.instance

    @property
    def kind(self) -> TemplateKind:
        return self.handles.kind

    def site_pos(self, data: mujoco.MjData, handle: str) -> np.ndarray:
        """World position of a handle's site (m)."""
        return np.array(data.site_xpos[self.site_ids[handle]], dtype=np.float64)


class TemplateBuilder(ABC):
    """Base class of a template's physics builder (module docstring)."""

    kind: ClassVar[TemplateKind]

    @abstractmethod
    def add(self, ctx: TemplateContext) -> TemplateHandles:
        """Author the instance into ``ctx.spec`` at its pose (module docstring rules)."""

    @abstractmethod
    def state(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        """The template's quantities (:attr:`ObjectTemplate.quantities`) now, in their units."""

    def passive(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        """Controls of the template's own actuators (detent, torque ramp); default: none."""
        return np.zeros(len(obj.actuator_ids))


_BUILDERS: dict[TemplateKind, TemplateBuilder] = {}
_LOCK = threading.RLock()


def register_template_builder(builder: TemplateBuilder, *, replace: bool = False) -> TemplateBuilder:
    if not isinstance(builder, TemplateBuilder):
        raise TypeError("register_template_builder takes a TemplateBuilder")
    kind = TemplateKind(builder.kind)
    with _LOCK:
        if kind in _BUILDERS and _BUILDERS[kind] is not builder and not replace:
            raise ValueError(f"template {kind.value!r} already has a builder")
        _BUILDERS[kind] = builder
    return builder


def get_template_builder(kind: TemplateKind | str) -> TemplateBuilder:
    """The builder of ``kind``, importing ``rtt.studio.templates.<kind>`` on first use."""
    k = TemplateKind(kind)
    with _LOCK:
        if k in _BUILDERS:
            return _BUILDERS[k]
    module = f"{__name__}.{k.value}"
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module:
            raise
        raise TemplateUnavailable(f"template {k.value!r} is not simulated yet (no module {module})") from None
    with _LOCK:
        if k not in _BUILDERS:
            raise TemplateUnavailable(f"importing {module} did not register a builder for {k.value!r}")
        return _BUILDERS[k]


@contextlib.contextmanager
def override_template_builder(builder: TemplateBuilder) -> Iterator[TemplateBuilder]:
    """Temporarily register ``builder`` for its kind (tests), restoring the previous one."""
    kind = TemplateKind(builder.kind)
    with _LOCK:
        previous = _BUILDERS.get(kind)
        _BUILDERS[kind] = builder
    try:
        yield builder
    finally:
        with _LOCK:
            if previous is None:
                _BUILDERS.pop(kind, None)
            else:
                _BUILDERS[kind] = previous


def builder_kinds() -> tuple[TemplateKind, ...]:
    """Kinds whose builder module exists (imports each one)."""
    out = []
    for k in TemplateKind:
        try:
            get_template_builder(k)
        except TemplateUnavailable:
            continue
        out.append(k)
    return tuple(out)
