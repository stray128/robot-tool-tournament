"""Family registry: one implementation per :class:`~rtt.effectors.types.EffectorFamily`.

A family module (``rtt.effectors.<family value>``, :data:`~rtt.effectors.types.FAMILY_MODULES`)
subclasses :class:`EffectorFamilyImpl` and calls :func:`register_family` at import time.
:func:`get_family` imports that module on first use, so a spawned worker process finds every
family without a shared list that each builder track would have to edit.

Two ways to run an episode:

* **Native** (every new family): implement :meth:`EffectorFamilyImpl.build`,
  :meth:`~EffectorFamilyImpl.primitive` and :meth:`~EffectorFamilyImpl.criterion`; the
  inherited :meth:`~EffectorFamilyImpl.run_episode` hands them to the shared runner
  (:func:`rtt.effectors.runner.run_native_episode`), which owns the wrist, phases, checks,
  scorer and recording, so every family is judged by the same code.
* **Delegated** (the parallel-jaw reference only): :meth:`~EffectorFamilyImpl.run_episode` is
  overridden to call the frozen v2 episode (:func:`rtt.sim.episode.run_episode_v2`) unchanged,
  so the product layer reproduces the pre-registered numbers exactly.
"""

from __future__ import annotations

import contextlib
import importlib
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from rtt.effectors.types import (
    FAMILY_FIDELITY,
    FAMILY_MODULES,
    BuildContext,
    EffectorBuild,
    EffectorEpisodeResult,
    EffectorFamily,
    EffectorSpec,
    Fidelity,
    GraspCriterion,
    GraspPrimitive,
    MorphologyRecipe,
    ParamSpec,
    RegistrationReference,
)

if TYPE_CHECKING:
    from rtt.parts.types import PartSpec
    from rtt.sim.objects import Placement
    from rtt.sim.perturb import Perturbation
    from rtt.sim.task import TaskV2

__all__ = [
    "EffectorFamilyImpl",
    "UnknownFamilyError",
    "get_family",
    "override_family",
    "register_family",
    "registered_families",
    "unregister_family",
]


class UnknownFamilyError(LookupError):
    """No implementation is registered (or importable) for a family."""


class EffectorFamilyImpl(ABC):
    """Base class of a family implementation (see the module docstring).

    Class attributes:

    * ``family``: the :class:`EffectorFamily` it implements.
    * ``reference_mode``: how registration is referenced (:class:`RegistrationReference`):
      ``"analytic"`` (the wrist-aimed pose) or ``"nominal_episode"`` (repeatability against
      the same effector's nominal certificate episode). Jaws are analytic; a family whose
      grasp does not centre the part on the wrist axis is ``"nominal_episode"``.
    """

    family: ClassVar[EffectorFamily]
    reference_mode: ClassVar[Literal["analytic", "nominal_episode"]] = "nominal_episode"

    @property
    def fidelity(self) -> Fidelity:
        return FAMILY_FIDELITY[self.family]

    # ------------------------------------------------------------------ what the family offers

    @abstractmethod
    def presets(self) -> tuple[EffectorSpec, ...]:
        """The effectors the tournament tries by default (catalog models x primitives, presets)."""

    def param_space(self, kind: str) -> tuple[ParamSpec, ...]:
        """The designable parameters of generator ``kind`` (empty: not designable)."""
        return ()

    def recipe_kinds(self) -> tuple[str, ...]:
        """Generator kinds that :meth:`param_space` describes (empty: not designable)."""
        return ()

    def spec_from_recipe(self, recipe: MorphologyRecipe) -> EffectorSpec:
        """The :class:`EffectorSpec` of a recipe for one of this family's generators."""
        raise NotImplementedError(f"family {self.family.value!r} has no designable generators")

    # ------------------------------------------------------------------ the native path

    def build(self, ctx: BuildContext) -> EffectorBuild:
        """Add the effector under ``ctx.mount`` and describe it (native families)."""
        raise NotImplementedError(f"family {self.family.value!r} does not build into the shared scene")

    def primitive(self, spec: EffectorSpec, build: EffectorBuild) -> GraspPrimitive:
        """The grasp primitive named by ``spec.primitive`` (native families)."""
        raise NotImplementedError(f"family {self.family.value!r} has no native primitive")

    def criterion(self, spec: EffectorSpec, build: EffectorBuild) -> GraspCriterion:
        """The engagement criterion (native families)."""
        raise NotImplementedError(f"family {self.family.value!r} has no native criterion")

    def run_episode(
        self,
        spec: EffectorSpec,
        part: PartSpec,
        placement: Placement,
        seed: int,
        *,
        task: TaskV2 | None = None,
        perturbation: Perturbation | None = None,
        record: bool = False,
        reference: RegistrationReference | None = None,
        condition_id: str = "",
        scene: Any = None,
    ) -> EffectorEpisodeResult:
        """One scored episode (default: the shared native runner)."""
        from rtt.effectors.runner import run_native_episode

        return run_native_episode(
            self,
            spec,
            part,
            placement,
            seed,
            task=task,
            perturbation=perturbation,
            record=record,
            reference=reference,
            condition_id=condition_id,
            scene=scene,
        )

    def check_spec(self, spec: EffectorSpec) -> None:
        """Raise ``ValueError`` if ``spec`` is not this family's."""
        if spec.family != self.family:
            raise ValueError(f"spec is for family {spec.family.value!r}, not {self.family.value!r}")


_REGISTRY: dict[EffectorFamily, EffectorFamilyImpl] = {}
_LOCK = threading.RLock()


def register_family(impl: EffectorFamilyImpl, *, replace: bool = False) -> EffectorFamilyImpl:
    """Register ``impl`` for ``impl.family`` (a second registration needs ``replace=True``)."""
    if not isinstance(impl, EffectorFamilyImpl):
        raise TypeError("register_family takes an EffectorFamilyImpl instance")
    family = EffectorFamily(impl.family)
    if impl.reference_mode not in ("analytic", "nominal_episode"):
        raise ValueError(f"reference_mode must be analytic or nominal_episode, got {impl.reference_mode!r}")
    with _LOCK:
        if family in _REGISTRY and _REGISTRY[family] is not impl and not replace:
            raise ValueError(
                f"family {family.value!r} is already registered ({type(_REGISTRY[family]).__name__})"
            )
        _REGISTRY[family] = impl
    return impl


def unregister_family(family: EffectorFamily | str) -> None:
    """Forget a registration (tests); the module stays imported, so re-register explicitly."""
    with _LOCK:
        _REGISTRY.pop(EffectorFamily(family), None)


def get_family(family: EffectorFamily | str) -> EffectorFamilyImpl:
    """The implementation of ``family``, importing ``rtt.effectors.<family>`` on first use."""
    fam = EffectorFamily(family)
    with _LOCK:
        if fam in _REGISTRY:
            return _REGISTRY[fam]
    module = FAMILY_MODULES[fam]
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module:
            raise
        raise UnknownFamilyError(f"family {fam.value!r} is not built yet (no module {module})") from None
    with _LOCK:
        if fam not in _REGISTRY:
            raise UnknownFamilyError(f"importing {module} did not register family {fam.value!r}")
        return _REGISTRY[fam]


def registered_families() -> tuple[EffectorFamily, ...]:
    """Families registered in this process (in enum order); does not import anything."""
    with _LOCK:
        return tuple(f for f in EffectorFamily if f in _REGISTRY)


def available_families() -> tuple[EffectorFamily, ...]:
    """Families that :func:`get_family` can load (imports every family module that exists)."""
    out = []
    for fam in EffectorFamily:
        try:
            get_family(fam)
        except UnknownFamilyError:
            continue
        out.append(fam)
    return tuple(out)


@contextlib.contextmanager
def override_family(impl: EffectorFamilyImpl) -> Iterator[EffectorFamilyImpl]:
    """Temporarily register ``impl`` for its family (tests), restoring the previous one."""
    family = EffectorFamily(impl.family)
    with _LOCK:
        previous = _REGISTRY.get(family)
        _REGISTRY[family] = impl
    try:
        yield impl
    finally:
        with _LOCK:
            if previous is None:
                _REGISTRY.pop(family, None)
            else:
                _REGISTRY[family] = previous


def presets_of(families: Sequence[EffectorFamily | str]) -> tuple[EffectorSpec, ...]:
    """Every preset of the given families, in the given order."""
    return tuple(spec for fam in families for spec in get_family(fam).presets())
