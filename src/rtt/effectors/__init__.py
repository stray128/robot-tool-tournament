"""Multi-effector product layer: for each part, every end-effector family, one recommendation.

**Product mode, not pre-registered.** The pre-registered parallel-jaw experiment
(``prereg-v1``) is untouched; this layer reuses its scene, checks and scorer read-only and
labels everything it produces with :data:`PRODUCT_MODE_LABEL`. See ``docs/EFFECTORS.md``.

Frozen contract (contract version :data:`CONTRACT_VERSION`):

* :mod:`rtt.effectors.types`: specs, builds, primitives, criteria, results, morphology recipes.
* :mod:`rtt.effectors.registry`: :func:`register_family`, :func:`get_family` and the
  :class:`EffectorFamilyImpl` base class.
* :mod:`rtt.effectors.scene`: the shared world, wrist and part around one effector.
* :mod:`rtt.effectors.runner`: the shared episode runner, jobs and replays.
* :mod:`rtt.effectors.certificate`: the 18-test robustness certificate for any effector.
* :mod:`rtt.effectors.primitives`: synergy primitive, contact readouts, contact criterion.
* :mod:`rtt.effectors.parallel_jaw`: the reference family (the frozen v2 jaw path).

Importing this package loads only the types and the registry; the runner, scene and
certificate (which import MuJoCo) load on first attribute access.
"""

from __future__ import annotations

import importlib
from typing import Any

from rtt.effectors.registry import (
    EffectorFamilyImpl,
    UnknownFamilyError,
    available_families,
    get_family,
    override_family,
    register_family,
    registered_families,
    unregister_family,
)
from rtt.effectors.types import (
    ALLOWED_LICENSES,
    BRAND_TOKENS,
    CONTRACT_VERSION,
    NAME_PREFIX,
    PRODUCT_MODE_LABEL,
    AssetSource,
    BuildContext,
    ContractError,
    EffectorBuild,
    EffectorEpisodeResult,
    EffectorFamily,
    EffectorSpec,
    Engagement,
    EntrantGroup,
    Fidelity,
    GraspCriterion,
    GraspPrimitive,
    InvalidEffector,
    MorphologyRecipe,
    ParamSpec,
    PrimitivePlan,
    RegistrationReference,
    Requirement,
    VisualAsset,
    check_display_name,
    find_brand_tokens,
)

_LAZY = {
    "run_effector_episode": "rtt.effectors.runner",
    "run_effector_jobs": "rtt.effectors.runner",
    "run_native_episode": "rtt.effectors.runner",
    "EffectorJob": "rtt.effectors.runner",
    "build_effector_scene": "rtt.effectors.scene",
    "EffectorScene": "rtt.effectors.scene",
    "certify_effector": "rtt.effectors.certificate",
    "EffectorCertificate": "rtt.effectors.certificate",
    "SynergyPrimitive": "rtt.effectors.primitives",
    "ContactCriterion": "rtt.effectors.primitives",
}

__all__ = [
    "ALLOWED_LICENSES",
    "BRAND_TOKENS",
    "CONTRACT_VERSION",
    "NAME_PREFIX",
    "PRODUCT_MODE_LABEL",
    "AssetSource",
    "BuildContext",
    "ContactCriterion",
    "ContractError",
    "EffectorBuild",
    "EffectorCertificate",
    "EffectorEpisodeResult",
    "EffectorFamily",
    "EffectorFamilyImpl",
    "EffectorJob",
    "EffectorScene",
    "EffectorSpec",
    "Engagement",
    "EntrantGroup",
    "Fidelity",
    "GraspCriterion",
    "GraspPrimitive",
    "InvalidEffector",
    "MorphologyRecipe",
    "ParamSpec",
    "PrimitivePlan",
    "RegistrationReference",
    "Requirement",
    "SynergyPrimitive",
    "UnknownFamilyError",
    "VisualAsset",
    "available_families",
    "build_effector_scene",
    "certify_effector",
    "check_display_name",
    "find_brand_tokens",
    "get_family",
    "override_family",
    "register_family",
    "registered_families",
    "run_effector_episode",
    "run_effector_jobs",
    "run_native_episode",
    "unregister_family",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'rtt.effectors' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)
