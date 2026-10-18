"""Library items as effector entrants (product mode): the compile and attach gate.

* every item with a native model or a rigid proxy is an entrant of an existing family, or says
  why not;
* every entrant builds in the shared scene on the training parts with no ``ContractError``
  (a part wider than an opening is a recorded refusal, not a failure);
* library entrants are not default presets: the catalog group stays at the founder's 12;
* episodes are deterministic in process and in a spawned worker (gate C3).
"""

from __future__ import annotations

from typing import Any

import pytest

from rtt.effectors.catalog.library_models import LIBRARY_MODELS
from rtt.effectors.catalog.models import CATALOG_MODELS
from rtt.effectors.registry import presets_of
from rtt.effectors.runner import EffectorJob, run_effector_jobs
from rtt.effectors.types import EffectorFamily, check_display_name
from rtt.library.brands import check_library_name
from rtt.library.entrants import GATE_PARTS, entrant_reason, gate_item, item_specs, library_entrants
from rtt.library.records import item_keys, read_item
from rtt.parts.library import get_part
from rtt.sim import objects as objs
from rtt.sim.task import default_task_v2

KEYS = item_keys()
ENTRANT_KEYS = [k for k in KEYS if item_specs(k)]


def test_every_simulated_item_is_an_entrant_or_says_why() -> None:
    for key in KEYS:
        item = read_item(key)
        if item.sim_status in ("native", "proxy"):
            assert item_specs(key) or entrant_reason(key), key
        else:
            assert not item_specs(key), key
    assert len(ENTRANT_KEYS) >= 15


def test_library_models_have_vendored_items_and_generic_names() -> None:
    assert not set(LIBRARY_MODELS) & set(CATALOG_MODELS)
    for key, cm in LIBRARY_MODELS.items():
        assert key in KEYS and read_item(key).sim.get("model_file"), key
        assert not cm.in_presets
        check_display_name(cm.display_name)
        check_library_name(cm.display_name)
    for spec in library_entrants():
        check_display_name(spec.display_name)
        check_library_name(spec.display_name)


def test_library_entrants_are_not_default_presets() -> None:
    catalog = presets_of([EffectorFamily.CATALOG_HAND, EffectorFamily.CATALOG_GRIPPER])
    assert len(catalog) == 12  # founder decision of 2026-10-17, unchanged
    assert not {s.model for s in catalog} & set(LIBRARY_MODELS)
    ids = [s.effector_id for s in library_entrants()]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("key", ENTRANT_KEYS)
def test_entrants_compile_and_attach_on_the_training_parts(key: str) -> None:
    results = gate_item(key, parts=GATE_PARTS)
    assert results
    for r in results:
        assert r.error is None, r.error
        assert r.passed and (r.built or r.probe), f"{key} built on no part: {r.refused}"
        assert set(r.built) | set(r.refused) == set(GATE_PARTS)
        assert r.mass_kg is not None and 0 < r.mass_kg <= 5.0


@pytest.fixture(scope="module")
def task() -> Any:
    return default_task_v2()


def test_a_library_entrant_is_deterministic_in_process_and_in_workers(task: Any) -> None:
    spec = item_specs("gripper-2f-force-48")[0]
    part = get_part("block")
    jobs = [EffectorJob(spec, part, objs.Placement(), 0, task=task)] * 2
    here = run_effector_jobs(jobs, workers=1)
    there = run_effector_jobs(jobs[:1], workers=2)
    assert here[0].traj_sha == here[1].traj_sha == there[0].traj_sha
    assert here[0].held  # the force-controlled gripper holds the block at nominal
