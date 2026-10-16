"""The v2 diversity bars under the full section-13 validators (spec section 12, rule D6).

:func:`rtt.search.diversity.threshold_rule` on the ``freeform-v2`` space with the training
parts and ``task_v2`` as the validation context, so the 2,000 random designs it draws pairs
from are exactly the designs arm A can simulate (schema, analytic, build, printability and
seat checks on all six parts). The result is written to ``configs/diversity_v2.json`` with
the hashes of the inputs it depends on; :data:`rtt.search.diversity.MIN_DISTANCE_V2` and
:data:`~rtt.search.diversity.EXPLORE_MIN_DISTANCE_V2` are pinned to it
(``tests/test_search_dedup.py`` checks they agree).

Usage::

    python -m rtt.search.threshold_v2 --workers 14 --out configs/diversity_v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

__all__ = ["DEFAULT_OUT", "compute", "main"]

DEFAULT_OUT: Final = Path(__file__).resolve().parents[3] / "configs" / "diversity_v2.json"
SCHEMA: Final = "rtt.diversity-rule/1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute(workers: int | None = None) -> dict[str, Any]:
    """The threshold rule on the training parts under ``task_v2``, with its inputs' hashes."""
    from rtt.parts.library import DEFAULT_PARTS_FILE, training_parts
    from rtt.parts.suite import DEFAULT_TRAIN_V2
    from rtt.search.diversity import threshold_rule
    from rtt.search.space import V2, get_space
    from rtt.search.space_v2 import V2Context, _generator_version
    from rtt.sim.task import DEFAULT_TASK_V2_PATH, default_task_v2

    space = get_space(V2)
    context = V2Context(parts=tuple(training_parts()), task=default_task_v2(), label="train_v2")
    t0 = time.perf_counter()
    rule = threshold_rule(space, context, workers=workers)
    return {
        "schema": SCHEMA,
        "rule": "t: at most 1 % of 10,000 random valid Sobol pairs lie strictly within t (rounded down to"
        " 0.001); explore bar: median same-cell distance of the pairs (rounded down to 0.01)",
        "validators": "schema + geometry (spec section 13) on the training parts, task_v2",
        **rule.to_json(),
        "valid_rate": round(rule.pool / rule.draws, 6),
        "generator_version": _generator_version(),
        "inputs_sha256": {
            "configs/parts_v2.yaml": _sha(DEFAULT_PARTS_FILE),
            "configs/task_v2.yaml": _sha(DEFAULT_TASK_V2_PATH),
            "configs/suites/train_v2.yaml": _sha(DEFAULT_TRAIN_V2),
        },
        "part_ids": list(context.part_ids),
        "wall_s": round(time.perf_counter() - t0, 1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rtt.search.threshold_v2", description=__doc__.split("\n")[0]
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="validation processes (default: in-process)"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    result = compute(args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    keys = ("min_distance", "explore_min_distance", "pool", "draws", "valid_rate", "wall_s")
    print(json.dumps({k: result[k] for k in keys}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
