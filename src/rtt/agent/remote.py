"""In-container entry for agent jobs: ``python -m rtt.agent.remote --run-dir /out/runs/<run_name>``.

:class:`rtt.agent.jobs.NebiusJobRunner` uploads ``inputs/part.stl``, ``inputs/part.json``
(the intake options) and ``inputs/designs.json`` to the run's bucket prefix, then starts
this module in the pinned image. It loads the part with the same backend the laptop uses
(:func:`rtt.agent.backends.default_part_backend`), simulates every design on CPU MuJoCo
and writes ``results.json`` (one :class:`~rtt.search.base.DesignScore` per design) and
``manifest.json``. Files are staged locally and copied whole to the bucket mount, which
rejects seeks and renames (see :func:`rtt.cloud.entry.publish`); the manifest goes last.

Exit status: 0 on success, 2 on an exception (the manifest then carries the error).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rtt.design.schema import FingerDesign

__all__ = ["main", "run_remote"]


def run_remote(
    run_dir: str | Path, *, stage_dir: str | Path | None = None, workers: int | None = None
) -> int:
    """Evaluate ``<run_dir>/inputs`` and publish the results into ``run_dir``; return the exit status."""
    from rtt.agent.backends import default_part_backend
    from rtt.cloud.entry import MANIFEST, base_manifest, copy_file, publish, utc_iso, write_json

    out = Path(run_dir)
    inputs = out / "inputs"
    stage = Path(stage_dir) if stage_dir is not None else Path(tempfile.mkdtemp(prefix="rtt-agent-"))
    stage.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    manifest: dict[str, Any] = base_manifest("agent-part", out, sys.argv)
    status = 0
    try:
        meta = json.loads((inputs / "part.json").read_text(encoding="utf-8"))
        raw = json.loads((inputs / "designs.json").read_text(encoding="utf-8"))
        designs = [FingerDesign.model_validate({k: v for k, v in d.items() if k != "design_id"}) for d in raw]
        work = Path(tempfile.mkdtemp(prefix="rtt-agent-part-"))
        backend = default_part_backend(work, workers=workers)
        part_id = str(meta["part_id"])
        info = backend.intake(
            inputs / "part.stl",
            part_id=part_id,
            name=str(meta.get("name") or part_id),
            units=meta.get("units", "mm"),
            density=meta.get("density"),
        )
        t_sim = time.time()
        scores = backend.evaluate(part_id, designs)
        write_json(
            stage / "results.json",
            {
                "status": "complete",
                "part": info.to_json(),
                "backend": backend.name,
                "scores": [s.to_json(include_episodes=True) for s in scores],
                "sim_s": round(time.time() - t_sim, 3),
            },
        )
        manifest["part_id"] = part_id
        manifest["designs"] = [d.design_id() for d in designs]
    except Exception as exc:  # noqa: BLE001 - recorded in the manifest, reported by exit status
        status = 2
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["traceback_tail"] = traceback.format_exc()[-2000:]
    published = publish(stage, out)
    names = published.pop("names")
    manifest["publish"] = published
    manifest["timing_s"] = {"total": round(time.time() - t0, 2)}
    manifest["finished_at"] = utc_iso()
    manifest["exit_status"] = status
    manifest["files"] = sorted(names) + [MANIFEST]
    write_json(stage / MANIFEST, manifest)
    copy_file(stage / MANIFEST, out / MANIFEST)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rtt.agent.remote", description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="run directory on the bucket mount")
    parser.add_argument("--stage", default=None, help="local stage directory (default: a temp dir)")
    parser.add_argument("--workers", type=int, default=None, help="CPU worker processes (default: all)")
    args = parser.parse_args(argv)
    return run_remote(args.run_dir, stage_dir=args.stage, workers=args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
