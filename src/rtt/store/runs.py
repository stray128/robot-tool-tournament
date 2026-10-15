"""Search results on disk: JSONL logs written as a run progresses, Parquet tables at the end.

Per run (``<out>/runs/<run_id>/``), appended round by round so a crashed job keeps
what it finished:

* ``history.jsonl``: one line per simulated design (proposal, score, episodes).
* ``rejections.jsonl``: every proposal that never reached the simulator.
* ``llm_calls.jsonl``: one line per model call (request hash, tokens, $, latency,
  the model's analysis). Message bodies live in the response cache.

Per arm (``<out>/results/``), written by :func:`write_tables`:

* ``designs.parquet`` / ``designs.jsonl``: one flat row per simulated design.
* ``episodes.parquet``: one row per simulated episode.
* ``rejections.parquet``: one row per rejected proposal.

Rows contain no timestamps, so a cache replay reproduces them byte for byte.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from rtt.design.schema import GEOMETRY_PARAMS

__all__ = [
    "append_jsonl",
    "design_row",
    "episode_rows",
    "read_jsonl",
    "read_table",
    "rejection_row",
    "write_jsonl",
    "write_parquet",
    "write_tables",
]

FAILURE_COLUMNS = (
    "missed_grasp",
    "slip_during_lift",
    "ejected_on_close",
    "unstable_hold",
    "finger_env_collision",
)


def append_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    """Append records as JSON lines (sorted keys, no NaN)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    return target


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    """Replace ``path`` with the given JSON lines."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")
    return append_jsonl(target, records)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    with p.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def design_row(
    run: Mapping[str, Any], evaluated: Mapping[str, Any], families: Sequence[str]
) -> dict[str, Any]:
    """Flatten one ``Evaluated.to_json()`` record plus run identity into a table row."""
    proposal = evaluated["proposal"]
    score = evaluated["score"]
    design = proposal["design"]
    row: dict[str, Any] = {
        "arm": run["arm"],
        "run_id": run["run_id"],
        "run_index": run["run_index"],
        "run_seed": run["run_seed"],
        "round": evaluated["round"],
        "eval_index": evaluated["index"],
        "label": evaluated["label"],
        "design_id": proposal["design_id"],
        "origin": proposal["origin"],
        "operator": proposal["operator"],
        "island": proposal["island"],
        "slot": proposal["slot"],
        "parents": ",".join(proposal["parents"]),
        "via": proposal["via"],
        "fallback_reason": proposal["fallback_reason"],
        "request_hash": proposal["request_hash"],
        "hypothesis": proposal["hypothesis"],
        "name": design.get("name"),
    }
    for key in GEOMETRY_PARAMS:
        row[key] = design[key]
    row.update(
        {
            "mass_g": score["mass_g"],
            "success": score["success"],
            "best_k": score["best_k"],
            "hold_fraction": score["hold_fraction"],
            "n_episodes": score["n_episodes"],
        }
    )
    for k, value in enumerate(score["success_by_k"]):
        row[f"success_k{k}"] = value
    for family in families:
        row[f"success_{family}"] = score["per_family"].get(family)
    for mode in FAILURE_COLUMNS:
        row[f"fail_{mode}"] = int(score["failure_modes"].get(mode, 0))
    return row


def episode_rows(run: Mapping[str, Any], evaluated: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per episode of one simulated design."""
    base = {
        "arm": run["arm"],
        "run_id": run["run_id"],
        "run_index": run["run_index"],
        "round": evaluated["round"],
        "eval_index": evaluated["index"],
        "design_id": evaluated["proposal"]["design_id"],
    }
    return [{**base, **episode} for episode in evaluated["score"].get("episodes", [])]


def rejection_row(run: Mapping[str, Any], rejection: Mapping[str, Any]) -> dict[str, Any]:
    params = rejection.get("params")
    return {
        "arm": run["arm"],
        "run_id": run["run_id"],
        "run_index": run["run_index"],
        "round": rejection["round"],
        "island": rejection.get("island"),
        "slot": rejection.get("slot"),
        "stage": rejection["stage"],
        "reason": rejection["reason"],
        "errors": " | ".join(rejection.get("errors", [])),
        "params": json.dumps(params, sort_keys=True) if params is not None else None,
        "request_hash": rejection.get("request_hash"),
    }


def write_parquet(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write rows (dicts with the same keys) as one Parquet table."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(r) for r in rows])
    pq.write_table(table, target)
    return target


def read_table(path: str | Path) -> list[dict[str, Any]]:
    """Rows of a Parquet file as dicts."""
    return pq.read_table(Path(path)).to_pylist()


def write_tables(
    out_dir: str | Path,
    runs: Sequence[Mapping[str, Any]],
    histories: Sequence[Sequence[Mapping[str, Any]]],
    rejections: Sequence[Sequence[Mapping[str, Any]]],
    families: Sequence[str],
) -> dict[str, Path]:
    """Write the arm-level tables for ``runs`` (parallel lists of run identity, history and rejections)."""
    results = Path(out_dir)
    designs: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for run, history, run_rejections in zip(runs, histories, rejections, strict=True):
        for item in history:
            designs.append(design_row(run, item, families))
            episodes.extend(episode_rows(run, item))
        rejected.extend(rejection_row(run, r) for r in run_rejections)
    paths = {
        "designs_parquet": write_parquet(results / "designs.parquet", designs),
        "designs_jsonl": write_jsonl(results / "designs.jsonl", designs),
    }
    if episodes:
        paths["episodes_parquet"] = write_parquet(results / "episodes.parquet", episodes)
    if rejected:
        paths["rejections_parquet"] = write_parquet(results / "rejections.parquet", rejected)
    return paths
