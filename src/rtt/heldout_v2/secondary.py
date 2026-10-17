"""The secondary family of PREREG A6/A10: S2, H1 and H2, Holm-adjusted, in one results file.

``python -m rtt.heldout_v2.secondary --random <A dir> --nemotron <C dir> --heldout <H1 dir>
--heldout-real <H2 dir> --out DIR`` (also ``rtt analyze-secondary``) writes
``secondary_v2.json`` (schema ``rtt.secondary_v2/1``). Every test is a function of
``src/rtt/analysis/s1.py`` (hash in PREREG A5), on the runs
:func:`rtt.analysis.tau.load_runs_v2` returns for each arm:

* **S2**: ``permutation_test_means`` of :func:`~rtt.analysis.s1.best_so_far_area` (the mean
  best-so-far training score over designs 9-40), C > A.
* **H1**, **H2**: ``permutation_test_means`` of the run-level held-out means of each run's
  frozen pick (``heldout_v2.json``, ``heldout_real_v2.json``), C > A. Each search run must
  appear in the held-out report exactly once, with the frozen record the search wrote (same
  arm, seed and record hash), and the reports must not be exploratory.
* **Holm** (:func:`~rtt.analysis.s1.holm`) over the members that ran, each rejected when its
  adjusted p-value is at most 0.05. Arm Z is exploratory, so S3 is not a member.

Reported alongside, outside the family: the H2 width bands (at most 50 mm and above 50 mm
across the jaw, a planned split tested the same way; no claim rests on a band alone), and
the lift-only versions of S2, H1 and H2 (exploratory). At R = 32 the permutation tests use
200,000 seeded relabellings (seed 20261016), the defaults of ``permutation_test_means``;
runs enter each test in seed order, so the result is reproducible.

Refused unless the runs are the registered design (:func:`rtt.analysis.tau.check_confirmatory`:
arm labels, seeds 1001-1032 in both arms, one task and suite), except with ``--exploratory``,
which labels the output so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from rtt.heldout_v2 import common

__all__ = ["ALPHA", "FAMILY", "OUT", "SCHEMA", "analyze", "main"]

SCHEMA: Final = "rtt.secondary_v2/1"
OUT: Final = "secondary_v2.json"
ALPHA: Final = 0.05
FAMILY: Final = ("H1", "H2", "S2")
SEEDS_MAIN: Final = common.REPO_ROOT / "configs" / "seeds_main.yaml"
H1_REPORT: Final = "heldout_v2.json"
H2_REPORT: Final = "heldout_real_v2.json"
BANDS: Final = ("width_le_50mm", "width_gt_50mm")


def _seeds(path: Path) -> list[int]:
    import yaml

    return [int(s) for s in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("seeds", [])]


def _report_path(path: str | Path, name: str) -> Path:
    p = Path(path)
    for candidate in (p, p / name, p / "procedural" / name, p / "real" / name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{p}: no {name}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _test(a: Sequence[float], c: Sequence[float]) -> dict[str, Any]:
    from rtt.analysis.s1 import permutation_test_means

    res = permutation_test_means(list(a), list(c))
    return {
        "mean_a": sum(a) / len(a),
        "mean_c": sum(c) / len(c),
        "diff_c_minus_a": res.statistic,
        "p": res.p_value,
        "method": res.method,
        "n_a": res.n_a,
        "n_c": res.n_c,
        "values_a": list(a),
        "values_c": list(c),
    }


def _heldout_rows(
    report: Mapping[str, Any], arm: str, seeds: Sequence[int], frozen: Mapping[tuple[str, int], str]
) -> list[Mapping[str, Any]]:
    """The report's run rows of ``arm``, in ``seeds`` order, checked against the search's picks."""
    rows = {int(r["run_seed"]): r for r in (report.get("arms", {}).get(arm, {}).get("runs") or [])}
    if sorted(rows) != sorted(seeds):
        raise ValueError(
            f"{report.get('endpoint')}: arm {arm} has held-out runs for seeds {sorted(rows)}, "
            f"the search has {sorted(seeds)}"
        )
    for seed in seeds:
        want = frozen.get((arm, seed))
        if want is None or rows[seed]["frozen_sha256"] != want:
            raise ValueError(
                f"{report.get('endpoint')}: {arm} seed {seed} was scored from another frozen record than"
                " the search wrote"
            )
    return [rows[s] for s in seeds]


def _check_report(report: Mapping[str, Any], schema: str, confirmatory: bool) -> list[str]:
    if report.get("schema") != schema:
        raise ValueError(f"expected a {schema} report, got {report.get('schema')!r}")
    reasons = list(report.get("exploratory_reasons") or [])
    if report.get("exploratory") and confirmatory:
        raise ValueError(f"{report.get('endpoint')}: the held-out report is exploratory ({reasons})")
    return reasons


def analyze(
    random_dirs: Sequence[str | Path],
    nemotron_dirs: Sequence[str | Path],
    *,
    heldout: str | Path | None = None,
    heldout_real: str | Path | None = None,
    seeds_file: str | Path = SEEDS_MAIN,
    confirmatory: bool = True,
) -> dict[str, Any]:
    """The secondary family from the fetched search outputs and the held-out reports."""
    from rtt.analysis.s1 import best_so_far_area, holm
    from rtt.analysis.tau import check_confirmatory, check_same_task, load_runs_v2

    dirs = [*random_dirs, *nemotron_dirs]
    runs_a = sorted((r for d in random_dirs for r in load_runs_v2(d)), key=lambda r: r.run_seed)
    runs_c = sorted((r for d in nemotron_dirs for r in load_runs_v2(d)), key=lambda r: r.run_seed)
    lift_a = sorted(
        (r for d in random_dirs for r in load_runs_v2(d, score_key="lift_success")), key=lambda r: r.run_seed
    )
    lift_c = sorted(
        (r for d in nemotron_dirs for r in load_runs_v2(d, score_key="lift_success")),
        key=lambda r: r.run_seed,
    )
    notes = []
    if confirmatory:
        check_confirmatory(runs_a, runs_c, _seeds(Path(seeds_file)))
        check_same_task(dirs, [*runs_a, *runs_c])
    else:
        notes.append("exploratory: the confirmatory checks on arms, seeds and task were not applied")
    seeds_a = [r.run_seed for r in runs_a]
    seeds_c = [r.run_seed for r in runs_c]
    frozen = {(f.arm, f.run_seed): f.content_sha256 for f in common.collect_frozen_v2(dirs)}

    tests: dict[str, Any] = {}
    exploratory: dict[str, Any] = {}
    tests["S2"] = {
        "endpoint": "area under the best-so-far training curve, designs 9-40 (best_so_far_area), C > A",
        **_test([best_so_far_area(r.scores) for r in runs_a], [best_so_far_area(r.scores) for r in runs_c]),
    }
    exploratory["S2_lift_only"] = _test(
        [best_so_far_area(r.scores) for r in lift_a], [best_so_far_area(r.scores) for r in lift_c]
    )
    inputs: dict[str, Any] = {
        "random": [str(d) for d in random_dirs],
        "nemotron": [str(d) for d in nemotron_dirs],
        "seeds_a": seeds_a,
        "seeds_c": seeds_c,
    }
    for name, path, schema, file in (
        ("H1", heldout, "rtt.heldout/2", H1_REPORT),
        ("H2", heldout_real, "rtt.heldout_real/2", H2_REPORT),
    ):
        if path is None:
            notes.append(f"{name} not given: not a member of this Holm family")
            continue
        report_file = _report_path(path, file)
        report = json.loads(report_file.read_text(encoding="utf-8"))
        reasons = _check_report(report, schema, confirmatory)
        if reasons:
            notes.append(f"{name} report is exploratory: {reasons}")
        rows_a = _heldout_rows(report, "random", seeds_a, frozen)
        rows_c = _heldout_rows(report, "nemotron", seeds_c, frozen)
        inputs[name] = {
            "report": str(report_file),
            "sha256": _sha256(report_file),
            "suite": report["suite"]["digest"],
        }
        tests[name] = {
            "endpoint": f"{name}: run-level mean success_v2 of each run's frozen pick, C > A",
            **_test([r["success"] for r in rows_a], [r["success"] for r in rows_c]),
        }
        exploratory[f"{name}_lift_only"] = _test(
            [r["lift_success"] for r in rows_a], [r["lift_success"] for r in rows_c]
        )
        if name == "H2":
            bands = {}
            for band in BANDS:
                for key, label in (("success", band), ("lift_success", f"{band}_lift_only")):
                    a = [r["by_band"][key][band] for r in rows_a]
                    c = [r["by_band"][key][band] for r in rows_c]
                    if any(v is None for v in [*a, *c]):
                        bands[label] = {"skipped": "no object in this band"}
                    else:
                        bands[label] = _test(a, c)
            exploratory["H2_width_bands"] = {
                "note": "planned split (PREREG A6, option A): reported for both bands, outside the Holm family",
                "band_edge_mm": report["suite"].get("band_edge_mm"),
                "objects_by_band": report["suite"].get("objects_by_band"),
                **bands,
            }
    members = {k: tests[k]["p"] for k in FAMILY if k in tests}
    adjusted = holm(members)
    for k, p_adj in adjusted.items():
        tests[k]["p_holm"] = p_adj
        tests[k]["rejected"] = bool(p_adj <= ALPHA)
    return {
        "schema": SCHEMA,
        "confirmatory": confirmatory,
        "alpha": ALPHA,
        "family": sorted(members),
        "not_run": {"S3": "arm Z is exploratory (PREREG A2, A6)"},
        "tests": tests,
        "exploratory": exploratory,
        "inputs": inputs,
        "notes": notes,
        "heldout_code": common.code_listing(),
    }


def main(argv: Sequence[str] | None = None, *, prog: str = "python -m rtt.heldout_v2.secondary") -> int:
    from rtt.store.manifest import code_facts, write_json_atomic

    parser = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    parser.add_argument("--random", nargs="+", type=Path, required=True, help="arm A search output(s)")
    parser.add_argument("--nemotron", nargs="+", type=Path, required=True, help="arm C search output(s)")
    parser.add_argument("--heldout", type=Path, default=None, help="H1 report or its directory")
    parser.add_argument("--heldout-real", type=Path, default=None, help="H2 report or its directory")
    parser.add_argument("--seeds", type=Path, default=SEEDS_MAIN, help="confirmatory seed list")
    parser.add_argument("--exploratory", action="store_true", help="skip the confirmatory checks (labelled)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = analyze(
            args.random,
            args.nemotron,
            heldout=args.heldout,
            heldout_real=args.heldout_real,
            seeds_file=args.seeds,
            confirmatory=not args.exploratory,
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    result["code"] = code_facts()
    path = write_json_atomic(args.out / OUT, result)
    for name in FAMILY:
        t = result["tests"].get(name)
        if t is None:
            print(f"{name}: not run")
            continue
        print(
            f"{name}: mean A {t['mean_a']:.4f}, C {t['mean_c']:.4f}, diff {t['diff_c_minus_a']:+.4f},"
            f" p {t['p']:.4g}, Holm {t['p_holm']:.4g} -> {'reject H0' if t['rejected'] else 'keep H0'}"
        )
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
