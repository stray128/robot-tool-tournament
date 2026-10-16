"""Run gate GV-H (the hero screen) on the laptop.

Every training part at the seven hero conditions (:data:`rtt.analysis.hero_rule.HERO_CONDITIONS`,
three jitter seeds each, from :func:`rtt.parts.suite.hero_screen_suite`) with the stock blade,
the configurator imprint and the designed candidates; the verdicts are
:func:`rtt.analysis.hero_rule.evaluate_part`'s. The training parts are procedural and
unbranded, so each passes the brand screen.

The designed candidates are fixed before the screen runs: the two reference entrants and the
``--top`` best Sobol designs of a GV3 report (by registered score, then lift-only score, then
lower mass), read from that report's ``gv3_designs.json``. The rule picks the best of them
per part.

    python -m rtt.analysis.gvh_run --gv3 runs/gv3 --out runs/gvh [--top 5] [--workers 14]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rtt.analysis import hero_rule as hr


def candidates(gv3_dir: Path, top: int) -> list[tuple[str, Any]]:
    from rtt.design.freeform import FingerRecipe, load_entrants_v2, reference_entrants

    report = json.loads((gv3_dir / "gv3_report.json").read_text(encoding="utf-8"))
    designs = {d["name"]: d for d in json.loads((gv3_dir / "gv3_designs.json").read_text(encoding="utf-8"))}
    ranked = sorted(
        report["sobol"], key=lambda r: (-r["score"], -r["lift_score"], r["mass_g"] or 0.0, r["name"])
    )
    out: list[tuple[str, Any]] = [(e.name, e) for e in reference_entrants(load_entrants_v2())]
    for row in ranked[:top]:
        out.append((row["name"], FingerRecipe.model_validate(designs[row["name"]]["design"])))
    return out


def run(
    gv3_dir: Path, out: Path, *, top: int = 5, workers: int | None = None, native_ccd: bool | None = None
) -> hr.HeroScreen:
    from rtt.design.freeform import load_entrants_v2
    from rtt.parts.suite import hero_screen_suite
    from rtt.sim.evaluate import evaluate_designs_v2
    from rtt.sim.task import default_task_v2

    task = default_task_v2()
    if native_ccd is not None:
        task = task.with_overrides(physics={"native_ccd": bool(native_ccd)})
    entrants = {e.name: e for e in load_entrants_v2()}
    cands = candidates(gv3_dir, top)
    lanes: list[tuple[str, Any]] = [
        ("stock", entrants["stock"]),
        ("imprint", entrants["imprint_configurator"]),
    ]
    lanes += [(f"designed:{name}", d) for name, d in cands]
    suite = hero_screen_suite(hr.HERO_CONDITIONS)
    records = [
        r.to_dict() for r in evaluate_designs_v2([d for _, d in lanes], suite, workers=workers, task=task)
    ]
    per = suite.n_episodes
    by_part: dict[str, dict[str, Any]] = {}
    for k, (lane, _) in enumerate(lanes):
        for rec in records[k * per : (k + 1) * per]:
            part = by_part.setdefault(
                rec["part_id"],
                {"part_id": rec["part_id"], "eligible": True, "stock": [], "imprint": [], "designed": {}},
            )
            if lane == "stock":
                part["stock"].append(rec)
            elif lane == "imprint":
                part["imprint"].append(rec)
            else:
                part["designed"].setdefault(lane.split(":", 1)[1], []).append(rec)
    screen_in = {"parts": list(by_part.values())}
    out.mkdir(parents=True, exist_ok=True)
    (out / "hero_screen_input.json").write_text(json.dumps(screen_in, default=float) + "\n", encoding="utf-8")
    screen = hr._screen_from_json(screen_in, hr.HeroRule())
    doc = screen.to_json()
    doc["candidates"] = [name for name, _ in cands]
    doc["native_ccd"] = bool(task.physics.native_ccd)
    (out / "hero_screen.json").write_text(json.dumps(doc, indent=1, default=float) + "\n", encoding="utf-8")
    return screen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--gv3", type=Path, required=True, help="a GV3 output directory (gv3_report.json, gv3_designs.json)"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--native-ccd", choices=("true", "false"), default=None)
    args = ap.parse_args(argv)
    native = None if args.native_ccd is None else args.native_ccd == "true"
    screen = run(args.gv3, args.out, top=args.top, workers=args.workers, native_ccd=native)
    for v in screen.verdicts:
        tag = "imprint hero" if v.imprint_hero else "hero" if v.hero else "-"
        print(f"{v.part_id:<24}{tag:<14}" + "; ".join(v.reasons))
    print(f"heroes: {list(screen.heroes)}; hook mode: {screen.hook_mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
