"""``python -m rtt.library``: the open effector library's commands.

python -m rtt.library check              verify every item, the licence gate and lock.json
python -m rtt.library gate [KEY ...]     compile and attach every entrant on the six
                                         training parts (and one nominal episode each)
python -m rtt.library export-web         write web/public/data/library/index.json
"""

from __future__ import annotations

import argparse
import json
import sys

from rtt.library.entrants import gate_item
from rtt.library.records import item_keys, verify_library
from rtt.library.web import export_web


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rtt.library", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    g = sub.add_parser("gate")
    g.add_argument("keys", nargs="*")
    g.add_argument("--no-nominal", action="store_true")
    e = sub.add_parser("export-web")
    e.add_argument("--no-gate", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "check":
        problems = verify_library()
        for p in problems:
            print(p)
        print(f"{len(item_keys())} items: {'ok' if not problems else f'{len(problems)} problem(s)'}")
        return 1 if problems else 0
    if args.cmd == "gate":
        bad = 0
        for key in args.keys or item_keys():
            for r in gate_item(key, nominal=not args.no_nominal):
                d = r.to_dict()
                held = sum(1 for n in d["nominal"].values() if n["held"])
                print(
                    f"{key:36s} {r.primitive:6s} {'PASS' if r.passed else 'FAIL'} built {len(r.built)}/6 "
                    f"refused {len(r.refused)} held {held} {r.error or ''}"
                )
                bad += not r.passed
        return 1 if bad else 0
    if args.cmd == "export-web":
        path = export_web(gate=not args.no_gate)
        data = json.loads(path.read_text())
        print(f"wrote {path} ({len(data['items'])} items, {data['counts']['entrants']} entrant specs)")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
