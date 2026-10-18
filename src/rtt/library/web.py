"""The catalog page's data: ``web/public/data/library/index.json`` (product mode).

:func:`export_web` reads every item's ``SOURCE.json``, ``catalog.json`` (listed and excluded
items) and ``lock.json``, runs the compile and attach gate (:mod:`rtt.library.entrants`, with one
nominal episode per built part) and writes one JSON file the site renders at build time. Every
string that reaches the screen is checked against the library's brand list
(:mod:`rtt.library.brands`, URLs aside): makers' names stay on the credits page, which reads
``SOURCE.json`` directly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from rtt.library.brands import find_library_brands
from rtt.library.entrants import entrant_reason, gate_item, item_specs
from rtt.library.licence import GREEN_LICENCES, LICENCE_URLS, Rating, classify
from rtt.library.mounts import CONFIRMED_PATTERNS, ISO_PATTERNS
from rtt.library.records import LIBRARY_DIR, REPO_ROOT, item_keys, read_item, read_lock
from rtt.library.taxonomy import CAPABILITIES, CAPABILITY_NOTES, CLASS_TITLES, KIND_TITLES, SIM_STATUS_TITLES

__all__ = ["INDEX_FORMAT", "INDEX_PATH", "LABEL", "export_web", "web_index"]

INDEX_FORMAT: Final = "rtt-web-library"
INDEX_PATH: Final = REPO_ROOT / "web" / "public" / "data" / "library" / "index.json"
#: The effector layer's label (equal to :data:`rtt.effectors.types.PRODUCT_MODE_LABEL`).
LABEL: Final = "Product mode: not part of the pre-registered experiment"
ROLE_TITLES: Final = {
    "print": "Print file",
    "cad": "CAD",
    "sim": "Simulation file",
    "derived": "Simulation file (derived)",
    "doc": "Documentation",
    "licence": "Licence text",
}
_URL = re.compile(r"https?://\S+")


def _screen(value: Any, where: str) -> None:
    """Refuse a brand word anywhere in ``value`` (strings, recursively), URLs aside."""
    if isinstance(value, str):
        hits = find_library_brands(_URL.sub(" ", value))
        if hits:
            raise ValueError(f"{where}: on-screen text names a brand ({', '.join(sorted(set(hits)))})")
    elif isinstance(value, Mapping):
        for k, v in value.items():
            if k in ("url", "source", "upstream_url", "repository", "href"):
                continue
            _screen(v, f"{where}.{k}")
    elif isinstance(value, list | tuple):
        for i, v in enumerate(value):
            _screen(v, f"{where}[{i}]")


def _licences(item: Any) -> list[dict[str, Any]]:
    return [
        {
            "spdx": str(lic["spdx"]),
            "covers": str(lic.get("covers", "")),
            "url": LICENCE_URLS.get(str(lic["spdx"])),
        }
        for lic in item.licence
    ]


def _downloads(item: Any) -> list[dict[str, Any]]:
    out = []
    for f in item.files:
        if f.role == "licence":
            continue
        out.append(
            {
                "path": f.path,
                "name": Path(f.path).name,
                "role": f.role,
                "role_title": ROLE_TITLES[f.role],
                "bytes": f.size,
                "license": f.license,
                "sha256": f.sha256,
                "modified": bool(f.changes),
            }
        )
    return out


def _gate_summary(results: list[Any]) -> dict[str, Any]:
    if not results:
        return {"ran": False}
    entries = []
    for r in results:
        d = r.to_dict()
        held = [p for p, n in d["nominal"].items() if n["held"]]
        d["nominal_held"] = held
        entries.append(d)
    return {"ran": True, "passed": all(r.passed for r in results), "entrants": entries}


def _item(key: str, gate: bool) -> dict[str, Any]:
    item = read_item(key)
    repo, commit = item.upstream["repository"], item.upstream["commit"]
    generated = commit == "generated"
    sim = dict(item.sim)
    specs = item_specs(key)
    results = gate_item(key, nominal=True) if gate and specs else []
    reason = entrant_reason(key)
    entry = {
        "key": key,
        "display_name": item.display_name,
        "summary": item.summary,
        "kind": item.kind,
        "kind_title": KIND_TITLES[item.kind],
        "class": item.item_class,
        "class_title": CLASS_TITLES[item.item_class],
        "capabilities": list(item.capabilities),
        "licences": _licences(item),
        "rating": item.rating.value,
        "mount": item.mount.to_dict(),
        "specs": [s.to_dict() for s in item.specs],
        "sim": {
            "status": item.sim_status,
            "title": SIM_STATUS_TITLES[item.sim_status],
            "source": str(sim.get("source", "")),
            "fidelity": sim.get("fidelity"),
            "note": str(sim.get("note", "")),
            "reason": str(sim.get("reason", "")),
            "family": sim.get("family"),
            "n_actuators": sim.get("n_actuators"),
            "mass_kg": sim.get("mass_kg"),
            "entrant": bool(specs),
            "entrant_reason": reason,
        },
        "gate": _gate_summary(results),
        "preview": (
            {
                "url": "data/library/glb/" + Path(str(item.preview["path"])).name,
                "bytes": int(item.preview["size"]),
            }
            if item.preview
            else None
        ),
        "downloads": _downloads(item),
        "links": [
            {
                "label": str(s.get("reason", "")).split(":")[0],
                "reason": str(s.get("reason", "")),
                "url": s["url"],
                "bytes": int(s.get("size", 0)),
            }
            for s in item.skipped
        ],
        "upstream": (
            {"generated": True}
            if generated
            else {
                "repository": repo,
                "commit": commit,
                "url": f"{repo}/tree/{commit}",
                "extra": [
                    {
                        "repository": e["repository"],
                        "commit": e["commit"],
                        "url": f"{e['repository']}/tree/{e['commit']}",
                    }
                    for e in item.upstream.get("extra", ())  # type: ignore[union-attr]
                ],
            }
        ),
        "notes": list(item.notes),
        "credit_anchor": f"library-{key}",
    }
    _screen({k: v for k, v in entry.items() if k not in ("upstream",)}, f"items.{key}")
    return entry


def web_index(*, gate: bool = True) -> dict[str, Any]:
    """The index the catalog page renders (module docstring)."""
    catalog = json.loads((LIBRARY_DIR / "catalog.json").read_text(encoding="utf-8"))
    items = [_item(k, gate) for k in item_keys()]
    excluded = catalog["excluded"]
    for e in excluded:
        if classify(str(e["license"])) is Rating.GREEN and e["rating"] == "red":
            raise ValueError(f"excluded item {e['name']!r} is red but its licence {e['license']} is green")
    data = {
        "format": INDEX_FORMAT,
        "version": 1,
        "label": LABEL,
        "audit": catalog["audit"],
        "counts": {
            "items": len(items),
            "entrants": sum(len(i["gate"].get("entrants", [])) for i in items),
            "entrant_items": sum(1 for i in items if i["sim"]["entrant"]),
            "by_sim_status": {s: sum(1 for i in items if i["sim"]["status"] == s) for s in SIM_STATUS_TITLES},
            "listed": len(catalog["listed"]),
            "excluded_amber": sum(1 for e in excluded if e["rating"] == "amber"),
            "excluded_red": sum(1 for e in excluded if e["rating"] == "red"),
        },
        "capabilities": [{"tag": c, "note": CAPABILITY_NOTES[c]} for c in CAPABILITIES],
        "green_licences": list(GREEN_LICENCES),
        "iso_patterns": [
            {**ISO_PATTERNS[d].to_dict(), "confirmed": d in CONFIRMED_PATTERNS} for d in sorted(ISO_PATTERNS)
        ],
        "items": items,
        "listed": catalog["listed"],
        "excluded": excluded,
        "already_vendored": catalog["already_vendored"],
        "provenance": {
            "manifest_sha256": read_lock()["manifest_sha256"],
            "lock": "assets/effectors/library/lock.json",
        },
    }
    _screen({k: data[k] for k in ("listed", "excluded", "already_vendored", "capabilities")}, "catalog")
    return data


def export_web(path: Path = INDEX_PATH, *, gate: bool = True) -> Path:
    data = web_index(gate=gate)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
