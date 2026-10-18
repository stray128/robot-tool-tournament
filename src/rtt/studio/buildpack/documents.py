"""The text files of a build pack: bill of materials, assembly notes, certificate, credits, README.

Every function takes plain JSON data (a finalist of :meth:`rtt.studio.search.TournamentReport.to_dict`
and the report around it), so a pack can be rebuilt from a saved report. On-screen text is
brand-free (display names, generic words; the certificate is checked with
:func:`rtt.effectors.types.find_brand_tokens`); credit lines with upstream names go only into
``CREDITS.txt``, the one credits file of the pack.
"""

from __future__ import annotations

import csv
import html
import io
from collections.abc import Mapping, Sequence
from typing import Any, Final

from rtt.effectors.types import PRODUCT_MODE_LABEL, find_brand_tokens
from rtt.studio.types import BomLine

__all__ = [
    "DISCLAIMER",
    "assembly_notes",
    "bom_csv",
    "bom_lines",
    "certificate_html",
    "credits_text",
    "readme_text",
]

DISCLAIMER: Final = (
    "Simulated in MuJoCo and untested on hardware. A recommendation is a simulation ranking under "
    "the stated conditions, not a hardware guarantee."
)
#: Fasteners per bolted interface (ISO 9409-1-50-4-M6 pattern) [assumed].
SCREWS_PER_INTERFACE: Final = 4


def _tool_rows(setup: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(setup["tools"])


def _actuation(tool: Mapping[str, Any], facts: Mapping[str, Any]) -> Mapping[str, Any]:
    return dict(facts.get(tool["effector_id"], {}).get("actuation") or {})


def _spec_note(tool: Mapping[str, Any], facts: Mapping[str, Any]) -> str:
    """What to buy or build: the simulated tool's key numbers (brand-free)."""
    act = _actuation(tool, facts)
    f = facts.get(tool["effector_id"], {})
    bits = [f"simulated as the {tool['family'].replace('_', ' ')} family, {tool['primitive']} primitive"]
    if act.get("opening_mm"):
        bits.append(f"opening {float(act['opening_mm']):.0f} mm")
    for key, unit, label in (
        ("linear_force_n_max", "N", "grip force"),
        ("joint_torque_nm_max", "N m", "joint torque"),
        ("holding_force_max_n", "N", "holding force"),
        ("vacuum_kpa", "kPa", "vacuum"),
        ("cable_tension_n", "N", "cable tension"),
    ):
        if act.get(key):
            bits.append(f"{label} {float(act[key]):.3g} {unit}")
    if f.get("mass_kg"):
        bits.append(f"{1000 * float(f['mass_kg']):.0f} g")
    return "; ".join(bits)[:200]


def bom_lines(
    setup: Mapping[str, Any],
    facts: Mapping[str, Any],
    printed: Sequence[Mapping[str, Any]],
    upstream: Mapping[str, Mapping[str, str]],
) -> list[BomLine]:
    """The bill of materials. ``facts``: effector id -> probe facts (mass, actuation); ``printed``:
    the generated printable parts' facts; ``upstream``: effector id -> {url, license}."""
    out: list[BomLine] = []
    kind = setup["kind"]
    for k, tool in enumerate(_tool_rows(setup)):
        up = upstream.get(tool["effector_id"], {})
        note = _spec_note(tool, facts)
        if up:
            note = (note + "; upstream source in the manifest's links")[:200]
        out.append(
            BomLine(item=f"Tool {k}: {tool['display_name']}", qty=1, license=up.get("license"), note=note)
        )
        act = _actuation(tool, facts)
        if tool["family"] == "suction":
            out.append(
                BomLine(
                    item=f"Vacuum generator or pump for tool {k}",
                    qty=1,
                    note=f"at least {float(act.get('vacuum_kpa', 60)):.0f} kPa below ambient, with a valve the controller switches",
                )
            )
        if tool["family"] == "soft_arm":
            out.append(
                BomLine(
                    item=f"Cable actuator for tool {k}",
                    qty=1,
                    note=f"at least {float(act.get('cable_tension_n', 50)):.0f} N cable tension",
                )
            )
    for part in printed:
        out.append(
            BomLine(
                item=f"Printed {part['kind'].replace('_', ' ')}: {part['file']}",
                qty=1,
                note=(
                    f"{part['mass_kg'] * 1000:.0f} g at {part['density_kg_m3']:.0f} kg/m^3 "
                    "(carbon-filled polymer [assumed])"
                ),
            )
        )
    interfaces = 0
    if kind == "single":
        interfaces = 1
    elif kind == "multi_head":
        interfaces = 1 + len(setup["tools"])
    else:
        n = len(setup["tools"])
        interfaces = 1 + 2 * n
        changer = setup.get("changer") or {}
        out.append(
            BomLine(
                item="Automatic tool changer, master plate (wrist side)",
                qty=1,
                note=(
                    f"{float(changer.get('changeover_s', 6.0)):g} s changeover, "
                    f"{float(changer.get('master_mass_kg', 0.25)) * 1000:.0f} g, stack "
                    f"{float(changer.get('stack_mm', 25.0)):g} mm [assumed]"
                ),
            )
        )
        out.append(
            BomLine(
                item="Automatic tool changer, tool plate",
                qty=n,
                note=f"one per tool, {float(changer.get('tool_plate_mass_kg', 0.1)) * 1000:.0f} g each [assumed]",
            )
        )
    out.append(
        BomLine(
            item="M6 socket head cap screw, 16 mm",
            qty=SCREWS_PER_INTERFACE * interfaces,
            note=f"{SCREWS_PER_INTERFACE} per bolted interface, ISO 9409-1-50-4-M6 pattern [assumed]",
        )
    )
    return out


def bom_csv(lines: Sequence[BomLine]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["item", "qty", "link", "license", "note"])
    for b in lines:
        w.writerow([b.item, b.qty, b.link or "", b.license or "", b.note])
    return buf.getvalue()


# --------------------------------------------------------------------------- notes


def _fmt_mount(m: Mapping[str, Any]) -> str:
    return (
        f"tilted {float(m['angle_deg']):g} deg about the bracket's x axis, mount plane "
        f"{float(m['standoff_mm']):g} mm from the wrist centre, {float(m['spin_deg']):g} deg spin"
    )


def assembly_notes(finalist: Mapping[str, Any], report: Mapping[str, Any], files: Sequence[str]) -> str:
    """``ASSEMBLY.md``: how the setup goes together and how the task runs on it."""
    setup = finalist["setup"]
    result = finalist["result"]
    task = report["task"]
    kind = setup["kind"]
    tools = _tool_rows(setup)
    lines = [
        f"# Assembly notes: {task['title']}",
        "",
        f"{PRODUCT_MODE_LABEL}. {DISCLAIMER}",
        "",
        f"Setup: **{setup.get('label') or kind}** ({kind.replace('_', ' ')}), setup id `{setup['setup_id'][:16]}`.",
        "",
        "## Parts",
        "",
    ]
    for k, t in enumerate(tools):
        lines.append(f"* Tool {k}: {t['display_name']} ({t['fidelity']} fidelity).")
    lines += ["", "## Mounting", ""]
    if kind == "single":
        lines.append(
            "Bolt the tool directly to the wrist flange (63 mm, ISO 9409-1-50-4-M6 pattern "
            "[assumed]), as in the simulation: its mount plane is the flange face."
        )
    elif kind == "multi_head":
        br = setup.get("bracket") or {}
        lines.append(
            "Print `bracket.stl` and bolt its hub face to the wrist flange. The bracket frame: origin "
            "at the centre of the flange face, z toward the wrist, x the axis the heads are tilted about."
        )
        for k, m in enumerate(setup["mounts"]):
            lines.append(f"* Head {k} ({tools[k]['display_name']}): {_fmt_mount(m)}.")
        lines.append(
            f"* Hub {br.get('hub_thickness_mm', 10):g} mm, arms {br.get('arm_width_mm', 24):g} mm across, "
            f"adapters {br.get('adapter_thickness_mm', 8):g} mm."
        )
        lines.append(
            f"* To use head k the wrist tilts the whole bracket by minus that head's angle, so the head "
            f"points straight down on the roll axis; each switch costs {float(setup.get('present_s') or 1.5):g} s "
            "[assumed]."
        )
    else:
        ch = setup.get("changer") or {}
        lines.append(
            "Bolt the tool changer's master plate to the wrist flange. Print one changer plate per tool "
            "(`changer_plate_tool<k>.stl`) and bolt it between the changer's tool plate and the tool."
        )
        lines.append(
            f"* Changer stack {float(ch.get('stack_mm', 25)):g} mm, changeover "
            f"{float(ch.get('changeover_s', 6)):g} s [assumed]."
        )
    lines += [
        "",
        "## The task on this setup",
        "",
        "| Step | Primitive | Object | Tool | Success |",
        "|---|---|---|---|---|",
    ]
    by_step = result["step_success"]
    for s in task["steps"]:
        tool = setup["assignment"][s["id"]]
        lines.append(
            f"| {s['id']} | {s['primitive']} | {s['object']} | {tool}: {tools[tool]['display_name']} | "
            f"{100 * float(by_step.get(s['id'], 0)):.0f}% |"
        )
    lines += [
        "",
        (
            f"Full-task completion {100 * float(result['completion_rate']):.0f}% over {result['n_runs']} runs; "
            f"median cycle {float(result['cycle_time_s']):.1f} s including changes."
        ),
        "",
        "## Before you build",
        "",
        "* Check every [assumed] value (bolt pattern, changer, bracket material) against your arm and tools.",
        "* The simulation's wrist floats: it has no arm reach or arm collision model. Check reach and clearance on your arm.",
        "* Run the task slowly on hardware first; the controllers are scripted and open-loop on placement.",
        "",
        "## Files",
        "",
    ]
    lines += [f"* `{f}`" for f in files]
    lines.append("")
    return "\n".join(lines)


def readme_text(report: Mapping[str, Any], finalist: Mapping[str, Any], files: Sequence[str]) -> str:
    setup = finalist["setup"]
    out = [
        f"Build pack: {report['task']['title']}",
        "",
        PRODUCT_MODE_LABEL,
        DISCLAIMER,
        "",
        f"Setup: {setup.get('label') or setup['kind']} ({setup['kind']}).",
        f"Task id: {report['task_id']}",
        f"Setup id: {setup['setup_id']}",
        "",
        "Contents (manifest.json lists every file with its sha256, licence and source):",
        *[f"  {f}" for f in files],
        "",
        "STEP files are not provided: no CAD kernel is installed where packs are made. The STL files",
        "are in millimetres; the MJCF folder loads in MuJoCo as is (mjcf/setup_tool<k>.xml).",
        "",
        DISCLAIMER,
        "",
    ]
    return "\n".join(out)


def credits_text(records: Sequence[Mapping[str, Any]]) -> str:
    """``CREDITS.txt``: the credit line, copyright and licence of every third-party file (the one
    place upstream names may appear), then the generated files' licence."""
    out = ["Credits and licences", "", PRODUCT_MODE_LABEL, ""]
    seen: set[tuple[str, str]] = set()
    for r in records:
        key = (r["attribution"], r["license"])
        if key in seen:
            continue
        seen.add(key)
        out += [
            f"* {r['attribution']}",
            f"  Licence: {r['license']}. {r['copyright']}",
            f"  Upstream: {r['upstream_url']} at {r['upstream_commit']}",
        ]
        if r.get("changes"):
            out.append(f"  Changes: {'; '.join(r['changes'])}")
        out.append("")
    out += [
        "Generated files (bracket, changer plates, scene description, certificate, notes, bill of",
        "materials): CC0-1.0, generated by Effector Lab Studio.",
        "",
    ]
    return "\n".join(out)


def _brands(text: str) -> list[str]:
    """Brand tokens in on-screen text: the contract's list, plus the open library's additions
    when the library package is installed."""
    hits = find_brand_tokens(text)
    try:
        from rtt.library.brands import find_library_brands  # type: ignore[import-not-found]
    except ImportError:
        return hits
    return hits + [h for h in find_library_brands(text) if h not in hits]


# --------------------------------------------------------------------------- certificate


_CSS: Final = """
body{font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:#f7f7f5;color:#1d1f21}
main{max-width:980px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:24px;margin:8px 0 4px}h2{font-size:18px;margin:28px 0 8px}
.label{background:#fff3cd;border:1px solid #e0c97a;padding:8px 12px;border-radius:6px;font-weight:600}
table{border-collapse:collapse;width:100%;margin:8px 0;background:#fff}
th,td{border:1px solid #d9d9d6;padding:5px 8px;text-align:left;vertical-align:top;font-size:13px}
th{background:#eeeeea}.num{text-align:right;font-variant-numeric:tabular-nums}
.tier-approximation{color:#8a5a00}.tier-not_simulated{color:#a00}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;background:#e8eef7;font-size:12px}
code{font-size:12px}small{color:#555}
"""


def _e(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _pct(x: Any) -> str:
    try:
        return f"{100.0 * float(x):.0f}%"
    except (TypeError, ValueError):
        return "n/a"


def _table(head: Sequence[str], rows: Sequence[Sequence[Any]], num: Sequence[int] = ()) -> str:
    th = "".join(f"<th>{_e(h)}</th>" for h in head)
    body = []
    for r in rows:
        cells = "".join(f'<td class="num">{c}</td>' if i in num else f"<td>{c}</td>" for i, c in enumerate(r))
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def certificate_html(
    finalist: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    used: Mapping[str, str],
    printed: Sequence[Mapping[str, Any]],
) -> str:
    """``certificate.html``: self-contained (no scripts, no external files), brand-free."""
    setup, result, runs = finalist["setup"], finalist["result"], finalist["runs"]
    task = report["task"]
    tools = _tool_rows(setup)
    rec = report.get("recommendation") or {}
    recommended = (rec.get("recommendation") or {}).get("setup_id") == setup["setup_id"]
    parts = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        '<meta name=viewport content="width=device-width,initial-scale=1">',
        f"<title>Build certificate: {_e(task['title'])}</title><style>{_CSS}</style></head><body><main>",
        f'<p class="label">{_e(PRODUCT_MODE_LABEL)}. {_e(DISCLAIMER)}</p>',
        f"<h1>Build certificate: {_e(task['title'])}</h1>",
        f"<p>{_e(task.get('description') or '')}</p>",
        f"<p><span class=pill>{_e(setup['kind'].replace('_', ' '))}</span> <b>{_e(setup.get('label') or '')}</b>"
        + (" &middot; the Studio's recommendation" if recommended else "")
        + "</p>",
    ]
    if rec.get("evidence") != "accepted":
        parts.append(f"<p><b>Not evidence:</b> {_e(rec.get('reason') or '')}</p>")
    # headline
    weakest = result["weakest_step"]
    parts.append("<h2>Result</h2>")
    parts.append(
        _table(
            ["Full-task completion", "Runs", "Weakest step", "Median cycle (with changes)", "Fidelity"],
            [
                [
                    _pct(result["completion_rate"]),
                    _e(result["n_runs"]),
                    f"{_e(weakest['step_id'])} ({_pct(weakest['success_rate'])})",
                    f"{float(result['cycle_time_s']):.1f} s",
                    f'<span class="tier-{_e(result["fidelity"])}">{_e(result["fidelity"])}</span>',
                ]
            ],
        )
    )
    if recommended and rec.get("recommendation", {}).get("why"):
        parts.append(f"<p><b>Why:</b> {_e(rec['recommendation']['why'])}</p>")
    # tools
    comp = result["complexity"]
    parts.append("<h2>Setup</h2>")
    parts.append(
        _table(
            ["Tool", "Effector", "Family", "Fidelity", "Steps"],
            [
                [
                    _e(k),
                    _e(t["display_name"]),
                    _e(t["family"].replace("_", " ")),
                    _e(t["fidelity"]),
                    _e(", ".join(s for s, v in setup["assignment"].items() if v == k)),
                ]
                for k, t in enumerate(tools)
            ],
        )
    )
    parts.append(
        _table(
            ["Tools", "Actuators", "Carried (heaviest segment)", "Built in total", "Changes per run"],
            [
                [
                    _e(comp["n_tools"]),
                    _e(comp["n_actuators"]),
                    f"{float(comp['tool_mass_kg']):.3f} kg",
                    f"{float(comp['total_mass_kg']):.3f} kg",
                    _e(comp["n_changes"]),
                ]
            ],
            num=(0, 1, 4),
        )
    )
    if setup["kind"] == "multi_head":
        parts.append(
            _table(
                ["Head", "Angle", "Standoff", "Lateral", "Spin"],
                [
                    [
                        _e(k),
                        f"{m['angle_deg']:g} deg",
                        f"{m['standoff_mm']:g} mm",
                        f"{m['lateral_mm']:g} mm",
                        f"{m['spin_deg']:g} deg",
                    ]
                    for k, m in enumerate(setup["mounts"])
                ],
            )
        )
    for p in printed:
        parts.append(
            f"<p><small>{_e(p['file'])}: {_e(p['description'])}; {p['mass_kg'] * 1000:.0f} g printed; "
            f"watertight single solid: {_e(p['watertight'])}.</small></p>"
        )
    # per step
    parts.append("<h2>Per step</h2>")
    step_rows = []
    by_id = {s["id"]: s for s in task["steps"]}
    modes: dict[str, dict[str, int]] = {}
    caveats: dict[str, list[str]] = {}
    tiers: dict[str, str] = {}
    for run in runs:
        for o in run["steps"]:
            if o["failure_mode"]:
                modes.setdefault(o["step_id"], {}).setdefault(o["failure_mode"], 0)
                modes[o["step_id"]][o["failure_mode"]] += 1
            caveats.setdefault(o["step_id"], [])
            for c in o["caveats"]:
                if c not in caveats[o["step_id"]]:
                    caveats[o["step_id"]].append(c)
            tiers[o["step_id"]] = o["fidelity"]
    for sid, rate in result["step_success"].items():
        s = by_id[sid]
        tool = setup["assignment"][sid]
        prim = (
            s["primitive"]
            if used.get(sid, s["primitive"]) == s["primitive"]
            else f"{s['primitive']} (as {used[sid]})"
        )
        step_rows.append(
            [
                _e(sid),
                _e(prim),
                _e(s["object"]),
                _e(f"{tool}: {tools[tool]['display_name']}"),
                _pct(rate),
                _e(", ".join(f"{m} {n}x" for m, n in sorted(modes.get(sid, {}).items())) or "none"),
                f'<span class="tier-{_e(tiers.get(sid, ""))}">{_e(tiers.get(sid, ""))}</span>',
                _e(" ".join(caveats.get(sid, []))),
            ]
        )
    parts.append(
        _table(
            ["Step", "Primitive", "Object", "Tool", "Success", "Failures", "Fidelity", "Caveats"],
            step_rows,
            num=(4,),
        )
    )
    # conditions
    parts.append("<h2>Conditions</h2>")
    parts.append(
        "<p>Every compared setup ran the same conditions and seeds in the same order. A seed draws "
        "the friction jitter (+-10%) and a +-0.5 mm aim jitter; the placement errors below are hidden "
        "from the controllers.</p>"
    )
    cond_rows = []
    by_cond = {r["condition_id"]: r for r in runs}
    for c in report["conditions"]:
        offs = "; ".join(f"{k} {v[0]:+.2f}/{v[1]:+.2f}" for k, v in c["aim_offsets_mm"].items()) or "none"
        r = by_cond.get(c["condition_id"])
        cond_rows.append(
            [
                _e(c["condition_id"]),
                _e(c["seed"]),
                _e(offs),
                _e("completed" if r and r["completed"] else "failed"),
                f"{float(r['cycle_time_s']):.1f} s" if r else "",
                f"<code>{_e(r['traj_sha'][:16])}</code>" if r else "",
            ]
        )
    parts.append(
        _table(["Condition", "Seed", "Placement error x/y (mm)", "Run", "Cycle", "Trajectory"], cond_rows)
    )
    # comparison
    ranking = rec.get("ranking") or rec.get("reference_results") or []
    if ranking:
        parts.append("<h2>Compared setups</h2>")
        parts.append(
            _table(
                ["Setup", "Kind", "Completion", "Weakest step", "Cycle", "Tools", "Eligible"],
                [
                    [
                        _e(r["label"]) + (" (best single off-the-shelf)" if r.get("best_single") else ""),
                        _e(r["kind"].replace("_", " ")),
                        _pct(r["completion_rate"]),
                        f"{_e(r['weakest_step']['step_id'])} {_pct(r['weakest_step']['success_rate'])}",
                        f"{float(r['cycle_time_s']):.1f} s",
                        _e(r["complexity"]["n_tools"]),
                        _e("yes" if r["eligible"] else (r.get("reason") or "no")),
                    ]
                    for r in ranking
                ],
            )
        )
    vs = rec.get("versus_best_single")
    if vs and recommended:
        parts.append(
            f"<p>Against the best single off-the-shelf effector: completion {vs['completion_delta'] * 100:+.0f} points, "
            f"cycle {vs['cycle_delta_s']:+.1f} s, {vs['extra_tools']:+d} tools.</p>"
        )
    sens = report.get("sensitivity") or {}
    if sens.get("combos"):
        changed = [c for c in sens["combos"] if c["changed"]]
        text = (
            "the recommendation is the same for every pair"
            if not changed
            else "the recommendation changes for "
            + ", ".join(f"{c['tool_change_s']:g} s / {c['present_s']:g} s" for c in changed)
        )
        parts.append(
            "<h2>Sensitivity to change times</h2>"
            f"<p>Re-ranked with tool changes of 3, 6 and 12 s and head switches of 0.75, 1.5 and 3 s: {_e(text)}; "
            f"the order changes in {sens['order_changes']} of {len(sens['combos'])} pairs.</p>"
        )
    # assumptions
    consts = report.get("constants") or {}
    parts.append(
        "<h2>Assumed constants</h2><p>Equal for every candidate; none is sourced from a target arm or changer.</p>"
    )
    parts.append(
        _table(
            ["Constant", "Value", "Note"],
            [
                [
                    _e(k),
                    _e(f"{v['value']:g} {v['unit']}" if isinstance(v, Mapping) else v),
                    _e(v.get("note", "") if isinstance(v, Mapping) else ""),
                ]
                for k, v in consts.items()
            ],
        )
    )
    # caveats and provenance
    allcav = list(dict.fromkeys([*result.get("caveats", []), *report.get("fixed_caveats", [])]))
    parts.append("<h2>Caveats</h2><ul>" + "".join(f"<li>{_e(c)}</li>" for c in allcav) + "</ul>")
    fair = sorted({r["fairness_sha"] for r in runs})
    parts.append("<h2>Provenance</h2>")
    parts.append(
        _table(
            ["Item", "Value"],
            [
                ["Contract", _e(report.get("contract", ""))],
                ["Task id", f"<code>{_e(report['task_id'])}</code>"],
                ["Setup id", f"<code>{_e(setup['setup_id'])}</code>"],
                ["Fairness hash", f"<code>{_e(', '.join(fair))}</code>"],
                ["Controllers", _e(report.get("controllers", ""))],
                [
                    "Screen",
                    _e(
                        f"{report['screen']['runs']} runs on {len(report['screen']['conditions'])} conditions"
                    ),
                ],
            ],
        )
    )
    parts.append(f'<p class="label">{_e(PRODUCT_MODE_LABEL)}. {_e(DISCLAIMER)}</p></main></body></html>')
    page = "\n".join(parts)
    hits = _brands(html.unescape(page))
    if hits:
        raise ValueError(f"the certificate would name a brand on screen: {sorted(set(hits))}")
    return page
