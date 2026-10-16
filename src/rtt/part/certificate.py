"""Self-contained HTML certificate for a part run (no scripts, no external files).

:func:`render_certificate_html` turns the facts gathered by
:func:`rtt.part.pack.collect_run` into one HTML page: the part (preview image and
intake facts), the chosen finger design (parameters, the proposer's own rationale,
the jaw preview), the robustness battery (drop-weighted score, the placement-error
curve as inline SVG and every test), and provenance (search arm, model, job, hashes,
cost). Every piece of text that came from a model or a file name is HTML-escaped.

The page says, at the top and the bottom, that the results are simulated and the
design is untested on hardware.
"""

from __future__ import annotations

import base64
import html
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

__all__ = ["DISCLAIMER", "placement_curve_svg", "render_certificate_html"]

DISCLAIMER: Final = (
    "Simulated, untested on hardware. These results come from a physics simulation (MuJoCo) of an "
    "open-source parallel gripper; print and test before relying on them."
)

_PARAM_LABELS: Final = {
    "finger_length_mm": ("Finger length", "mm"),
    "finger_thickness_mm": ("Thickness at root", "mm"),
    "finger_width_mm": ("Width at root", "mm"),
    "bend_angle_deg": ("Bend angle", "deg"),
    "bend_start_frac": ("Bend position", "of length"),
    "inner_face": ("Inner face", ""),
    "inner_face_depth_mm": ("Face recess depth", "mm"),
    "tip_profile": ("Tip profile", ""),
    "tip_size_mm": ("Tip feature size", "mm"),
    "taper_ratio": ("Taper (tip / root)", ""),
}

_CSS: Final = """
:root{color-scheme:light;--surface:#fcfcfb;--panel:#f4f3f0;--rule:#e2e0db;--ink:#0b0b0b;--ink-2:#52514e;
--ink-3:#77756f;--accent:#2a78d6;--series-1:#2a78d6;--series-2:#eb6834;--good:#0ca30c;--bad:#d03b3b;
--warn-bg:#fff6e0;--warn-ink:#6b4a00;--grid:#e7e5e0}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--surface:#1a1a19;
--panel:#232321;--rule:#3a3936;--ink:#ffffff;--ink-2:#c3c2b7;--ink-3:#9a988f;--accent:#3987e5;
--series-1:#3987e5;--series-2:#d95926;--warn-bg:#3a2f12;--warn-ink:#f6d58a;--grid:#33322f}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--panel:#232321;--rule:#3a3936;--ink:#ffffff;
--ink-2:#c3c2b7;--ink-3:#9a988f;--accent:#3987e5;--series-1:#3987e5;--series-2:#d95926;--warn-bg:#3a2f12;
--warn-ink:#f6d58a;--grid:#33322f}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--surface);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,
"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 48px}
h1{font-size:30px;line-height:1.15;margin:4px 0 6px;letter-spacing:-.01em}
h2{font-size:19px;margin:0 0 14px;letter-spacing:-.005em}
h3{font-size:14px;margin:18px 0 8px;color:var(--ink-2);text-transform:uppercase;letter-spacing:.06em}
.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3)}
.sub{color:var(--ink-2);margin:0}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px;word-break:break-all}
.notice{background:var(--warn-bg);color:var(--warn-ink);border-radius:8px;padding:10px 14px;margin:18px 0 26px;
font-weight:600}
section{border-top:1px solid var(--rule);padding:26px 0}
.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:28px;align-items:start}
@media (max-width:760px){.grid{grid-template-columns:minmax(0,1fr)}}
.figure{background:var(--panel);border-radius:10px;padding:8px;text-align:center}
.figure img{max-width:100%;height:auto;display:block;margin:0 auto}
.figure p{margin:4px 0 2px;font-size:12.5px;color:var(--ink-3)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:600;color:var(--ink-2);font-size:12.5px}
td.num,th.num{text-align:right}
.kv th{width:46%;font-weight:500}
.scroll{overflow-x:auto}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:6px 0 18px}
.tile{background:var(--panel);border-radius:10px;padding:14px 16px}
.tile .v{font-size:28px;font-weight:650;line-height:1.1}
.tile .l{font-size:12.5px;color:var(--ink-2)}
.status{display:inline-flex;align-items:center;gap:6px;font-weight:600}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.dot.good{background:var(--good)}.dot.bad{background:var(--bad)}
blockquote{margin:0 0 12px;padding:10px 14px;border-left:3px solid var(--accent);background:var(--panel);
border-radius:0 8px 8px 0;color:var(--ink)}
blockquote .who{display:block;font-size:12px;color:var(--ink-3);margin-top:6px}
.curves{margin-top:10px;gap:18px}
svg text{font-family:inherit}
.warnings li{color:var(--ink-2)}
footer{border-top:1px solid var(--rule);padding-top:18px;color:var(--ink-3);font-size:12.5px}
@media print{.notice{border:1px solid #c9a227}section{break-inside:avoid}}
"""


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _pct(x: float | None, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100.0 * x:.{digits}f}%"


def _num(x: Any, digits: int = 1) -> str:
    if isinstance(x, bool) or x is None:
        return _e(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{x:.0f}"
    return _e(x)


def _img(png: bytes | None, alt: str, caption: str) -> str:
    if not png:
        return f'<div class="figure"><p>{_e(caption)} (no preview)</p></div>'
    data = base64.b64encode(png).decode("ascii")
    return (
        f'<figure class="figure" style="margin:0"><img src="data:image/png;base64,{data}" alt="{_e(alt)}">'
        f"<p>{_e(caption)}</p></figure>"
    )


def _status(ok: bool) -> str:
    cls, word = ("good", "Held") if ok else ("bad", "Dropped")
    return f'<span class="status"><span class="dot {cls}" aria-hidden="true"></span>{word}</span>'


def _grasp_text(axis: str, offset_mm: float) -> str:
    side = "across the length" if axis == "wide" else "across the narrow side"
    where = "at the centre" if abs(offset_mm) < 1e-9 else f"{abs(offset_mm):.0f} mm off centre"
    return f"{side}, {where}"


def _curve_panel(key: str, title: str, pts: Sequence[tuple[float, float]], n_poses: int) -> str:
    w, h = 340, 220
    left, right, top, bottom = 46, 14, 34, 42
    pw, ph = w - left - right, h - top - bottom
    errors = [e for e, _ in pts] or [0.0]
    lo_e, hi_e = min(errors + [-3.0]), max(errors + [3.0])

    def sx(e: float) -> float:
        return left + (e - lo_e) / (hi_e - lo_e) * pw

    def sy(v: float) -> float:
        return top + (1.0 - v) * ph

    out = [
        (
            f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-labelledby="pc-{key}-t" '
            'style="display:block;max-width:420px">'
        ),
        f'<title id="pc-{key}-t">{_e(title)}: share of poses held vs placement error</title>',
        f'<text x="{left}" y="16" font-size="13" font-weight="600" fill="var(--ink)">{_e(title)}</text>',
    ]
    for v in (0.0, 0.5, 1.0):
        out.append(
            f'<line x1="{left}" x2="{left + pw}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="var(--grid)" />'
        )
        out.append(
            f'<text x="{left - 8}" y="{sy(v) + 4:.1f}" text-anchor="end" font-size="11.5" fill="var(--ink-3)">'
            f"{int(v * 100)}%</text>"
        )
    for e in errors:
        tick = "0" if e == 0 else f"{e:+g}"
        out.append(
            f'<text x="{sx(e):.1f}" y="{top + ph + 17}" text-anchor="middle" font-size="11.5" '
            f'fill="var(--ink-3)">{tick}</text>'
        )
    out.append(
        f'<text x="{left + pw / 2:.1f}" y="{h - 6}" text-anchor="middle" font-size="11.5" fill="var(--ink-2)">'
        "placement error (mm)</text>"
    )
    if pts:
        path = " ".join(f"{'M' if i == 0 else 'L'}{sx(e):.1f},{sy(s):.1f}" for i, (e, s) in enumerate(pts))
        out.append(
            f'<path d="{path}" fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round" />'
        )
        for e, s in pts:
            tip = f"{title}, {e:+g} mm: held in {round(s * n_poses)} of {n_poses} pose(s)"
            out.append(
                f'<g><title>{_e(tip)}</title><circle cx="{sx(e):.1f}" cy="{sy(s):.1f}" r="10" fill="transparent" />'
                f'<circle cx="{sx(e):.1f}" cy="{sy(s):.1f}" r="4.5" fill="var(--series-1)" '
                'stroke="var(--surface)" stroke-width="2" /></g>'
            )
    out.append("</svg>")
    return "".join(out)


def placement_curve_svg(curve: Mapping[str, Sequence[Sequence[float]]], n_poses: int) -> str:
    """Two inline-SVG panels (finger-width axis, closing axis): share of certified poses still
    held vs placement error. Same y scale in both; the test table below holds every value."""
    panels = []
    for key, title in (("x", "Along the finger width (x)"), ("y", "Along the closing axis (y)")):
        pts = sorted((float(e), float(s)) for e, s in curve.get(key, ()))
        panels.append(f"<div>{_curve_panel(key, title, pts, n_poses)}</div>")
    return f'<div class="grid curves">{"".join(panels)}</div>'


def _battery_text(settings: Mapping[str, Any]) -> str:
    """The stress-test battery in words, from the certificate's own recorded settings."""
    try:
        err = max(float(e) for e in settings["placement_errors_mm"])
        shake = settings["shake"]
        amp_mm, freq = 1000.0 * float(shake["amplitude_m"]), float(shake["frequency_hz"])
        g = (2.0 * math.pi * freq) ** 2 * amp_mm / 1000.0 / 9.81
        hold = float(settings["post_perturb_hold_s"])
    except (KeyError, TypeError, ValueError):
        return _e(
            "Each pose is grasped with no random jitter, then moved off the aim point, shaken and pushed; "
            "the test table below lists every condition and result."
        )
    push = "at its top" if (settings.get("push") or {}).get("at_top", True) else "at its centre of mass"
    return _e(
        f"Each pose is grasped with no random jitter, then the part is moved off the aim point by up to "
        f"+-{err:g} mm along each gripper axis, shaken along x, y and z after the lift ({amp_mm:g} mm at "
        f"{freq:g} Hz, about {g:.1f} g) and pushed sideways {push}. A test passes only if the part is still "
        f"held {hold:g} s after the disturbance."
    ).replace("+-", "&plusmn;")


def _kv_rows(rows: Sequence[tuple[str, str]]) -> str:
    return "".join(f"<tr><th>{_e(k)}</th><td>{v}</td></tr>" for k, v in rows)


def render_certificate_html(ctx: Mapping[str, Any]) -> str:
    """The certificate page for the facts in ``ctx`` (see :func:`rtt.part.pack.collect_run`)."""
    part = ctx["part"]
    cert = ctx["certificate"]
    frozen = ctx["frozen"]
    design = frozen["design"]
    manifest = ctx.get("manifest") or {}
    proposal = ctx.get("proposal") or {}
    images = ctx.get("images") or {}
    n_poses = len(cert["poses"])
    score = float(cert["score"])
    primary = cert["poses"][0] if cert["poses"] else {}
    tol = primary.get("tolerance_mm") or {}

    def tol_txt(axis: str) -> str:
        v = tol.get(axis)
        return "fails nominal" if v is None else f"&plusmn;{_num(float(v))} mm"

    # ---- part
    ex = part["extents_mm"]
    pose_rows = "".join(
        f"<tr><td>{_e(p['rank'])}</td><td class='num'>{_pct(p['probability'])}</td>"
        f"<td>{' x '.join(_num(v, 0) for v in p['extents_mm'])} mm</td>"
        f"<td class='num'>{_e(p['n_pieces'])}</td>"
        f"<td>{_e('; '.join(_grasp_text(c['axis'], c['offset_mm']) for c in p['candidates']))}</td></tr>"
        for p in part["poses"]
    )
    part_rows = _kv_rows(
        [
            ("File", _e(part["source_name"])),
            ("Size as uploaded", f"{' x '.join(_num(v) for v in ex)} mm"),
            ("Units", f"{_e(part['units'])}{' (detected)' if part['units_detected'] else ''}"),
            (
                "Mass",
                (
                    f"{_num(part['mass_g'])} g of {_e(part['material'])} "
                    f"({_num(part['density_kg_m3'], 0)} kg/m&sup3;)"
                    f"{', approximate' if part['mass_is_approximate'] else ''}"
                ),
            ),
            ("Closed solid", "yes" if part["watertight"] else "no (convex pieces used)"),
            ("Repairs", _e("; ".join(part["repairs"]) or "none")),
            ("Collision model", f"{_e(part['decomposition'])} convex decomposition"),
            ("SHA-256", f'<span class="mono">{_e(part["source_sha256"])}</span>'),
        ]
    )
    warnings = "".join(f"<li>{_e(w)}</li>" for w in part.get("warnings", []))

    # ---- design
    param_rows = []
    for key, (label, unit) in _PARAM_LABELS.items():
        value = design.get(key)
        shown = _e(value).replace("_", " ") if isinstance(value, str) else _num(value, 2)
        param_rows.append(f"<tr><th>{_e(label)}</th><td class='num'>{shown}</td><td>{_e(unit)}</td></tr>")
    origin = frozen.get("origin", "")
    origin_txt = {
        "entrant": "a manual starting design",
        "sobol": "the shared random starting round",
        "random": "random search",
        "nemotron": "Nemotron, with simulation feedback",
        "zeroshot": "Nemotron, without feedback",
        "fallback_random": "a random fill-in during the Nemotron search",
    }.get(origin, origin)
    quotes = []
    if proposal.get("hypothesis"):
        who = "Nemotron's hypothesis for this design" if origin in ("nemotron", "zeroshot") else "Design note"
        quotes.append(f"<blockquote>{_e(proposal['hypothesis'])}<span class='who'>{who}</span></blockquote>")
    if design.get("rationale") and design.get("rationale") != proposal.get("hypothesis"):
        quotes.append(
            f"<blockquote>{_e(design['rationale'])}<span class='who'>Design rationale</span></blockquote>"
        )
    if proposal.get("analysis"):
        quotes.append(
            f"<blockquote>{_e(proposal['analysis'])}<span class='who'>Nemotron's analysis of the results "
            "before proposing it</span></blockquote>"
        )
    if not quotes:
        why = {
            "fallback_random": "a random fill-in, used when the proposer's own designs failed the validator or "
            "were too close to designs already tested",
            "random": "random search",
        }.get(origin, "the random starting round")
        quotes.append(f"<p class='sub'>This design came from {_e(why)}, so it has no written rationale.</p>")
    training_rows = _kv_rows(
        [
            ("Found by", _e(origin_txt)),
            (
                "Training success",
                f"{_pct(frozen['training_success'])} of {_e(ctx.get('episodes_per_design') or '?')} grasps",
            ),
            (
                "Grasp height setting",
                f"k = {_e(frozen['controller_setting'])}"
                + (
                    f" ({_pct(frozen['grasp_height_frac'])} of the reachable band)"
                    if frozen.get("grasp_height_frac") is not None
                    else ""
                ),
            ),
            ("Finger mass", f"{_num(frozen['mass_g'])} g (PLA)"),
            ("Design id", f'<span class="mono">{_e(frozen["design_id"])}</span>'),
        ]
    )

    # ---- tests
    test_rows = []
    for pc in cert["poses"]:
        for t in pc["tests"]:
            p = t["params"]
            if t["category"] == "placement":
                cond = (
                    "aimed at the grasp point"
                    if not p.get("axis")
                    else f"{p['error_mm']:+g} mm along {p['axis']}"
                )
            elif t["category"] == "shake":
                cond = f"{p['amplitude_m'] * 1000:g} mm at {p['frequency_hz']:g} Hz along {p['axis']}"
            else:
                cond = f"{p['impulse_ns']:g} N s push along {p['axis']} at the top"
            test_rows.append(
                f"<tr><td>{_e(pc['pose_rank'])}</td><td>{_e(t['category'])}</td><td>{_e(cond)}</td>"
                f"<td>{_status(bool(t['success']))}</td><td>{_e((t['failure_mode'] or '').replace('_', ' '))}</td>"
                f"<td class='num'>{_num(t['max_lift_m'] * 1000.0, 0)}</td>"
                f"<td class='num'>{_num(t['hold_time_s'], 2)}</td></tr>"
            )
    pose_scores = "".join(
        f"<tr><td>{_e(pc['pose_rank'])}</td><td class='num'>{_pct(pc['probability'])}</td>"
        f"<td>{_e(_grasp_text(pc['candidate'].split('-')[0], pc['offset_mm']))}</td><td class='num'>{_e(pc['passed'])}</td>"
        + "".join(
            f"<td class='num'>{_pct(pc['category_scores'].get(c))}</td>"
            for c in ("placement", "shake", "push")
        )
        + "</tr>"
        for pc in cert["poses"]
    )

    # ---- provenance
    model = manifest.get("model") or {}
    job = manifest.get("job") or {}
    code = manifest.get("code") or {}
    llm = manifest.get("llm") or {}
    evaluator = manifest.get("evaluator") or {}
    prov = [
        (
            "Search arm",
            (
                f"{_e(manifest.get('arm', frozen.get('arm')))}, budget {_e(manifest.get('budget'))} designs "
                f"x {_e(manifest.get('runs', 1))} run(s), seed {_e(frozen.get('run_seed'))}"
            ),
        ),
        (
            "Model",
            _e(model.get("model_id") or "none (no model used)")
            + (f" via {_e(manifest.get('provider'))}" if manifest.get("provider") else ""),
        ),
        (
            "Model spend",
            f"${llm.get('cost_usd', 0.0):.4f} ({_e(llm.get('calls', 0))} calls, "
            f"{_e(llm.get('prompt_tokens', 0))} + {_e(llm.get('completion_tokens', 0))} tokens)"
            if llm
            else "n/a",
        ),
        (
            "Simulator",
            (
                f"MuJoCo, executor {_e(manifest.get('executor', 'cpu'))}, task digest "
                f"<span class='mono'>{_e(evaluator.get('task_hash'))}</span>"
            ),
        ),
        (
            "Training suite",
            (
                f"{_e(evaluator.get('n_conditions'))} conditions, digest "
                f"<span class='mono'>{_e(evaluator.get('suite_hash'))}</span>"
            ),
        ),
        (
            "Compute",
            _e(
                " / ".join(v for v in (job.get("platform"), job.get("preset"), job.get("region")) if v)
                or "local run"
            ),
        ),
        (
            "Job id",
            f"<span class='mono'>{_e(job.get('job_id'))}</span>"
            if job.get("job_id")
            else "local run (no job)",
        ),
        ("Code", f"<span class='mono'>{_e(code.get('git_sha') or 'unknown')}</span>"),
        (
            "Frozen design",
            f"<span class='mono'>{_e(frozen.get('content_sha256'))}</span> at {_e(frozen.get('frozen_at'))}",
        ),
        ("Search wall time", f"{_num(manifest.get('wall_s'), 1)} s"),
        ("Certificate created", _e(cert.get("created_at"))),
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grasp Certificate</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<div class="eyebrow">Grasp certificate &middot; simulated</div>
<h1>Fingers for {_e(part["name"])}</h1>
<p class="sub">Design <span class="mono">{_e(frozen["design_id"])}</span>, chosen by {_e(origin_txt)} and
stress-tested on {n_poses} resting pose(s) of the part.</p>
<div class="notice" role="note">{_e(DISCLAIMER)}</div>

<div class="tiles">
<div class="tile"><div class="v">{_pct(score)}</div><div class="l">robustness, weighted by how often each pose occurs</div></div>
<div class="tile"><div class="v">{_e(cert["passed"])} / {_e(cert["n_tests"])}</div><div class="l">stress tests held</div></div>
<div class="tile"><div class="v">{_pct(frozen["training_success"])}</div><div class="l">training grasps succeeded</div></div>
<div class="tile"><div class="v">{tol_txt("y")}</div><div class="l">placement error tolerated across the jaw (pose 0)</div></div>
</div>

<section>
<h2>The part</h2>
<div class="grid">
<div>{_img(images.get("part"), "The uploaded part in its most likely resting pose", "Most likely resting pose")}</div>
<div><table class="kv">{part_rows}</table></div>
</div>
<h3>Resting poses tested</h3>
<div class="scroll"><table>
<tr><th>Pose</th><th class="num">Share of drops</th><th>Length x width x height</th><th class="num">Convex pieces</th><th>Grasps tried</th></tr>
{pose_rows}
</table></div>
{f'<h3>Intake notes</h3><ul class="warnings">{warnings}</ul>' if warnings else ""}
</section>

<section>
<h2>The finger design</h2>
<div class="grid">
<div>
{"".join(quotes)}
<table class="kv">{training_rows}</table>
</div>
<div>{_img(images.get("jaws"), "The two printed jaws on their guide rods, fully open", "Printed jaw pair on the guide rods, fully open")}</div>
</div>
<h3>Parameters</h3>
<div class="scroll"><table>{"".join(param_rows)}</table></div>
</section>

<section>
<h2>Stress tests</h2>
<p class="sub">{_battery_text(primary.get("settings") or {})}</p>
<h3>Placement-error curve</h3>
<p class="sub">Share of the tested resting poses still held when the part sits that far from where the
gripper aims.</p>
{placement_curve_svg(cert["placement_curve"], n_poses)}
<h3>By pose</h3>
<div class="scroll"><table>
<tr><th>Pose</th><th class="num">Share of drops</th><th>Grasp</th><th class="num">Passed</th><th class="num">Placement</th><th class="num">Shake</th><th class="num">Push</th></tr>
{pose_scores}
</table></div>
<h3>Every test</h3>
<div class="scroll"><table>
<tr><th>Pose</th><th>Kind</th><th>Condition</th><th>Result</th><th>Failure</th><th class="num">Max lift (mm)</th><th class="num">Hold (s)</th></tr>
{"".join(test_rows)}
</table></div>
</section>

<section>
<h2>Provenance</h2>
<div class="scroll"><table class="kv">{_kv_rows(prov)}</table></div>
</section>

<footer>
<p><strong>{_e(DISCLAIMER)}</strong></p>
<p>Generated by Robot Tool Tournament (Apache-2.0). Jaw clamp dimensions marked "assumed" in manifest.json are not
documented; check them against your gripper before printing.</p>
</footer>
</main>
</body>
</html>
"""
