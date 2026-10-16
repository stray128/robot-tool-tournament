"""The hero rule: which parts the video and the site may show as showcase cases.

Fixed before calibration (gate GV-H) and hashed into ``docs/PREREG.md``: the digest of
:class:`HeroRule` and this file's SHA-256. A failure is never staged: every input is a
recorded simulation episode, and every ambiguity is resolved against qualification.

**Screen conditions** (:data:`HERO_CONDITIONS`): placement errors of +-1, +-2 and +-3 mm
across the jaw (palm x), and a 5 degree yaw at zero offset; 7 conditions in all. The six
translation errors are the "non-zero errors" of criterion (b).

**A part is a hero** when it passes the brand screen (scans with visible branding never
qualify) and

* **(a) the stock blade fails**: ``success_v2`` fails on at least 4 of the 7 conditions,
  and the failure is *visible* on at least 2 of the 7: a slip during the lift, an ejection
  on closing, a drop during the hold, or a tilt above 10 degrees;
* **(c) a designed recipe succeeds**: the best designed recipe passes ``success_v2`` with
  a planar offset of at most 0.5 mm on at least 6 of the 7 conditions. "Best" is the
  candidate with the most such passes (then the lower mean offset, then the design ID).

**An imprint hero** is a hero for which also

* **(b) the configurator imprint jams or stays off-centre**: a seat shortfall of at least
  2 mm or a planar offset of at least 1 mm on at least 4 of the 6 translation errors.

**Measurements.** Offset is the planar offset sqrt(dx^2 + dy^2) of the part in the palm
frame at the registration moment (end of closing + 0.25 s); shortfall is ``seat_short_mm``
at the same moment; tilt is the largest tilt recorded for the episode (at registration or
after the hold). ``success_v2`` is the registered-grasp criterion of ``docs/TASK_V2.md``.

**Episodes per condition.** A condition may hold several episodes (jitter seeds). It
counts as showing a property only if a strict majority of its episodes do; ties count
against qualification. An episode flagged ``physics_invalid`` never counts as a stock or
imprint failure and never counts as a designed pass.

**Outcome.** Heroes are labelled "showcase parts chosen because the stock finger fails".
If fewer than 3 parts qualify, the hook uses the imprint-vs-designed offset contrast only
(``hook_mode = "offset_contrast_only"``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

__all__ = [
    "HERO_CONDITIONS",
    "HERO_LABEL",
    "TRANSLATION_CONDITIONS",
    "HeroRule",
    "HeroScreen",
    "PartVerdict",
    "ScreenEpisode",
    "apply_hero_rule",
    "evaluate_part",
    "main",
]

#: condition -> (offset across the jaw in mm, yaw in degrees)
HERO_CONDITIONS: Final[dict[str, tuple[float, float]]] = {
    "x-3": (-3.0, 0.0),
    "x-2": (-2.0, 0.0),
    "x-1": (-1.0, 0.0),
    "x+1": (1.0, 0.0),
    "x+2": (2.0, 0.0),
    "x+3": (3.0, 0.0),
    "yaw+5": (0.0, 5.0),
}
TRANSLATION_CONDITIONS: Final[tuple[str, ...]] = ("x-3", "x-2", "x-1", "x+1", "x+2", "x+3")
HERO_LABEL: Final = "showcase parts chosen because the stock finger fails"

#: Failure modes that a viewer sees (v1 long names; v2 short names are normalised to these).
VISIBLE_MODES: Final[frozenset[str]] = frozenset({"slip_during_lift", "ejected_on_close", "unstable_hold"})
_MODE_ALIASES: Final[dict[str, str]] = {
    "slip": "slip_during_lift",
    "ejected": "ejected_on_close",
    "unstable": "unstable_hold",
    "drop": "unstable_hold",
    "dropped": "unstable_hold",
    "missed": "missed_grasp",
}
HookMode = Literal["hero_triptych", "offset_contrast_only"]


@dataclass(frozen=True)
class HeroRule:
    """Thresholds of the hero rule (spec section 8); :meth:`digest` goes into PREREG."""

    stock_fail_min: int = 4
    stock_visible_min: int = 2
    visible_tilt_deg: float = 10.0
    imprint_short_mm: float = 2.0
    imprint_offset_mm: float = 1.0
    imprint_min: int = 4
    designed_pass_min: int = 6
    designed_offset_mm: float = 0.5
    min_heroes: int = 3

    def digest(self) -> str:
        """SHA-256 of the canonical JSON of the thresholds, conditions and aggregation rules."""
        blob = json.dumps(
            {
                "rule": "hero",
                "version": 1,
                "thresholds": asdict(self),
                "conditions": HERO_CONDITIONS,
                "translation_conditions": list(TRANSLATION_CONDITIONS),
                "visible_modes": sorted(VISIBLE_MODES),
                "aliases": _MODE_ALIASES,
                "offset": "planar sqrt(dx^2+dy^2), palm frame, close + 0.25 s",
                "aggregation": "strict majority of a condition's episodes; ties and physics_invalid against",
                "label": HERO_LABEL,
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True)
class ScreenEpisode:
    """One screen episode, reduced to what the rule reads."""

    condition: str
    success_v2: bool
    failure_mode: str | None = None
    tilt_deg: float = 0.0
    seat_short_mm: float = 0.0
    offset_mm: float = 0.0
    physics_invalid: bool = False

    def __post_init__(self) -> None:
        if self.condition not in HERO_CONDITIONS:
            raise ValueError(
                f"unknown hero condition {self.condition!r} (expected one of {list(HERO_CONDITIONS)})"
            )
        for name in ("tilt_deg", "seat_short_mm", "offset_mm"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")

    @property
    def mode(self) -> str | None:
        return _MODE_ALIASES.get(self.failure_mode, self.failure_mode) if self.failure_mode else None

    @classmethod
    def from_record(cls, rec: Mapping[str, Any], *, condition: str | None = None) -> ScreenEpisode:
        """From a v2 episode record (``RECORD_VERSION = 2`` field names)."""
        cond = condition if condition is not None else str(rec.get("hero_condition", rec.get("condition")))
        if "success_v2" in rec:
            ok = bool(rec["success_v2"])
        else:
            ok = bool(rec["success"]) and bool(rec["registered"])
        tilts = [
            float(rec[k]) for k in ("tilt_max_deg", "reg_tilt_deg", "hold_tilt_deg") if rec.get(k) is not None
        ]
        mode = rec.get("failure_mode")
        invalid = bool(rec.get("bad_qacc")) or mode == "physics_invalid"
        return cls(
            condition=cond,
            success_v2=ok and not invalid,
            failure_mode=None if mode is None else str(mode),
            tilt_deg=max(tilts) if tilts else 0.0,
            seat_short_mm=float(rec.get("seat_short_mm") or 0.0),
            offset_mm=math.hypot(float(rec.get("reg_dx_mm") or 0.0), float(rec.get("reg_dy_mm") or 0.0)),
            physics_invalid=invalid,
        )


@dataclass(frozen=True)
class PartVerdict:
    part_id: str
    eligible: bool
    stock_fails: int
    stock_visible: int
    a_ok: bool
    imprint_hits: int | None
    b_ok: bool
    best_design: str | None
    design_passes: int
    c_ok: bool
    hero: bool
    imprint_hero: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class HeroScreen:
    rule_digest: str
    verdicts: tuple[PartVerdict, ...]
    heroes: tuple[str, ...]
    imprint_heroes: tuple[str, ...]
    hook_mode: HookMode
    label: str = HERO_LABEL

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "rtt.hero_screen/1",
            "rule_digest": self.rule_digest,
            "label": self.label,
            "hook_mode": self.hook_mode,
            "heroes": list(self.heroes),
            "imprint_heroes": list(self.imprint_heroes),
            "verdicts": [asdict(v) for v in self.verdicts],
        }


def _by_condition(
    episodes: Iterable[ScreenEpisode], conditions: Sequence[str], role: str
) -> dict[str, list[ScreenEpisode]]:
    out: dict[str, list[ScreenEpisode]] = {c: [] for c in conditions}
    for ep in episodes:
        if ep.condition in out:
            out[ep.condition].append(ep)
    missing = [c for c, eps in out.items() if not eps]
    if missing:
        raise ValueError(f"{role}: no episodes for conditions {missing}")
    return out


def _majority(eps: Sequence[ScreenEpisode], prop: Any) -> bool:
    return 2 * sum(1 for e in eps if prop(e)) > len(eps)


def evaluate_part(
    part_id: str,
    stock: Iterable[ScreenEpisode],
    designed: Mapping[str, Iterable[ScreenEpisode]],
    imprint: Iterable[ScreenEpisode] | None = None,
    *,
    eligible: bool = True,
    rule: HeroRule | None = None,
) -> PartVerdict:
    """Apply criteria (a), (b) and (c) to one part's screen episodes."""
    rule = rule or HeroRule()
    conds = list(HERO_CONDITIONS)
    reasons: list[str] = []

    stock_by = _by_condition(stock, conds, f"{part_id} stock")

    def stock_fail(e: ScreenEpisode) -> bool:
        return not e.physics_invalid and not e.success_v2

    def stock_visible(e: ScreenEpisode) -> bool:
        return stock_fail(e) and (e.mode in VISIBLE_MODES or e.tilt_deg > rule.visible_tilt_deg)

    fails = sum(_majority(stock_by[c], stock_fail) for c in conds)
    visible = sum(_majority(stock_by[c], stock_visible) for c in conds)
    a_ok = fails >= rule.stock_fail_min and visible >= rule.stock_visible_min
    if not a_ok:
        reasons.append(
            f"(a) stock fails {fails}/7 (need {rule.stock_fail_min}), visibly {visible}/7 (need {rule.stock_visible_min})"
        )

    imprint_hits: int | None = None
    b_ok = False
    if imprint is not None:
        imp_by = _by_condition(imprint, TRANSLATION_CONDITIONS, f"{part_id} imprint")

        def jammed_or_off(e: ScreenEpisode) -> bool:
            return not e.physics_invalid and (
                e.seat_short_mm >= rule.imprint_short_mm or e.offset_mm >= rule.imprint_offset_mm
            )

        imprint_hits = sum(_majority(imp_by[c], jammed_or_off) for c in TRANSLATION_CONDITIONS)
        b_ok = imprint_hits >= rule.imprint_min
        if not b_ok:
            reasons.append(f"(b) imprint short or off-centre on {imprint_hits}/6 (need {rule.imprint_min})")
    else:
        reasons.append("(b) imprint not screened")

    def designed_pass(e: ScreenEpisode) -> bool:
        return not e.physics_invalid and e.success_v2 and e.offset_mm <= rule.designed_offset_mm

    ranking: list[tuple[int, float, str]] = []
    for design_id, eps in designed.items():
        by = _by_condition(eps, conds, f"{part_id} design {design_id}")
        passes = sum(_majority(by[c], designed_pass) for c in conds)
        offsets = [e.offset_mm for c in conds for e in by[c]]
        ranking.append((-passes, sum(offsets) / len(offsets), design_id))
    if ranking:
        neg_passes, _, best = min(ranking)
        passes = -neg_passes
    else:
        best, passes = None, 0
    c_ok = passes >= rule.designed_pass_min
    if not c_ok:
        reasons.append(
            f"(c) best designed recipe passes {passes}/7 with offset <= {rule.designed_offset_mm} mm (need {rule.designed_pass_min})"
        )
    if not eligible:
        reasons.append("brand screen: branding visible")
    hero = eligible and a_ok and c_ok
    return PartVerdict(
        part_id=part_id,
        eligible=eligible,
        stock_fails=fails,
        stock_visible=visible,
        a_ok=a_ok,
        imprint_hits=imprint_hits,
        b_ok=b_ok,
        best_design=best,
        design_passes=passes,
        c_ok=c_ok,
        hero=hero,
        imprint_hero=hero and b_ok,
        reasons=tuple(reasons),
    )


def apply_hero_rule(verdicts: Sequence[PartVerdict], rule: HeroRule | None = None) -> HeroScreen:
    """Collect per-part verdicts into the hero set and the hook mode."""
    rule = rule or HeroRule()
    ids = [v.part_id for v in verdicts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate parts in the hero screen: {ids}")
    heroes = tuple(v.part_id for v in verdicts if v.hero)
    return HeroScreen(
        rule_digest=rule.digest(),
        verdicts=tuple(verdicts),
        heroes=heroes,
        imprint_heroes=tuple(v.part_id for v in verdicts if v.imprint_hero),
        hook_mode="hero_triptych" if len(heroes) >= rule.min_heroes else "offset_contrast_only",
    )


def _screen_from_json(data: Mapping[str, Any], rule: HeroRule) -> HeroScreen:
    """``{"parts": [{"part_id", "eligible", "stock": [rec], "imprint": [rec] | null, "designed": {id: [rec]}}]}``."""
    verdicts = []
    for part in data["parts"]:
        imprint = part.get("imprint")
        verdicts.append(
            evaluate_part(
                str(part["part_id"]),
                [ScreenEpisode.from_record(r) for r in part["stock"]],
                {k: [ScreenEpisode.from_record(r) for r in v] for k, v in part["designed"].items()},
                None if imprint is None else [ScreenEpisode.from_record(r) for r in imprint],
                eligible=bool(part.get("eligible", True)),
                rule=rule,
            )
        )
    return apply_hero_rule(verdicts, rule)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rtt.analysis.hero_rule", description=__doc__.splitlines()[0])
    ap.add_argument(
        "screen", nargs="?", type=Path, help="screen episodes JSON (omit to print the rule digest)"
    )
    ap.add_argument("--out", type=Path, default=None, help="write hero_screen.json here")
    args = ap.parse_args(argv)
    rule = HeroRule()
    if args.screen is None:
        print(json.dumps({"thresholds": asdict(rule), "digest": rule.digest()}, indent=1))
        return 0
    screen = _screen_from_json(json.loads(args.screen.read_text(encoding="utf-8")), rule)
    for v in screen.verdicts:
        tag = "imprint hero" if v.imprint_hero else "hero" if v.hero else "-"
        print(f"{v.part_id:<24}{tag:<14}" + "; ".join(v.reasons))
    print(f"heroes: {list(screen.heroes)}; hook mode: {screen.hook_mode}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "hero_screen.json").write_text(
            json.dumps(screen.to_json(), indent=1) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
