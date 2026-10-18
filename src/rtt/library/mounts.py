"""Mounting interfaces: ISO 9409-1 flange patterns and the item mount record.

The ISO text is copyrighted; dimension values are facts. Every value carries a status:

* ``sourced``: read from a published source today (manufacturer datasheets, catalogue pages);
* ``secondary``: from one secondary source only, so **unverified** until checked against a
  purchased copy of ISO 9409-1:2004;
* ``assumed``: our engineering assumption (for example, a dowel equal to the thread size).

Nothing here claims conformance with the standard. Generated adapters (:mod:`rtt.library.adapters`)
print the status of every dimension they use.

An item's :class:`Mount` says how it attaches: an ISO pattern where one is known, otherwise a
plain description of its own interface (a wrist servo horn, a threaded stud, an adapter plate).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Literal

__all__ = [
    "COBOT_MAX_THREAD_DEPTH_MM",
    "CONFIRMED_PATTERNS",
    "ISO_PATTERNS",
    "DimStatus",
    "Dimension",
    "FlangePattern",
    "Mount",
    "iso_pattern",
]

DimStatus = Literal["sourced", "secondary", "assumed"]


@dataclass(frozen=True)
class Dimension:
    """One dimension value with its provenance (mm)."""

    value: float
    status: DimStatus
    source: str = ""

    @property
    def verified(self) -> bool:
        return self.status == "sourced"

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "status": self.status, "source": self.source}


@dataclass(frozen=True)
class FlangePattern:
    """An ISO 9409-1 bolt pattern: pitch circle ``d1``, ``n`` holes of thread ``thread``,
    centring diameter ``d2`` (H7 on the robot side) and a dowel hole on the pitch circle."""

    designation: str
    d1: Dimension  # pitch circle diameter
    n_holes: int
    thread: str  # metric thread, e.g. "M6"
    thread_d: float  # nominal thread diameter (mm)
    d2: Dimension  # centring diameter
    dowel_d: Dimension  # dowel pin diameter
    seen_on: str = ""
    notes: tuple[str, ...] = field(default=())

    @property
    def unverified(self) -> tuple[str, ...]:
        """Names of the dimensions that are not ``sourced``."""
        return tuple(n for n in ("d1", "d2", "dowel_d") if not getattr(self, n).verified)

    def to_dict(self) -> dict[str, Any]:
        return {
            "designation": self.designation,
            "d1_mm": self.d1.to_dict(),
            "n_holes": self.n_holes,
            "thread": self.thread,
            "d2_mm": self.d2.to_dict(),
            "dowel_mm": self.dowel_d.to_dict(),
            "seen_on": self.seen_on,
            "unverified": list(self.unverified),
            "notes": list(self.notes),
        }


_GRASPMONKEY = "secondary catalogue table (centring diameters); check against ISO 9409-1:2004"
_ASSUMED_DOWEL = "assumed equal to the thread size"

#: Patterns the research could tie to published sources. 63 mm is left out: sources disagree
#: on its hole count.
ISO_PATTERNS: Final[Mapping[str, FlangePattern]] = MappingProxyType(
    {
        p.designation: p
        for p in (
            FlangePattern(
                "ISO 9409-1-31.5-4-M5",
                Dimension(31.5, "sourced", "CAD catalogue, tool-plate maker and gripper coupling listings"),
                4,
                "M5",
                5.0,
                Dimension(20.0, "secondary", _GRASPMONKEY),
                Dimension(5.0, "assumed", _ASSUMED_DOWEL),
                seen_on="small gripper couplings, tool plates",
                notes=("one secondary source lists this pattern as M6; three primary sources give M5",),
            ),
            FlangePattern(
                "ISO 9409-1-40-4-M6",
                Dimension(40.0, "sourced", "tool-plate maker listing"),
                4,
                "M6",
                6.0,
                Dimension(25.0, "secondary", _GRASPMONKEY),
                Dimension(6.0, "assumed", _ASSUMED_DOWEL),
                seen_on="tool plates",
            ),
            FlangePattern(
                "ISO 9409-1-50-4-M6",
                Dimension(50.0, "sourced", "cobot datasheets (several makers), gripper flange listings"),
                4,
                "M6",
                6.0,
                Dimension(31.5, "secondary", _GRASPMONKEY),
                Dimension(6.0, "sourced", "cobot tool-flange drawings show a 6 mm dowel hole"),
                seen_on="most 3 to 20 kg cobots, cobot gripper flanges",
            ),
            FlangePattern(
                "ISO 9409-1-80-6-M8",
                Dimension(80.0, "sourced", "large cobot and industrial arm datasheets"),
                6,
                "M8",
                8.0,
                Dimension(50.0, "secondary", _GRASPMONKEY),
                Dimension(8.0, "assumed", _ASSUMED_DOWEL),
                seen_on="20 to 35 kg cobots, mid-size industrial arms",
            ),
        )
    }
)

#: The two patterns the library ships pre-built adapters for (founder brief, 2026-10-18).
CONFIRMED_PATTERNS: Final[tuple[str, ...]] = ("ISO 9409-1-50-4-M6", "ISO 9409-1-31.5-4-M5")

#: A cobot maker's rule: tool-flange bolts must not reach deeper than 8 mm into the flange
#: (an M6 flange rule; ``sourced`` from the maker's support article in the licence audit).
COBOT_MAX_THREAD_DEPTH_MM: Final = 8.0


def iso_pattern(designation: str) -> FlangePattern:
    try:
        return ISO_PATTERNS[designation]
    except KeyError:
        raise KeyError(f"unknown ISO 9409-1 pattern {designation!r}; known: {sorted(ISO_PATTERNS)}") from None


@dataclass(frozen=True)
class Mount:
    """How a library item attaches (``SOURCE.json`` ``mount``).

    * ``interface``: plain words, brand-free (it goes on screen);
    * ``iso_pattern``: an ISO 9409-1 designation when the item mounts on one (or ``None``);
    * ``status``: ``sourced``, ``assumed`` or ``unknown`` for that pattern claim;
    * ``adapter``: the library key of a generated adapter that fits (or ``None``);
    * ``source``: where the claim comes from (a URL), when sourced.
    """

    interface: str
    iso_pattern: str | None = None
    status: Literal["sourced", "assumed", "unknown"] = "unknown"
    adapter: str | None = None
    source: str = ""
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.iso_pattern is not None and self.iso_pattern not in ISO_PATTERNS:
            raise ValueError(f"unknown ISO pattern {self.iso_pattern!r}")
        if self.status not in ("sourced", "assumed", "unknown"):
            raise ValueError("mount status must be sourced, assumed or unknown")
        if not self.interface.strip():
            raise ValueError("a mount needs an interface description")
        object.__setattr__(self, "notes", tuple(self.notes))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Mount:
        return cls(
            interface=str(raw["interface"]),
            iso_pattern=raw.get("iso_pattern"),
            status=raw.get("status", "unknown"),
            adapter=raw.get("adapter"),
            source=str(raw.get("source", "")),
            notes=tuple(raw.get("notes", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "iso_pattern": self.iso_pattern,
            "status": self.status,
            "adapter": self.adapter,
            "source": self.source,
            "notes": list(self.notes),
        }
