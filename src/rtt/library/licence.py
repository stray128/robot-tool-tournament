"""The library's licence gate: which SPDX licences may ship, the chain rule, and credit lines.

Three lanes (``docs/LIBRARY.md`` section 3):

* **green** (:data:`GREEN_LICENCES`): permissive licences granted by the copyright holder or
  traceable to them. Files in this lane ship with attribution.
* **amber** (:data:`AMBER_LICENCES`): allowed only after review (share-alike, reciprocal or
  copyleft licences). The founder decided on 2026-10-18 that amber items stay **out** of the
  library and its print packs; the catalog page lists them with the reason.
* **red**: everything else. Any non-commercial (``-NC``) or no-derivatives (``-ND``)
  variant, no licence at all, "personal use only", platform or vendor terms that forbid
  redistribution (:data:`DENIED_REFS`), and unresolved conflicts between two statements.

**Chain rule.** A file is only as clean as the worst link in its derivation chain
(:func:`chain_rating`). A link is either the licence the file itself carries
(``relation = "licensed"``) or the licence of something its geometry was derived from
(``relation = "derived_from"``). Geometry that copies a non-commercial design is amber at best
(simulation only, never printable) even when the derived file carries a permissive licence.

Copyright licences cover files, not patents or trademarks. Makers' names stay off screen
(:mod:`rtt.library.brands`); credits name them (:func:`attribution_line`).
"""

from __future__ import annotations

import enum
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

__all__ = [
    "AMBER_LICENCES",
    "DENIED_REFS",
    "GREEN_LICENCES",
    "LICENCE_URLS",
    "ChainLink",
    "LicenceDecision",
    "Rating",
    "attribution_line",
    "chain_rating",
    "classify",
    "gate",
    "licence_url",
]


class Rating(enum.StrEnum):
    """Licence lanes, best first."""

    GREEN = "green"
    AMBER = "amber"
    RED = "red"

    @property
    def rank(self) -> int:
        return _RANK[self]


_RANK: Final = {Rating.GREEN: 0, Rating.AMBER: 1, Rating.RED: 2}

#: Allowed automatically: permissive licences on the files we would ship.
GREEN_LICENCES: Final[tuple[str, ...]] = (
    "CC0-1.0",
    "CC-BY-3.0",
    "CC-BY-4.0",
    "MIT",
    "MIT-0",
    "0BSD",
    "ISC",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "BSD-3-Clause-Clear",
    "Zlib",
    "CERN-OHL-P-2.0",
)

#: Allowed only after review: share-alike, weakly or strongly reciprocal, copyleft.
AMBER_LICENCES: Final[tuple[str, ...]] = (
    "CC-BY-SA-3.0",
    "CC-BY-SA-4.0",
    "CERN-OHL-W-2.0",
    "CERN-OHL-S-2.0",
    "GPL-2.0-only",
    "GPL-2.0-or-later",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
    "LGPL-2.1-only",
    "LGPL-2.1-or-later",
    "LGPL-3.0-only",
    "LGPL-3.0-or-later",
    "MPL-2.0",
)

#: Terms that forbid redistribution or carry no licence (SPDX ``LicenseRef-`` ids we use).
DENIED_REFS: Final[tuple[str, ...]] = (
    "NONE",
    "NOASSERTION",
    "LicenseRef-all-rights-reserved",
    "LicenseRef-personal-use-only",
    "LicenseRef-platform-eula",  # model-sharing sites whose terms are non-sublicensable
    "LicenseRef-vendor-cad-portal",  # vendor CAD download portals
    "LicenseRef-simulator-asset-eula",  # simulator asset packs that may not be redistributed
    "LicenseRef-conflicting-statements",
)

_NC_ND: Final = re.compile(r"(^|-)(NC|ND)(-|$)", re.IGNORECASE)

#: Licence texts, for credit lines (CC BY requires the licence's URI or a link to it).
LICENCE_URLS: Final[dict[str, str]] = {
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC-BY-3.0": "https://creativecommons.org/licenses/by/3.0/",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "MIT": "https://spdx.org/licenses/MIT.html",
    "MIT-0": "https://spdx.org/licenses/MIT-0.html",
    "0BSD": "https://spdx.org/licenses/0BSD.html",
    "ISC": "https://spdx.org/licenses/ISC.html",
    "Apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
    "BSD-2-Clause": "https://spdx.org/licenses/BSD-2-Clause.html",
    "BSD-3-Clause": "https://spdx.org/licenses/BSD-3-Clause.html",
    "BSD-3-Clause-Clear": "https://spdx.org/licenses/BSD-3-Clause-Clear.html",
    "Zlib": "https://spdx.org/licenses/Zlib.html",
    "CERN-OHL-P-2.0": "https://ohwr.org/cern_ohl_p_v2.txt",
}


def licence_url(spdx: str) -> str | None:
    """The licence text's URL (``None`` for an id outside the green lane)."""
    return LICENCE_URLS.get(spdx)


def classify(spdx: str) -> Rating:
    """The lane of one SPDX id (exact match, case-sensitive as SPDX ids are)."""
    if not isinstance(spdx, str) or not spdx.strip():
        return Rating.RED
    s = spdx.strip()
    if s in DENIED_REFS or _NC_ND.search(s):
        return Rating.RED
    if s in GREEN_LICENCES:
        return Rating.GREEN
    if s in AMBER_LICENCES:
        return Rating.AMBER
    return Rating.RED  # unknown: denied until someone reviews it


@dataclass(frozen=True)
class ChainLink:
    """One link of a file's derivation chain.

    ``relation``: ``"licensed"`` (the file's own licence) or ``"derived_from"`` (the licence of
    the design, model or CAD the file's geometry was made from). ``what`` and ``source`` say
    which upstream statement this is (a LICENSE file, a README sentence, a listing page).
    """

    spdx: str
    relation: Literal["licensed", "derived_from"] = "licensed"
    what: str = ""
    source: str = ""

    def rating(self) -> Rating:
        r = classify(self.spdx)
        if self.relation == "derived_from" and r is Rating.RED and _NC_ND.search(self.spdx or ""):
            return Rating.AMBER  # sim geometry copying an NC design: simulation only, never printed
        return r

    def to_dict(self) -> dict[str, str]:
        return {"spdx": self.spdx, "relation": self.relation, "what": self.what, "source": self.source}

    @classmethod
    def from_dict(cls, raw: dict[str, str]) -> ChainLink:
        return cls(
            spdx=str(raw["spdx"]),
            relation=raw.get("relation", "licensed"),  # type: ignore[arg-type]
            what=str(raw.get("what", "")),
            source=str(raw.get("source", "")),
        )


def chain_rating(links: Iterable[ChainLink]) -> Rating:
    """The worst link's rating (an empty chain is red: no licence was found)."""
    worst = None
    for link in links:
        r = link.rating()
        if worst is None or r.rank > worst.rank:
            worst = r
    return worst if worst is not None else Rating.RED


@dataclass(frozen=True)
class LicenceDecision:
    """What the gate decided for one file or item."""

    rating: Rating
    shippable: bool  # may be redistributed in this repository and on the site
    printable: bool  # may go into a print pack
    reasons: tuple[str, ...] = field(default=())


def gate(links: Sequence[ChainLink], *, allow_amber: bool = False) -> LicenceDecision:
    """Apply the gate to one derivation chain.

    Green ships and prints. Amber ships only with ``allow_amber`` (never in this build: the
    founder excluded amber items on 2026-10-18), and never prints when the amber link is a
    non-commercial design. Red never ships.
    """
    rating = chain_rating(links)
    reasons = []
    for link in links:
        r = link.rating()
        if r is not Rating.GREEN:
            reasons.append(
                f"{link.relation} {link.spdx or 'no licence'} ({r.value}){': ' + link.what if link.what else ''}"
            )
    if not links:
        reasons.append("no licence statement found")
    nc_design = any(link.relation == "derived_from" and _NC_ND.search(link.spdx or "") for link in links)
    if rating is Rating.GREEN:
        return LicenceDecision(rating, True, True, ())
    if rating is Rating.AMBER and allow_amber:
        return LicenceDecision(rating, True, not nc_design, tuple(reasons))
    return LicenceDecision(rating, False, False, tuple(reasons))


def attribution_line(
    title: str,
    author: str,
    source_url: str,
    spdx: str,
    *,
    commit: str | None = None,
    retrieved: str | None = None,
    changes: Sequence[str] = (),
) -> str:
    """One credit line in the library's format (title, author, source, licence, changes).

    >>> attribution_line("Example gripper", "A. Maker", "https://example.org/g", "CC-BY-4.0",
    ...                  commit="abc1234def", changes=["convex pieces (CoACD 0.02, seed 0)"])
    '"Example gripper" by A. Maker, https://example.org/g (commit abc1234), licensed CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/). Modified by Effector Lab: convex pieces (CoACD 0.02, seed 0).'
    """
    where = f" (commit {commit[:7]})" if commit else (f" (retrieved {retrieved})" if retrieved else "")
    url = licence_url(spdx)
    lic = f"licensed {spdx}" + (f" ({url})" if url else "")
    line = f'"{title}" by {author}, {source_url}{where}, {lic}.'
    if changes:
        line += " Modified by Effector Lab: " + "; ".join(changes) + "."
    else:
        line += " Unmodified."
    return line
