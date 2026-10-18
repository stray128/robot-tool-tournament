"""The library's licence gate (product mode): lanes, the chain rule, credit lines."""

from __future__ import annotations

import pytest

from rtt.effectors.types import ALLOWED_LICENSES
from rtt.library.licence import (
    AMBER_LICENCES,
    DENIED_REFS,
    GREEN_LICENCES,
    LICENCE_URLS,
    ChainLink,
    Rating,
    attribution_line,
    chain_rating,
    classify,
    gate,
)


@pytest.mark.parametrize("spdx", GREEN_LICENCES)
def test_green_lane_licences_are_green(spdx: str) -> None:
    assert classify(spdx) is Rating.GREEN
    assert LICENCE_URLS[spdx].startswith("https://")


@pytest.mark.parametrize("spdx", AMBER_LICENCES)
def test_share_alike_and_reciprocal_licences_need_review(spdx: str) -> None:
    assert classify(spdx) is Rating.AMBER


@pytest.mark.parametrize(
    "spdx",
    [
        "CC-BY-NC-4.0",
        "CC-BY-NC-SA-4.0",
        "CC-BY-NC-3.0",
        "CC-BY-ND-4.0",
        "CC-BY-NC-ND-4.0",
        "LicenseRef-something-else",
        "",
        *DENIED_REFS,
    ],
)
def test_non_commercial_no_derivatives_unknown_and_eulas_are_red(spdx: str) -> None:
    assert classify(spdx) is Rating.RED


def test_the_contract_vendoring_list_is_inside_the_green_lane() -> None:
    assert set(ALLOWED_LICENSES) <= set(GREEN_LICENCES)


def test_a_file_is_only_as_clean_as_the_worst_link() -> None:
    clean = ChainLink("MIT")
    assert chain_rating([clean, ChainLink("Apache-2.0", "derived_from")]) is Rating.GREEN
    assert chain_rating([clean, ChainLink("CC-BY-SA-4.0")]) is Rating.AMBER
    assert chain_rating([clean, ChainLink("CC-BY-NC-4.0")]) is Rating.RED
    assert chain_rating([]) is Rating.RED  # no licence found


def test_sim_geometry_copying_a_non_commercial_design_is_simulation_only() -> None:
    links = [ChainLink("MIT"), ChainLink("CC-BY-NC-SA-4.0", "derived_from", "hardware CAD")]
    assert chain_rating(links) is Rating.AMBER
    d = gate(links, allow_amber=True)
    assert d.shippable and not d.printable  # never in a print pack
    assert not gate(links).shippable  # and excluded by default (founder, 2026-10-18)


def test_the_gate_ships_green_only_by_default() -> None:
    assert gate([ChainLink("CERN-OHL-P-2.0")]).shippable
    assert gate([ChainLink("CERN-OHL-P-2.0")]).printable
    d = gate([ChainLink("CERN-OHL-S-2.0")])
    assert not d.shippable and d.rating is Rating.AMBER and d.reasons
    assert not gate([]).shippable and gate([]).reasons == ("no licence statement found",)


def test_credit_line_format() -> None:
    line = attribution_line(
        "Example gripper",
        "A. Maker",
        "https://example.org/g",
        "CC-BY-4.0",
        commit="abc1234def",
        changes=["convex pieces (CoACD 0.02, seed 0)"],
    )
    assert line == (
        '"Example gripper" by A. Maker, https://example.org/g (commit abc1234), licensed CC-BY-4.0 '
        "(https://creativecommons.org/licenses/by/4.0/). Modified by Effector Lab: convex pieces (CoACD 0.02, seed 0)."
    )
    assert attribution_line("T", "A", "https://x", "MIT", retrieved="2026-10-18").endswith("Unmodified.")
