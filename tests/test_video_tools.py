"""scripts/video: the hook shot list and the caption/readout tools (no rendering here)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from rtt.sim.replay import OVERLAY_CHANNELS, Replay

REPO = Path(__file__).resolve().parents[1]


def _overlay_module():
    spec = importlib.util.spec_from_file_location(
        "rtt_video_overlay", REPO / "scripts" / "video" / "overlay.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rtt_video_overlay"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_shot_list_is_ten_seconds_of_known_layouts() -> None:
    spec = yaml.safe_load((REPO / "scripts" / "video" / "shots.yaml").read_text(encoding="utf-8"))
    assert spec["fps"] == 30 and spec["size"] == [1280, 720]
    assert sum(s["seconds"] for s in spec["shots"]) == pytest.approx(10.0)
    assert [s["layout"] for s in spec["shots"]] == ["single", "triptych", "sequence", "card"]
    for shot in spec["shots"]:
        ro = [shot.get("readout")] + [p.get("readout") for p in shot.get("panels", [])]
        assert all(r is None or r["channel"] in OVERLAY_CHANNELS for r in ro)
    tri = spec["shots"][1]["panels"]
    assert [p["tint"] for p in tri] == ["red", "amber", "green"]


def test_readouts_and_frame_spans() -> None:
    ov = _overlay_module()
    frames = np.zeros((90, 4, 7), dtype=np.float32)
    header = {
        "fps": 30,
        "bodies": [{"name": n} for n in ("palm", "finger_left", "finger_right", "object")],
        "overlays": {"channels": ["offset_mm", "seat_short_mm"]},
    }
    overlays = np.stack([np.linspace(2.0, 0.05, 90), np.full(90, 5.0)], axis=1).astype(np.float16)
    replay = Replay(frames=frames, header=header, overlays=overlays)
    assert ov.readout_text(replay, "offset_mm", "{start:.1f} -> {value:.2f} mm", 89) == "2.0 -> 0.05 mm"
    assert ov.readout_text(replay, "hold_s", "{value}", 0) is None
    idx = ov.frame_indices(replay, [1.0, 2.0], 31)
    assert idx[0] == 30 and idx[-1] == 60 and len(idx) == 31
    assert ov.frame_indices(replay, "all", 3).tolist() == [0, 44, 89]
    card = ov._card("title", (320, 180))
    assert card.shape == (180, 320, 3)


# --------------------------------------------------------------------------- the v2 shot list

import json
import re
import shutil
import subprocess

from rtt.effectors.types import find_brand_tokens
from rtt.library.brands import find_library_brands

SHOTS = REPO / "scripts" / "video" / "shots.json"
SCRIPT_DOC = REPO / "docs" / "VIDEO_SCRIPT.md"
PENDING = re.compile(r"\{\{\s*pending:[^}]*\}\}", re.IGNORECASE)
#: Sponsor terms may show on screen (founder decision, 27 Oct 2026); these other names may not.
SPONSOR_TERMS = ("NVIDIA", "Nemotron", "Nebius", "Token Factory")
NOT_ON_SCREEN = re.compile(
    r"(?<![a-z0-9])(openai|claude|anthropic|github|youtube|mujoco)(?![a-z0-9])",
    re.IGNORECASE,
)


OVERRIDABLE = ("page", "on_screen", "voiceover", "caption", "tag", "setup", "timeline")


def _resolve(raw: dict, on: tuple[str, ...]) -> dict:
    """The rule of scripts/video/shots.mjs: a flagged shot's fallback replaces it while its flag is off."""
    flags = {**raw["flags"], **dict.fromkeys(on, True)}
    shots = []
    for s in raw["shots"]:
        s = json.loads(json.dumps(s))
        fallback = s.pop("fallback", None)
        if s.get("flag"):
            s["variant"] = "flag" if flags[s["flag"]] else "fallback"
            if not flags[s["flag"]]:
                s.update({k: v for k, v in (fallback or {}).items() if k in OVERRIDABLE})
        shots.append(s)
    return {**raw, "flags": flags, "shots": shots}


def _raw_shots() -> dict:
    return json.loads(SHOTS.read_text(encoding="utf-8"))


def _variants() -> list[dict]:
    raw = _raw_shots()
    return [_resolve(raw, ()), _resolve(raw, tuple(raw["flags"]))]


def _spoken_words(text: str) -> int:
    """Words outside the markers, plus the length a marker states ("about 12 words")."""
    n = len(PENDING.sub(" ", text).split())
    for m in PENDING.findall(text):
        about = re.search(r"about (\d+) words", m)
        n += int(about.group(1)) if about else 10
    return n


def test_the_shot_list_is_back_to_back_under_the_limit_with_every_flag_state() -> None:
    raw = _raw_shots()
    assert raw["format"] == "rtt-video-shots/2"
    assert raw["max_duration_s"] <= 179
    for spec in _variants():
        t = 0
        for s in spec["shots"]:
            assert s["start"] == pytest.approx(t), s["id"]
            assert s["duration"] > 0 and s["voiceover"].strip()
            assert isinstance(s["setup"], list) and isinstance(s["timeline"], list)
            assert s["tag"] is None or s["tag"] in spec["labels"]
            t += s["duration"]
        assert t <= raw["max_duration_s"]
        on_screen = sum(s["duration"] for s in spec["shots"] if s["page"])
        assert on_screen / t > 0.9  # the product is on screen for nearly all of the cut


def test_a_flagged_shot_has_a_fallback_and_keeps_its_slot() -> None:
    raw = _raw_shots()
    flagged = [s for s in raw["shots"] if s.get("flag")]
    assert flagged and all(s["flag"] in raw["flags"] and s.get("fallback") for s in flagged)
    assert not any(raw["flags"].values())  # every flag is off by default
    off, on = _variants()
    for s in flagged:
        a = next(x for x in off["shots"] if x["id"] == s["id"])
        b = next(x for x in on["shots"] if x["id"] == s["id"])
        assert (a["start"], a["duration"]) == (b["start"], b["duration"])
        assert a["variant"] == "fallback" and b["variant"] == "flag"
        assert a["voiceover"] == s["fallback"]["voiceover"] and b["voiceover"] == s["voiceover"]


def test_every_voiceover_line_is_in_the_script_word_for_word() -> None:
    doc = " ".join(SCRIPT_DOC.read_text(encoding="utf-8").split())
    for spec in _variants():
        for s in spec["shots"]:
            assert " ".join(s["voiceover"].split()) in doc, s["id"]


def test_no_brand_on_screen_and_the_platform_names_are_spoken() -> None:
    for spec in _variants():
        burned = [s.get("caption") or "" for s in spec["shots"]]
        burned += [spec["labels"][s["tag"]] for s in spec["shots"] if s.get("tag")]
        burned += [
            spec["thumbnail"]["title"],
            spec["thumbnail"]["subtitle"],
            spec["end_card"]["title"],
            *spec["end_card"]["lines"],
        ]
        spoken = [s["voiceover"] for s in spec["shots"]]
        for text in burned + spoken:
            assert not find_brand_tokens(text) and not find_library_brands(text), text
        for text in burned:
            assert not NOT_ON_SCREEN.search(text), text
        said = " ".join(spoken)
        for phrase in ("Nebius Token Factory", "Nemotron", "Nebius Serverless Jobs"):
            assert phrase in said
        assert "simulated" in said
        assert "Nebius" in spec["end_card"]["credit"] and "AI" in " ".join(spec["end_card"]["lines"])


def test_the_pace_leaves_room_to_breathe() -> None:
    for spec in _variants():
        total_w = 0
        for s in spec["shots"]:
            w = _spoken_words(s["voiceover"])
            total_w += w
            assert w / s["duration"] <= 2.6, (s["id"], w, s["duration"])  # about 156 words a minute at most
        total_s = sum(s["duration"] for s in spec["shots"])
        assert total_w / total_s * 60 <= 150


def test_the_node_rules_agree_with_the_python_reading() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    tool = REPO / "scripts" / "video" / "shots.mjs"
    raw = _raw_shots()
    for on, spec in zip(((), tuple(raw["flags"])), _variants(), strict=True):
        args = [a for f in on for a in ("--flag", f)]
        out = subprocess.run([node, str(tool), "resolve", *args], capture_output=True, text=True, check=True)
        assert [s["voiceover"] for s in json.loads(out.stdout)["shots"]] == [
            s["voiceover"] for s in spec["shots"]
        ]
        assert (
            subprocess.run([node, str(tool), "check", *args], check=False, capture_output=True).returncode
            == 0
        )
        markers = any(PENDING.search(s["voiceover"] + (s.get("caption") or "")) for s in spec["shots"])
        final = subprocess.run([node, str(tool), "check", "--final", *args], check=False, capture_output=True)
        assert final.returncode == (1 if markers else 0)


def test_sponsor_terms_may_show_and_vendor_brands_may_not(tmp_path: Path) -> None:
    from rtt.effectors.types import BRAND_TOKENS

    for term in SPONSOR_TERMS:
        assert (
            not find_brand_tokens(term) and not find_library_brands(term) and not NOT_ON_SCREEN.search(term)
        )
    vendor = BRAND_TOKENS[0]
    assert find_brand_tokens(f"a {vendor} gripper")
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    tool = REPO / "scripts" / "video" / "shots.mjs"
    raw = _raw_shots()

    def check(caption: str) -> int:
        doc = json.loads(json.dumps(raw))
        doc["shots"][1]["caption"] = caption
        f = tmp_path / "shots.json"
        f.write_text(json.dumps(doc), encoding="utf-8")
        return subprocess.run(
            [node, str(tool), "check", "--shots", str(f)], check=False, capture_output=True
        ).returncode

    assert check("NVIDIA Nemotron plans on Nebius Token Factory") == 0
    assert check(f"Recorded with a {vendor} gripper") == 1
    assert check("Voice by OpenAI") == 1
