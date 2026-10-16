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
