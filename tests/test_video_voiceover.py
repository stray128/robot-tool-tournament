"""scripts/video/voiceover.py with a stub speech client: no key, no network, no spend."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import wave
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RATE = 24_000


def _vo():
    spec = importlib.util.spec_from_file_location(
        "rtt_video_voiceover", REPO / "scripts" / "video" / "voiceover.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rtt_video_voiceover"] = mod
    spec.loader.exec_module(mod)
    return mod


class _Resp:
    def __init__(self, content: bytes) -> None:
        self.content = content


class StubClient:
    """``client.audio.speech.create(...)``: 0.06 s of a quiet tone per character, recorded."""

    def __init__(
        self, seconds_per_char: float = 0.06, long_for: str | None = None, fail_with: str | None = None
    ) -> None:
        self.calls: list[dict] = []
        self.seconds_per_char = seconds_per_char
        self.long_for = long_for
        self.fail_with = fail_with
        outer = self

        class _Speech:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                if outer.fail_with:
                    raise RuntimeError(outer.fail_with)
                n = len(kwargs["input"])
                secs = (
                    n
                    * outer.seconds_per_char
                    * (10 if outer.long_for and outer.long_for in kwargs["input"] else 1)
                )
                frames = int(secs * RATE)
                tone = bytearray()
                for i in range(frames):
                    v = int(1000 * math.sin(2 * math.pi * 220 * i / RATE))
                    tone += v.to_bytes(2, "little", signed=True)
                return _Resp(bytes(tone))

        class _Audio:
            speech = _Speech()

        self.audio = _Audio()


def _shots(tmp: Path, *, marker: bool = False) -> Path:
    shots = {
        "format": "rtt-video-shots/2",
        "script": "docs/VIDEO_SCRIPT.md",
        "max_duration_s": 179,
        "flags": {"hosted": False},
        "labels": {},
        "shots": [
            {
                "id": "a1",
                "start": 0,
                "duration": 4,
                "voiceover": "First line, short.",
                "setup": [],
                "timeline": [],
            },
            {
                "id": "a2",
                "start": 4,
                "duration": 5,
                "voiceover": "Second line." + (" {{pending: a result, about 4 words}}" if marker else ""),
                "setup": [],
                "timeline": [],
            },
            {
                "id": "a3",
                "start": 9,
                "duration": 3,
                "flag": "hosted",
                "voiceover": "Hosted line.",
                "setup": [],
                "timeline": [],
                "fallback": {"voiceover": "Local line."},
            },
        ],
    }
    f = tmp / "shots.json"
    f.write_text(json.dumps(shots), encoding="utf-8")
    return f


def _guard(tmp: Path, spent: float = 0.0):
    vo = _vo()
    vr = vo._vision_review()
    f = tmp / "spend.json"
    if spent:
        f.write_text(json.dumps({vo.SPEND_NAME: {"spent_usd": spent, "calls": 1}}), encoding="utf-8")
    return vr.SpendGuard(f, caps={vo.SPEND_NAME: vo.HARD_CAP_USD})


def _seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, RATE)
        return w.getnframes() / RATE


def test_lines_in_shot_order_per_line_wavs_and_a_track_as_long_as_the_cut(tmp_path: Path) -> None:
    vo = _vo()
    stub = StubClient()
    res = vo.run(
        out=tmp_path / "out",
        shots_file=_shots(tmp_path),
        client=stub,
        guard=_guard(tmp_path),
        log=lambda *_: None,
    )
    assert [c["input"] for c in stub.calls] == ["First line, short.", "Second line.", "Local line."]
    assert all(
        c["response_format"] == "pcm" and c["model"] == vo.DEFAULT_MODEL and c["voice"] == vo.DEFAULT_VOICE
        for c in stub.calls
    )
    assert all(c["instructions"] == vo.DEFAULT_INSTRUCTIONS for c in stub.calls)
    out = tmp_path / "out"
    for i in ("a1", "a2", "a3"):
        assert _seconds(out / "lines" / f"{i}.wav") > 0
    assert _seconds(out / "voiceover.wav") == pytest.approx(12.0, abs=1 / RATE)
    m = json.loads((out / "voiceover.json").read_text())
    assert [ln["id"] for ln in m["lines"]] == ["a1", "a2", "a3"]
    assert [ln["at_s"] for ln in m["lines"]] == [0.35, 4.35, 9.35]
    assert m["flags"] == [] and m["draft"] is False and res.problems == []
    assert m["cost"]["guard_estimate_this_run_usd"] > 0


def test_a_flag_speaks_the_flagged_line(tmp_path: Path) -> None:
    vo = _vo()
    stub = StubClient()
    vo.run(
        out=tmp_path / "o",
        shots_file=_shots(tmp_path),
        flags=["hosted"],
        client=stub,
        guard=_guard(tmp_path),
        log=lambda *_: None,
    )
    assert stub.calls[-1]["input"] == "Hosted line."
    assert json.loads((tmp_path / "o" / "voiceover.json").read_text())["flags"] == ["hosted"]
    with pytest.raises(vo.VoiceoverError, match="unknown flag"):
        vo.run(
            out=tmp_path / "o",
            shots_file=_shots(tmp_path),
            flags=["nope"],
            client=stub,
            guard=_guard(tmp_path),
            log=lambda *_: None,
        )


def test_pending_markers_are_refused_unless_a_draft_speaks_a_stand_in(tmp_path: Path) -> None:
    vo = _vo()
    stub = StubClient()
    shots = _shots(tmp_path, marker=True)
    with pytest.raises(vo.VoiceoverError, match="pending markers in a2"):
        vo.run(out=tmp_path / "o", shots_file=shots, client=stub, guard=_guard(tmp_path), log=lambda *_: None)
    assert stub.calls == []
    vo.run(
        out=tmp_path / "o",
        shots_file=shots,
        draft=True,
        pending_say="Result to come.",
        client=stub,
        guard=_guard(tmp_path),
        log=lambda *_: None,
    )
    assert stub.calls[1]["input"] == "Second line. Result to come."
    m = json.loads((tmp_path / "o" / "voiceover.json").read_text())
    assert m["draft"] is True and m["lines"][1]["pending"] == 1


def test_the_spend_guard_refuses_before_any_call(tmp_path: Path) -> None:
    vo = _vo()
    vr = vo._vision_review()
    stub = StubClient()
    with pytest.raises(vr.SpendCapReached):
        vo.run(
            out=tmp_path / "o",
            shots_file=_shots(tmp_path),
            client=stub,
            guard=_guard(tmp_path, spent=1.9999),
            log=lambda *_: None,
        )
    assert stub.calls == []
    with pytest.raises(vo.VoiceoverError, match="at most 2.00"):
        vo.run(
            out=tmp_path / "o",
            shots_file=_shots(tmp_path),
            cap_usd=5.0,
            client=stub,
            guard=_guard(tmp_path),
            log=lambda *_: None,
        )
    with pytest.raises(vo.VoiceoverError, match="no price"):
        vo.run(
            out=tmp_path / "o",
            shots_file=_shots(tmp_path),
            model="some-new-tts",
            client=stub,
            guard=_guard(tmp_path),
            log=lambda *_: None,
        )
    assert stub.calls == []


def test_the_estimate_is_by_characters_and_recorded(tmp_path: Path) -> None:
    vo = _vo()
    guard = _guard(tmp_path)
    vo.run(
        out=tmp_path / "o", shots_file=_shots(tmp_path), client=StubClient(), guard=guard, log=lambda *_: None
    )
    chars = sum(
        len(t) + len(vo.DEFAULT_INSTRUCTIONS) for t in ("First line, short.", "Second line.", "Local line.")
    )
    assert guard.spent(vo.SPEND_NAME) == pytest.approx(
        chars * vo.PRICE_USD_PER_MCHAR[vo.DEFAULT_MODEL] / 1e6, abs=1e-6
    )


def test_cached_lines_cost_nothing(tmp_path: Path) -> None:
    vo = _vo()
    guard = _guard(tmp_path)
    vo.run(
        out=tmp_path / "o", shots_file=_shots(tmp_path), client=StubClient(), guard=guard, log=lambda *_: None
    )
    spent = guard.spent(vo.SPEND_NAME)
    again = StubClient()
    res = vo.run(
        out=tmp_path / "o", shots_file=_shots(tmp_path), client=again, guard=guard, log=lambda *_: None
    )
    assert again.calls == [] and guard.spent(vo.SPEND_NAME) == pytest.approx(spent)
    assert res.manifest["cost"]["guard_estimate_this_run_usd"] == 0


def test_an_overrun_is_reported(tmp_path: Path) -> None:
    vo = _vo()
    res = vo.run(
        out=tmp_path / "o",
        shots_file=_shots(tmp_path),
        client=StubClient(long_for="Second"),
        guard=_guard(tmp_path),
        log=lambda *_: None,
    )
    assert any(p.startswith("a2 runs") for p in res.problems)
    m = json.loads((tmp_path / "o" / "voiceover.json").read_text())
    assert m["lines"][2]["at_s"] > 9.35  # the next line waits for the long one


def test_the_key_never_reaches_the_output(tmp_path: Path, monkeypatch, capsys) -> None:
    vo = _vo()
    secret = "sk-test-NEVERPRINTME0123456789"
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps({"openai": secret}), encoding="utf-8")
    seen = {}

    def fake_client(key):
        seen["key"] = key.reveal()
        assert secret not in repr(key) and secret not in str(key)
        return StubClient()

    monkeypatch.setattr(vo, "openai_client", fake_client)
    out = tmp_path / "out"
    code = vo.main(
        [
            "--out",
            str(out),
            "--shots",
            str(_shots(tmp_path)),
            "--keys",
            str(keys),
            "--spend-file",
            str(tmp_path / "s.json"),
        ]
    )
    assert code == 0 and seen["key"] == secret
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    for f in out.rglob("*"):
        if f.is_file():
            assert secret.encode() not in f.read_bytes()


def test_an_api_error_is_redacted_and_releases_its_reservation(tmp_path: Path) -> None:
    vo = _vo()
    guard = _guard(tmp_path)
    with pytest.raises(vo.VoiceoverError) as err:
        vo.run(
            out=tmp_path / "o",
            shots_file=_shots(tmp_path),
            client=StubClient(fail_with="401: bad key sk-abcdef1234567890"),
            guard=guard,
            log=lambda *_: None,
        )
    assert "sk-abcdef1234567890" not in str(err.value)
    assert guard.spent(vo.SPEND_NAME) == 0 and guard._reserved.get(vo.SPEND_NAME, 0) == 0


def test_the_real_shot_list_needs_a_draft_while_markers_remain(tmp_path: Path) -> None:
    vo = _vo()
    raw = json.loads((REPO / "scripts" / "video" / "shots.json").read_text(encoding="utf-8"))
    lines = vo.lines_of(vo.resolve_shots(raw, ()), draft=True, pending_say="Result to come.")
    assert [ln.id for ln in lines] == [s["id"] for s in raw["shots"]]
    assert all("{{" not in ln.spoken for ln in lines)
    res = vo.run(
        out=tmp_path / "o",
        client=StubClient(),
        guard=_guard(tmp_path),
        dry_run=True,
        draft=True,
        log=lambda *_: None,
    )
    assert res.manifest["dry_run"] and res.manifest["estimate_usd"] < 0.2


def test_the_python_reading_of_flags_matches_the_node_rules() -> None:
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    vo = _vo()
    shots = REPO / "scripts" / "video" / "shots.json"
    raw = json.loads(shots.read_text(encoding="utf-8"))
    for on in ((), tuple(raw["flags"])):
        args = [a for f in on for a in ("--flag", f)]
        out = subprocess.run(
            [node, str(REPO / "scripts" / "video" / "shots.mjs"), "resolve", *args],
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(out.stdout)["shots"] == vo.resolve_shots(raw, on)["shots"]
