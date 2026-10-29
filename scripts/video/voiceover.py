#!/usr/bin/env python
"""AI voice-over for the demo video: one WAV per voice-over line of ``shots.json``, and one timed
``voiceover.wav`` that ``scripts/video/assemble.sh --voiceover`` lays over the picture.

::

    python scripts/video/voiceover.py --out DIR [--shots FILE] [--flag NAME]...
        [--model gpt-4o-mini-tts-2025-12-15] [--voice marin] [--instructions TEXT] [--speed X]
        [--draft [--pending-say TEXT]] [--keys FILE] [--cap-usd 2.0] [--spend-file FILE]
        [--dry-run]

The lines are read from ``shots.json`` in shot order, with its flags resolved by the same rule as
``scripts/video/shots.mjs`` (a flagged shot's ``fallback`` replaces it while its flag is off).
Each line is sent to the OpenAI text-to-speech API (``/v1/audio/speech``) as raw 24 kHz, 16-bit
mono PCM and written as ``DIR/lines/<shot id>.wav``. ``DIR/voiceover.wav`` places each line at its
shot's start plus a short lead-in (``--lead-s``), later only when the line before it is still
speaking, and is exactly as long as the cut. ``DIR/voiceover.json`` records the model, the voice,
the flags, each line's place, length and any overrun of its shot, and the cost; ``assemble.sh``
reads it to refuse a voice-over made with other flags, other shots, or as a draft.

**Pending markers.** A line that still holds a ``{{pending: ...}}`` marker is refused, because the
marker would be spoken. ``--draft`` speaks each marker as ``--pending-say`` instead, so the voice and
the pacing can be judged early; the output is marked as a draft, and the final cut refuses it.

**Key.** Read at run time from a JSON file's ``openai`` entry: ``--keys``, else ``RTT_API_KEYS``,
else ``api_keys.json`` at this checkout's root, else at the main checkout's root (git-ignored). It
is held in an object that never prints, and it is never logged, written or sent anywhere but to
the API. Error messages are redacted.

**Spend guard.** Estimated by characters before every call: (the line's characters plus the
instructions' characters) times :data:`PRICE_USD_PER_MCHAR` for the model. The run refuses to start
when the total estimate would pass the cap (``--cap-usd``, at most :data:`HARD_CAP_USD`), and each
call reserves its estimate first. The running total is shared by every checkout and worktree:
``--spend-file``, else ``RTT_TTS_SPEND_FILE``, else ``<git common dir>/rtt/tts_spend.json``. A spend
file that cannot be trusted stops the run (the guard of ``scripts/replay_vision_review.py``). A line
already made with the same model, voice, instructions and text is reused from ``DIR/cache/`` at no
cost.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

HERE: Final = Path(__file__).resolve().parent
REPO: Final = HERE.parents[1]
DEFAULT_SHOTS: Final = HERE / "shots.json"
FORMAT: Final = "rtt-video-voiceover/1"

DEFAULT_MODEL: Final = "gpt-4o-mini-tts-2025-12-15"
DEFAULT_VOICE: Final = "marin"
DEFAULT_INSTRUCTIONS: Final = (
    "Speak as a calm, clear narrator of a short product demo. Keep a steady, unhurried pace, a warm "
    "and even tone, and plain emphasis. Pause briefly at each full stop. Read numbers as written."
)
#: Dollars per million input characters that the guard estimates with. tts-1 and tts-1-hd are
#: billed by character at these published rates. gpt-4o-mini-tts is billed by tokens (text in, audio
#: out; OpenAI's own estimate is about $0.015 a minute of audio): its rate here is **[assumed]**, set
#: at twice the published per-minute estimate expressed per character, on purpose, so the guard
#: stops early rather than late. A model that is not listed is refused unless a rate is given.
PRICE_USD_PER_MCHAR: Final[Mapping[str, float]] = {
    "tts-1": 15.0,
    "tts-1-hd": 30.0,
    "gpt-4o-mini-tts": 30.0,
    "gpt-4o-mini-tts-2025-12-15": 30.0,
}
#: OpenAI's published estimate for the token-billed model, dollars per minute of audio: the
#: "likely" cost reported beside the guard's estimate.
USD_PER_AUDIO_MINUTE: Final[Mapping[str, float]] = {
    "gpt-4o-mini-tts": 0.015,
    "gpt-4o-mini-tts-2025-12-15": 0.015,
}
#: This script never spends more than this, whatever ``--cap-usd`` says (founder decision).
HARD_CAP_USD: Final = 2.0
SPEND_NAME: Final = "openai_tts"
SAMPLE_RATE: Final = 24_000  # the API's pcm format: 24 kHz, 16-bit signed little-endian, mono
PENDING: Final = re.compile(r"\{\{\s*pending:[^}]*\}\}", re.IGNORECASE)
OVERRIDABLE: Final = ("page", "on_screen", "voiceover", "caption", "tag", "setup", "timeline")


class VoiceoverError(RuntimeError):
    """The run cannot go on; the message says why (never with a key in it)."""


def _vision_review() -> Any:
    """``scripts/replay_vision_review.py``: its key holder, key loader, spend guard and redaction."""
    name = "rtt_replay_vision_review"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "replay_vision_review.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- the shot list


def resolve_shots(spec: Mapping[str, Any], on: Sequence[str] = ()) -> dict[str, Any]:
    """``spec`` with the flags in ``on`` turned on: the rule of ``shots.mjs`` (``resolveShots``)."""
    flags = dict(spec.get("flags") or {})
    for name in on:
        if name not in flags:
            raise VoiceoverError(f"unknown flag {name}: shots.json lists {', '.join(flags) or 'none'}")
        flags[name] = True
    shots = []
    for s in spec["shots"]:
        s = copy.deepcopy(dict(s))
        fallback = s.pop("fallback", None)
        if s.get("flag"):
            if flags.get(s["flag"]):
                s["variant"] = "flag"
            else:
                s["variant"] = "fallback"
                for k in OVERRIDABLE:
                    if fallback and k in fallback:
                        s[k] = fallback[k]
        shots.append(s)
    return {**spec, "flags": flags, "shots": shots}


@dataclass(frozen=True)
class Line:
    """One voice-over line: its shot, its slot on the cut and the text that is spoken."""

    id: str
    start_s: float
    duration_s: float
    script: str  # as written in shots.json
    spoken: str  # what is sent (pending markers replaced in a draft)
    pending: int  # how many markers the script line holds


def lines_of(spec: Mapping[str, Any], *, draft: bool, pending_say: str) -> list[Line]:
    out = []
    for s in spec["shots"]:
        text = str(s.get("voiceover") or "").strip()
        if not text:
            raise VoiceoverError(f"{s['id']} has no voice-over line")
        n = len(PENDING.findall(text))
        spoken = PENDING.sub(pending_say, text) if draft else text
        spoken = re.sub(r"\s+", " ", spoken).strip()
        out.append(Line(s["id"], float(s["start"]), float(s["duration"]), text, spoken, n))
    return out


# --------------------------------------------------------------------------- audio


def pcm_seconds(pcm: bytes) -> float:
    return len(pcm) / 2 / SAMPLE_RATE


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    os.replace(tmp, path)


def read_wav_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, SAMPLE_RATE):
            raise VoiceoverError(f"{path.name} is not 24 kHz 16-bit mono")
        return w.readframes(w.getnframes())


@dataclass
class Placed:
    line: Line
    at_s: float
    audio_s: float
    slot_end_s: float = field(init=False)
    overrun_s: float = field(init=False)

    def __post_init__(self) -> None:
        self.slot_end_s = self.line.start_s + self.line.duration_s
        self.overrun_s = max(0.0, self.at_s + self.audio_s - self.slot_end_s)


def place(
    lines: Sequence[Line], audio_s: Mapping[str, float], *, lead_s: float, gap_s: float
) -> list[Placed]:
    """Each line at its shot's start plus ``lead_s``, or ``gap_s`` after the line before it ends,
    whichever is later."""
    out: list[Placed] = []
    prev_end = -math.inf
    for ln in lines:
        at = max(ln.start_s + lead_s, prev_end + gap_s)
        p = Placed(ln, round(at, 3), audio_s[ln.id])
        out.append(p)
        prev_end = p.at_s + p.audio_s
    return out


def mix(placed: Sequence[Placed], pcm: Mapping[str, bytes], total_s: float) -> bytes:
    """One mono track as long as the cut (longer only when the last line runs past it)."""
    end = max([total_s] + [p.at_s + p.audio_s for p in placed])
    buf = bytearray(round(end * SAMPLE_RATE) * 2)
    for p in placed:
        i = round(p.at_s * SAMPLE_RATE) * 2
        data = pcm[p.line.id]
        buf[i : i + len(data)] = data
    return bytes(buf)


# --------------------------------------------------------------------------- the API


class SpeechClient(Protocol):
    """What this script needs of a client: the OpenAI SDK's ``client.audio.speech.create``."""

    audio: Any


def openai_client(key: Any) -> SpeechClient:
    from openai import OpenAI

    return OpenAI(api_key=key.reveal(), max_retries=3, timeout=120.0)


def synthesize(
    client: SpeechClient, text: str, *, model: str, voice: str, instructions: str | None, speed: float | None
) -> bytes:
    """Raw 24 kHz 16-bit mono PCM of ``text``."""
    kwargs: dict[str, Any] = {"model": model, "voice": voice, "input": text, "response_format": "pcm"}
    if instructions and not model.startswith("tts-1"):
        kwargs["instructions"] = instructions
    if speed is not None:
        kwargs["speed"] = speed
    resp = client.audio.speech.create(**kwargs)
    data = getattr(resp, "content", None)
    if data is None and hasattr(resp, "read"):
        data = resp.read()
    if not isinstance(data, bytes | bytearray) or not data:
        raise VoiceoverError("the speech API returned no audio")
    if len(data) % 2:
        data = data[:-1]
    return bytes(data)


def price_per_char(model: str, override: float | None = None) -> float:
    if override is not None:
        if not (math.isfinite(override) and override > 0):
            raise VoiceoverError(
                f"a price is a positive number of dollars per million characters, got {override!r}"
            )
        return override / 1e6
    if model not in PRICE_USD_PER_MCHAR:
        raise VoiceoverError(
            f"no price is listed for {model!r}: pass --price-per-mchar or choose one of {sorted(PRICE_USD_PER_MCHAR)}"
        )
    return PRICE_USD_PER_MCHAR[model] / 1e6


def estimate_usd(text: str, instructions: str | None, model: str, per_char: float) -> float:
    chars = len(text) + (len(instructions or "") if not model.startswith("tts-1") else 0)
    return chars * per_char


def cache_key(line: Line, model: str, voice: str, instructions: str | None, speed: float | None) -> str:
    doc = {
        "model": model,
        "voice": voice,
        "instructions": instructions if not model.startswith("tts-1") else None,
        "speed": speed,
        "text": line.spoken,
        "format": "pcm",
        "rate": SAMPLE_RATE,
    }
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode("utf-8")).hexdigest()


def git_common_dir() -> Path | None:
    return _vision_review()._git_common_dir(REPO)


def default_spend_path() -> Path:
    env = os.environ.get("RTT_TTS_SPEND_FILE")
    if env:
        return Path(env)
    common = git_common_dir()
    if common is not None:
        return common / "rtt" / "tts_spend.json"
    return REPO / "runs" / "video" / "tts_spend.json"


def default_keys_path() -> Path:
    env = os.environ.get("RTT_API_KEYS")
    if env:
        return Path(env)
    here = REPO / "api_keys.json"
    if here.exists():
        return here
    common = git_common_dir()
    if common is not None and common.name == ".git":
        return common.parent / "api_keys.json"
    return here


# --------------------------------------------------------------------------- the run


@dataclass
class Result:
    manifest: dict[str, Any]
    problems: list[str]


def run(
    *,
    out: Path,
    shots_file: Path = DEFAULT_SHOTS,
    flags: Sequence[str] = (),
    model: str = DEFAULT_MODEL,
    voice: str = DEFAULT_VOICE,
    instructions: str | None = DEFAULT_INSTRUCTIONS,
    speed: float | None = None,
    draft: bool = False,
    pending_say: str = "Result to come.",
    cap_usd: float = HARD_CAP_USD,
    price_per_mchar: float | None = None,
    lead_s: float = 0.35,
    gap_s: float = 0.25,
    client: SpeechClient | None = None,
    client_factory: Any = None,
    guard: Any = None,
    dry_run: bool = False,
    log: Any = print,
) -> Result:
    """Make the voice-over. ``client`` (or ``client_factory()``, called only when a line must be
    made) speaks; ``guard`` is a ``SpendGuard`` (the shared file when ``None``)."""
    if not (math.isfinite(cap_usd) and 0 < cap_usd <= HARD_CAP_USD):
        raise VoiceoverError(f"--cap-usd must be above 0 and at most {HARD_CAP_USD:.2f}")
    raw = json.loads(Path(shots_file).read_text(encoding="utf-8"))
    spec = resolve_shots(raw, flags)
    lines = lines_of(spec, draft=draft, pending_say=pending_say)
    with_markers = [ln.id for ln in lines if ln.pending]
    if with_markers and not draft:
        raise VoiceoverError(
            f"pending markers in {', '.join(with_markers)}: fill them in shots.json and docs/VIDEO_SCRIPT.md, "
            "or pass --draft to speak each as a stand-in"
        )
    total_s = float(sum(s["duration"] for s in spec["shots"]))
    per_char = price_per_char(model, price_per_mchar)

    out = Path(out)
    cache = out / "cache"
    keys = {ln.id: cache_key(ln, model, voice, instructions, speed) for ln in lines}
    todo = [ln for ln in lines if not (cache / f"{keys[ln.id]}.wav").exists()]
    estimates = {ln.id: estimate_usd(ln.spoken, instructions, model, per_char) for ln in lines}
    need = sum(estimates[ln.id] for ln in todo)
    vr = _vision_review()
    if guard is None:
        guard = vr.SpendGuard(default_spend_path(), caps={SPEND_NAME: cap_usd})
    else:
        guard.caps = {**guard.caps, SPEND_NAME: min(cap_usd, guard.caps.get(SPEND_NAME, cap_usd))}
    spent = guard.spent(SPEND_NAME)
    log(
        f"voiceover: {len(lines)} lines, {sum(len(ln.spoken) for ln in lines)} characters, {len(todo)} to make "
        f"({len(lines) - len(todo)} cached); model {model}, voice {voice}; estimate ${need:.4f} by characters, "
        f"${spent:.4f} spent of the ${guard.caps[SPEND_NAME]:.2f} cap"
    )
    if spent + need > guard.caps[SPEND_NAME]:
        raise vr.SpendCapReached(
            f"the estimate ${need:.4f} with ${spent:.4f} already spent would pass the ${guard.caps[SPEND_NAME]:.2f} cap"
        )
    if dry_run:
        return Result({"dry_run": True, "estimate_usd": round(need, 6), "lines": [ln.id for ln in todo]}, [])

    made_usd = 0.0
    for ln in todo:
        if client is None:
            client = client_factory() if client_factory else None
            if client is None:
                raise VoiceoverError("no speech client")
        est = estimates[ln.id]
        guard.reserve(SPEND_NAME, est)
        try:
            pcm = synthesize(
                client, ln.spoken, model=model, voice=voice, instructions=instructions, speed=speed
            )
        except VoiceoverError:
            guard.release(SPEND_NAME, est)
            raise
        except Exception as exc:  # noqa: BLE001  (any SDK error: reported redacted, never with a key)
            guard.release(SPEND_NAME, est)
            raise VoiceoverError(f"{ln.id}: the speech API failed: {vr._redact(str(exc))[:300]}") from None
        guard.commit(SPEND_NAME, est, est)
        made_usd += est
        write_wav(cache / f"{keys[ln.id]}.wav", pcm)
        log(f"  made {ln.id}: {pcm_seconds(pcm):.2f} s, ${est:.4f}")

    pcm_by_id = {ln.id: read_wav_pcm(cache / f"{keys[ln.id]}.wav") for ln in lines}
    for ln in lines:
        write_wav(out / "lines" / f"{ln.id}.wav", pcm_by_id[ln.id])
    audio_s = {i: pcm_seconds(b) for i, b in pcm_by_id.items()}
    placed = place(lines, audio_s, lead_s=lead_s, gap_s=gap_s)
    write_wav(out / "voiceover.wav", mix(placed, pcm_by_id, total_s))

    problems = []
    for p in placed:
        if p.overrun_s > 0.05:
            problems.append(
                f"{p.line.id} runs {p.overrun_s:.2f} s past its shot ({p.audio_s:.2f} s of speech in {p.line.duration_s:.0f} s)"
            )
    end = max(p.at_s + p.audio_s for p in placed)
    if end > total_s + 0.5:
        problems.append(
            f"the voice-over ends at {end:.2f} s, past the cut's {total_s:.0f} s: assemble.sh will refuse it"
        )
    audio_total = sum(audio_s.values())
    rate = USD_PER_AUDIO_MINUTE.get(model)
    manifest = {
        "format": FORMAT,
        "shots": str(Path(shots_file).name),
        "model": model,
        "voice": voice,
        "instructions": instructions if not model.startswith("tts-1") else None,
        "speed": speed,
        "flags": sorted(k for k, v in spec["flags"].items() if v),
        "draft": bool(draft and with_markers),
        "pending_say": pending_say if draft and with_markers else None,
        "sample_rate": SAMPLE_RATE,
        "total_s": total_s,
        "speech_s": round(audio_total, 3),
        "words_per_minute": round(sum(len(ln.spoken.split()) for ln in lines) / (audio_total / 60), 1)
        if audio_total
        else None,
        "lines": [
            {
                "id": p.line.id,
                "start_s": p.line.start_s,
                "slot_end_s": p.slot_end_s,
                "at_s": p.at_s,
                "audio_s": round(p.audio_s, 3),
                "overrun_s": round(p.overrun_s, 3),
                "chars": len(p.line.spoken),
                "pending": p.line.pending,
                "file": f"lines/{p.line.id}.wav",
                "text": p.line.spoken,
            }
            for p in placed
        ],
        "cost": {
            "guard_estimate_this_run_usd": round(made_usd, 6),
            "guard_estimate_all_lines_usd": round(sum(estimates.values()), 6),
            "likely_usd_all_lines": round(audio_total / 60 * rate, 6)
            if rate
            else round(sum(estimates.values()), 6),
            "basis": "characters x the guard's rate; likely = audio minutes x OpenAI's published per-minute estimate"
            if rate
            else "characters x the published per-character rate",
            "spent_total_usd": round(guard.spent(SPEND_NAME), 6),
            "cap_usd": guard.caps[SPEND_NAME],
        },
        "problems": problems,
    }
    (out / "voiceover.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return Result(manifest, problems)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True, help="output folder, outside the repository")
    ap.add_argument("--shots", default=str(DEFAULT_SHOTS))
    ap.add_argument("--flag", action="append", default=[], help="turn on a flagged shot (repeatable)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument(
        "--instructions", default=DEFAULT_INSTRUCTIONS, help="voice direction (not for tts-1 models)"
    )
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--draft", action="store_true", help="speak pending markers as --pending-say")
    ap.add_argument("--pending-say", default="Result to come.")
    ap.add_argument("--keys", default=None, help="JSON file with an 'openai' entry")
    ap.add_argument("--cap-usd", type=float, default=HARD_CAP_USD)
    ap.add_argument(
        "--price-per-mchar",
        type=float,
        default=None,
        help="dollars per million characters, for an unlisted model",
    )
    ap.add_argument("--spend-file", default=None)
    ap.add_argument("--lead-s", type=float, default=0.35)
    ap.add_argument("--gap-s", type=float, default=0.25)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the estimate; no key, no call")
    args = ap.parse_args(argv)

    out = Path(args.out).resolve()
    if out == REPO or REPO in out.parents:
        print("voiceover: --out must be outside the repository", file=sys.stderr)
        return 2
    vr = _vision_review()
    try:
        guard = vr.SpendGuard(
            Path(args.spend_file) if args.spend_file else default_spend_path(),
            caps={SPEND_NAME: min(args.cap_usd, HARD_CAP_USD)},
        )

        def factory() -> SpeechClient:
            keys = vr.load_keys(args.keys or default_keys_path(), names=("openai",))
            return openai_client(keys["openai"])

        res = run(
            out=out,
            shots_file=Path(args.shots),
            flags=args.flag,
            model=args.model,
            voice=args.voice,
            instructions=args.instructions or None,
            speed=args.speed,
            draft=args.draft,
            pending_say=args.pending_say,
            cap_usd=args.cap_usd,
            price_per_mchar=args.price_per_mchar,
            lead_s=args.lead_s,
            gap_s=args.gap_s,
            client_factory=factory,
            guard=guard,
            dry_run=args.dry_run,
        )
    except (VoiceoverError, vr.ReviewSetupError, vr.SpendCapReached) as exc:
        print(f"voiceover: {vr._redact(str(exc))}", file=sys.stderr)
        return 2
    m = res.manifest
    if m.get("dry_run"):
        print(f"dry run: {len(m['lines'])} lines to make, estimate ${m['estimate_usd']:.4f}; nothing sent")
        return 0
    c = m["cost"]
    print(
        f"wrote {out / 'voiceover.wav'} ({m['total_s']:.0f} s, {m['speech_s']:.1f} s of speech, "
        f"{m['words_per_minute']} words a minute){' DRAFT' if m['draft'] else ''}"
    )
    print(
        f"cost: ${c['guard_estimate_this_run_usd']:.4f} this run by the guard's character estimate; "
        f"likely ${c['likely_usd_all_lines']:.4f} for all lines; ${c['spent_total_usd']:.4f} spent of ${c['cap_usd']:.2f}"
    )
    for p in res.problems:
        print(f"  check: {p}")
    return 1 if any("assemble.sh will refuse" in p for p in res.problems) else 0


if __name__ == "__main__":
    sys.exit(main())
