#!/usr/bin/env python
"""Advisory vision review of Studio replays: two independent vision models look at frames of the drawn
replay around cuts and contact onsets and say whether one solid visibly passes through another or a
tool comes apart.

::

    python scripts/replay_vision_review.py --reports DIR [--heroes h1,h2,h3] [--out FILE]
        [--keys FILE] [--spend-file FILE] [--max-frames-per-lane N]
    python scripts/replay_vision_review.py --smoke [--keys FILE]      # one tiny call to each reviewer

It never blocks an export. The replay gate (:mod:`rtt.studio.replay_qa`) does that, with
measurements; this is a second pair of eyes, and a frame is flagged only when **both** reviewers
flag it (one reviewer cannot flag anything). Flags go to a review list, a JSON file next to the
export (``--out``; ``scripts/export_studio_web.py --vision-review`` writes it beside its QA report),
for a person to look at. Nothing here changes a replay.

Keys are read at run time from a JSON file with one entry for each reviewer (:data:`KEY_ENTRIES`):
``--keys``, or ``RTT_API_KEYS``, or ``api_keys.json`` at the repository root (git-ignored). They are held in
objects that do not print and are never logged, written or sent anywhere but to their own service. The
model of each reviewer is configurable (``--model-a``, ``--model-b``); reviewer B's is chosen from its
service's own model list at run time (the first of :data:`OPENAI_PREFERENCE` it lists), and the run stops
with a clear message when none is there.

Spend is capped for the whole plan: :data:`DEFAULT_CAPS_USD` per reviewer, kept as a running total in a
git-ignored file shared by every checkout and worktree of the repository (``--spend-file``,
``RTT_VISION_SPEND_FILE``, or ``<git common dir>/rtt/vision_spend.json``, seeded from the older
per-checkout ``runs/vision_review/spend.json``). The guard fails closed: a spend file that is missing
counts as nothing spent (a first run), one that cannot be read, parsed or trusted (a non-dictionary, a
negative or non-finite number) stops the review with ``spend file unreadable: nothing is spent`` and is
never reset. Each look reserves its worst-case cost first and is not made when that would pass the cap;
the review then stops and says so. Every reviewer must have a guard. The prices in
:data:`PRICES_USD_PER_MTOK` are what the guard estimates with; a model with no listed price is refused
unless a price is given (``--price-a``, ``--price-b``), and the second reviewer's are ``[assumed]``,
set high on purpose.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

#: The entries of the keys file, reviewer A's and reviewer B's.
KEY_ENTRIES: Final = ("claude", "openai")
#: Spend ceilings (US dollars) for the whole plan, per reviewer (keyed by the keys file's entry names).
DEFAULT_CAPS_USD: Final[Mapping[str, float]] = {"claude": 60.0, "openai": 30.0}
#: A reviewer flags a frame at this severity (0 none, 1 subtle, 2 clearly visible, 3 glaring) or more.
MIN_SEVERITY: Final = 2
DEFAULT_CLAUDE_MODEL: Final = "claude-sonnet-5-5"
#: Tried in this order against the service's model list; the first one listed is used.
OPENAI_PREFERENCE: Final = ("gpt-5.5", "gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4.1", "gpt-4o")
#: Dollars per million (input, output) tokens the guard estimates with; a model that is not listed is
#: refused (:func:`price_of`) unless a price is given. Reviewer A's are the published rates. Reviewer
#: B's are **[assumed]**: set at 10 / 40, above any published rate I could confirm, on purpose, so the
#: guard stops early rather than late.
PRICES_USD_PER_MTOK: Final[Mapping[str, tuple[float, float]]] = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-haiku-5-5": (0.1, 0.5),
    "gpt-5.5": (10.0, 40.0),  # [assumed]
    "gpt-5.4": (10.0, 40.0),  # [assumed]
    "gpt-5.2": (10.0, 40.0),  # [assumed]
    "gpt-5.1": (10.0, 40.0),  # [assumed]
    "gpt-5": (10.0, 40.0),  # [assumed]
    "gpt-4.1": (10.0, 40.0),  # [assumed]
    "gpt-4o": (10.0, 40.0),  # [assumed]
}
#: Worst case a single look is reserved at: tokens of the image and prompt in, the output cap out.
EST_INPUT_TOKENS: Final = 1500
MAX_OUTPUT_TOKENS: Final = 2000

PROMPT: Final = """This is one frame of a 3D replay of a physics simulation: a robot end effector (orange) working on objects (blue and white) on a table (brown); dark parts are the wrist and its bracket, and a dark frame at the back may be a display-only tool stand. The frame is drawn from simple solid shapes.

Judge ONLY geometric plausibility as a viewer would see it:
1. interpenetration: does any solid visibly pass through, sink into, or overlap another solid (a tool part inside another tool part or inside an object, an object inside a wall, a part through the table)?
2. detached: are parts of one tool visibly separated or floating apart from where they should attach?
Ignore lighting, labels, framing and missing textures. Be strict about what is actually visible; if it is ambiguous because of perspective, say so and keep the severity low.

Reply with JSON only, no prose: {"interpenetration": true|false, "detached": true|false, "severity": 0-3 (0 none, 1 subtle, 2 clearly visible, 3 glaring), "where": "<short description or empty>", "confidence": 0.0-1.0}"""


class ReviewSetupError(RuntimeError):
    """The review cannot start (no keys, no model); the message says what to fix."""


class SpendCapReached(RuntimeError):
    """A look was not made because it could pass a reviewer's spend cap."""


class SpendFileError(SpendCapReached):
    """The spend file cannot be trusted (unreadable, unparsable, or not a record of non-negative spends):
    nothing is spent, and the file is left as it is."""


# --------------------------------------------------------------------------- keys


class _Secret:
    """A credential that does not print: ``repr`` and ``str`` hide it; only :meth:`reveal` shows it."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "<hidden>"

    __str__ = __repr__


def default_keys_path() -> Path:
    return Path(os.environ.get("RTT_API_KEYS") or REPO / "api_keys.json")


def load_keys(
    path: str | Path | None = None, names: Sequence[str] = KEY_ENTRIES, *, flag: str = "--keys"
) -> dict[str, _Secret]:
    """The entries ``names`` of the keys file, read now, as :class:`_Secret` objects. ``flag`` is the
    command-line option that names the file, for the message when it cannot be read."""
    where = Path(path) if path else default_keys_path()
    try:
        doc = json.loads(where.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReviewSetupError(
            f"cannot read the keys file {where.name} ({type(exc).__name__}): pass {flag} FILE or set "
            f"RTT_API_KEYS to a JSON file with the entries {list(names)}"
        ) from None
    missing = [n for n in names if not isinstance(doc, dict) or not str(doc.get(n) or "").strip()]
    if missing:
        raise ReviewSetupError(f"the keys file has no entry for {missing}: it needs {list(names)}")
    return {n: _Secret(str(doc[n]).strip()) for n in names}


# --------------------------------------------------------------------------- spend guard


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a number a spend file may hold")


class SpendGuard:
    """A running total of what each reviewer has cost, kept in a JSON file that survives runs, and a
    cap per reviewer. :meth:`reserve` refuses (raises :class:`SpendCapReached`) a look whose worst
    case would pass the cap; :meth:`commit` records what it really cost; reservations are not
    written down (a crashed run costs at most what it spent).

    The file is the total for the whole plan, shared by every process: it is read and rewritten under
    an exclusive ``flock`` (``<file>.lock``) and replaced atomically. It fails closed: no file is zero
    spent, but a file that cannot be read, parsed or trusted raises :class:`SpendFileError` and is
    never overwritten. ``seed_from``: older spend files whose totals are added when this file does
    not exist yet (the first use of a new location)."""

    def __init__(
        self,
        path: str | Path,
        caps: Mapping[str, float] | None = None,
        *,
        seed_from: Sequence[str | Path] = (),
    ) -> None:
        self.path = Path(path)
        self.caps = dict(DEFAULT_CAPS_USD if caps is None else caps)
        self._reserved: dict[str, float] = {}
        self._unrecorded: dict[str, float] = {}  # paid for, but the file could not be written
        self._seed_from = [Path(p) for p in seed_from]

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_name(self.path.name + ".lock"), "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _unreadable(self, why: str) -> SpendFileError:
        return SpendFileError(f"spend file unreadable: nothing is spent ({self.path.name}: {why})")

    def _load(self, path: Path) -> dict[str, dict[str, Any]] | None:
        """The record in ``path`` (``None`` when there is no such file); :class:`SpendFileError` for one
        that is not a dictionary of ``{"spent_usd": finite number >= 0, "calls": integer >= 0}``."""
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise self._unreadable(type(exc).__name__) from None
        try:
            doc = json.loads(text, parse_constant=_no_constant)
        except ValueError:
            raise self._unreadable("not valid JSON") from None
        if not isinstance(doc, dict):
            raise self._unreadable("not a dictionary")
        for name, row in doc.items():
            spent = row.get("spent_usd") if isinstance(row, dict) else None
            calls = row.get("calls", 0) if isinstance(row, dict) else None
            if (
                isinstance(spent, bool)
                or not isinstance(spent, int | float)
                or not math.isfinite(spent)
                or spent < 0
                or isinstance(calls, bool)
                or not isinstance(calls, int)
                or calls < 0
            ):
                raise self._unreadable(f"the entry for {name!r} is not a non-negative spend")
        return doc

    def _read(self) -> dict[str, dict[str, Any]]:
        doc = self._load(self.path)
        if doc is not None:
            return doc
        if self._seed_from:
            return self._seed()
        return {}

    def _seed(self) -> dict[str, dict[str, Any]]:
        """The sum of the older files (the first use of this location), written down."""
        total: dict[str, dict[str, Any]] = {}
        for old in self._seed_from:
            doc = self._load(old)
            for name, row in (doc or {}).items():
                mine = total.setdefault(name, {"spent_usd": 0.0, "calls": 0})
                mine["spent_usd"] = round(mine["spent_usd"] + float(row["spent_usd"]), 6)
                mine["calls"] += int(row.get("calls", 0))
        if total:
            self._write(total)
        return total

    def _write(self, doc: Mapping[str, Any]) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def spent(self, name: str) -> float:
        with self._locked():
            row = self._read().get(name, {})
        return float(row.get("spent_usd", 0.0)) + self._unrecorded.get(name, 0.0)

    def reserve(self, name: str, usd: float) -> None:
        cap = self.caps.get(name)
        if cap is None:
            raise SpendCapReached(f"no spend cap is set for reviewer {name!r}: nothing is spent without one")
        if not math.isfinite(usd) or usd < 0:
            raise SpendCapReached(
                f"reviewer {name!r}: a look cannot be estimated at {usd!r}: nothing is spent"
            )
        held, spent = self._reserved.get(name, 0.0), self.spent(name)
        if spent + held + usd > cap:
            raise SpendCapReached(
                f"reviewer {name!r}: {spent:.4f} spent + {held + usd:.4f} reserved would pass the {cap:.2f} cap"
            )
        self._reserved[name] = held + usd

    def release(self, name: str, usd: float) -> None:
        self._reserved[name] = max(0.0, self._reserved.get(name, 0.0) - usd)

    def commit(self, name: str, reserved: float, actual: float) -> None:
        """Record what a look cost. When the file cannot be read it is left alone and the cost is kept
        in memory (counted by :meth:`spent`), and :class:`SpendFileError` stops the caller."""
        self.release(name, reserved)
        try:
            with self._locked():
                doc = self._read()
                row = dict(doc.get(name, {}))
                row["spent_usd"] = round(float(row.get("spent_usd", 0.0)) + float(actual), 6)
                row["calls"] = int(row.get("calls", 0)) + 1
                doc[name] = row
                self._write(doc)
        except SpendFileError:
            self._unrecorded[name] = self._unrecorded.get(name, 0.0) + float(actual)
            raise
        self._unrecorded.pop(name, None)


def _git_common_dir(repo: Path) -> Path | None:
    """The git common directory of ``repo``: the main checkout's ``.git`` for a worktree too."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    path = Path(out.stdout.strip())
    return path if path.is_absolute() else (repo / path).resolve()


def default_spend_path(repo: Path = REPO) -> Path:
    """The one spend file every checkout and worktree of the repository shares: ``RTT_VISION_SPEND_FILE``,
    else ``<git common dir>/rtt/vision_spend.json`` (never part of a commit), else a file next to the keys
    file (``*api_keys*.json`` is git-ignored)."""
    env = os.environ.get("RTT_VISION_SPEND_FILE")
    if env:
        return Path(env)
    common = _git_common_dir(repo)
    if common is not None:
        return common / "rtt" / "vision_spend.json"
    return default_keys_path().with_name("api_keys_vision_spend.json")


def legacy_spend_files(repo: Path = REPO) -> list[Path]:
    """The older per-checkout spend files (``runs/vision_review/spend.json`` of this checkout and of the
    main one) that exist: what :func:`default_spend_guard` seeds a new shared file from."""
    roots = [repo]
    common = _git_common_dir(repo)
    if common is not None and common.name == ".git":
        roots.append(common.parent)
    seen: dict[Path, None] = {}
    for root in roots:
        f = (root / "runs" / "vision_review" / "spend.json").resolve()
        if f.is_file():
            seen[f] = None
    return list(seen)


def default_spend_guard(repo: Path = REPO) -> SpendGuard:
    """The shared guard at :func:`default_spend_path`, seeded from the older per-checkout files the
    first time (and only when no path was named by ``RTT_VISION_SPEND_FILE``)."""
    seed = [] if os.environ.get("RTT_VISION_SPEND_FILE") else legacy_spend_files(repo)
    return SpendGuard(default_spend_path(repo), seed_from=seed)


# --------------------------------------------------------------------------- reviewers


@dataclass(frozen=True)
class FrameRef:
    """One frame to look at: where it is (lane and time), why it was chosen, and its PNG."""

    frame_id: str
    label: str
    lane: str
    time_s: float
    png: bytes
    reason: str = ""


class Reviewer(Protocol):
    """What :func:`review_frames` needs of a reviewer: a ``name`` (the spend cap's key) and a
    ``review(frame, prompt)`` that returns the verdict as a dict (``interpenetration``, ``detached``,
    ``severity``, ``where``, ``confidence``, ``cost_usd``) or raises."""

    name: str

    def review(self, frame: FrameRef, prompt: str) -> dict[str, Any]: ...


def _as_bool(value: Any) -> bool:
    """A reply's yes or no, strictly: ``false``, ``"false"``, ``"no"``, ``0`` and ``""`` are no (``bool``
    of the string ``"false"`` would be yes). Raises ``ValueError`` for anything it cannot read."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        word = value.strip().lower()
        if word in ("true", "yes", "1"):
            return True
        if word in ("false", "no", "0", ""):
            return False
    if isinstance(value, int | float) and value in (0, 1):
        return bool(value)
    raise ValueError(f"not a yes or no: {value!r}")


def parse_verdict(text: str) -> dict[str, Any]:
    """The verdict of a reply: the JSON object in it, with each field coerced. Raises ``ValueError``."""
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e < s:
        raise ValueError("the reply holds no JSON object")
    raw = json.loads(text[s : e + 1])
    return {
        "interpenetration": _as_bool(raw.get("interpenetration", False)),
        "detached": _as_bool(raw.get("detached", False)),
        "severity": max(0, min(3, int(raw.get("severity", 0)))),
        "where": str(raw.get("where", ""))[:200],
        "confidence": max(0.0, min(1.0, float(raw.get("confidence", 0.0)))),
    }


_TOKEN = re.compile(r"(sk-|key-|Bearer\s+)[A-Za-z0-9_\-*.]{4,}")


def _redact(text: str) -> str:
    """``text`` without anything shaped like a credential (a service may echo part of a key)."""
    return _TOKEN.sub("<hidden>", text)


def _post_json(url: str, headers: Mapping[str, str], body: Mapping[str, Any], timeout: float = 120.0) -> Any:
    """POST JSON, retrying a rate limit or a server error a few times (HTTPS, standard library only)."""
    data = json.dumps(body).encode("utf-8")
    last: Exception | None = None
    for attempt in range(4):
        req = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = _redact(exc.read().decode("utf-8", "replace")[:300])
            last = RuntimeError(f"HTTP {exc.code}: {detail}")
            if exc.code not in (408, 409, 429) and exc.code < 500:
                raise last from None
        except (urllib.error.URLError, TimeoutError) as exc:
            last = RuntimeError(f"network error: {type(exc).__name__}")
        time.sleep(1.5 * (attempt + 1))
    assert last is not None
    raise last


def _get_json(url: str, headers: Mapping[str, str], timeout: float = 60.0) -> Any:
    req = urllib.request.Request(url, headers=dict(headers))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        raise ReviewSetupError(f"the model list could not be read (HTTP {exc.code})") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ReviewSetupError(f"the model list could not be read ({type(exc).__name__})") from None


def price_of(model: str, override: tuple[float, float] | None = None) -> tuple[float, float]:
    """Dollars per million (input, output) tokens the guard estimates ``model`` at: ``override`` when
    given, else the model's entry in :data:`PRICES_USD_PER_MTOK`. A model with neither is refused:
    nothing is spent at a price nobody wrote down."""
    if override is not None:
        if len(override) != 2 or not all(math.isfinite(v) and v > 0 for v in override):
            raise ReviewSetupError(f"a price is two positive numbers (input, output), got {override!r}")
        return (float(override[0]), float(override[1]))
    if model not in PRICES_USD_PER_MTOK:
        raise ReviewSetupError(
            f"no price is listed for the model {model!r}: pass its dollars per million tokens "
            f"(--price-a / --price-b IN,OUT) or choose a listed model"
        )
    return PRICES_USD_PER_MTOK[model]


class _HttpsReviewer:
    name = ""
    model = ""

    def __init__(self, key: _Secret, model: str, price: tuple[float, float]) -> None:
        self._key, self.model, self.price = key, model, price

    def estimate_usd(self, frame: FrameRef | None = None) -> float:
        """The worst a look can cost: the image and prompt in, the output cap out."""
        return (EST_INPUT_TOKENS * self.price[0] + MAX_OUTPUT_TOKENS * self.price[1]) / 1e6

    def _cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.price[0] + tokens_out * self.price[1]) / 1e6


class ClaudeReviewer(_HttpsReviewer):
    """Reviewer A: the Messages API over HTTPS, an image block and the prompt."""

    name = "claude"
    URL = "https://api.anthropic.com/v1/messages"

    def review(self, frame: FrameRef, prompt: str) -> dict[str, Any]:
        body = {
            "model": self.model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "output_config": {"effort": "low"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.standard_b64encode(frame.png).decode("ascii"),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        out = _post_json(self.URL, {"x-api-key": self._key.reveal(), "anthropic-version": "2023-06-01"}, body)
        usage = out.get("usage") or {}
        cost = self._cost(
            int(usage.get("input_tokens", EST_INPUT_TOKENS)),
            int(usage.get("output_tokens", MAX_OUTPUT_TOKENS)),
        )
        if out.get("stop_reason") == "refusal":
            raise _Costed("the reviewer declined to look at the frame", cost)
        text = "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")
        try:
            verdict = parse_verdict(text)
        except ValueError as exc:
            raise _Costed(f"unreadable reply ({exc})", cost) from None
        return {**verdict, "cost_usd": cost}


class OpenAIReviewer(_HttpsReviewer):
    """Reviewer B: the Chat Completions API over HTTPS, a text part and an image part."""

    name = "openai"
    URL = "https://api.openai.com/v1/chat/completions"

    def review(self, frame: FrameRef, prompt: str) -> dict[str, Any]:
        data = base64.standard_b64encode(frame.png).decode("ascii")
        body: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": MAX_OUTPUT_TOKENS,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
                    ],
                }
            ],
        }
        if self.model.startswith(("gpt-5", "o")):
            body["reasoning_effort"] = "low"
        out = _post_json(self.URL, {"authorization": f"Bearer {self._key.reveal()}"}, body)
        usage = out.get("usage") or {}
        cost = self._cost(
            int(usage.get("prompt_tokens", EST_INPUT_TOKENS)),
            int(usage.get("completion_tokens", MAX_OUTPUT_TOKENS)),
        )
        text = ((out.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        try:
            verdict = parse_verdict(text)
        except ValueError as exc:
            raise _Costed(f"unreadable reply ({exc})", cost) from None
        return {**verdict, "cost_usd": cost}


class _Costed(RuntimeError):
    """A look that failed after it was paid for: the message and what it cost."""

    def __init__(self, message: str, cost_usd: float) -> None:
        super().__init__(message)
        self.cost_usd = cost_usd


def choose_openai_model(listing: Sequence[str], prefer: str | None = None) -> str:
    """The second reviewer's model, from the service's own list: ``prefer`` when given (it must be
    listed), else the first of :data:`OPENAI_PREFERENCE` that is."""
    have = set(listing)
    if prefer is not None:
        if prefer not in have:
            raise ReviewSetupError(f"the model {prefer!r} is not in the service's model list")
        return prefer
    for model in OPENAI_PREFERENCE:
        if model in have:
            return model
    raise ReviewSetupError(
        f"no vision-capable model of {list(OPENAI_PREFERENCE)} is in the service's model list "
        f"({len(have)} models listed): pass --model-b"
    )


def list_openai_models(key: _Secret) -> list[str]:
    out = _get_json("https://api.openai.com/v1/models", {"authorization": f"Bearer {key.reveal()}"})
    return sorted(str(m.get("id")) for m in out.get("data", []) if m.get("id"))


def make_reviewers(
    keys: Mapping[str, _Secret],
    *,
    claude_model: str = DEFAULT_CLAUDE_MODEL,
    openai_model: str | None = None,
    claude_price: tuple[float, float] | None = None,
    openai_price: tuple[float, float] | None = None,
) -> list[Reviewer]:
    """The two reviewers: A (``claude``) and B (``openai``, its model chosen from the live model list).
    A model whose price is not listed needs its price given (:func:`price_of`)."""
    price_a = price_of(claude_model, claude_price)
    model_b = choose_openai_model(list_openai_models(keys["openai"]), openai_model)
    return [
        ClaudeReviewer(keys["claude"], claude_model, price_a),
        OpenAIReviewer(keys["openai"], model_b, price_of(model_b, openai_price)),
    ]


# --------------------------------------------------------------------------- the review


def _flags(verdict: Mapping[str, Any]) -> bool:
    return (
        bool(verdict.get("interpenetration") or verdict.get("detached"))
        and int(verdict.get("severity", 0)) >= MIN_SEVERITY
    )


NOTE: Final = (
    "Two independent vision models looked at frames of the drawn replay around cuts and contact "
    "onsets. A frame is listed only when both flagged a visible overlap or a detached part. This list is "
    "advice for a person to look at: it never blocks an export."
)


def review_frames(
    frames: Sequence[FrameRef],
    reviewers: Sequence[Reviewer],
    *,
    guards: Mapping[str, SpendGuard] | None = None,
    estimate_usd: Mapping[str, float] | None = None,
    log: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Every frame to every reviewer; the review list (advisory: ``blocks_export`` is false).

    A frame is flagged only when every reviewer flagged it and there are at least two (one reviewer
    cannot flag). A reviewer that fails on a frame (a service error, an unreadable reply) gives no
    opinion on it, so the frame is not flagged. ``guards`` (a :class:`SpendGuard` per reviewer name) is
    required for every reviewer: each look is reserved first and the review stops, saying so, before a
    cap would be passed or when the spend file cannot be trusted.

    Raises:
        ReviewSetupError: a reviewer has no guard (nothing is spent without one).
    """
    unguarded = [r.name for r in reviewers if not guards or r.name not in guards]
    if unguarded:
        raise ReviewSetupError(f"no spend guard for reviewer(s) {unguarded}: nothing is spent without one")
    labels = [chr(ord("A") + i) for i in range(len(reviewers))]
    say = log or (lambda _m: None)
    flagged: list[dict[str, Any]] = []
    disagreed = errors = 0
    cost = dict.fromkeys(labels, 0.0)
    stopped = False
    stop_reason = ""
    looked = 0

    def estimate(r: Reviewer, frame: FrameRef) -> float:
        if estimate_usd is not None and r.name in estimate_usd:
            return float(estimate_usd[r.name])
        fn = getattr(r, "estimate_usd", None)
        return float(fn(frame)) if callable(fn) else 0.0

    for frame in frames:
        held: dict[str, float] = {}
        try:
            for r in reviewers:
                est = estimate(r, frame)
                guards[r.name].reserve(r.name, est)  # type: ignore[index]
                held[r.name] = est
        except SpendCapReached as exc:
            for name, est in held.items():
                guards[name].release(name, est)  # type: ignore[index]
            say(f"  the review stops here: {exc}")
            stopped, stop_reason = True, str(exc)
            break
        verdicts: dict[str, dict[str, Any]] = {}
        for label, r in zip(labels, reviewers, strict=True):
            try:
                v = r.review(frame, PROMPT)
                actual = float(v.get("cost_usd", 0.0))
                verdicts[label] = v
            except _Costed as exc:
                actual = exc.cost_usd
                errors += 1
                say(f"  reviewer {label}: no opinion on {frame.frame_id} ({exc})")
            except Exception as exc:  # noqa: BLE001 - advisory: a failing reviewer gives no opinion
                actual = 0.0
                errors += 1
                say(f"  reviewer {label}: no opinion on {frame.frame_id} ({type(exc).__name__})")
            cost[label] += actual
            try:
                guards[r.name].commit(r.name, held.get(r.name, 0.0), actual)  # type: ignore[index]
            except SpendFileError as exc:
                say(f"  the review stops here: {exc}")
                stopped, stop_reason = True, str(exc)
        looked += 1
        flags = {label: _flags(v) for label, v in verdicts.items()}
        if len(verdicts) == len(reviewers) and len(reviewers) >= 2:
            if all(flags.values()):
                flagged.append(
                    {
                        "frame_id": frame.frame_id,
                        "label": frame.label,
                        "lane": frame.lane,
                        "time_s": round(frame.time_s, 3),
                        "reason": frame.reason,
                        "reviewers": sorted(verdicts),
                        "findings": {
                            label: {
                                "severity": v["severity"],
                                "where": v["where"],
                                "confidence": v["confidence"],
                            }
                            for label, v in verdicts.items()
                        },
                    }
                )
            elif any(flags.values()):
                disagreed += 1
        if stopped:  # the spend file cannot be trusted any more: no more looks
            break
    return {
        "format": "rtt-replay-review",
        "version": 1,
        "advisory": True,
        "blocks_export": False,
        "note": NOTE,
        "summary": {
            "frames": len(frames),
            "frames_looked_at": looked,
            "reviewers": len(reviewers),
            "flagged": len(flagged),
            "reviewers_disagreed": disagreed,
            "errors": errors,
            "stopped_by_spend_cap": stopped,
            "stop_reason": stop_reason,
            "cost_usd": {k: round(v, 6) for k, v in cost.items()},
        },
        "flagged": flagged,
    }


# --------------------------------------------------------------------------- choosing frames


def frame_times(
    cuts_s: Sequence[tuple[float, float]],
    onsets_s: Sequence[float],
    duration_s: float,
    max_frames: int = 24,
) -> list[float]:
    """Times (sorted, distinct) to look at: the last frame before each cut and a moment after the first
    frame of the next segment (``cuts_s`` are ``(before, after)`` pairs), and just after each contact
    onset or release in ``onsets_s`` (events closer than a quarter second count once). At most
    ``max_frames``: cuts first, the contacts spread evenly over the rest. No events, no frames."""
    out: list[float] = []
    for before, after in cuts_s:
        out += [float(before), min(float(after) + 0.05, duration_s)]
    contacts: list[float] = []
    for t in sorted(float(x) for x in onsets_s):
        if not contacts or t - contacts[-1] >= 0.25:
            contacts.append(t)
    room = max(0, max_frames - len(out))
    if len(contacts) > room:
        pick = (
            [contacts[round(i * (len(contacts) - 1) / max(room - 1, 1))] for i in range(room)] if room else []
        )
        contacts = sorted(set(pick))
    out += [min(t + 0.02, duration_s) for t in contacts]
    out = sorted({round(t, 3) for t in out if 0.0 <= t <= duration_s})
    return out[:max_frames] if len(out) > max_frames else out


def review_lanes(
    lanes: Sequence[tuple[str, Any]],
    reviewers: Sequence[Reviewer],
    *,
    guards: Mapping[str, SpendGuard] | None = None,
    max_frames_per_lane: int = 24,
    log: Callable[[str], Any] = print,
) -> dict[str, Any]:
    """The review list of lanes ``(hero, LaneBuild)``: frames are drawn around each lane's cuts and
    settled contact events (:func:`rtt.studio.replay.contact_times`) and reviewed together."""
    from rtt.studio import replay as sr
    from rtt.studio.replay_qa import render_frames

    frames: list[FrameRef] = []
    for hero, lane in lanes:
        header, rec = lane.replay.header, lane.recording
        t = header["t"]
        cuts = [(float(t[c - 1]), float(t[c])) for c in header.get("cuts", [])]
        onsets = sr.contact_times(rec.contact_log)
        times = frame_times(cuts, onsets, float(t[-1]), max_frames_per_lane)
        if not times:
            continue
        pngs = render_frames(lane, times)
        for when, png in zip(times, pngs, strict=True):
            reason = "cut" if any(b - 0.01 <= when <= a + 0.1 for b, a in cuts) else "contact"
            frames.append(
                FrameRef(
                    f"{hero}/{lane.stem}@{when:.3f}",
                    f"{hero} {lane.stem} at {when:.2f} s ({reason})",
                    f"{hero}/{lane.stem}",
                    float(when),
                    png,
                    reason,
                )
            )
        log(f"  {hero}/{lane.stem}: {len(times)} frames drawn")
    return review_frames(frames, reviewers, guards=guards, log=log)


# --------------------------------------------------------------------------- command line


def _smoke_png() -> bytes:
    """A small synthetic frame: an orange block sunk into a brown table (a blue box)."""
    import io

    from PIL import Image, ImageDraw

    im = Image.new("RGB", (320, 240), (200, 200, 205))
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 160, 320, 240], fill=(120, 100, 80))
    dr.rectangle([100, 120, 220, 190], fill=(60, 120, 230))
    dr.rectangle([150, 60, 190, 170], fill=(250, 140, 30))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _parse_price(text: str | None) -> tuple[float, float] | None:
    """``"IN,OUT"`` as two numbers (dollars per million tokens), or ``None``."""
    if not text:
        return None
    try:
        a, b = (float(x) for x in text.split(","))
    except ValueError:
        raise ReviewSetupError(f"a price is IN,OUT (two numbers), got {text!r}") from None
    return (a, b)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--reports", type=Path, help="folder with <case_id>/report.json (as export_studio_web.py)"
    )
    ap.add_argument("--heroes", default="h1,h2,h3")
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "studio_export" / "replay_review.json")
    ap.add_argument("--keys", type=Path, default=None)
    ap.add_argument("--spend-file", type=Path, default=None)
    ap.add_argument("--model-a", default=DEFAULT_CLAUDE_MODEL, help="reviewer A's model")
    ap.add_argument(
        "--model-b",
        default=None,
        help="reviewer B's model (default: the first preferred one its service lists)",
    )
    ap.add_argument(
        "--price-a", default=None, help="IN,OUT dollars per million tokens, for a model not listed"
    )
    ap.add_argument(
        "--price-b", default=None, help="IN,OUT dollars per million tokens, for a model not listed"
    )
    ap.add_argument("--max-frames-per-lane", type=int, default=24)
    ap.add_argument(
        "--smoke", action="store_true", help="one tiny look by each reviewer, to prove the wiring"
    )
    args = ap.parse_args(argv)
    try:
        keys = load_keys(args.keys)
        reviewers = make_reviewers(
            keys,
            claude_model=args.model_a,
            openai_model=args.model_b,
            claude_price=_parse_price(args.price_a),
            openai_price=_parse_price(args.price_b),
        )
    except ReviewSetupError as exc:
        print(f"replay_vision_review: {exc}", file=sys.stderr)
        return 2
    guard = SpendGuard(args.spend_file) if args.spend_file else default_spend_guard()
    guards = {r.name: guard for r in reviewers}
    if args.smoke:
        frame = FrameRef("smoke", "synthetic frame", "smoke", 0.0, _smoke_png(), "smoke")
        review = review_frames([frame], reviewers, guards=guards, log=print)
        print(json.dumps(review["summary"], indent=1))
        try:
            print("spent so far:", {r.name: round(guard.spent(r.name), 6) for r in reviewers})
        except SpendFileError as exc:
            print(f"replay_vision_review: {exc}", file=sys.stderr)
            return 1
        return 0 if review["summary"]["errors"] == 0 and not review["summary"]["stopped_by_spend_cap"] else 1
    if args.reports is None:
        print("replay_vision_review: pass --reports DIR (or --smoke)", file=sys.stderr)
        return 2
    import export_studio_web as ex

    from rtt.studio import replay as sr
    from rtt.studio.replay import ReplayMismatchError, staging_store

    lanes: list[tuple[str, Any]] = []
    import tempfile

    try:
        with tempfile.TemporaryDirectory(prefix="rtt-vision-") as tmp:
            store = staging_store(Path(tmp) / "raw")
            for hero in [h.strip() for h in args.heroes.split(",") if h.strip()]:
                case_id = dict(ex.HEROES)[hero]
                report, _ = ex.load_report(args.reports, case_id)
                failures: list[str] = []
                built = sr.export_lanes(
                    hero, report, store, code_version=ex.code_version(), failures=failures
                )
                if failures:
                    raise ReplayMismatchError("; ".join(failures))
                lanes += [(hero, lane) for lane in built]
            review = review_lanes(
                lanes, reviewers, guards=guards, max_frames_per_lane=args.max_frames_per_lane
            )
    except (ReplayMismatchError, ex.ExportError, RuntimeError) as exc:
        print(f"replay_vision_review: {exc}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(review, indent=1) + "\n", encoding="utf-8")
    s = review["summary"]
    print(f"{s['flagged']} of {s['frames']} frames flagged by both reviewers; review list: {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
