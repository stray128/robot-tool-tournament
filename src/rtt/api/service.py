"""The service behind the HTTP API: parts, agent sessions, runs and print packs.

:class:`AgentService` owns the per-process state: the part backend, the shared local job
runner, the daily spend book, the passcode gate, and the sessions in flight (each runs
:class:`rtt.agent.engineer.GraspEngineer` in its own thread). :mod:`rtt.api.app` is a
thin HTTP layer over it, and :mod:`rtt.agent` (``python -m rtt.agent``) drives it from
the terminal.

Two modes:

* **live** (default): Nemotron on Token Factory. Starting a session needs the passcode,
  because every agent turn costs money; the session and day caps apply.
* **offline** (``ApiSettings.offline``): the scripted transport of
  :mod:`rtt.agent.fakes` plays the agent, so no model is called and nothing is billed
  (its token counts go to an in-memory day book). Physics stays real unless
  ``fake_physics`` is set. Offline sessions are recorded as ``scripted-offline``.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from rtt.agent.backends import PartBackend, PartInfo, PartNotFoundError, default_part_backend
from rtt.agent.caps import (
    AgentCaps,
    CapError,
    DailySpendBook,
    GuardedLedger,
    PasscodeGate,
    PasscodeRequiredError,
    SpendGuard,
)
from rtt.agent.effector_tool import EffectorJobRunner, designer_factory, tournament_job
from rtt.agent.engineer import DEFAULT_GOAL, AgentLLM, GraspEngineer, SessionResult
from rtt.agent.jobs import JobRunner, LocalJobRunner
from rtt.agent.proposer import NemotronDesignProposer
from rtt.agent.recorder import SessionRecorder, load_session, redact_paths
from rtt.agent.store import AgentStore, check_id, new_id
from rtt.agent.summarizer import LightningSummarizer
from rtt.agent.tools import Toolbox
from rtt.design.schema import FingerDesign, load_entrants

__all__ = [
    "DEFAULT_ALLOWED_HOSTS",
    "DEFAULT_CORS_ORIGINS",
    "AgentService",
    "ApiSettings",
    "BusyError",
    "PartNotFoundError",
    "SessionView",
    "TargetUnavailableError",
    "UploadError",
]

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_ROOT: Final = REPO_ROOT / "runs" / "agent"
DEFAULT_ENTRANTS: Final = REPO_ROOT / "configs" / "entrants.yaml"
DEFAULT_CORS_ORIGINS: Final = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)
#: Host headers the API answers by default (loopback; ``testserver`` is the test client's).
DEFAULT_ALLOWED_HOSTS: Final = ("127.0.0.1", "localhost", "::1", "testserver")
_NAME_RE: Final = re.compile(r"[^A-Za-z0-9 _.-]+")
#: Thinnest upload accepted, as the convex hull's ``2 x volume / area`` in mm (see :func:`check_solid`).
MIN_THICKNESS_MM: Final = 0.1


class BusyError(CapError):
    """The server already runs its maximum number of sessions."""

    code = "busy"


class UploadError(ValueError):
    """The upload is not an acceptable STL."""


class TargetUnavailableError(CapError):
    """A configured job target (Nebius) could not be set up, so no session was started."""

    code = "target_unavailable"


def _flag(env: Mapping[str, str], name: str) -> bool:
    return env.get(name, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ApiSettings:
    """Server settings (``from_env`` reads ``RTT_AGENT_*`` variables).

    Attributes:
        root: Where parts, sessions, results and print packs are written.
        day_book: The daily spend file (``None``: ``<root>/daily_spend.json``).
            :meth:`from_env` pins it to ``runs/agent/daily_spend.json`` (or
            ``RTT_AGENT_DAY_BOOK``) whatever the root, so another ``--root`` does not
            start a fresh day.
        max_upload_bytes: Largest accepted STL.
        caps: Per-session limits (and the day cap).
        cors_origins: Browser origins allowed to call the API (local web dev).
        allowed_hosts: ``Host`` header values the API answers (``"*"``: any).
        max_active_sessions: Sessions that may run at once on this server.
        enable_nebius: Offer ``target='nebius'`` to the agent.
        offline: Scripted agent instead of Nemotron (no model calls, no spend).
        fake_physics: Synthetic physics instead of MuJoCo (tests and UI work only).
        workers: CPU worker processes per simulation (``None``: all CPUs).
        require_passcode: Uploads and offline sessions need the passcode too, not only
            live sessions. ``python -m rtt.api`` sets it when it listens beyond loopback
            (Addendum A: part intake is "local or passcode only");
            ``RTT_AGENT_REQUIRE_PASSCODE=1`` forces it.
    """

    root: Path = DEFAULT_ROOT
    day_book: Path | None = None
    max_upload_bytes: int = 20 * 1024 * 1024
    caps: AgentCaps = field(default_factory=AgentCaps)
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS
    max_active_sessions: int = 1
    enable_nebius: bool = False
    offline: bool = False
    fake_physics: bool = False
    workers: int | None = None
    require_passcode: bool = False

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ApiSettings:
        env = os.environ if environ is None else environ
        caps = AgentCaps(
            session_usd=float(env.get("RTT_AGENT_SESSION_USD", "0.50")),
            day_usd=float(env.get("RTT_AGENT_DAY_USD", "2.00")),
        )
        origins = env.get("RTT_AGENT_CORS", "")
        hosts = env.get("RTT_AGENT_ALLOWED_HOSTS", "")
        workers = env.get("RTT_AGENT_WORKERS", "")
        return cls(
            root=Path(env.get("RTT_AGENT_ROOT") or DEFAULT_ROOT),
            day_book=Path(env.get("RTT_AGENT_DAY_BOOK") or DEFAULT_ROOT / "daily_spend.json"),
            max_upload_bytes=int(float(env.get("RTT_AGENT_MAX_UPLOAD_MB", "20")) * 1024 * 1024),
            caps=caps,
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()) or DEFAULT_CORS_ORIGINS,
            allowed_hosts=tuple(h.strip() for h in hosts.split(",") if h.strip()) or DEFAULT_ALLOWED_HOSTS,
            workers=int(workers) if workers else None,
            enable_nebius=_flag(env, "RTT_AGENT_NEBIUS"),
            offline=_flag(env, "RTT_AGENT_OFFLINE"),
            fake_physics=_flag(env, "RTT_AGENT_FAKE_PHYSICS"),
            require_passcode=_flag(env, "RTT_AGENT_REQUIRE_PASSCODE"),
        )


def check_stl(data: bytes) -> None:
    """Accept binary STL (size matches its triangle count) or ASCII STL; else :class:`UploadError`."""
    if len(data) < 84:
        raise UploadError("file too small to be an STL")
    count = int.from_bytes(data[80:84], "little")
    if count > 0 and len(data) == 84 + 50 * count:
        return
    head = data[:512].lstrip().lower()
    if head.startswith(b"solid") and b"facet" in data[:4096].lower():
        return
    raise UploadError("not an STL file (binary size does not match its triangle count, and not ASCII STL)")


def check_solid(data: bytes, units: str = "mm") -> float:
    """Refuse a flat or near-flat mesh before intake; returns its hull thickness in mm.

    The hull thickness is ``2 x volume / area`` of the convex hull: about the thickness
    of a plate, a third of a cube's side, half a rod's diameter. A mesh with (almost) no
    thickness has a centre of mass that is never strictly inside its hull, and the
    stable-pose search of the intake (``trimesh.poses.compute_stable_poses``, which
    rejection-samples the centre of mass) then loops forever: one tiny upload would pin
    a CPU core and an API thread for good. Such a part cannot be gripped anyway.

    Raises:
        UploadError: unreadable, no triangles, or thinner than :data:`MIN_THICKNESS_MM`.
    """
    import io

    import numpy as np
    import trimesh

    try:
        mesh = trimesh.load(io.BytesIO(data), file_type="stl", force="mesh")
    except Exception as exc:  # noqa: BLE001 - any parser failure is a bad upload
        raise UploadError(f"could not read the STL ({type(exc).__name__})") from None
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise UploadError("the STL holds no triangles")
    points = np.asarray(mesh.vertices, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)] * (1000.0 if units == "m" else 1.0)
    try:
        hull = trimesh.convex.convex_hull(points)
        thickness = 2.0 * float(hull.volume) / float(hull.area)
    except Exception:  # noqa: BLE001 - qhull refuses flat point sets: no thickness
        thickness = 0.0
    if not math.isfinite(thickness) or thickness < MIN_THICKNESS_MM:
        raise UploadError(
            f"the part is flat or almost flat (about {max(thickness, 0.0):.3g} mm thick); "
            f"upload a solid at least {MIN_THICKNESS_MM:g} mm thick"
        )
    return thickness


def clean_name(name: str | None, fallback: str) -> str:
    """A display name with only letters, digits, space, dot, dash and underscore (max 60)."""
    cleaned = _NAME_RE.sub("", (name or "").strip())[:60].strip(" .")
    return cleaned or fallback


class SessionView:
    """Read access to a session: live (its recorder) or finished (its file)."""

    def __init__(self, recorder: SessionRecorder | None = None, doc: Mapping[str, Any] | None = None) -> None:
        self._recorder = recorder
        self._doc = dict(doc) if doc is not None else None

    @property
    def done(self) -> bool:
        return self._recorder.done if self._recorder is not None else True

    def to_json(self) -> dict[str, Any]:
        if self._recorder is not None:
            return self._recorder.to_json()
        assert self._doc is not None
        return dict(self._doc)

    def wait_events(self, after: int, timeout_s: float) -> list[dict[str, Any]]:
        if self._recorder is not None:
            return self._recorder.wait_events(after, timeout_s)
        assert self._doc is not None
        return [e for e in self._doc.get("events", []) if int(e.get("seq", 0)) > after]


@dataclass
class _Running:
    recorder: SessionRecorder
    thread: threading.Thread
    stop: threading.Event
    result: SessionResult | None = None
    effectors: EffectorJobRunner | None = None  # the session's own runner (live designer), if any


class AgentService:
    """Parts, sessions, runs and print packs for one server process.

    Args:
        settings: Server settings.
        backend: Part backend (default: :func:`rtt.agent.backends.default_part_backend`,
            or :class:`rtt.agent.fakes.FakePartBackend` with ``fake_physics``).
        gate: Passcode gate (default: from ``RTT_AGENT_PASSCODE_SHA256``).
        transport_factory: Builds the model transport for a session (default: Token
            Factory from ``.env``; offline: the scripted transport).
        nebius_runner: Runner for ``target='nebius'`` (default: built from
            ``~/.config/rtt`` when ``enable_nebius``).
        baselines: Entrant designs offered to the agent (default ``configs/entrants.yaml``).
        designer_factory: ``factory(budget_usd)`` -> the Nemotron morphology designer for
            ``recommend_effector(design=true)`` in live sessions (default: track B3's designer on
            Token Factory, built by :func:`rtt.agent.effector_tool.designer_factory`, with the
            sessions' transport factory when one is given). Offline sessions never get one.
    """

    def __init__(
        self,
        settings: ApiSettings | None = None,
        *,
        backend: PartBackend | None = None,
        gate: PasscodeGate | None = None,
        transport_factory: Callable[[], Any] | None = None,
        nebius_runner: JobRunner | None = None,
        baselines: Sequence[FingerDesign] | None = None,
        designer_factory: Callable[[float], Any] | None = None,
    ) -> None:
        self.settings = settings or ApiSettings()
        self.store = AgentStore(self.settings.root)
        if backend is None:
            if self.settings.fake_physics:
                from rtt.agent.fakes import FakePartBackend

                backend = FakePartBackend(self.store.parts_dir)
            else:
                backend = default_part_backend(self.store.parts_dir, workers=self.settings.workers)
        self.backend = backend
        self.gate = gate if gate is not None else PasscodeGate.from_env()
        if transport_factory is None and self.settings.offline:
            from rtt.agent.fakes import ScriptedAgentTransport

            transport_factory = ScriptedAgentTransport
        self.transport_factory = transport_factory
        caps = self.settings.caps
        self.day_book = DailySpendBook(
            None if self.settings.offline else (self.settings.day_book or self.store.daily_spend_path),
            caps.day_usd,
        )
        self.local_runner = LocalJobRunner(self.backend.evaluate, max_workers=caps.max_concurrent_jobs)
        # recommend_effector (product mode): tournaments on this machine, outputs under
        # <root>/effectors/<part_id>/. They need the simulator backend's parts (resolve_upload);
        # the offline fake cannot run them, so the tool is not offered there. This shared runner
        # has no designer (offline sessions); a live session, which needed the passcode, gets
        # its own runner with the Nemotron designer (_session_effectors).
        self._effectors_on = callable(getattr(self.backend, "_get", None))
        self.effectors: EffectorJobRunner | None = (
            EffectorJobRunner(tournament_job(self.backend, self.store.root / "effectors"), designs=False)
            if self._effectors_on
            else None
        )
        self._designer_factory = designer_factory
        self._nebius = nebius_runner
        self.baselines = list(baselines) if baselines is not None else load_entrants(DEFAULT_ENTRANTS)
        self._sessions: dict[str, _Running] = {}
        self._starting = 0  # sessions between the busy check and their thread start
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"AgentService(root={str(self.store.root)!r}, offline={self.settings.offline})"

    def _session_effectors(self, offline: bool) -> EffectorJobRunner | None:
        """The effector runner of one session: the shared designer-free one offline, else the
        session's own with a live Nemotron designer (each job's client capped at its reservation;
        the toolbox keeps the session's design spend under its design cap, the session cap and
        the day cap)."""
        if not self._effectors_on or offline:
            return self.effectors
        root = self.store.root / "effectors"
        factory = self._designer_factory or designer_factory(root, transport=self.transport_factory)
        return EffectorJobRunner(tournament_job(self.backend, root, designer_factory=factory), designs=True)

    # ------------------------------------------------------------------ parts

    def upload_part(
        self,
        data: bytes,
        *,
        name: str | None = None,
        units: str = "mm",
        density: float | None = None,
        passcode: str | None = None,
    ) -> PartInfo:
        """Validate and register an uploaded STL; returns what the agent will see.

        Raises:
            PasscodeRequiredError / PasscodeLockedError: ``require_passcode`` is set and
                the passcode is wrong or missing (checked before anything is parsed).
            UploadError: not an acceptable STL.
        """
        if self.settings.require_passcode and not self.gate.check(passcode):
            raise PasscodeRequiredError("this server needs the passcode for uploads")
        if len(data) > self.settings.max_upload_bytes:
            raise UploadError(f"file is larger than {self.settings.max_upload_bytes} bytes")
        if units not in ("mm", "m"):
            raise UploadError("units must be 'mm' or 'm'")
        if density is not None and not 50.0 <= density <= 25_000.0:
            raise UploadError("density_kg_m3 must be between 50 and 25000")
        check_stl(data)
        check_solid(data, units)
        part_id = new_id("part")
        part_dir = self.store.part_dir(part_id)
        part_dir.mkdir(parents=True, exist_ok=True)
        source = part_dir / "source.stl"
        source.write_bytes(data)
        try:
            return self.backend.intake(
                source, part_id=part_id, name=clean_name(name, part_id), units=units, density=density
            )
        except (ValueError, TypeError) as exc:
            shutil.rmtree(part_dir, ignore_errors=True)
            raise UploadError(redact_paths(f"could not load the part: {exc}")[:400]) from exc
        except BaseException:
            shutil.rmtree(part_dir, ignore_errors=True)
            raise

    def part(self, part_id: str) -> PartInfo:
        check_id("part", part_id)
        return self.backend.info(part_id)

    # ------------------------------------------------------------------ sessions

    def runners(self) -> dict[str, JobRunner]:
        """Job targets offered to the agent. Offline sessions need no passcode, so they
        never get a paid target."""
        runners: dict[str, JobRunner] = {"local": self.local_runner}
        if self.settings.enable_nebius and not self.settings.offline:
            if self._nebius is None:
                from rtt.agent.jobs import NebiusJobRunner

                try:
                    self._nebius = NebiusJobRunner.from_config(self.store.parts_dir)
                except Exception as exc:  # noqa: BLE001 - reported as a 503, never as "no such part"
                    raise TargetUnavailableError(
                        redact_paths(
                            f"the nebius target is enabled but not usable: {type(exc).__name__}: {exc}"
                        )[:300]
                    ) from None
            runners["nebius"] = self._nebius
        return runners

    def active_sessions(self) -> int:
        with self._lock:
            return self._active_locked()

    def _active_locked(self) -> int:
        return sum(1 for r in self._sessions.values() if r.thread.is_alive())

    def start_session(self, part_id: str, *, passcode: str | None = None, goal: str | None = None) -> str:
        """Start the agent on ``part_id`` in a background thread; returns the session ID.

        Raises:
            PasscodeRequiredError / PasscodeLockedError: live mode (or ``require_passcode``)
                without the right passcode.
            PartNotFoundError: unknown or malformed part ID.
            BusyError: the server already runs ``max_active_sessions`` sessions.
            TargetUnavailableError: the Nebius target is enabled but could not be set up.
        """
        offline = self.settings.offline
        # The passcode is checked before the part is looked up: a lookup can re-run the
        # whole intake (after a restart), which an unauthenticated request must not cause.
        needs_passcode = not offline or self.settings.require_passcode
        if needs_passcode and not self.gate.check(passcode):
            raise PasscodeRequiredError(
                "a live session calls paid models; the passcode is required"
                if not offline
                else "this server needs the passcode for every session"
            )
        unlocked = True
        try:
            part = self.part(part_id)
        except ValueError:
            raise PartNotFoundError(part_id) from None
        with self._lock:  # check and claim a slot atomically: concurrent requests cannot both pass
            if self._active_locked() + self._starting >= self.settings.max_active_sessions:
                raise BusyError("another session is running; try again when it finishes")
            self._starting += 1
        try:
            return self._launch_session(part, offline=offline, unlocked=unlocked, goal=goal)
        finally:
            with self._lock:
                self._starting -= 1

    def _launch_session(self, part: PartInfo, *, offline: bool, unlocked: bool, goal: str | None) -> str:
        part_id = part.part_id
        caps = self.settings.caps
        # Targets first: if one cannot be set up, no session file is left behind as "running".
        runners = self.runners()
        session_id = new_id("session")
        guard = SpendGuard(caps.session_usd, self.day_book, session_id, billing=not offline)
        ledger = GuardedLedger(guard, None if offline else self.store.ledger_path)
        transport = self.transport_factory() if self.transport_factory is not None else None
        goal_text = (goal or "").strip()[:500] or DEFAULT_GOAL
        recorder = SessionRecorder(
            session_id,
            self.store.session_path(session_id),
            meta={"part": part.to_json(), "goal": goal_text, "offline": offline},
        )
        proposer = NemotronDesignProposer.create(ledger, transport=transport)
        effectors = self._session_effectors(offline)
        toolbox = Toolbox(
            session_id=session_id,
            part_id=part_id,
            backend=self.backend,
            proposer=proposer,
            runners=runners,
            store=self.store,
            caps=caps,
            guard=guard,
            unlocked=unlocked,
            ledger=ledger,
            recorder=recorder,
            baselines=self.baselines,
            offline=offline,
            effectors=effectors,
        )
        stop = threading.Event()
        engineer = GraspEngineer(
            AgentLLM.create(ledger, transport=transport),
            toolbox,
            recorder,
            caps=caps,
            goal=goal_text,
            summarizer=LightningSummarizer.create(ledger, transport=transport),
            stop=stop,
            offline=offline,
        )
        own = effectors if effectors is not None and effectors is not self.effectors else None
        running = _Running(recorder, threading.Thread(target=lambda: None), stop, effectors=own)

        def run() -> None:
            try:
                running.result = engineer.run()
            finally:
                if own is not None:  # the session's jobs were settled or cancelled by the toolbox
                    own.shutdown()

        running.thread = threading.Thread(target=run, name=f"rtt-agent-{session_id}", daemon=True)
        with self._lock:  # registered and alive before the slot claimed in start_session is freed
            self._sessions[session_id] = running
            running.thread.start()
        return session_id

    def session(self, session_id: str) -> SessionView:
        """A live session, or a finished one read from its file (``KeyError`` if neither)."""
        check_id("session", session_id)
        with self._lock:
            running = self._sessions.get(session_id)
        if running is not None:
            return SessionView(recorder=running.recorder)
        path = self.store.session_path(session_id)
        if not path.is_file():
            raise KeyError(session_id)
        return SessionView(doc=load_session(path))

    def stop_session(self, session_id: str) -> bool:
        """Ask a running session to stop after its current step; ``False`` if it is not running."""
        check_id("session", session_id)
        with self._lock:
            running = self._sessions.get(session_id)
        if running is None or not running.thread.is_alive():
            return False
        running.stop.set()
        return True

    def wait_session(self, session_id: str, timeout_s: float | None = None) -> SessionResult | None:
        """Block until a session thread ends (for the CLI and tests)."""
        with self._lock:
            running = self._sessions.get(session_id)
        if running is None:
            return None
        running.thread.join(timeout_s)
        return running.result

    # ------------------------------------------------------------------ outputs

    def run(self, run_id: str) -> dict[str, Any]:
        check_id("run", run_id)
        return self.store.load_run(run_id)

    def print_pack(self, pack_id: str) -> Path:
        path = self.store.pack_zip(pack_id)
        if not path.is_file():
            raise FileNotFoundError(pack_id)
        return path

    def shutdown(self) -> None:
        with self._lock:
            running = list(self._sessions.values())
        for item in running:
            item.stop.set()
            if item.effectors is not None:
                item.effectors.shutdown()
        self.local_runner.shutdown()
        if self.effectors is not None:
            self.effectors.shutdown()
