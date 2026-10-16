"""Grasp Engineer: a bounded Nemotron tool-calling agent (product mode).

Product mode only: these sessions are not part of the pre-registered experiment and are
labelled so in every recording.

* :mod:`rtt.agent.engineer`: the tool loop (Nemotron 3 Super, ``tool_choice="auto"``).
* :mod:`rtt.agent.tools`: the eight tools and every hard cap.
* :mod:`rtt.agent.schemas`: tool JSON schemas and argument validation.
* :mod:`rtt.agent.caps`: tool-call, concurrency, dollar and passcode limits.
* :mod:`rtt.agent.backends`: part intake and physics (MuJoCo via ``rtt.sim`` or ``rtt.part``).
* :mod:`rtt.agent.jobs`: local and Nebius simulation jobs; :mod:`rtt.agent.remote` runs
  inside the Nebius job.
* :mod:`rtt.agent.proposer` (Super) and :mod:`rtt.agent.summarizer` (Lightning).
* :mod:`rtt.agent.recorder`: the session file used for playback.
* :mod:`rtt.agent.fakes`: offline stand-ins for tests and dry runs.

See ``docs/AGENT.md``.
"""

from rtt.agent.caps import (
    HARD_MAX_CONCURRENT_JOBS,
    HARD_MAX_TOOL_CALLS,
    AgentCaps,
    DailySpendBook,
    GuardedLedger,
    PasscodeGate,
    SpendGuard,
    passcode_digest,
)
from rtt.agent.recorder import PRODUCT_MODE_LABEL, SessionRecorder, load_session
from rtt.agent.schemas import TOOL_NAMES, TOOL_SPECS, openai_tools, validate_args

__all__ = [
    "HARD_MAX_CONCURRENT_JOBS",
    "HARD_MAX_TOOL_CALLS",
    "PRODUCT_MODE_LABEL",
    "TOOL_NAMES",
    "TOOL_SPECS",
    "AgentCaps",
    "DailySpendBook",
    "GuardedLedger",
    "PasscodeGate",
    "SessionRecorder",
    "SpendGuard",
    "load_session",
    "openai_tools",
    "passcode_digest",
    "validate_args",
]
