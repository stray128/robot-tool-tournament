"""``python -m rtt.agent``: run one Grasp Engineer session from the terminal.

::

    python -m rtt.agent run --stl part.stl [--name NAME] [--units mm] [--goal TEXT]
                            [--offline] [--fake-physics] [--session-usd 0.30]

A live session (the default) calls Nemotron on Token Factory and needs the spend
passcode: it is read from the ``RTT_AGENT_PASSCODE`` environment variable (never a
command-line argument, so it stays out of shell history) and checked against
``RTT_AGENT_PASSCODE_SHA256``. Steps are printed as they happen; the session file,
results and print pack land under ``runs/agent/`` (or ``--root``).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _line(event: dict[str, Any]) -> str:
    kind = event["type"]
    head = f"[{event['t_s']:7.1f}s] {kind:<16}"
    if kind == "tool.call":
        return f"{head} {event['name']}({json.dumps(event.get('args'), separators=(',', ':'))})"
    if kind in ("tool.result", "tool.refused"):
        body = (
            event.get("result")
            if event.get("ok")
            else {"error": event.get("error"), "code": event.get("code")}
        )
        text = json.dumps(body, separators=(",", ":"))
        return f"{head} {event['name']} ok={event['ok']} {text[:300]}"
    if kind == "llm.call":
        return (
            f"{head} {event['role']} {event['model']} in={event['prompt_tokens']} "
            f"out={event['completion_tokens']} ${event['cost_usd']:.5f} {event['latency_s']}s"
        )
    if kind == "agent.message":
        calls = ", ".join(c["name"] for c in event.get("tool_calls", []))
        return f"{head} {(event.get('content') or '').strip()[:300]} {('-> ' + calls) if calls else ''}"
    if kind == "summary":
        return f"{head} {event['text']}"
    if kind == "job.update":
        return f"{head} {event['job_id']} {event['state']} est ${event.get('est_usd', 0):.4f}"
    if kind == "session.finished":
        return f"{head} {event['status']} ({event.get('reason') or 'done'}) spend ${event['spend']['spent_usd']:.4f}"
    return head


def main(argv: Sequence[str] | None = None, *, prog: str = "python -m rtt.agent") -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="upload a part and run one session")
    run.add_argument("--stl", required=True, type=Path)
    run.add_argument("--name", default=None)
    run.add_argument("--units", choices=("mm", "m"), default="mm")
    run.add_argument("--density", type=float, default=None, help="kg/m^3 (default: the task's)")
    run.add_argument("--goal", default=None)
    run.add_argument("--root", type=Path, default=None, help="output root (default runs/agent)")
    run.add_argument("--offline", action="store_true", help="scripted agent: no model calls, no spend")
    run.add_argument("--fake-physics", action="store_true", help="synthetic physics instead of MuJoCo")
    run.add_argument("--session-usd", type=float, default=None, help="session spend cap (USD)")
    run.add_argument("--workers", type=int, default=None, help="CPU workers per simulation")
    args = parser.parse_args(argv)

    from dataclasses import replace

    from rtt.api.service import AgentService, ApiSettings

    settings = ApiSettings.from_env()
    caps = settings.caps
    if args.session_usd is not None:
        caps = replace(caps, session_usd=args.session_usd)
    settings = replace(
        settings,
        caps=caps,
        offline=settings.offline or args.offline,
        fake_physics=settings.fake_physics or args.fake_physics,
        root=args.root or settings.root,
        workers=args.workers if args.workers is not None else settings.workers,
    )
    service = AgentService(settings)
    try:
        passcode = os.environ.get("RTT_AGENT_PASSCODE")
        part = service.upload_part(
            args.stl.read_bytes(),
            name=args.name or args.stl.stem,
            units=args.units,
            density=args.density,
            passcode=passcode,
        )
        print(
            f"part {part.part_id}: {part.size_mm} mm, grip {part.grip_width_mm} mm, fits={part.fits_gripper}"
        )
        session_id = service.start_session(part.part_id, passcode=passcode, goal=args.goal)
        print(f"session {session_id} -> {service.store.session_path(session_id)}")
        view = service.session(session_id)
        last = 0
        while True:
            events = view.wait_events(last, 1.0)
            for event in events:
                print(_line(event), flush=True)
                last = event["seq"]
            if view.done and not view.wait_events(last, 0.0):
                break
            time.sleep(0.05)
        service.wait_session(session_id, 5.0)
        doc = view.to_json()
        return 0 if doc.get("status") == "completed" else 1
    finally:
        service.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
