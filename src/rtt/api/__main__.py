"""``python -m rtt.api``: serve the Grasp Engineer API on this machine (127.0.0.1 by default)."""

from __future__ import annotations

import argparse
import ipaddress
from collections.abc import Sequence
from dataclasses import replace


def is_loopback(host: str) -> bool:
    """Whether ``host`` (a bind address or name) only accepts connections from this machine."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False  # another host name: assume it is reachable from elsewhere


def main(argv: Sequence[str] | None = None, *, prog: str = "python -m rtt.api") -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: local only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--offline", action="store_true", help="scripted agent, no model calls, no spend")
    parser.add_argument("--fake-physics", action="store_true", help="synthetic physics (UI work only)")
    parser.add_argument("--nebius", action="store_true", help="offer target='nebius' to the agent")
    args = parser.parse_args(argv)

    import uvicorn

    from rtt.api.app import create_app
    from rtt.api.service import ApiSettings

    settings = ApiSettings.from_env()
    settings = replace(
        settings,
        offline=settings.offline or args.offline,
        fake_physics=settings.fake_physics or args.fake_physics,
        enable_nebius=settings.enable_nebius or args.nebius,
    )
    if args.host not in settings.allowed_hosts and args.host not in ("0.0.0.0", "::"):
        # bound to one named address: answer requests made to it (see RTT_AGENT_ALLOWED_HOSTS)
        settings = replace(settings, allowed_hosts=(*settings.allowed_hosts, args.host))
    if not is_loopback(args.host):
        # reachable from other machines: part intake is "local or passcode only" (Addendum A),
        # so uploads and offline sessions need the passcode too
        settings = replace(settings, require_passcode=True)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
