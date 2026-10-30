"""``python -m rtt.runner serve [--fake]`` (also ``rtt runner serve``); see :mod:`rtt.runner.local`."""

from __future__ import annotations

from rtt.runner.local import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(prog="python -m rtt.runner"))
