#!/usr/bin/env python
"""Probe the NVIDIA Nemotron models on Token Factory (wrapper around :mod:`rtt.llm.probe`).

Runs live by default (cache first; only cache misses are billed, under the spend
cap). Prefer ``rtt probe-llm`` (summary), ``rtt probe-llm --live`` or
``rtt probe-llm --replay``.

Usage:
    python scripts/probe_llm.py [--calls 2] [--cap-usd 0.15] [--thinking-off] [--replay]
"""

from rtt.llm.probe import main

if __name__ == "__main__":
    raise SystemExit(main())
