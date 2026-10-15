#!/usr/bin/env python
"""Token Factory smoke test: list models (free) and make one tiny Nano call.

Prints counts, latency, token usage and estimated cost only; never the key and
never the response text. Exit status is non-zero on any failure.

Usage:
    python scripts/tf_smoke.py [--env-file .env] [--skip-call]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rtt.llm import MODELS, NEMOTRON_NANO, Ledger, LLMClient, LLMError, OpenAITransport
from rtt.llm.providers import MissingCredentialsError

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--skip-call", action="store_true", help="only list models (free)")
    parser.add_argument("--budget-usd", type=float, default=0.001, help="spend cap for the tiny call")
    args = parser.parse_args(argv)

    try:
        transport = OpenAITransport.from_env(args.env_file if args.env_file.is_file() else None)
    except MissingCredentialsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    started = time.monotonic()
    listing = transport.list_models(timeout_s=30.0)
    list_latency = time.monotonic() - started
    if listing.status_code != 200 or listing.body is None:
        print(f"models: HTTP {listing.status_code}", file=sys.stderr)
        return 1
    ids = {str(m.get("id")) for m in listing.body.get("data", []) if isinstance(m, dict)}
    nvidia = sorted(i for i in ids if i.lower().startswith("nvidia/"))
    print(f"models: {len(ids)} listed, {len(nvidia)} nvidia, latency {list_latency:.2f}s")
    for model_id in MODELS:
        print(f"  {'ok     ' if model_id in ids else 'MISSING'} {model_id}")

    if args.skip_call:
        return 0

    ledger = Ledger(budget_usd=args.budget_usd)
    client = LLMClient(
        NEMOTRON_NANO, temperature=0.0, top_p=1.0, max_tokens=64, ledger=ledger, transport=transport
    )
    messages = [{"role": "user", "content": "Reply with the single word: ready"}]
    try:
        _, meta = client.chat(messages)
    except LLMError as exc:
        print(f"call: failed ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1
    print(
        f"call: {meta.model} latency {meta.latency_s:.2f}s, tokens {meta.prompt_tokens} in / "
        f"{meta.completion_tokens} out, est ${meta.cost_usd:.6f}, finish={meta.finish_reason}, "
        f"reasoning_separate={meta.reasoning_separate}, attempts={meta.attempts}"
    )
    if meta.rate_limit_headers:
        print("rate-limit headers: " + ", ".join(sorted(meta.rate_limit_headers)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
