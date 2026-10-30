"""``rtt runner serve``: the runner on 127.0.0.1, against fakes or real keys.

* ``--fake``: no network at all. The planner answers every description with the Task Engineer's
  worked example (a valid plan, judged by the real checks); jobs live in memory, queue for a few
  seconds, run for a few more and finish with a summary built from the plan. State is in memory.
* without ``--fake``: the real Task Engineer and real Nebius jobs. The free tier's key comes from
  ``NEBIUS_API_KEY`` in the environment or ``--env-file``; jobs use the local ``nebius`` CLI
  profile and the uk-south2 settings and S3 profile of ``~/.config/rtt/nebius.env``
  (:mod:`rtt.cloud.config`). Real jobs cost money: the caps still apply, and state (so the spend
  ledger) is in memory unless ``--table`` names a DynamoDB table.

The server sends CORS headers itself (a Function URL does that in AWS) and lets the site's dev
server (``http://localhost:4341``) call it. The sweeper runs every ``--sweep-s`` seconds.
"""

from __future__ import annotations

import argparse
import base64
import json
import secrets as pysecrets
import threading
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from rtt.runner.app import Runner
from rtt.runner.caps import KillSwitch
from rtt.runner.config import NebiusJobConfig, RunnerConfig
from rtt.runner.engineer import Planner
from rtt.runner.logs import configure_logging
from rtt.runner.nebius import FakeNebius, JobBackend, NebiusBackend
from rtt.runner.secrets import StaticSecrets
from rtt.runner.store import DynamoStore, MemoryStore, Store

__all__ = ["FakePlannerTransport", "build_local_runner", "fake_summary", "main", "serve"]


class FakePlannerTransport:
    """A model that always submits the Task Engineer's worked example (no network)."""

    def __init__(self) -> None:
        from rtt.studio.engineer.examples import worked_example

        self.args = worked_example().to_tool_args()

    def chat_completions(self, payload: Mapping[str, Any], timeout_s: float) -> Any:
        from rtt.llm.providers import TransportResponse

        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "submit_task_plan", "arguments": json.dumps(self.args)},
                }
            ],
        }
        body = {
            "id": "fake",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        return TransportResponse(200, {}, body)

    def list_models(self, timeout_s: float) -> Any:
        from rtt.llm.providers import TransportResponse

        return TransportResponse(200, {}, {"data": []})


def fake_summary(run_id: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    """What a fake job "found": a made-up single-tool pick over the plan's steps (labelled fake)."""
    from rtt.runner.summary import PRODUCT_MODE_LABEL, SUMMARY_FORMAT

    steps = [s.get("id") for s in plan.get("steps") or []]
    tool = "Fake gripper (local fake run)"
    return {
        "format": SUMMARY_FORMAT,
        "version": 1,
        "label": PRODUCT_MODE_LABEL,
        "run_id": run_id,
        "status": "done",
        "task_id": plan.get("task_id"),
        "title": plan.get("title"),
        "outcome": "recommended",
        "evidence": "fake",
        "pick": {"setup_id": "fake", "label": tool, "kind": "single", "tools": [tool]},
        "closest": None,
        "completion": {"k": 8, "n": 8},
        "why": "This is a local fake run: no physics ran.",
        "baseline": {
            "setup_id": "fake",
            "label": tool,
            "kind": "single",
            "tools": [tool],
            "completion": {"k": 8, "n": 8},
            "basis": "best_single",
        },
        "baseline_note": None,
        "ruled_out": {
            "candidates": 0,
            "pruned": 0,
            "pruned_for_some_steps": 0,
            "screened": 0,
            "setups_rejected": 0,
            "finalists": 1,
            "finalists_not_eligible": 0,
        },
        "steps": [{"step": s, "tool": tool, "success_rate": 1.0} for s in steps],
        "conditions": 8,
        "code_version": "fake",
        "timings": {},
    }


def _real_jobs(cfg: RunnerConfig) -> JobBackend:
    """Nebius jobs with the laptop's CLI profile and S3 profile (uk-south2 settings)."""
    import boto3

    from rtt.cloud.config import load_config
    from rtt.cloud.nebius_jobs import NebiusCLI

    home = load_config()
    region = home.for_region(cfg.nebius.region)
    region.require("project_id", "subnet_id", "bucket_id", "bucket_name", "image", "s3_profile")
    neb = replace(
        cfg.nebius,
        project_id=region.project_id,
        subnet_id=region.subnet_id,
        bucket_id=region.bucket_id,
        bucket_name=region.bucket_name,
        image=region.image,
        s3_endpoint=region.s3_endpoint or cfg.nebius.s3_endpoint,
        cli=region.cli,
    )
    s3 = boto3.Session(profile_name=region.s3_profile).client("s3", endpoint_url=neb.s3_endpoint)
    return NebiusBackend(neb, NebiusCLI(region.cli, region.profile), s3, cfg.job_env())


def build_local_runner(*, fake: bool, env_file: str | Path | None = None, table: str | None = None) -> Runner:
    cfg = replace(RunnerConfig.from_env(), emit_cors=True, kill_switch_param=None)
    store: Store = DynamoStore(table) if table else MemoryStore()
    if fake:
        cfg = replace(
            cfg,
            nebius=NebiusJobConfig(
                project_id="fake",
                subnet_id="fake",
                bucket_id="fake",
                bucket_name="fake",
                image="fake@sha256:0",
            ),
        )
        backend = FakeNebius(timeline=(5.0, 10.0), summary_for=fake_summary)
        planner = Planner(cfg, transport_factory=lambda provider, key: FakePlannerTransport())
        secrets = StaticSecrets({cfg.tokenfactory_secret: pysecrets.token_hex(16)})
        return Runner(cfg, store, planner, lambda: backend, secrets, KillSwitch(cfg))
    from rtt.llm.providers import load_api_key

    secrets = StaticSecrets({cfg.tokenfactory_secret: load_api_key(env_file)})
    return Runner(cfg, store, Planner(cfg), lambda: _real_jobs(cfg), secrets, KillSwitch(cfg))


def _handler_for(runner: Runner) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "effector-lab-runner"

        def _serve(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            limit = runner.cfg.max_body_bytes + 1
            body = self.rfile.read(min(length, limit)) if length > 0 else b""
            path = self.path.split("?", 1)[0]
            event = {
                "rawPath": path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": base64.b64encode(body).decode(),
                "isBase64Encoded": True,
                "requestContext": {
                    "requestId": pysecrets.token_hex(8),
                    "http": {"method": self.command, "path": path, "sourceIp": self.client_address[0]},
                },
            }
            out = runner.handle_event(event)
            data = out["body"].encode()
            self.send_response(out["statusCode"])
            for k, v in out["headers"].items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        do_GET = do_POST = do_OPTIONS = _serve

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def serve(runner: Runner, host: str = "127.0.0.1", port: int = 8787, sweep_s: float = 30.0) -> None:
    server = ThreadingHTTPServer((host, port), _handler_for(runner))
    stop = threading.Event()

    def sweeper() -> None:
        while not stop.wait(sweep_s):
            runner.sweep()

    threading.Thread(target=sweeper, daemon=True).start()
    print(json.dumps({"serving": f"http://{host}:{port}", "at": datetime.now(UTC).isoformat()}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()


def main(argv: Sequence[str] | None = None, prog: str = "rtt runner") -> int:
    ap = argparse.ArgumentParser(prog=prog, description="The hosted runner, locally (docs/RUNNER.md).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("serve", help="serve the runner API on 127.0.0.1")
    sv.add_argument("--fake", action="store_true", help="fake model and fake jobs: no network, no cost")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8787)
    sv.add_argument("--env-file", type=Path, default=None, help=".env with NEBIUS_API_KEY (free tier)")
    sv.add_argument("--table", default=None, help="a DynamoDB table instead of memory")
    sv.add_argument("--sweep-s", type=float, default=30.0)
    args = ap.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("error: rtt runner serve listens on localhost only")
        return 2
    configure_logging()
    runner = build_local_runner(fake=args.fake, env_file=args.env_file, table=args.table)
    serve(runner, args.host, args.port, args.sweep_s)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
