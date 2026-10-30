"""Fakes and a harness for the hosted runner's tests (``tests/test_runner_*.py``): no network.

* :class:`FakeModel`: an OpenAI-compatible transport that answers each call with the next scripted
  item (tool arguments, or an HTTP status whose error body echoes the key it was given, the worst
  a provider could do).
* :class:`Harness`: a :class:`~rtt.runner.app.Runner` over a :class:`MemoryStore`, a
  :class:`FakeNebius`, the fake model, static secrets, a kill switch read from a dict and a clock
  the test moves.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

from botocore.exceptions import ClientError

from rtt.llm.client import TokenBucket
from rtt.llm.providers import TransportResponse
from rtt.runner.app import Runner
from rtt.runner.caps import KillSwitch
from rtt.runner.config import NebiusJobConfig, ProviderSpec, RunnerConfig
from rtt.runner.engineer import Planner
from rtt.runner.nebius import FakeNebius
from rtt.runner.secrets import StaticSecrets
from rtt.runner.store import DynamoStore, MemoryStore
from rtt.runner.summary import PRODUCT_MODE_LABEL, SUMMARY_FORMAT
from rtt.studio.engineer.examples import worked_example

ORIGIN = "https://stray128.github.io"
VISITOR_KEY = "vk-SENTINEL-7f3a9c1e5b2d4f6a8c0e"
FREE_KEY = "tf-OURS-SENTINEL-0b1c2d3e4f5a6b7c"
START = datetime(2026, 10, 28, 9, 0, 0, tzinfo=UTC)


def good_args() -> dict[str, Any]:
    return worked_example().to_tool_args()


def good_plan() -> dict[str, Any]:
    return worked_example().to_dict()


class FakeModel:
    """Scripted model (module docstring); ``key`` is what the runner handed the transport."""

    def __init__(self, answers: Sequence[Any], key: str = "") -> None:
        self.answers = answers if isinstance(answers, list) else list(answers)  # shared script
        self.key = key
        self.payloads: list[dict[str, Any]] = []

    def chat_completions(self, payload: Mapping[str, Any], timeout_s: float) -> TransportResponse:
        self.payloads.append(json.loads(json.dumps(payload)))
        item = self.answers.pop(0) if self.answers else good_args()
        if isinstance(item, int):
            return TransportResponse(
                item,
                {},
                {"error": {"message": f"key {self.key} is not valid"}},
                error_text=f"bad key {self.key}",
            )
        if isinstance(item, BaseException):
            raise item
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "submit_task_plan", "arguments": json.dumps(item)},
                }
            ],
        }
        body = {
            "id": f"r{len(self.payloads)}",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
            "usage": {"prompt_tokens": 18000, "completion_tokens": 2000},
        }
        return TransportResponse(200, {}, body)

    def list_models(self, timeout_s: float) -> TransportResponse:
        return TransportResponse(200, {}, {"data": []})


def summary_for(run_id: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": SUMMARY_FORMAT,
        "version": 1,
        "label": PRODUCT_MODE_LABEL,
        "run_id": run_id,
        "status": "done",
        "outcome": "recommended",
        "pick": {"setup_id": "s1", "label": "Gripper", "kind": "single", "tools": ["Gripper"]},
        "completion": {"k": 8, "n": 8},
        "why": "best",
        "baseline": {
            "setup_id": "s1",
            "label": "Gripper",
            "kind": "single",
            "tools": ["Gripper"],
            "completion": {"k": 8, "n": 8},
            "basis": "R01",
        },
        "ruled_out": {"candidates": 3, "pruned": 1},
        "steps": [{"step": s["id"], "tool": "Gripper", "success_rate": 1.0} for s in plan.get("steps") or []],
        "code_version": "abc123",
    }


class _DynamoExceptions:
    class TransactionCanceledException(ClientError):
        pass


_ADD = re.compile(r"^ADD (?P<adds>.+) SET expires_at = :e$")
_CLAUSE = re.compile(r"^\(attribute_not_exists\((#a\d+)\) OR (#a\d+) <= (:lim\d+)\)$")


class FakeDynamoClient:
    """A DynamoDB client in memory that enforces what the real service does and a stub cannot:

    * a transaction may not hold two actions on one item (``ValidationException``), and
    * each action's ``ConditionExpression`` is evaluated (``attribute_not_exists(pk)``,
      ``#v = :v``, and capped ``ADD`` clauses joined by ``AND``); any failure cancels the whole
      transaction with per-action ``CancellationReasons``.

    ``fail_next`` makes the next transaction raise a service error (an outage)."""

    exceptions = _DynamoExceptions

    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}
        self.transactions: list[list[str]] = []
        self.rejected_duplicates = 0
        self.fail_next = 0

    @staticmethod
    def _n(value: Mapping[str, str]) -> int:
        return int(value["N"])

    def _condition_ok(self, item: dict[str, Any] | None, op: Mapping[str, Any]) -> bool:
        cond = op.get("ConditionExpression")
        if not cond:
            return True
        names = op.get("ExpressionAttributeNames") or {}
        values = op.get("ExpressionAttributeValues") or {}
        if cond == "attribute_not_exists(pk)":
            return item is None
        if cond == "#v = :v":
            return item is not None and item.get(names["#v"]) == values[":v"]
        for clause in cond.split(" AND "):
            m = _CLAUSE.match(clause)
            assert m and m.group(1) == m.group(2), f"unknown condition {clause!r}"
            attr = names[m.group(1)]
            if item is not None and attr in item and self._n(item[attr]) > self._n(values[m.group(3)]):
                return False
        return True

    def transact_write_items(self, TransactItems: list[dict[str, Any]]) -> dict[str, Any]:
        if self.fail_next:
            self.fail_next -= 1
            raise ClientError(
                {"Error": {"Code": "InternalServerError", "Message": "scripted"}}, "TransactWriteItems"
            )
        keys = []
        for t in TransactItems:
            ((_, op),) = t.items()
            keys.append((op.get("Key") or op.get("Item"))["pk"]["S"])
        self.transactions.append(keys)
        if len(set(keys)) != len(keys):
            self.rejected_duplicates += 1
            raise ClientError(
                {
                    "Error": {
                        "Code": "ValidationException",
                        "Message": "Transaction request cannot include multiple operations on one item",
                    }
                },
                "TransactWriteItems",
            )
        reasons = [
            {"Code": "None" if self._condition_ok(self.items.get(k), op) else "ConditionalCheckFailed"}
            for k, (op,) in zip(keys, (t.values() for t in TransactItems), strict=True)
        ]
        if any(r["Code"] != "None" for r in reasons):
            raise _DynamoExceptions.TransactionCanceledException(
                {
                    "Error": {"Code": "TransactionCanceledException", "Message": "cancelled"},
                    "CancellationReasons": reasons,
                },
                "TransactWriteItems",
            )
        for k, t in zip(keys, TransactItems, strict=True):
            ((kind, op),) = t.items()
            if kind == "Put":
                self.items[k] = json.loads(json.dumps(op["Item"]))
                continue
            names, values = op["ExpressionAttributeNames"], op["ExpressionAttributeValues"]
            m = _ADD.match(op["UpdateExpression"])
            assert m, op["UpdateExpression"]
            item = self.items.setdefault(k, {"pk": {"S": k}})
            for part in m.group("adds").split(", "):
                name, value = part.split(" ")
                attr = names[name]
                item[attr] = {"N": str(self._n(item.get(attr, {"N": "0"})) + self._n(values[value]))}
            item["expires_at"] = values[":e"]
        return {}

    def get_item(self, TableName: str, Key: Mapping[str, Any], ConsistentRead: bool = True) -> dict[str, Any]:
        item = self.items.get(Key["pk"]["S"])
        return {"Item": json.loads(json.dumps(item))} if item else {}

    def query(self, **kwargs: Any) -> dict[str, Any]:
        return {"Items": [i for i in self.items.values() if (i.get("active") or {}).get("S") == "run"]}


class Harness:
    def __init__(
        self, cfg: RunnerConfig | None = None, answers: Sequence[Any] = (), *, dynamo: bool = False
    ) -> None:
        self.now = START
        base = cfg or RunnerConfig()
        self.cfg = replace(
            base,
            kill_switch_param="/test/kill",
            kill_switch_ttl_s=0.0,
            nebius=base.nebius
            if not base.nebius.missing()
            else NebiusJobConfig(
                project_id="project-x",
                subnet_id="subnet-x",
                bucket_id="bucket-x",
                bucket_name="b",
                image="cr.example/reg/rtt-sim@sha256:" + "0" * 64,
            ),
        )
        self.dynamo = FakeDynamoClient() if dynamo else None
        self.store: Any = DynamoStore("runner-test", self.dynamo) if dynamo else MemoryStore()
        self.nebius = FakeNebius(clock=lambda: self.now)
        self.models: list[FakeModel] = []
        self.answers = list(answers)
        self.params = {"/test/kill": "off"}
        self.kill = KillSwitch(self.cfg, reader=lambda name: self.params[name])
        self.planner = Planner(self.cfg, transport_factory=self._transport, limiter=TokenBucket(1e12))
        self.secrets = StaticSecrets({self.cfg.tokenfactory_secret: FREE_KEY})
        self.runner = Runner(
            self.cfg,
            self.store,
            self.planner,
            lambda: self.nebius,
            self.secrets,
            self.kill,
            clock=lambda: self.now,
        )

    def _transport(self, provider: ProviderSpec, key: str) -> FakeModel:
        model = FakeModel(self.answers, key)
        self.models.append(model)
        return model

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)

    def call(
        self,
        method: str,
        path: str,
        body: Any = None,
        *,
        ip: str = "203.0.113.7",
        origin: str | None = ORIGIN,
        raw: bytes | None = None,
    ) -> tuple[int, Any, dict[str, str]]:
        headers = {"content-type": "application/json"}
        if origin is not None:
            headers["origin"] = origin
        data = raw if raw is not None else (b"" if body is None else json.dumps(body).encode())
        event = {
            "rawPath": path,
            "headers": headers,
            "body": data.decode("utf-8", errors="replace"),
            "isBase64Encoded": False,
            "requestContext": {
                "requestId": "req-1",
                "http": {"method": method, "path": path, "sourceIp": ip},
            },
        }
        out = self.runner.handle_event(event)
        payload = json.loads(out["body"]) if out["body"] else None
        return out["statusCode"], payload, out["headers"]

    def plan(
        self,
        provider: str = "free",
        key: str | None = None,
        ip: str = "203.0.113.7",
        text: str = "Open a drawer.",
    ) -> tuple[int, Any]:
        status, body, _ = self.call(
            "POST", "/plan", {"description": text, "provider": provider, "key": key}, ip=ip
        )
        return status, body

    def start_run(self, plan: Any = None, ip: str = "203.0.113.7") -> tuple[int, Any]:
        status, body, _ = self.call(
            "POST", "/runs", {"plan": plan if plan is not None else good_plan()}, ip=ip
        )
        return status, body
