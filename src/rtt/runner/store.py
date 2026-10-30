"""State of the hosted runner: counters, the spend ledger and run records.

One DynamoDB table (on demand), keyed by ``pk`` alone, with a TTL on ``expires_at`` for every
item and a sparse index (``active``) over the runs still in flight. :class:`MemoryStore` is the same
store in memory for tests and ``rtt runner serve``.

Every write goes through :meth:`Store.commit`, one all-or-nothing transaction of

* :class:`Bump`: add ``amount`` to a number on an item, refused when the result would pass
  ``limit`` (DynamoDB conditions cannot add, so the check is ``value <= limit - amount``; a bump
  that could never fit, ``limit - amount < 0``, is refused before anything is sent), and
* at most one run record write, guarded by its version (optimistic concurrency): a new run must
  not exist yet; an update must find the version it read.

DynamoDB refuses a transaction with two actions on one item, so :func:`plan_transaction` groups
the bumps by item into one update each (``ADD #a0 :n0, #a1 :n1 SET expires_at = :e``, the capped
attributes' conditions joined by ``AND``). Both stores run the same plan, and it refuses a run
write whose item a bump also touches, so a transaction that DynamoDB would reject fails in memory
too.

So a run's reservation, its per-IP count and its record are written together, and a run is
settled exactly once. A refused bump raises :class:`LimitReached` naming its cap; a lost race on a
run raises :class:`Conflict`.

Items (``pk``):

* ``count#<name>#<day>`` / ``count#<name>#<ip hash>#<day>``: ``n``.
* ``spend#nebius#day#<day>`` and ``spend#nebius#total``: ``committed`` (settled spend plus open
  reservations) and ``settled``, in micro-dollars.
* ``spend#llm#day#<day>``: ``micros`` spent on free-tier plans.
* ``run#<run_id>``: ``doc`` (the record as JSON), ``version``, ``active`` (only while in flight).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

__all__ = [
    "ACTIVE_INDEX",
    "Bump",
    "Conflict",
    "DynamoStore",
    "ItemUpdate",
    "LimitReached",
    "MemoryStore",
    "RunWrite",
    "Store",
    "plan_transaction",
]

ACTIVE_INDEX: Final = "active-runs"


class LimitReached(Exception):
    """A bump would pass its limit; ``bump`` names which."""

    def __init__(self, bump: Bump) -> None:
        super().__init__(f"limit reached: {bump.cap}")
        self.bump = bump


class Conflict(Exception):
    """The run record changed since it was read (or a new run id already exists)."""


@dataclass(frozen=True)
class Bump:
    key: str
    attr: str
    amount: int
    expires_at: int
    limit: int | None = None
    cap: str = ""

    @property
    def capped(self) -> bool:
        return self.limit is not None and self.amount > 0

    def allows(self, current: int) -> bool:
        return not self.capped or current <= self.limit - self.amount  # type: ignore[operator]


@dataclass(frozen=True)
class AttrAdd:
    """The total added to one attribute of one item, and the capped bump that guards it."""

    attr: str
    amount: int
    capped: Bump | None


@dataclass(frozen=True)
class ItemUpdate:
    """Every bump on one item, as one update: one ``ADD`` per attribute."""

    key: str
    adds: tuple[AttrAdd, ...]
    expires_at: int

    @property
    def first_capped(self) -> Bump | None:
        return next((a.capped for a in self.adds if a.capped is not None), None)


def plan_transaction(bumps: Sequence[Bump], run: RunWrite | None = None) -> list[ItemUpdate]:
    """The bumps grouped into one :class:`ItemUpdate` per item (in first-seen order).

    Raises :class:`LimitReached` for a capped bump that can never fit (``limit - amount < 0``),
    and ``ValueError`` for a run write on an item a bump also touches (DynamoDB would reject it)
    or for two capped bumps on one item: a failed condition names its action, not the clause, so
    each item may carry at most one cap for :class:`LimitReached` to name the right one.
    """
    for b in bumps:
        if b.capped and b.limit - b.amount < 0:  # type: ignore[operator]
            raise LimitReached(b)
    order: list[str] = []
    by_key: dict[str, dict[str, list[Bump]]] = {}
    for b in bumps:
        if b.key not in by_key:
            order.append(b.key)
            by_key[b.key] = {}
        by_key[b.key].setdefault(b.attr, []).append(b)
    if run is not None and run_key(run.run_id) in by_key:
        raise ValueError("a run write and a bump target the same item")
    out: list[ItemUpdate] = []
    for key in order:
        adds: list[AttrAdd] = []
        for attr, group in by_key[key].items():
            capped = [b for b in group if b.capped]
            if len(capped) > 1:
                raise ValueError(f"two capped bumps on {key} {attr}")
            total = sum(b.amount for b in group)
            if total != 0 or capped:
                adds.append(AttrAdd(attr, total, capped[0] if capped else None))
        if sum(1 for a in adds if a.capped is not None) > 1:
            raise ValueError(f"two capped bumps on {key}")
        if adds:
            out.append(
                ItemUpdate(key, tuple(adds), max(b.expires_at for g in by_key[key].values() for b in g))
            )
    return out


def _fits(current: int, add: AttrAdd) -> bool:
    b = add.capped
    return b is None or current <= b.limit - add.amount  # type: ignore[operator]


@dataclass(frozen=True)
class RunWrite:
    """Write ``record`` as run ``record["run_id"]``; ``expect_version`` ``None`` creates it."""

    record: Mapping[str, Any]
    expect_version: int | None
    active: bool
    expires_at: int

    @property
    def run_id(self) -> str:
        return str(self.record["run_id"])


class Store(Protocol):
    def commit(self, bumps: Sequence[Bump] = (), run: RunWrite | None = None) -> None: ...

    def numbers(self, key: str) -> dict[str, int]: ...

    def get_run(self, run_id: str) -> tuple[dict[str, Any], int] | None: ...

    def active_runs(self) -> list[tuple[dict[str, Any], int]]: ...


def run_key(run_id: str) -> str:
    return f"run#{run_id}"


class MemoryStore:
    """The store in one process (tests, ``rtt runner serve``). TTLs are recorded, not enforced."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.items: dict[str, dict[str, Any]] = {}

    def commit(self, bumps: Sequence[Bump] = (), run: RunWrite | None = None) -> None:
        updates = plan_transaction(bumps, run)
        with self._lock:
            for u in updates:
                item = self.items.get(u.key, {})
                for add in u.adds:
                    if not _fits(int(item.get(add.attr, 0)), add):
                        raise LimitReached(add.capped)  # type: ignore[arg-type]
            if run is not None:
                existing = self.items.get(run_key(run.run_id))
                if run.expect_version is None and existing is not None:
                    raise Conflict(run.run_id)
                if run.expect_version is not None and (
                    existing is None or int(existing["version"]) != run.expect_version
                ):
                    raise Conflict(run.run_id)
            for u in updates:
                item = self.items.setdefault(u.key, {"pk": u.key})
                for add in u.adds:
                    item[add.attr] = int(item.get(add.attr, 0)) + add.amount
                item["expires_at"] = u.expires_at
            if run is not None:
                version = 1 if run.expect_version is None else run.expect_version + 1
                item = {
                    "pk": run_key(run.run_id),
                    "doc": json.dumps(run.record, sort_keys=True),
                    "version": version,
                    "expires_at": run.expires_at,
                }
                if run.active:
                    item["active"] = "run"
                self.items[run_key(run.run_id)] = item

    def numbers(self, key: str) -> dict[str, int]:
        with self._lock:
            item = self.items.get(key, {})
            return {k: int(v) for k, v in item.items() if k not in ("pk", "doc", "active")}

    def get_run(self, run_id: str) -> tuple[dict[str, Any], int] | None:
        with self._lock:
            item = self.items.get(run_key(run_id))
            if item is None or "doc" not in item:
                return None
            return json.loads(item["doc"]), int(item["version"])

    def active_runs(self) -> list[tuple[dict[str, Any], int]]:
        with self._lock:
            return [
                (json.loads(i["doc"]), int(i["version"]))
                for i in self.items.values()
                if i.get("active") == "run" and "doc" in i
            ]


class DynamoStore:
    """The store on DynamoDB through the low-level client (numbers travel as strings, no floats).

    ``client``: a boto3 DynamoDB client (tests pass a stubbed one).
    """

    def __init__(self, table: str, client: Any | None = None) -> None:
        self.table = table
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("dynamodb")
        return self._client

    def _update_op(self, u: ItemUpdate) -> dict[str, Any]:
        names: dict[str, str] = {}
        values: dict[str, Any] = {":e": {"N": str(u.expires_at)}}
        adds: list[str] = []
        conditions: list[str] = []
        for i, add in enumerate(u.adds):
            names[f"#a{i}"] = add.attr
            values[f":n{i}"] = {"N": str(add.amount)}
            adds.append(f"#a{i} :n{i}")
            if add.capped is not None:
                values[f":lim{i}"] = {"N": str(add.capped.limit - add.amount)}  # type: ignore[operator]
                conditions.append(f"(attribute_not_exists(#a{i}) OR #a{i} <= :lim{i})")
        update: dict[str, Any] = {
            "TableName": self.table,
            "Key": {"pk": {"S": u.key}},
            "UpdateExpression": "ADD " + ", ".join(adds) + " SET expires_at = :e",
            "ExpressionAttributeNames": names,
            "ExpressionAttributeValues": values,
        }
        if conditions:
            update["ConditionExpression"] = " AND ".join(conditions)
        return {"Update": update}

    def _run_op(self, run: RunWrite) -> dict[str, Any]:
        version = 1 if run.expect_version is None else run.expect_version + 1
        item: dict[str, Any] = {
            "pk": {"S": run_key(run.run_id)},
            "doc": {"S": json.dumps(run.record, sort_keys=True)},
            "version": {"N": str(version)},
            "expires_at": {"N": str(run.expires_at)},
            "created_at": {"S": str(run.record.get("created_at", ""))},
        }
        if run.active:
            item["active"] = {"S": "run"}
        put: dict[str, Any] = {"TableName": self.table, "Item": item}
        if run.expect_version is None:
            put["ConditionExpression"] = "attribute_not_exists(pk)"
        else:
            put["ConditionExpression"] = "#v = :v"
            put["ExpressionAttributeNames"] = {"#v": "version"}
            put["ExpressionAttributeValues"] = {":v": {"N": str(run.expect_version)}}
        return {"Put": put}

    def commit(self, bumps: Sequence[Bump] = (), run: RunWrite | None = None) -> None:
        updates = plan_transaction(bumps, run)
        ops = [self._update_op(u) for u in updates]
        if run is not None:
            ops.append(self._run_op(run))
        if not ops:
            return
        try:
            self.client.transact_write_items(TransactItems=ops)
        except self.client.exceptions.TransactionCanceledException as exc:
            reasons = exc.response.get("CancellationReasons") or []
            for i, reason in enumerate(reasons):
                if (reason or {}).get("Code") == "ConditionalCheckFailed":
                    if i < len(updates):
                        # plan_transaction allows at most one capped attribute per item, so the
                        # failed condition of this action is that bump's
                        capped = updates[i].first_capped
                        if capped is None:  # an uncapped update has no condition to fail
                            raise
                        raise LimitReached(capped) from None
                    raise Conflict(run.run_id if run else "") from None
            raise

    def numbers(self, key: str) -> dict[str, int]:
        resp = self.client.get_item(TableName=self.table, Key={"pk": {"S": key}}, ConsistentRead=True)
        item = resp.get("Item") or {}
        return {k: int(v["N"]) for k, v in item.items() if "N" in v}

    def get_run(self, run_id: str) -> tuple[dict[str, Any], int] | None:
        resp = self.client.get_item(
            TableName=self.table, Key={"pk": {"S": run_key(run_id)}}, ConsistentRead=True
        )
        item = resp.get("Item")
        if not item or "doc" not in item:
            return None
        return json.loads(item["doc"]["S"]), int(item["version"]["N"])

    def active_runs(self) -> list[tuple[dict[str, Any], int]]:
        out: list[tuple[dict[str, Any], int]] = []
        kwargs: dict[str, Any] = {
            "TableName": self.table,
            "IndexName": ACTIVE_INDEX,
            "KeyConditionExpression": "#a = :a",
            "ExpressionAttributeNames": {"#a": "active"},
            "ExpressionAttributeValues": {":a": {"S": "run"}},
        }
        while True:
            resp = self.client.query(**kwargs)
            for item in resp.get("Items") or []:
                if "doc" in item:
                    out.append((json.loads(item["doc"]["S"]), int(item["version"]["N"])))
            last = resp.get("LastEvaluatedKey")
            if not last:
                return out
            kwargs["ExclusiveStartKey"] = last
