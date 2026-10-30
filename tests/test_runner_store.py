"""The runner's store: the in-memory store's transactions, and the DynamoDB store's requests and
error mapping against a stubbed client (botocore's Stubber; no network)."""

from __future__ import annotations

import json
from typing import Any

import boto3
import pytest
from botocore.stub import Stubber

from rtt.runner.store import (
    ACTIVE_INDEX,
    Bump,
    Conflict,
    DynamoStore,
    LimitReached,
    MemoryStore,
    RunWrite,
    plan_transaction,
)
from runner_support import FakeDynamoClient

TABLE = "runner-test"


def run(run_id: str = "r1", status: str = "queued") -> dict[str, Any]:
    return {"run_id": run_id, "status": status, "created_at": "2026-10-28T09:00:00Z"}


def test_memory_store_bumps_are_all_or_nothing() -> None:
    s = MemoryStore()
    a = Bump("count#a", "n", 1, 100, 2, "cap_a")
    b = Bump("count#b", "n", 1, 100, 1, "cap_b")
    s.commit([a, b])
    with pytest.raises(LimitReached) as info:
        s.commit([a, b])  # b is full: a must not move either
    assert info.value.bump.cap == "cap_b"
    assert s.numbers("count#a") == {"n": 1, "expires_at": 100}
    s.commit([a])
    with pytest.raises(LimitReached):
        s.commit([a])
    s.commit([Bump("count#a", "n", -1, 100)])  # a release has no limit
    assert s.numbers("count#a")["n"] == 1


def test_memory_store_limit_counts_the_amount() -> None:
    s = MemoryStore()
    s.commit([Bump("spend", "committed", 400, 1, 1000, "spend")])
    with pytest.raises(LimitReached):
        s.commit([Bump("spend", "committed", 700, 1, 1000, "spend")])
    s.commit([Bump("spend", "committed", 600, 1, 1000, "spend")])
    assert s.numbers("spend")["committed"] == 1000


def test_memory_store_run_versions() -> None:
    s = MemoryStore()
    s.commit(run=RunWrite(run(), None, True, 99))
    with pytest.raises(Conflict):
        s.commit(run=RunWrite(run(), None, True, 99))
    rec, version = s.get_run("r1")  # type: ignore[misc]
    assert version == 1 and rec["status"] == "queued"
    s.commit(run=RunWrite(run(status="running"), 1, True, 99))
    with pytest.raises(Conflict):
        s.commit(run=RunWrite(run(status="done"), 1, False, 99))
    assert [r["status"] for r, _ in s.active_runs()] == ["running"]
    s.commit(run=RunWrite(run(status="done"), 2, False, 99))
    assert s.active_runs() == [] and s.get_run("r1")[0]["status"] == "done"  # type: ignore[index]
    assert s.get_run("nope") is None


def _client() -> Any:
    return boto3.client(
        "dynamodb", region_name="ap-south-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )


def test_dynamo_commit_builds_one_update_per_item() -> None:
    """Two bumps on one item (a settlement: committed and settled) become one update with two
    ADDs; DynamoDB refuses two actions on one item in a transaction."""
    client = _client()
    store = DynamoStore(TABLE, client)
    bumps = [
        Bump("count#runs#ip#d", "n", 1, 500, 2, "runs_per_ip_per_day"),
        Bump("spend#nebius#total", "committed", 440000, 900, 10_000_000, "spend_total_usd"),
        Bump("spend#nebius#total", "settled", 5, 950),
    ]
    expected = {
        "TransactItems": [
            {
                "Update": {
                    "TableName": TABLE,
                    "Key": {"pk": {"S": "count#runs#ip#d"}},
                    "UpdateExpression": "ADD #a0 :n0 SET expires_at = :e",
                    "ExpressionAttributeNames": {"#a0": "n"},
                    "ExpressionAttributeValues": {":e": {"N": "500"}, ":n0": {"N": "1"}, ":lim0": {"N": "1"}},
                    "ConditionExpression": "(attribute_not_exists(#a0) OR #a0 <= :lim0)",
                }
            },
            {
                "Update": {
                    "TableName": TABLE,
                    "Key": {"pk": {"S": "spend#nebius#total"}},
                    "UpdateExpression": "ADD #a0 :n0, #a1 :n1 SET expires_at = :e",
                    "ExpressionAttributeNames": {"#a0": "committed", "#a1": "settled"},
                    "ExpressionAttributeValues": {
                        ":e": {"N": "950"},
                        ":n0": {"N": "440000"},
                        ":lim0": {"N": "9560000"},
                        ":n1": {"N": "5"},
                    },
                    "ConditionExpression": "(attribute_not_exists(#a0) OR #a0 <= :lim0)",
                }
            },
            {
                "Put": {
                    "TableName": TABLE,
                    "Item": {
                        "pk": {"S": "run#r1"},
                        "doc": {"S": json.dumps(run(), sort_keys=True)},
                        "version": {"N": "1"},
                        "expires_at": {"N": "99"},
                        "created_at": {"S": "2026-10-28T09:00:00Z"},
                        "active": {"S": "run"},
                    },
                    "ConditionExpression": "attribute_not_exists(pk)",
                }
            },
        ]
    }
    with Stubber(client) as stub:
        stub.add_response("transact_write_items", {}, expected)
        store.commit(bumps, RunWrite(run(), None, True, 99))
        stub.assert_no_pending_responses()


def test_plan_transaction_groups_by_item_and_refuses_what_dynamodb_would() -> None:
    bumps = [
        Bump("spend#day", "committed", -100, 5),
        Bump("spend#total", "committed", -100, 9),
        Bump("spend#day", "settled", 300, 7),
        Bump("spend#day", "committed", 50, 6),  # same attribute twice: summed
    ]
    updates = plan_transaction(bumps)
    assert [u.key for u in updates] == ["spend#day", "spend#total"]
    day = updates[0]
    assert [(a.attr, a.amount) for a in day.adds] == [("committed", -50), ("settled", 300)]
    assert day.expires_at == 7
    with pytest.raises(ValueError, match="same item"):
        plan_transaction([Bump("run#r1", "n", 1, 1)], RunWrite(run(), None, True, 1))
    with pytest.raises(ValueError, match="two capped"):
        plan_transaction([Bump("k", "n", 1, 1, 5, "a"), Bump("k", "n", 1, 1, 5, "b")])
    with pytest.raises(ValueError, match="two capped"):  # one cap per item, so a failure names it
        plan_transaction([Bump("k", "n", 1, 1, 5, "a"), Bump("k", "m", 1, 1, 5, "b")])
    # a cap that can never be met is refused before anything is sent (a cap of 0 disables)
    for bump in (Bump("k", "n", 1, 1, 0, "zero"), Bump("k", "committed", 440_000, 1, 300_000, "small")):
        with pytest.raises(LimitReached) as info:
            plan_transaction([bump])
        assert info.value.bump is bump
    assert plan_transaction([Bump("k", "n", 0, 1)]) == []


@pytest.mark.parametrize("kind", ["memory", "dynamo"])
def test_both_stores_settle_two_attributes_of_one_item_in_one_transaction(kind: str) -> None:
    fake = FakeDynamoClient()
    store: Any = MemoryStore() if kind == "memory" else DynamoStore(TABLE, fake)
    store.commit([Bump("spend#day", "committed", 440_000, 9, 2_000_000, "spend_per_day_usd")])
    store.commit([Bump("spend#day", "committed", -300_000, 9), Bump("spend#day", "settled", 140_000, 9)])
    assert store.numbers("spend#day")["committed"] == 140_000
    assert store.numbers("spend#day")["settled"] == 140_000
    assert fake.rejected_duplicates == 0
    for zero in (Bump("count#x", "n", 1, 1, 0, "zero"), Bump("spend#x", "committed", 5, 1, 4, "tiny")):
        with pytest.raises(LimitReached):
            store.commit([zero])
    assert store.numbers("count#x") == {} and store.numbers("spend#x") == {}


def test_the_fake_client_rejects_two_actions_on_one_item() -> None:
    """Guard on the fake itself: it fails the way DynamoDB does, so the store tests above would
    catch a regression to one action per bump."""
    from botocore.exceptions import ClientError

    fake = FakeDynamoClient()
    op = {
        "Update": {
            "TableName": TABLE,
            "Key": {"pk": {"S": "k"}},
            "UpdateExpression": "ADD #a0 :n0 SET expires_at = :e",
            "ExpressionAttributeNames": {"#a0": "n"},
            "ExpressionAttributeValues": {":n0": {"N": "1"}, ":e": {"N": "1"}},
        }
    }
    with pytest.raises(ClientError, match="multiple operations on one item"):
        fake.transact_write_items(TransactItems=[op, op])
    assert fake.rejected_duplicates == 1


def test_dynamo_maps_cancellations_to_the_failed_bump_or_the_run() -> None:
    client = _client()
    store = DynamoStore(TABLE, client)
    bumps = [Bump("a", "n", 1, 1, 3, "cap_a"), Bump("b", "n", 1, 1, 3, "cap_b")]
    with Stubber(client) as stub:
        stub.add_client_error(
            "transact_write_items",
            service_error_code="TransactionCanceledException",
            http_status_code=400,
            response_meta={},
            modeled_fields={"CancellationReasons": [{"Code": "None"}, {"Code": "ConditionalCheckFailed"}]},
        )
        with pytest.raises(LimitReached) as info:
            store.commit(bumps)
        assert info.value.bump.cap == "cap_b"
        stub.add_client_error(
            "transact_write_items",
            service_error_code="TransactionCanceledException",
            http_status_code=400,
            modeled_fields={"CancellationReasons": [{"Code": "None"}, {"Code": "ConditionalCheckFailed"}]},
        )
        with pytest.raises(Conflict):
            store.commit(bumps[:1], RunWrite(run(), 3, True, 1))


def test_dynamo_update_of_a_run_checks_its_version_and_drops_active() -> None:
    client = _client()
    store = DynamoStore(TABLE, client)
    with Stubber(client) as stub:
        stub.add_response(
            "transact_write_items",
            {},
            {
                "TransactItems": [
                    {
                        "Put": {
                            "TableName": TABLE,
                            "Item": {
                                "pk": {"S": "run#r1"},
                                "doc": {"S": json.dumps(run(status="done"), sort_keys=True)},
                                "version": {"N": "4"},
                                "expires_at": {"N": "7"},
                                "created_at": {"S": "2026-10-28T09:00:00Z"},
                            },
                            "ConditionExpression": "#v = :v",
                            "ExpressionAttributeNames": {"#v": "version"},
                            "ExpressionAttributeValues": {":v": {"N": "3"}},
                        }
                    }
                ]
            },
        )
        store.commit(run=RunWrite(run(status="done"), 3, False, 7))


def test_dynamo_reads() -> None:
    client = _client()
    store = DynamoStore(TABLE, client)
    with Stubber(client) as stub:
        stub.add_response(
            "get_item",
            {"Item": {"pk": {"S": "spend#nebius#total"}, "committed": {"N": "12"}, "settled": {"N": "10"}}},
            {"TableName": TABLE, "Key": {"pk": {"S": "spend#nebius#total"}}, "ConsistentRead": True},
        )
        assert store.numbers("spend#nebius#total") == {"committed": 12, "settled": 10}
        stub.add_response(
            "get_item",
            {"Item": {"pk": {"S": "run#r1"}, "doc": {"S": json.dumps(run())}, "version": {"N": "2"}}},
            {"TableName": TABLE, "Key": {"pk": {"S": "run#r1"}}, "ConsistentRead": True},
        )
        assert store.get_run("r1") == (run(), 2)
        stub.add_response(
            "get_item", {}, {"TableName": TABLE, "Key": {"pk": {"S": "run#r2"}}, "ConsistentRead": True}
        )
        assert store.get_run("r2") is None
        query = {
            "TableName": TABLE,
            "IndexName": ACTIVE_INDEX,
            "KeyConditionExpression": "#a = :a",
            "ExpressionAttributeNames": {"#a": "active"},
            "ExpressionAttributeValues": {":a": {"S": "run"}},
        }
        item = {"doc": {"S": json.dumps(run())}, "version": {"N": "1"}}
        stub.add_response("query", {"Items": [item], "LastEvaluatedKey": {"pk": {"S": "run#r1"}}}, query)
        stub.add_response("query", {"Items": [item]}, {**query, "ExclusiveStartKey": {"pk": {"S": "run#r1"}}})
        assert store.active_runs() == [(run(), 1), (run(), 1)]
