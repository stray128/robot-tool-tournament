"""Check a synthesized runner template against the rules it must keep (stdlib only).

``python3 check_template.py cdk.out/EffectorLabRunner.template.json``

Fails (exit 1, one line per broken rule) unless: the Function URL is public with CORS for exactly
the allowed origins and both invoke permissions (the second limited to Function URL calls); the
role may read exactly two secrets, over TLS, and touch only the runner's table, its index and the
kill switch; no wildcard action or resource is granted; the table has a TTL, point-in-time
recovery and encryption at rest; the log group keeps logs for a set time and the role may only
write to it (no managed policy); no environment variable looks like it holds a secret; the
function is x86_64 with reserved concurrency; the sweeper schedule targets the function; and a
monthly cost budget mails at least one address.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ORIGINS = ["https://stray128.github.io", "http://localhost:4341"]
SECRET_NAME = re.compile(r"(KEY|TOKEN|PASSWORD|CREDENTIAL)", re.IGNORECASE)
SECRET_IDS = ("RUNNER_TOKENFACTORY_SECRET", "RUNNER_NEBIUS_SECRET")


def _of(resources: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [r for r in resources.values() if r["Type"] == kind]


def check(template: dict[str, Any], origins: list[str] = ORIGINS) -> list[str]:
    res = template["Resources"]
    out: list[str] = []

    urls = _of(res, "AWS::Lambda::Url")
    if len(urls) != 1:
        return ["expected one Function URL"]
    url = urls[0]["Properties"]
    if url.get("AuthType") != "NONE":
        out.append("the Function URL must be public (AuthType NONE)")
    cors = url.get("Cors") or {}
    if sorted(cors.get("AllowOrigins") or []) != sorted(origins):
        out.append(f"CORS origins must be exactly {origins}, got {cors.get('AllowOrigins')}")
    if set(cors.get("AllowMethods") or []) - {"GET", "POST"}:
        out.append("CORS methods must be GET and POST only")
    if cors.get("AllowCredentials"):
        out.append("CORS must not allow credentials")

    perms = [p["Properties"] for p in _of(res, "AWS::Lambda::Permission")]
    if not any(
        p["Action"] == "lambda:InvokeFunctionUrl" and p.get("FunctionUrlAuthType") == "NONE" for p in perms
    ):
        out.append("missing lambda:InvokeFunctionUrl for the public URL")
    if not any(
        p["Action"] == "lambda:InvokeFunction" and p.get("InvokedViaFunctionUrl") is True for p in perms
    ):
        out.append("missing lambda:InvokeFunction limited to Function URL calls")
    for p in perms:
        if (
            p["Principal"] == "*"
            and p["Action"] == "lambda:InvokeFunction"
            and not p.get("InvokedViaFunctionUrl")
        ):
            out.append("a public lambda:InvokeFunction permission is not limited to the Function URL")
        if p["Principal"] == "events.amazonaws.com" and "SourceArn" not in p:
            out.append("the schedule's permission must name its rule")

    secrets_seen = 0
    for policy in _of(res, "AWS::IAM::Policy"):
        for st in policy["Properties"]["PolicyDocument"]["Statement"]:
            actions = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
            resources = st["Resource"] if isinstance(st["Resource"], list) else [st["Resource"]]
            if any(a == "*" or a.endswith(":*") for a in actions):
                out.append(f"wildcard action in {st.get('Sid')}: {actions}")
            if any(r == "*" for r in resources):
                out.append(f"wildcard resource in {st.get('Sid')}")
            if any(a.startswith("secretsmanager:") for a in actions):
                if actions != ["secretsmanager:GetSecretValue"]:
                    out.append(f"secrets: only GetSecretValue, got {actions}")
                secrets_seen += len(resources)
                if (st.get("Condition") or {}).get("Bool", {}).get("aws:SecureTransport") != "true":
                    out.append("secrets must be read over TLS (aws:SecureTransport)")
            if any(a.startswith("logs:") for a in actions):
                if set(actions) - {"logs:CreateLogStream", "logs:PutLogEvents"}:
                    out.append(f"logs: only CreateLogStream and PutLogEvents, got {actions}")
                if len(resources) != 1 or "Fn::GetAtt" not in json.dumps(resources[0]):
                    out.append("logs: the role may write only to the stack's own log group")
            if any(a.startswith("dynamodb:") for a in actions):
                allowed = {"dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:Query"}
                if set(actions) - allowed:
                    out.append(f"dynamodb: unexpected actions {sorted(set(actions) - allowed)}")
    if secrets_seen != 2:
        out.append(f"the role must read exactly two secrets, found {secrets_seen}")

    (table,) = _of(res, "AWS::DynamoDB::Table") or [None]
    if table is None:
        out.append("no table")
    else:
        props = table["Properties"]
        if props.get("BillingMode") != "PAY_PER_REQUEST":
            out.append("the table must be on demand")
        if not (props.get("TimeToLiveSpecification") or {}).get("Enabled"):
            out.append("the table needs a TTL")
        if not (props.get("PointInTimeRecoverySpecification") or {}).get("PointInTimeRecoveryEnabled"):
            out.append("the table needs point-in-time recovery")
        if not (props.get("SSESpecification") or {}).get("SSEEnabled"):
            out.append("the table must name its encryption at rest")
        if table.get("DeletionPolicy") != "Retain":
            out.append("the table (the spend ledger) must be retained on delete")

    groups = _of(res, "AWS::Logs::LogGroup")
    if not groups or any("RetentionInDays" not in g["Properties"] for g in groups):
        out.append("every log group needs a retention period")

    for role in _of(res, "AWS::IAM::Role"):
        if role["Properties"].get("ManagedPolicyArns"):
            out.append("a role carries a managed policy (the function's role must be inline-only)")

    for fn in _of(res, "AWS::Lambda::Function"):
        props = fn["Properties"]
        if int(props.get("ReservedConcurrentExecutions") or 0) < 1:
            out.append("the function needs reserved concurrency (a ceiling on parallel requests)")
        variables = (props.get("Environment") or {}).get("Variables") or {}
        if str(variables.get("RUNNER_LAMBDA_TIMEOUT_S")) != str(props.get("Timeout")):
            out.append("RUNNER_LAMBDA_TIMEOUT_S must equal the function timeout")
        if props.get("Architectures") != ["x86_64"]:
            out.append("the function must be x86_64 (the nebius CLI in the image is)")
        if props.get("Timeout", 0) < 300:
            out.append("the function timeout must cover a plan and its repair call")
        for name, value in ((props.get("Environment") or {}).get("Variables") or {}).items():
            if name in SECRET_IDS:  # the secrets' names or ARNs, never their values
                if not (
                    isinstance(value, str)
                    and value.startswith(("effector-lab/runner/", "arn:aws:secretsmanager:"))
                ):
                    out.append(f"environment variable {name} must name a secret, not hold one")
            elif SECRET_NAME.search(name):
                out.append(f"environment variable {name} looks like a secret")
            if isinstance(value, str) and value.startswith(("sk-", "AKIA")):
                out.append(f"environment variable {name} holds a key")

    budgets = [b["Properties"] for b in _of(res, "AWS::Budgets::Budget")]
    if not budgets:
        out.append("no AWS Budgets budget")
    for b in budgets:
        data = b["Budget"]
        if data.get("TimeUnit") != "MONTHLY" or data.get("BudgetType") != "COST":
            out.append("the budget must be a monthly cost budget")
        emails = [
            s.get("Address")
            for n in b.get("NotificationsWithSubscribers") or []
            for s in n.get("Subscribers") or []
            if s.get("SubscriptionType") == "EMAIL"
        ]
        if not emails or not all(emails):
            out.append("the budget must mail an address")

    rules = _of(res, "AWS::Events::Rule")
    if not any("rate(2 minutes)" in (r["Properties"].get("ScheduleExpression") or "") for r in rules):
        out.append("the sweeper schedule (rate 2 minutes) is missing")
    for r in rules:
        for target in r["Properties"].get("Targets") or []:
            if json.loads(target.get("Input") or "{}") != {"runner_task": "sweep"}:
                out.append("the schedule must send {'runner_task': 'sweep'}")
    return out


def main(argv: list[str]) -> int:
    path = Path(argv[1] if len(argv) > 1 else "cdk.out/EffectorLabRunner.template.json")
    origins = argv[2].split(",") if len(argv) > 2 else ORIGINS
    problems = check(json.loads(path.read_text(encoding="utf-8")), origins)
    for p in problems:
        print(f"FAIL: {p}")
    if not problems:
        print(f"OK: {path.name} keeps every rule")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
