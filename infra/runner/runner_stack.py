"""The hosted runner's AWS stack: one Lambda (container image) behind a Function URL.

* **Function URL** (auth ``NONE``, HTTPS only, buffered): planning calls take 17 to 40 s, past API
  Gateway's 29 s integration limit. CORS allows the site and its local dev server only.
* **Lambda** (container image from ``infra/runner/Dockerfile``): the runner (:mod:`rtt.runner`),
  the repository's dependencies and a pinned ``nebius`` CLI. No secret in the image or the
  environment: the two secrets are read at run time, and the role may read exactly those two.
* **DynamoDB** (on demand, TTL on every item, point-in-time recovery, retained on delete): daily
  counters, per-IP limits, run records and the spend ledger.
* **SSM parameter** ``/effector-lab/runner/kill-switch`` ("off"; "on" stops new plans and runs
  within about 30 s, and the next sweep, up to 2 minutes later, cancels the jobs in flight). The
  stack owns the value: a stack update can set it back to "off", so re-set it after a deploy.
* **EventBridge schedule** every two minutes: polls runs in flight, cancels jobs past the wall-time
  limit and settles their cost.
* **Logs**: the function's log group keeps 14 days; the role may write only to it.
* **Reserved concurrency** (default 10): at most 10 requests run at once, so a flood cannot run up
  Lambda time. The account needs at least 100 unreserved concurrency left after it (AWS's rule).
* **AWS Budgets**: a monthly cost budget (default $20) on the resources tagged
  ``project=effector-lab``, mailing ``alertEmail`` at 80 % actual and 100 % forecast.

The physics never runs on AWS: each run is one Nebius Serverless Job.

Context (``-c key=value`` at deploy time; nothing account-specific is committed):

* ``nebiusProjectId``, ``nebiusSubnetId``, ``nebiusBucketId``, ``nebiusBucketName``, ``jobImage``
  (``cr.uk-south2.nebius.cloud/<registry>/rtt-sim@sha256:...``): the Nebius job. While any is
  empty the runner plans but answers ``POST /runs`` with 503.
* ``tokenFactorySecretArn``, ``nebiusSecretArn``: the full secret ARNs (recommended); without
  them the secrets are found by name and the policy names ``<name>-??????``.
* ``secretsKmsKeyArn``: only if the secrets use a customer managed KMS key.
* ``jobPreset``: the ``cpu-d3`` preset (default ``32vcpu-128gb``; a smaller one such as
  ``4vcpu-16gb`` makes a cheap smoke test, and the caps price it the same way).
* ``alertEmail`` (REQUIRED, no default): where the budget alerts go.
* ``budgetUsd`` (default 20), ``budgetScope`` (``tag``, the default, or ``account``: the tag scope
  needs ``project`` activated as a cost allocation tag in the Billing console).
* ``reservedConcurrency`` (default 10; ``none`` reserves nothing).
* ``allowedOrigins`` (comma-separated), ``autoOff`` (``YYYY-MM-DD``), ``logRetentionDays``, and any
  cap as ``cap.<name>`` (``cap.spend_total_usd=10``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from aws_cdk import (
    ArnFormat,
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    Tags,
)
from aws_cdk import aws_budgets as budgets
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_ssm as ssm
from constructs import Construct

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORIGINS = "https://stray128.github.io,http://localhost:4341"
TOKENFACTORY_SECRET = "effector-lab/runner/tokenfactory"
NEBIUS_SECRET = "effector-lab/runner/nebius-sa"
KILL_SWITCH_PARAM = "/effector-lab/runner/kill-switch"
LAMBDA_TIMEOUT_S = 360
ACTIVE_INDEX = "active-runs"
CAPS = (
    "free_plans_per_day",
    "free_plans_per_ip_per_day",
    "keyed_plans_per_day",
    "keyed_plans_per_ip_per_day",
    "runs_per_ip_per_day",
    "spend_per_day_usd",
    "spend_total_usd",
    "plan_budget_usd",
    "job_wall_limit_s",
    "cancel_grace_s",
)
#: Everything in the repository the Lambda image does not need (the build context is the repo).
IMAGE_EXCLUDE = [
    ".git",
    ".venv",
    "**/.venv",
    "**/__pycache__",
    "**/*.pyc",
    ".env",
    ".env.*",
    "secrets",
    "runs",
    "outputs",
    "results",
    "data",
    "web",
    "tests",
    "docs",
    "jobs",
    "scripts",
    "Makefile",
    "infra/runner/cdk.out",
    "infra/runner/node_modules",
    ".hypothesis",
    ".pytest_cache",
    ".ruff_cache",
    "*.mp4",
]


def _ctx(scope: Construct, key: str, default: str = "") -> str:
    value = scope.node.try_get_context(key)
    return default if value is None else str(value)


def _secret_arn(stack: Stack, arn: str, name: str) -> str:
    """The secret's ARN for the policy: the full ARN when given, else ``<name>-??????`` (Secrets
    Manager appends six random characters to a name)."""
    if arn:
        return arn
    return stack.format_arn(
        service="secretsmanager",
        resource="secret",
        resource_name=f"{name}-??????",
        arn_format=ArnFormat.COLON_RESOURCE_NAME,
    )


class RunnerStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs: Any) -> None:
        super().__init__(scope, construct_id, **kwargs)

        token_arn = _ctx(self, "tokenFactorySecretArn")
        nebius_arn = _ctx(self, "nebiusSecretArn")
        kms_arn = _ctx(self, "secretsKmsKeyArn")
        origins = [o.strip() for o in _ctx(self, "allowedOrigins", DEFAULT_ORIGINS).split(",") if o.strip()]
        retention_days = int(_ctx(self, "logRetentionDays", "14"))
        reserved_raw = _ctx(self, "reservedConcurrency", "10").strip().lower()
        reserved = None if reserved_raw in ("none", "off", "") else int(reserved_raw)
        if reserved is not None and reserved < 1:
            raise ValueError("reservedConcurrency must be at least 1 (0 would stop the function); use 'none'")
        alert_email = _ctx(self, "alertEmail").strip()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", alert_email):
            raise ValueError("pass -c alertEmail=<address> (required: where the budget alerts go)")
        budget_usd = float(_ctx(self, "budgetUsd", "20"))
        budget_scope = _ctx(self, "budgetScope", "tag")
        if budget_scope not in ("tag", "account"):
            raise ValueError("budgetScope must be 'tag' or 'account'")
        preset = _ctx(self, "jobPreset", "32vcpu-128gb")
        if not re.fullmatch(r"(2|4|8|16|32|48|64|96|128|160|192|224|256)vcpu-\d+gb", preset):
            raise ValueError(f"jobPreset {preset!r} is not a cpu-d3 preset such as 32vcpu-128gb")

        # ------------------------------------------------------------ state
        table = dynamodb.Table(
            self,
            "State",
            partition_key=dynamodb.Attribute(name="pk", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="expires_at",
            encryption=dynamodb.TableEncryption.AWS_MANAGED,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True
            ),
            removal_policy=RemovalPolicy.RETAIN,
        )
        table.add_global_secondary_index(
            index_name=ACTIVE_INDEX,
            partition_key=dynamodb.Attribute(name="active", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="created_at", type=dynamodb.AttributeType.STRING),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        kill_switch = ssm.StringParameter(
            self,
            "KillSwitch",
            parameter_name=KILL_SWITCH_PARAM,
            string_value="off",
            description="Effector Lab runner kill switch: on stops new plans and runs",
        )

        # ------------------------------------------------------------ function
        log_group = logs.LogGroup(
            self,
            "Logs",
            retention=logs.RetentionDays(retention_days)
            if retention_days in {d.value for d in logs.RetentionDays}
            else logs.RetentionDays.TWO_WEEKS,
            removal_policy=RemovalPolicy.DESTROY,
        )

        env = {
            "RUNNER_TABLE_NAME": table.table_name,
            "RUNNER_KILL_SWITCH_PARAM": kill_switch.parameter_name,
            "RUNNER_ALLOWED_ORIGINS": ",".join(origins),
            "RUNNER_EMIT_CORS": "false",
            "RUNNER_AUTO_OFF": _ctx(self, "autoOff", "2026-12-15"),
            "RUNNER_TOKENFACTORY_SECRET": token_arn or TOKENFACTORY_SECRET,
            "RUNNER_NEBIUS_SECRET": nebius_arn or NEBIUS_SECRET,
            "RUNNER_NEBIUS_PROJECT_ID": _ctx(self, "nebiusProjectId"),
            "RUNNER_NEBIUS_SUBNET_ID": _ctx(self, "nebiusSubnetId"),
            "RUNNER_NEBIUS_BUCKET_ID": _ctx(self, "nebiusBucketId"),
            "RUNNER_NEBIUS_BUCKET_NAME": _ctx(self, "nebiusBucketName"),
            "RUNNER_NEBIUS_IMAGE": _ctx(self, "jobImage"),
            "RUNNER_NEBIUS_PRESET": preset,
            "RUNNER_NEBIUS_CLI": "/opt/nebius/nebius",
            "RUNNER_LAMBDA_TIMEOUT_S": str(LAMBDA_TIMEOUT_S),
        }
        for cap in CAPS:
            value = _ctx(self, f"cap.{cap}")
            if value:
                env[f"RUNNER_{cap.upper()}"] = value

        # the function's own role: no managed policy, logs only to its own group
        role = iam.Role(
            self,
            "RunnerRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Effector Lab runner - its log group, table, kill switch and two secrets",
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="Logs",
                actions=["logs:CreateLogStream", "logs:PutLogEvents"],
                resources=[log_group.log_group_arn],
            )
        )

        fn = lambda_.DockerImageFunction(
            self,
            "Runner",
            description="Effector Lab hosted runner - plans with the Task Engineer, runs searches on Nebius",
            code=lambda_.DockerImageCode.from_image_asset(
                str(REPO_ROOT),
                file="infra/runner/Dockerfile",
                exclude=IMAGE_EXCLUDE,
                platform=ecr_assets.Platform.LINUX_AMD64,
            ),
            architecture=lambda_.Architecture.X86_64,
            memory_size=1536,
            timeout=Duration.seconds(LAMBDA_TIMEOUT_S),
            environment=env,
            log_group=log_group,
            role=role,
            reserved_concurrent_executions=reserved,
        )

        # least privilege: the table and its index, the kill switch, exactly two secrets
        fn.add_to_role_policy(
            iam.PolicyStatement(
                sid="State",
                actions=[
                    "dynamodb:GetItem",
                    "dynamodb:PutItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:Query",
                ],
                resources=[table.table_arn, f"{table.table_arn}/index/{ACTIVE_INDEX}"],
            )
        )
        fn.add_to_role_policy(
            iam.PolicyStatement(
                sid="KillSwitch", actions=["ssm:GetParameter"], resources=[kill_switch.parameter_arn]
            )
        )
        fn.add_to_role_policy(
            iam.PolicyStatement(
                sid="Secrets",
                actions=["secretsmanager:GetSecretValue"],
                resources=[
                    _secret_arn(self, token_arn, TOKENFACTORY_SECRET),
                    _secret_arn(self, nebius_arn, NEBIUS_SECRET),
                ],
                conditions={"Bool": {"aws:SecureTransport": "true"}},
            )
        )
        if kms_arn:
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    sid="SecretsKey",
                    actions=["kms:Decrypt"],
                    resources=[kms_arn],
                    conditions={
                        "StringEquals": {"kms:ViaService": f"secretsmanager.{self.region}.amazonaws.com"}
                    },
                )
            )

        url = fn.add_function_url(
            auth_type=lambda_.FunctionUrlAuthType.NONE,
            invoke_mode=lambda_.InvokeMode.BUFFERED,
            cors=lambda_.FunctionUrlCorsOptions(
                allowed_origins=origins,
                allowed_methods=[lambda_.HttpMethod.GET, lambda_.HttpMethod.POST],
                allowed_headers=["content-type"],
                max_age=Duration.minutes(10),
            ),
        )

        # the sweeper
        rule = events.Rule(
            self,
            "Sweep",
            description="Effector Lab runner - poll runs in flight, cancel overdue jobs, settle spend",
            schedule=events.Schedule.rate(Duration.minutes(2)),
        )
        rule.add_target(
            targets.LambdaFunction(fn, event=events.RuleTargetInput.from_object({"runner_task": "sweep"}))
        )

        # a monthly cost budget with email alerts
        cost_filters = {"TagKeyValue": ["user:project$effector-lab"]} if budget_scope == "tag" else None
        budgets.CfnBudget(
            self,
            "Budget",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_type="COST",
                time_unit="MONTHLY",
                budget_limit=budgets.CfnBudget.SpendProperty(amount=budget_usd, unit="USD"),
                cost_filters=cost_filters,
            ),
            notifications_with_subscribers=[
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        comparison_operator="GREATER_THAN",
                        notification_type=kind,
                        threshold=threshold,
                        threshold_type="PERCENTAGE",
                    ),
                    subscribers=[
                        budgets.CfnBudget.SubscriberProperty(address=alert_email, subscription_type="EMAIL")
                    ],
                )
                for kind, threshold in (("ACTUAL", 80), ("FORECASTED", 100))
            ],
        )

        Tags.of(self).add("project", "effector-lab")
        Tags.of(self).add("component", "hosted-runner")

        CfnOutput(self, "RunnerUrl", value=url.url, description="Function URL of the runner")
        CfnOutput(self, "TableName", value=table.table_name)
        CfnOutput(self, "KillSwitchParameter", value=kill_switch.parameter_name)
        CfnOutput(self, "FunctionName", value=fn.function_name)
        CfnOutput(self, "LogGroupName", value=log_group.log_group_name)
