"""CDK app of the hosted runner (``docs/RUNNER.md``; the stack is in ``runner_stack.py``).

The account and region come from the deploying credentials (``CDK_DEFAULT_ACCOUNT`` and
``CDK_DEFAULT_REGION``) or from ``-c account=... -c region=...``; neither is written here.
"""

from __future__ import annotations

import os

import aws_cdk as cdk
from runner_stack import RunnerStack

app = cdk.App()
account = app.node.try_get_context("account") or os.environ.get("CDK_DEFAULT_ACCOUNT")
region = app.node.try_get_context("region") or os.environ.get("CDK_DEFAULT_REGION")
RunnerStack(
    app,
    "EffectorLabRunner",
    env=cdk.Environment(account=account, region=region) if account and region else None,
    description="Effector Lab hosted runner - Lambda Function URL, DynamoDB state, Nebius jobs",
)
app.synth()
