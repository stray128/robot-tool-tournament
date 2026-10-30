# syntax=docker/dockerfile:1
#
# Lambda image of the hosted runner (rtt.runner.lambda_handler.handler).
#
# Build context: the repository root (the CDK stack passes it with this file and an exclude
# list, so web/, tests/, docs/, data/, runs/ and every .env stay out). Contents:
#   * the AWS Lambda Python 3.12 base image (the interpreter the repository pins);
#   * the repository's runtime dependencies, exactly as uv.lock pins them (no dev tools);
#   * the repository itself under ${LAMBDA_TASK_ROOT}/app (src/, configs/, assets/), so code
#     that resolves files from the repository root finds them;
#   * the nebius CLI, pinned by version and checked against its SHA-256.
# No secret is baked in: the function reads its two secrets from Secrets Manager at run time.
#
# Local build (what `cdk deploy` does for you):
#   docker build --platform linux/amd64 -f infra/runner/Dockerfile -t effector-lab-runner .

ARG LAMBDA_BASE=public.ecr.aws/lambda/python:3.12
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.21

FROM ${UV_IMAGE} AS uv

FROM ${LAMBDA_BASE} AS nebius
# The release the repository's jobs were run with (docs/NEBIUS.md); the hash is of
# https://artifacts.nebius.cloud/cli/release/0.12.291/linux/x86_64/nebius.
ARG NEBIUS_CLI_VERSION=0.12.291
ARG NEBIUS_CLI_SHA256=a00fd5d26f4b3e09ad907e197b8573b4bf22cfe43ac5be468470ba82d1520450
RUN python3 - <<'PY'
import hashlib, os, urllib.request
version = os.environ["NEBIUS_CLI_VERSION"]
url = f"https://artifacts.nebius.cloud/cli/release/{version}/linux/x86_64/nebius"
data = urllib.request.urlopen(url, timeout=120).read()
digest = hashlib.sha256(data).hexdigest()
if digest != os.environ["NEBIUS_CLI_SHA256"]:
    raise SystemExit(f"nebius CLI {version}: sha256 {digest} does not match the pinned hash")
os.makedirs("/opt/nebius", exist_ok=True)
with open("/opt/nebius/nebius", "wb") as fh:
    fh.write(data)
os.chmod("/opt/nebius/nebius", 0o755)
PY

FROM ${LAMBDA_BASE}

COPY --from=nebius /opt/nebius/nebius /opt/nebius/nebius

# Dependencies first (cached until uv.lock changes), installed next to the handler.
COPY pyproject.toml uv.lock ${LAMBDA_TASK_ROOT}/app/
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    cd ${LAMBDA_TASK_ROOT}/app \
 && uv export --frozen --no-dev --no-emit-project --no-hashes -o /tmp/requirements.txt \
 && uv pip install --no-cache --python /var/lang/bin/python3.12 --target ${LAMBDA_TASK_ROOT} \
      -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt

# The repository (the build context's exclude list keeps everything else out).
COPY src ${LAMBDA_TASK_ROOT}/app/src
COPY configs ${LAMBDA_TASK_ROOT}/app/configs
COPY assets ${LAMBDA_TASK_ROOT}/app/assets
COPY README.md LICENSE NOTICE ${LAMBDA_TASK_ROOT}/app/
RUN python3 -m compileall -q ${LAMBDA_TASK_ROOT}/app/src

# HOME and MPLCONFIGDIR point at /tmp, Lambda's only writable path; the CLI keeps its profile
# under its own private HOME (rtt.runner.nebius).
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}/app/src \
    RUNNER_NEBIUS_CLI=/opt/nebius/nebius \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1

CMD ["rtt.runner.lambda_handler.handler"]
