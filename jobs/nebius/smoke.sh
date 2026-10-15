#!/usr/bin/env bash
# Smoke job: run the pinned image on Nebius and record its environment (GPU, driver,
# headless OpenGL renderer, Warp CUDA devices, Token Factory key presence) in
# runs/<run_id>/{probe,manifest}.json on the bucket, then fetch it to runs/cloud/.
#
# Usage: jobs/nebius/smoke.sh [extra `python -m rtt.cloud run probe` flags]
#   e.g. jobs/nebius/smoke.sh --dry-run
#        jobs/nebius/smoke.sh --platform gpu-h200-sxm --cap-usd 5
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require RTT_IMAGE NEBIUS_PROFILE NEBIUS_PROJECT_ID NEBIUS_SUBNET_ID NEBIUS_BUCKET_ID NEBIUS_BUCKET_NAME

exec "${RTT_PY}" -m rtt.cloud run probe "$@"
