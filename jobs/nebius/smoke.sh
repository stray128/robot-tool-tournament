#!/usr/bin/env bash
# Smoke job: run the pinned image on Nebius and record its environment (GPU, driver,
# headless OpenGL renderer, Warp CUDA devices, Token Factory key presence) in
# runs/<run_id>/{probe,manifest}.json on the bucket, then fetch it to runs/cloud/.
#
# Usage: jobs/nebius/smoke.sh [extra `python -m rtt.cloud run probe` flags]
#   e.g. jobs/nebius/smoke.sh --dry-run
#        jobs/nebius/smoke.sh --no-warp                 (uks2-cpu, the default target)
#        jobs/nebius/smoke.sh --platform gpu-h200-sxm --cap-usd 5
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# region settings (NEBIUS_*_ID or NEBIUS_*_ID_<SUFFIX>) are checked per target by the CLI
require RTT_IMAGE NEBIUS_PROFILE NEBIUS_TENANT_ID

exec "${RTT_PY}" -m rtt.cloud run probe "$@"
