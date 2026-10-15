#!/usr/bin/env bash
# Gate G1 on Nebius: run `rtt.cloud.entry m1` in the pinned image (CPU MuJoCo), writing
# replays, MP4s, report.json and manifest.json to runs/m1-<UTC timestamp>/ on the bucket,
# then wait, record the spend, fetch the run to runs/cloud/ and verify it.
#
# Usage: jobs/nebius/m1.sh [extra `python -m rtt.cloud run m1` flags]
#   e.g. jobs/nebius/m1.sh --dry-run
#        jobs/nebius/m1.sh --no-video --cap-usd 3
#        jobs/nebius/m1.sh --preemptible
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require RTT_IMAGE NEBIUS_PROFILE NEBIUS_PROJECT_ID NEBIUS_SUBNET_ID NEBIUS_BUCKET_ID NEBIUS_BUCKET_NAME

exec "${RTT_PY}" -m rtt.cloud run m1 "$@"
