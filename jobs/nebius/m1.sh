#!/usr/bin/env bash
# Gate G1 on Nebius: run `rtt.cloud.entry m1` in the pinned image (CPU MuJoCo), writing
# replays, MP4s, report.json and manifest.json to runs/m1-<UTC timestamp>/ on the bucket,
# then wait, record the spend, fetch the run to runs/cloud/ and verify it.
#
# The default target is uks2-cpu (CPU-only cpu-d3, uk-south2); without capacity there the
# job moves to eun1-gpu (H100, eu-north1).
#
# Usage: jobs/nebius/m1.sh [extra `python -m rtt.cloud run m1` flags]
#   e.g. jobs/nebius/m1.sh --dry-run
#        jobs/nebius/m1.sh --no-video --cap-usd 3
#        jobs/nebius/m1.sh --target eun1-gpu --preemptible
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# region settings (NEBIUS_*_ID or NEBIUS_*_ID_<SUFFIX>) are checked per target by the CLI
require RTT_IMAGE NEBIUS_PROFILE NEBIUS_TENANT_ID

exec "${RTT_PY}" -m rtt.cloud run m1 "$@"
