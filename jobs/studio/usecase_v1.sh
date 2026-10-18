#!/usr/bin/env bash
# Use-case benchmark v1 of the Studio (product mode, not part of the pre-registered experiment):
# scripts/studio_usecase_v1.py with the Nebius account settings loaded from outside the repository
# (jobs/nebius/env.sh). Protocol: docs/USECASE_V1.md.
#
# Usage: jobs/studio/usecase_v1.sh COMMAND [flags]
#   freeze-arms [--executor nebius|local|fake] [--reuse-b DIR] [--reuse-c DIR] [--refreeze]
#                                                development search (or reused reports, verified on
#                                                this code); writes the frozen arms
#   commit-secret                                draws the held-out secret, commits its SHA-256
#   finalize                                     checks everything, sets status: frozen
#   evaluate [--executor nebius|fake] [--out DIR] after the tag: held-out runs, parity runs, analysis
#   analyse [--results DIR]                      re-run the analysis of a results folder
#   fetch --run-id ID --out DIR                  download a fan-out run's bucket prefix (recovery)
#   dry-run --out DIR [--heroes all] [--reuse-b DIR] [--reuse-c DIR]
#                                                the whole pipeline on a throwaway seed, local fake cloud
#   estimate --measured DIR...                   the cost table from measured fan-out runs
#
# A Nebius executor (the default for freeze-arms and evaluate) needs RTT_IMAGE (push_image.sh),
# the profile and the uk-south2 settings; the CLI checks the region's settings per target.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../nebius/env.sh"

cmd="${1:?usage: jobs/studio/usecase_v1.sh COMMAND [flags]}"
needs_nebius=false
case "${cmd}" in
  freeze-arms | evaluate)
    [[ " $* " == *" --executor local "* || " $* " == *" --executor fake "* ]] || needs_nebius=true
    ;;
  fetch) needs_nebius=true ;;
esac
if [[ "${needs_nebius}" == true ]]; then
  require RTT_IMAGE NEBIUS_PROFILE NEBIUS_TENANT_ID
fi

export PYTHONPATH="${RTT_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${RTT_PY}" "${RTT_REPO_ROOT}/scripts/studio_usecase_v1.py" "$@"
