# Shared setup for jobs/nebius/*.sh; source it, do not run it.
#
# Loads the account settings that stay outside the repository:
#   $RTT_NEBIUS_ENV (default ~/.config/rtt/nebius.env)  NEBIUS_PROFILE, NEBIUS_*_ID, ...
#   ~/.config/rtt/image.env                              RTT_IMAGE (written by push_image.sh)
# Variables already set in the environment win over both files.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RTT_REPO_ROOT="${repo_root}"

_rtt_load_env_file() {
  local file="$1" line key value
  [[ -f "${file}" ]] || return 0
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line#export }"
    [[ "${line}" =~ ^[[:space:]]*(#|$) ]] && continue
    key="${line%%=*}"
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [[ -n "${!key:-}" ]] && continue # the environment wins
    value="${line#*=}"
    if [[ "${value}" =~ ^\"(.*)\"$ || "${value}" =~ ^\'(.*)\'$ ]]; then value="${BASH_REMATCH[1]}"; fi
    export "${key}=${value}"
  done <"${file}"
}

_rtt_load_env_file "${RTT_NEBIUS_ENV:-${HOME}/.config/rtt/nebius.env}"
_rtt_load_env_file "${HOME}/.config/rtt/image.env"

export NEBIUS_CLI="${NEBIUS_CLI:-${HOME}/.nebius/bin/nebius}"
export NEBIUS_REGION="${NEBIUS_REGION:-eu-north1}"
export RTT_PY="${RTT_PY:-${repo_root}/.venv/bin/python}"

# require VAR...: fail with the names (never the values) of unset variables.
require() {
  local missing=() var
  for var in "$@"; do
    [[ -n "${!var:-}" ]] || missing+=("${var}")
  done
  if ((${#missing[@]})); then
    echo "error: set ${missing[*]} (environment or ~/.config/rtt/nebius.env)" >&2
    return 1
  fi
}
