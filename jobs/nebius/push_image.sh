#!/usr/bin/env bash
# Push a locally built image to Nebius Container Registry and record its digest references.
#
# Usage: jobs/nebius/push_image.sh [--region REGION] [LOCAL_TAG]   (default tag: rtt-sim:<short git sha>)
#
# By default the image goes to the home registry (NEBIUS_REGISTRY_ID, in NEBIUS_REGION)
# and to every region that has its own registry (NEBIUS_REGISTRY_ID_<SUFFIX>, e.g.
# NEBIUS_REGISTRY_ID_UKSOUTH2). A job pulls only from a registry in its own project:
# uk-south2 jobs failed to pull from the eu-north1 registry (StartFailed), so uk-south2
# has its own copy. --region pushes to one registry only.
#
# Authentication uses the Nebius CLI as a Docker credential helper inside a throwaway
# DOCKER_CONFIG, so ~/.docker/config.json is never modified and no token is written to
# disk or printed. The pushed references (registry/path@sha256:...) are saved to
# ~/.config/rtt/image.env: RTT_IMAGE for the home region, RTT_IMAGE_<SUFFIX> for the others.
# Other lines of that file are kept. Jobs always run a digest, never a tag.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require NEBIUS_PROFILE NEBIUS_REGISTRY_ID

only=""
if [[ "${1:-}" == "--region" ]]; then
  only="${2:?--region needs a region, e.g. uk-south2}"
  shift 2
fi

head_sha="$(git -C "${RTT_REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
local_tag="${1:-rtt-sim:${head_sha:0:12}}"
docker image inspect "${local_tag}" >/dev/null
# the commit the image was built from (build_image.sh labels it), not the checkout's HEAD
git_sha="$(docker image inspect "${local_tag}" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
[[ -n "${git_sha}" && "${git_sha}" != "<no value>" ]] || git_sha="${head_sha}"

suffix_of() { local r="${1//-/}"; echo "${r^^}"; }

# region -> registry ID: home first, then every region with a NEBIUS_REGISTRY_ID_<SUFFIX>
declare -A registry_of=(["${NEBIUS_REGION}"]="${NEBIUS_REGISTRY_ID}")
order=("${NEBIUS_REGION}")
for region in eu-north1 eu-north2 eu-west1 eu-west2 eu-south1 uk-south1 uk-south2 us-central1 us-north1 me-west1; do
  var="NEBIUS_REGISTRY_ID_$(suffix_of "${region}")"
  if [[ -n "${!var:-}" ]]; then
    [[ -n "${registry_of[${region}]:-}" ]] || order+=("${region}")
    registry_of["${region}"]="${!var}"
  fi
done
if [[ -n "${only}" ]]; then
  [[ -n "${registry_of[${only}]:-}" ]] || {
    echo "error: no registry for ${only} (set NEBIUS_REGISTRY_ID_$(suffix_of "${only}"))" >&2
    exit 1
  }
  order=("${only}")
fi

cfg_dir="$(mktemp -d)"
trap 'rm -rf "${cfg_dir}"' EXIT
cat >"${cfg_dir}/docker-credential-nebius" <<EOF
#!/usr/bin/env bash
exec "${NEBIUS_CLI}" --profile "${NEBIUS_PROFILE}" --no-check-update registry docker-credential "\$@"
EOF
chmod 700 "${cfg_dir}/docker-credential-nebius"
helpers=""
for region in "${order[@]}"; do
  helpers+="${helpers:+, }\"cr.${region}.nebius.cloud\": \"nebius\""
done
printf '{"credHelpers": {%s}}\n' "${helpers}" >"${cfg_dir}/config.json"

image_env="${HOME}/.config/rtt/image.env"
mkdir -p "$(dirname "${image_env}")"
umask 077
touch "${image_env}"

# set_image_env KEY=VALUE...: replace those keys in image.env, keep every other line
set_image_env() {
  local tmp keys=() kv
  for kv in "$@"; do keys+=("${kv%%=*}"); done
  tmp="$(mktemp "${image_env}.XXXXXX")"
  grep -v -E "^(# written by jobs/nebius/push_image.sh|($(IFS='|'; echo "${keys[*]}"))=)" "${image_env}" >"${tmp}" || true
  {
    echo "# written by jobs/nebius/push_image.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ) (local only, never commit)"
    cat "${tmp}"
    printf '%s\n' "$@"
  } >"${tmp}.new"
  mv "${tmp}.new" "${image_env}"
  rm -f "${tmp}"
}

for region in "${order[@]}"; do
  registry_host="cr.${region}.nebius.cloud"
  repo="${registry_host}/${registry_of[${region}]#registry-}/rtt-sim"
  remote_tag="${repo}:${local_tag##*:}"
  docker tag "${local_tag}" "${remote_tag}"
  start=$(date +%s)
  PATH="${cfg_dir}:${PATH}" DOCKER_CONFIG="${cfg_dir}" docker push --quiet "${remote_tag}" >/dev/null
  elapsed=$(($(date +%s) - start))
  ref="$(docker image inspect "${remote_tag}" --format '{{range .RepoDigests}}{{println .}}{{end}}' |
    grep -m1 "^${repo}@sha256:")"
  digest="${ref#*@}"
  if [[ "${region}" == "${NEBIUS_REGION}" ]]; then
    set_image_env "RTT_IMAGE=${ref}" "RTT_IMAGE_DIGEST=${digest}" "RTT_IMAGE_GIT_SHA=${git_sha}"
    saved="RTT_IMAGE"
  else
    saved="RTT_IMAGE_$(suffix_of "${region}")"
    set_image_env "${saved}=${ref}"
  fi
  echo "pushed ${local_tag} to ${region} in ${elapsed} s: digest ${digest} (saved as ${saved})"
done
echo "references saved to ~/.config/rtt/image.env"
