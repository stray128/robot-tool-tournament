#!/usr/bin/env bash
# Push a locally built image to Nebius Container Registry and record its digest reference.
#
# Usage: jobs/nebius/push_image.sh [LOCAL_TAG]     (default: rtt-sim:<short git sha>)
#
# Authentication uses the Nebius CLI as a Docker credential helper inside a throwaway
# DOCKER_CONFIG, so ~/.docker/config.json is never modified and no token is written to
# disk or printed. The pushed reference (registry/path@sha256:...) is saved to
# ~/.config/rtt/image.env as RTT_IMAGE; jobs always run that digest, never a tag.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require NEBIUS_PROFILE NEBIUS_REGISTRY_ID

git_sha="$(git -C "${RTT_REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
local_tag="${1:-rtt-sim:${git_sha:0:12}}"
registry_host="cr.${NEBIUS_REGION}.nebius.cloud"
repo="${registry_host}/${NEBIUS_REGISTRY_ID#registry-}/rtt-sim"
remote_tag="${repo}:${local_tag##*:}"

docker image inspect "${local_tag}" >/dev/null

cfg_dir="$(mktemp -d)"
trap 'rm -rf "${cfg_dir}"' EXIT
cat >"${cfg_dir}/docker-credential-nebius" <<EOF
#!/usr/bin/env bash
exec "${NEBIUS_CLI}" --profile "${NEBIUS_PROFILE}" --no-check-update registry docker-credential "\$@"
EOF
chmod 700 "${cfg_dir}/docker-credential-nebius"
printf '{"credHelpers": {"%s": "nebius"}}\n' "${registry_host}" >"${cfg_dir}/config.json"

docker tag "${local_tag}" "${remote_tag}"
start=$(date +%s)
PATH="${cfg_dir}:${PATH}" DOCKER_CONFIG="${cfg_dir}" docker push --quiet "${remote_tag}" >/dev/null
elapsed=$(($(date +%s) - start))

ref="$(docker image inspect "${remote_tag}" --format '{{range .RepoDigests}}{{println .}}{{end}}' |
  grep -m1 "^${repo}@sha256:")"
digest="${ref#*@}"

image_env="${HOME}/.config/rtt/image.env"
mkdir -p "$(dirname "${image_env}")"
umask 077
{
  echo "# written by jobs/nebius/push_image.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ) (local only, never commit)"
  echo "RTT_IMAGE=${ref}"
  echo "RTT_IMAGE_DIGEST=${digest}"
  echo "RTT_IMAGE_GIT_SHA=${git_sha}"
} >"${image_env}"

echo "pushed ${local_tag} in ${elapsed} s"
echo "digest ${digest}"
echo "reference saved to ~/.config/rtt/image.env (RTT_IMAGE)"
