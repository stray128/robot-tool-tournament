#!/usr/bin/env bash
# Build the simulation image (docker/Dockerfile.sim).
#
# Usage: jobs/nebius/build_image.sh [--ref GIT_REF] [TAG]   (default tag: rtt-sim:<short git sha>)
#
#   no --ref     build from the working tree as it is now; uncommitted or untracked files
#                make the build "dirty" (git_dirty: true in every manifest).
#   --ref HEAD   build from a clean export of one commit (`git archive`): the image holds
#                exactly that commit, and nothing half-written in the working tree. Use
#                this for scored jobs.
#
# The git SHA and a dirty flag are baked into /app/BUILD_INFO.json so every job manifest
# can name the code it ran. The tag is local only; jobs always reference the pushed digest
# (see push_image.sh).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

ref=""
if [[ "${1:-}" == "--ref" ]]; then
  ref="${2:?--ref needs a git ref, e.g. HEAD}"
  shift 2
fi

context="${repo_root}"
if [[ -n "${ref}" ]]; then
  git_sha="$(git rev-parse --verify "${ref}^{commit}")"
  git_dirty=false
  git_rev="${git_sha}"
  context="$(mktemp -d)"
  trap 'rm -rf "${context}"' EXIT
  # tracked files of that commit only: no .env, no runs/, no untracked work in progress
  git archive --format=tar "${git_sha}" | tar -x -C "${context}"
else
  git_sha="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  # untracked files count: `COPY . .` puts them in the image
  if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then git_dirty=true; else git_dirty=false; fi
  git_rev="${git_sha}"
  [[ "${git_dirty}" == true ]] && git_rev="${git_sha}-dirty"
fi
tag="${1:-rtt-sim:${git_sha:0:12}}"

docker build \
  -f "${context}/docker/Dockerfile.sim" \
  --build-arg GIT_SHA="${git_sha}" \
  --build-arg GIT_DIRTY="${git_dirty}" \
  --build-arg GIT_REV="${git_rev}" \
  -t "${tag}" \
  "${context}"

echo "built ${tag} (git ${git_sha:0:12}, dirty=${git_dirty}${ref:+, from ${ref}})"
docker image inspect "${tag}" --format 'image id {{.Id}}, size {{.Size}} bytes'
