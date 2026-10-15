#!/usr/bin/env bash
# Build the simulation image (docker/Dockerfile.sim) from the repository root.
#
# Usage: jobs/nebius/build_image.sh [TAG]          (default tag: rtt-sim:<short git sha>)
#
# The git SHA and a dirty flag are baked into /app/BUILD_INFO.json so every job manifest
# can name the code it ran. The tag is local only; jobs always reference the pushed digest
# (see push_image.sh).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

git_sha="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
# untracked files count: `COPY . .` puts them in the image
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then git_dirty=true; else git_dirty=false; fi
git_rev="${git_sha}"
[[ "${git_dirty}" == true ]] && git_rev="${git_sha}-dirty"
tag="${1:-rtt-sim:${git_sha:0:12}}"

docker build \
  -f docker/Dockerfile.sim \
  --build-arg GIT_SHA="${git_sha}" \
  --build-arg GIT_DIRTY="${git_dirty}" \
  --build-arg GIT_REV="${git_rev}" \
  -t "${tag}" \
  .

echo "built ${tag} (git ${git_sha:0:12}, dirty=${git_dirty})"
docker image inspect "${tag}" --format 'image id {{.Id}}, size {{.Size}} bytes'
