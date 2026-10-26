#!/usr/bin/env bash
# Leak check: refuse private strings in the repository.
#
# The patterns live OUTSIDE the repository, so this script contains none of them:
#   * every non-empty, non-comment line of ~/.config/rtt/leak-patterns.txt
#     (override the path with RTT_LEAK_PATTERNS);
#   * the values of the resource-ID variables in ~/.config/rtt/nebius.env
#     (names ending in _ID or containing _ID_, NEBIUS_BUCKET_NAME[_<SUFFIX>] and
#     NEBIUS_SECRET_NAME_<SUFFIX>; a registry ID also without its "registry-" prefix, the
#     form it takes in image references; override the file with RTT_NEBIUS_ENV);
#   * every credential in the repository's .env (names containing KEY, TOKEN, SECRET or
#     PASSWORD), and the first 5 characters of each as a key-shape prefix, so a pasted or
#     rotated key of the same format is caught too;
#   * the held-out seed in secrets/heldout_seed.txt (override with RTT_HELDOUT_SEED_FILE),
#     whose SHA-256 is the only form that may be committed before the reveal, and likewise
#     the use-case benchmark v1 secret in secrets/usecase_v1_seed.txt (RTT_USECASE_SEED_FILE).
# Patterns are matched as fixed strings in text and binary files: case-insensitively,
# except the key-shape prefixes, which are matched exactly.
# Hits are reported as file:line plus the pattern's source and index, never the
# pattern or the matching text, so the report itself leaks nothing.
#
# Usage:
#   scripts/leak_check.sh            tracked + untracked (not ignored) files in the work tree
#   scripts/leak_check.sh --staged   the staged versions of files about to be committed
#   scripts/leak_check.sh --history  every commit reachable from any ref
#   scripts/leak_check.sh PATH...    the given files or directories
# Exit status: 0 clean, 1 hits found, 2 setup error (for example, no pattern file).
# Install as pre-commit and pre-push hooks with `make hooks`.
set -euo pipefail

repo="$(git rev-parse --show-toplevel)"
cd "$repo"
patterns_file="${RTT_LEAK_PATTERNS:-$HOME/.config/rtt/leak-patterns.txt}"
nebius_env="${RTT_NEBIUS_ENV:-$HOME/.config/rtt/nebius.env}"

if [[ ! -r "$patterns_file" ]]; then
  echo "leak_check: no pattern file at $patterns_file (set RTT_LEAK_PATTERNS)" >&2
  exit 2
fi

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
chmod 700 "$workdir"
pats="$workdir/patterns"        # case-insensitive
srcs="$workdir/sources"
cs_pats="$workdir/cs_patterns"  # exact case (key-shape prefixes)
cs_srcs="$workdir/cs_sources"
: >"$pats"; : >"$srcs"; : >"$cs_pats"; : >"$cs_srcs"

trim() {
  local p="$1"
  p="${p#"${p%%[![:space:]]*}"}"
  p="${p%"${p##*[![:space:]]}"}"
  printf '%s' "$p"
}

add_pattern() {  # $1 = source label, $2 = pattern
  local p
  p="$(trim "$2")"
  [[ ${#p} -ge 4 ]] || return 0      # very short strings would match everywhere
  printf '%s\n' "$p" >>"$pats"
  printf '%s\n' "$1" >>"$srcs"
}

add_exact() {  # $1 = source label, $2 = pattern (matched case-sensitively)
  local p
  p="$(trim "$2")"
  [[ ${#p} -ge 4 ]] || return 0
  printf '%s\n' "$p" >>"$cs_pats"
  printf '%s\n' "$1" >>"$cs_srcs"
}

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "${line// }" || "$line" == \#* ]] && continue
  add_pattern "leak-patterns.txt" "$line"
done <"$patterns_file"

env_values() {  # $1 = file, $2 = label, $3 = name filter (extended regex)
  [[ -r "$1" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    local name="${BASH_REMATCH[2]}" value="${BASH_REMATCH[3]}"
    [[ "$name" =~ $3 ]] || continue
    value="${value%\"}"; value="${value#\"}"; value="${value%\'}"; value="${value#\'}"
    add_pattern "$2:$name" "$value"
    # a registry ID appears in image references without its "registry-" prefix
    if [[ "$value" == registry-* ]]; then add_pattern "$2:$name path" "${value#registry-}"; fi
    if [[ -n "${4:-}" ]]; then add_exact "$2:$name prefix" "${value:0:5}"; fi
  done <"$1"
}
# per-region variants too (NEBIUS_BUCKET_NAME_UKSOUTH2, and NEBIUS_SECRET_NAME_<SUFFIX>,
# which holds a MysteryBox secret ID when a region references another project's secret)
env_values "$nebius_env" "nebius.env" '(_ID$|_ID_|^NEBIUS_BUCKET_NAME(_[A-Z0-9]+)?$|^NEBIUS_SECRET_NAME_[A-Z0-9]+$)'
env_values "$repo/.env" ".env" '(KEY|TOKEN|SECRET|PASSWORD)' prefix
# the build tools' API keys (api_keys.json, git-ignored): each value, and its first 10 characters
# as a key-shape prefix
if [[ -r "$repo/api_keys.json" ]]; then
  while IFS=$'\t' read -r name value; do
    [[ -n "$value" ]] || continue
    add_pattern "api_keys.json:$name" "$value"
    add_exact "api_keys.json:$name prefix" "${value:0:10}"
  done < <(python3 -c 'import json,sys; [print(k, v, sep="\t") for k, v in json.load(open(sys.argv[1])).items() if isinstance(v, str)]' "$repo/api_keys.json")
fi
# the held-out seed: until the results release reveals it, only its SHA-256 may appear
seed_file="${RTT_HELDOUT_SEED_FILE:-$repo/secrets/heldout_seed.txt}"
if [[ -r "$seed_file" ]]; then add_pattern "heldout seed" "$(head -n 1 "$seed_file")"; fi
# the use-case benchmark v1 secret, under the same rule until its results reveal it
usecase_seed="${RTT_USECASE_SEED_FILE:-$repo/secrets/usecase_v1_seed.txt}"
if [[ -r "$usecase_seed" ]]; then add_pattern "usecase-v1 seed" "$(head -n 1 "$usecase_seed")"; fi

n_patterns=$(( $(wc -l <"$pats") + $(wc -l <"$cs_pats") ))
mode="${1:-}"
hits="$workdir/hits"
: >"$hits"

report() {  # $1 = label, $2 = patterns, $3 = sources, $4 = "ci"|"cs"; stdin: "file:line:text"
  local label="$1" pfile="$2" sfile="$3" mode="$4"
  while IFS= read -r hit; do
    local where="${hit%%:*}" rest="${hit#*:}"
    local lineno="${rest%%:*}" text="${rest#*:}"
    local i=0 src=""
    while IFS= read -r p; do
      i=$((i + 1))
      if [[ "$mode" == ci && "${text,,}" == *"${p,,}"* ]] || [[ "$mode" == cs && "$text" == *"$p"* ]]; then
        src="$(sed -n "${i}p" "$sfile") #$i"
        break
      fi
    done <"$pfile"
    printf '%s%s:%s  [%s]\n' "$label" "$where" "$lineno" "${src:-pattern}" >>"$hits"
  done
}

grep_both() {  # $1 = label prefix; remaining args are passed to grep (files or nothing for stdin)
  local label="$1"
  shift
  if [[ -s "$pats" ]]; then
    grep -F -i -a -n -H -f "$pats" "$@" 2>/dev/null | report "$label" "$pats" "$srcs" ci || true
  fi
  if [[ -s "$cs_pats" ]]; then
    grep -F -a -n -H -f "$cs_pats" "$@" 2>/dev/null | report "$label" "$cs_pats" "$cs_srcs" cs || true
  fi
}

scan_files() {  # args: files to scan in the work tree
  [[ $# -gt 0 ]] || return 0
  grep_both "" -- "$@"
}

case "$mode" in
  --staged)
    staged="$workdir/staged"
    while IFS= read -r -d '' f; do
      [[ "$f" == ".env" ]] && continue
      mkdir -p "$staged/$(dirname "$f")"
      git show ":$f" >"$staged/$f" 2>/dev/null || continue
    done < <(git diff --cached --name-only --diff-filter=ACMR -z)
    if [[ -d "$staged" ]]; then
      (cd "$staged" && mapfile -d '' sf < <(find . -type f -print0) && grep_both "staged " -- "${sf[@]}")
    fi
    ;;
  --history)
    while read -r rev; do
      for pf in ci cs; do
        if [[ $pf == ci ]]; then pfile="$pats"; sfile="$srcs"; flags=(-i); else pfile="$cs_pats"; sfile="$cs_srcs"; flags=(); fi
        [[ -s "$pfile" ]] || continue
        git grep -F "${flags[@]}" -a -n -f "$pfile" "$rev" -- . ':(exclude).env' 2>/dev/null \
          | sed "s|^$rev:||" | report "${rev:0:10} " "$pfile" "$sfile" "$pf" || true
      done
    done < <(git rev-list --all)
    ;;
  "")
    mapfile -d '' files < <(git ls-files -co --exclude-standard -z | grep -z -v -x -F '.env' || true)
    existing=()
    for f in "${files[@]}"; do [[ -f "$f" ]] && existing+=("$f"); done
    scan_files "${existing[@]}"
    ;;
  -h|--help)
    sed -n '2,27p' "$0"
    exit 0
    ;;
  *)
    targets=()
    for t in "$@"; do
      if [[ -d "$t" ]]; then
        while IFS= read -r -d '' f; do targets+=("$f"); done < <(find "$t" -type f ! -name .env -print0)
      else
        targets+=("$t")
      fi
    done
    scan_files "${targets[@]}"
    ;;
esac

if [[ -s "$hits" ]]; then
  sort -u "$hits" >&2
  echo "leak_check: $(sort -u "$hits" | wc -l) hit(s) for $n_patterns private patterns; remove them before committing" >&2
  exit 1
fi
echo "leak_check: clean ($n_patterns private patterns, mode ${mode:-worktree})"
