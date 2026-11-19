#!/usr/bin/env bash
set -euo pipefail

# Dry-run by default. Use --apply only after reviewing the candidate list.
# Targets are limited to historical replay/probe directories under
# repair-sequential-20261017.

PROJECT_ROOT="${ITU_PROJECT_ROOT:-/home/zhangqingtian/project/itu}"
REPAIR_ROOT="${ITU_REPAIR_ROOT:-${PROJECT_ROOT}/build/repair-sequential-20261017}"
APPLY=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cleanup-repair-history.sh [--dry-run]
  bash scripts/cleanup-repair-history.sh --apply

Default is --dry-run. --apply is required for deletion.
Protected paths:
  repair-sequential-20261017/build-tree
  repair-sequential-20261017/ns-r10-rnnoise
  repair-sequential-20261017/final-r16-experimental
EOF
}

for arg in "$@"; do
  case "$arg" in
    --dry-run) APPLY=0 ;;
    --apply) APPLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$arg" >&2; usage >&2; exit 2 ;;
  esac
done

say() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ -d "$REPAIR_ROOT" && ! -L "$REPAIR_ROOT" ]] || \
  die "missing or unsafe repair root: $REPAIR_ROOT"

reference_is_used() {
  local needle="$1"
  if command -v rg >/dev/null 2>&1; then
    rg -q -F "$needle" \
      "$PROJECT_ROOT/src" "$PROJECT_ROOT/server" \
      "$PROJECT_ROOT/scripts" "$PROJECT_ROOT/tests" 2>/dev/null
    return $?
  fi
  command -v grep >/dev/null 2>&1 || die "rg or grep is required for reference checks"
  grep -R -F -q -- "$needle" \
    "$PROJECT_ROOT/src" "$PROJECT_ROOT/server" \
    "$PROJECT_ROOT/scripts" "$PROJECT_ROOT/tests" 2>/dev/null
}

assert_not_protected() {
  local path="$1"
  local protected
  for protected in \
    "$REPAIR_ROOT/build-tree" \
    "$REPAIR_ROOT/ns-r10-rnnoise" \
    "$REPAIR_ROOT/final-r16-experimental"; do
    if [[ "$path" == "$protected" || "$path" == "$protected/"* ]]; then
      die "refusing protected path: $path"
    fi
  done
}

candidate_dirs=()
while IFS= read -r -d '' path; do
  [[ -d "$path" && ! -L "$path" ]] || \
    die "candidate is not a regular directory: $path"
  assert_not_protected "$path"
  candidate_dirs+=("$path")
done < <(
  find "$REPAIR_ROOT" -mindepth 1 -maxdepth 1 -type d \
    \( -name '*-replay' \
       -o -name 'context-r*-probes' \
       -o -name '_archived-meta-20261022' \
       -o -name 'context-r43a-v5' \
       -o -name 'context-r43b-v5' \) \
    -print0
)

if (( ${#candidate_dirs[@]} == 0 )); then
  say "No matching historical directories found."
  exit 0
fi

for path in "${candidate_dirs[@]}"; do
  name="$(basename "$path")"
  if reference_is_used "build/repair-sequential-20261017/$name"; then
    die "candidate is referenced by project code: $path"
  fi
done

say "Repair root: $REPAIR_ROOT"
say "Mode: $([[ $APPLY -eq 1 ]] && printf APPLY || printf DRY-RUN)"
say "Candidates: ${#candidate_dirs[@]}"

total_bytes=0
for path in "${candidate_dirs[@]}"; do
  bytes="$(du -sb "$path" | awk '{print $1}')"
  total_bytes=$((total_bytes + bytes))
  if (( APPLY == 0 )); then
    say "DRY-RUN remove: $path"
  else
    rm -rf -- "$path"
    [[ ! -e "$path" && ! -L "$path" ]] || die "removal did not complete: $path"
    say "REMOVED: $path"
  fi
done

printf 'Candidate size: %.2f GiB\n' \
  "$(awk -v bytes="$total_bytes" 'BEGIN{print bytes/1073741824}')"

if (( APPLY == 0 )); then
  say "DRY-RUN complete; no files were deleted."
else
  say "APPLY complete."
fi
df -h "$REPAIR_ROOT" | sed -n '1,2p'
