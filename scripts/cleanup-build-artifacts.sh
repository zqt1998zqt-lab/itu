#!/usr/bin/env bash
set -euo pipefail

# Dry-run by default. Nothing is deleted unless --apply is passed.
# This script intentionally uses exact roots and refuses symlink targets.

PROJECT_ROOT="${ITU_PROJECT_ROOT:-/home/zhangqingtian/project/itu}"
BUILD_ROOT="${ITU_BUILD_ROOT:-${PROJECT_ROOT}/build}"
APPLY=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cleanup-build-artifacts.sh [--dry-run]
  bash scripts/cleanup-build-artifacts.sh --apply

Default is --dry-run. --apply is required for any deletion.
The script protects the active CLI build tree, acoustic-qualification,
ns-hybrid-v2, and case directories referenced by source/tests/scripts.
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

require_dir() {
  local path="$1"
  [[ -d "$path" && ! -L "$path" ]] || die "expected directory: $path"
}

require_regular() {
  local path="$1"
  [[ -f "$path" && ! -L "$path" ]] || die "expected regular file: $path"
}

assert_not_protected() {
  local path="$1"
  local protected
  for protected in \
    "$BUILD_ROOT/acoustic-qualification" \
    "$BUILD_ROOT/ns-hybrid-v2-20261115" \
    "$BUILD_ROOT/repair-sequential-20261017/build-tree"; do
    if [[ "$path" == "$protected" || "$path" == "$protected/"* ]]; then
      die "refusing protected path: $path"
    fi
  done
}

remove_path() {
  local path="$1"
  if [[ ! -e "$path" && ! -L "$path" ]]; then
    say "SKIP absent: $path"
    return 0
  fi
  [[ ! -L "$path" ]] || die "refusing symlink: $path"
  assert_not_protected "$path"
  if (( APPLY == 0 )); then
    say "DRY-RUN remove: $path"
    return 0
  fi
  rm -rf -- "$path"
  [[ ! -e "$path" && ! -L "$path" ]] || die "removal did not complete: $path"
  say "REMOVED: $path"
}

remove_regular() {
  local path="$1"
  if [[ ! -e "$path" && ! -L "$path" ]]; then
    say "SKIP absent: $path"
    return 0
  fi
  [[ ! -L "$path" ]] || die "refusing symlink: $path"
  assert_not_protected "$path"
  if (( APPLY == 0 )); then
    say "DRY-RUN remove file: $path"
    return 0
  fi
  rm -f -- "$path"
  [[ ! -e "$path" && ! -L "$path" ]] || die "file removal did not complete: $path"
  say "REMOVED file: $path"
}

assert_not_protected "$BUILD_ROOT/itu_audio_cli"
CLI_TARGET="$(readlink -f "$BUILD_ROOT/itu_audio_cli" 2>/dev/null || true)"
[[ -n "$CLI_TARGET" ]] || die "itu_audio_cli symlink is missing or broken"
case "$CLI_TARGET" in
  "$BUILD_ROOT/repair-sequential-20261017/build-tree/"*) ;;
  *) die "itu_audio_cli points outside the expected active build tree: $CLI_TARGET" ;;
esac

say "Project: $PROJECT_ROOT"
say "Build root: $BUILD_ROOT"
say "Active CLI target: $CLI_TARGET"
say "Mode: $([[ $APPLY -eq 1 ]] && printf APPLY || printf DRY-RUN)"
df -h "$BUILD_ROOT" | sed -n '1,2p'

# Batch 1: exact duplicate package and known transient build cache files.
DUPLICATE_ZIP="$BUILD_ROOT/measurement-aware-v2-repro-20261104T015000Z.zip"
remove_regular "$DUPLICATE_ZIP"

shopt -s nullglob
MDF_CACHE_DIRS=("$BUILD_ROOT"/mdf-yhat-build-*)
shopt -u nullglob
for dir in "${MDF_CACHE_DIRS[@]}"; do
  [[ -d "$dir" && ! -L "$dir" ]] || die "invalid mdf cache directory: $dir"
  while IFS= read -r -d '' file; do
    remove_regular "$file"
  done < <(find "$dir" -maxdepth 1 -type f \
    \( -name '*.gch' -o -name '*.o' -o -name '*.a' \) -print0)
done

# Batch 2: historical experiment directories with no active CLI target.
BATCH2_DIRS=(
  "$BUILD_ROOT/repair-sequential-20261017/context-r43a"
  "$BUILD_ROOT/repair-sequential-20261017/context-r43b"
  "$BUILD_ROOT/ns-backend-comparison-20261018"
  "$BUILD_ROOT/enhancement-chain-matrix-20261027-vcp-unified-raw075"
)
for dir in "${BATCH2_DIRS[@]}"; do
  require_dir "$dir"
  remove_path "$dir"
done

# Batch 3: keep source/test/script-referenced controlled-native cases.
CONTROLLED_ROOT="$BUILD_ROOT/controlled-native-full-20261017"
CASES_ROOT="$CONTROLLED_ROOT/cases"
require_dir "$CONTROLLED_ROOT"
require_dir "$CASES_ROOT"

declare -A KEEP=()
KEEP_NAMES=(
  "coverage-16000-development-9c7b62f0-snr+30"
  "vehicle-16000-development-m2-10"
  "vehicle-16000-holdout-m1-02"
)

if command -v rg >/dev/null 2>&1; then
  while IFS= read -r ref; do
    [[ -n "$ref" ]] || continue
    suffix="${ref#*/cases/}"
    [[ "$suffix" != "$ref" ]] || continue
    name="${suffix%%/*}"
    [[ -n "$name" ]] && KEEP_NAMES+=("$name")
  done < <(
    rg -o --no-filename \
      -g '!cleanup-build-artifacts.sh' \
      "build/controlled-native-full-20261017/cases/[^/[:space:]\"')]+" \
      "$PROJECT_ROOT/src" "$PROJECT_ROOT/scripts" "$PROJECT_ROOT/tests" \
      "$PROJECT_ROOT/server" 2>/dev/null || true
  )
fi

mapfile -t KEEP_NAMES < <(printf '%s\n' "${KEEP_NAMES[@]}" | sort -u)
for name in "${KEEP_NAMES[@]}"; do
  KEEP["$name"]=1
  [[ -d "$CASES_ROOT/$name" && ! -L "$CASES_ROOT/$name" ]] || \
    die "referenced controlled-native case is missing: $CASES_ROOT/$name"
done

say "Controlled-native keep set (${#KEEP_NAMES[@]}):"
printf '  %s\n' "${KEEP_NAMES[@]}"

delete_count=0
keep_count=0
while IFS= read -r -d '' case_dir; do
  name="${case_dir##*/}"
  if [[ -n "${KEEP[$name]+x}" ]]; then
    keep_count=$((keep_count + 1))
    continue
  fi
  remove_path "$case_dir"
  delete_count=$((delete_count + 1))
done < <(find "$CASES_ROOT" -mindepth 1 -maxdepth 1 -type d -print0)

say "Controlled-native cases kept: $keep_count"
say "Controlled-native cases targeted: $delete_count"

if (( APPLY == 0 )); then
  say "DRY-RUN complete; no files were deleted."
else
  say "APPLY complete."
fi
df -h "$BUILD_ROOT" | sed -n '1,2p'
