#!/usr/bin/env bash
set -euo pipefail
umask 077

project_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
target="$project_root/.env"
temporary="$(mktemp "${target}.tmp.XXXXXX")"

cleanup() {
  unset OPENROUTER_API_KEY
  rm -f "$temporary"
}
trap cleanup EXIT

printf 'Enter OpenRouter API key (input is hidden): '
IFS= read -r -s OPENROUTER_API_KEY
printf '\n'

if [[ -z "$OPENROUTER_API_KEY" ]]; then
  printf 'No key entered; no files changed.\n' >&2
  exit 1
fi

if [[ -f "$target" ]]; then
  grep -v '^OPENROUTER_API_KEY=' "$target" > "$temporary" || true
fi
printf 'OPENROUTER_API_KEY=%s\n' "$OPENROUTER_API_KEY" >> "$temporary"
chmod 600 "$temporary"
mv "$temporary" "$target"
printf 'Saved OpenRouter key to %s with owner-only permissions.\n' "$target"
