#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

has_unstaged_changes=0
has_staged_changes=0
has_untracked_files=0

if ! git diff --quiet; then
    has_unstaged_changes=1
fi

if ! git diff --cached --quiet; then
    has_staged_changes=1
fi

if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    has_untracked_files=1
fi

if [[ "$has_unstaged_changes" -eq 0 && "$has_staged_changes" -eq 0 && "$has_untracked_files" -eq 0 ]]; then
    echo "No changes to commit."
    exit 0
fi

message="${*:-snapshot $(date '+%Y-%m-%d %H:%M:%S')}"

git add -A
git commit -m "$message"
