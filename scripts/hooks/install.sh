#!/usr/bin/env bash
# Install this repo's hooks into .git/hooks (which git does not version).
set -eu
cd "$(dirname "$0")/../.."
for h in scripts/hooks/pre-commit; do
    n=$(basename "$h")
    cp "$h" ".git/hooks/$n"
    chmod +x ".git/hooks/$n"
    echo "installed .git/hooks/$n"
done
