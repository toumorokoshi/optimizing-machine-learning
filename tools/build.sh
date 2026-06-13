#!/bin/bash
set -e

# Find mdbook_wrapper.sh in RUNFILES_DIR or locally, following symlinks
WRAPPER=""
if [ -n "$RUNFILES_DIR" ]; then
    WRAPPER=$(find -L "$RUNFILES_DIR" -name mdbook_wrapper.sh -perm -u+x | head -n 1)
fi
if [ -z "$WRAPPER" ]; then
    WRAPPER=$(find -L . -name mdbook_wrapper.sh -perm -u+x | head -n 1)
fi
if [ -z "$WRAPPER" ]; then
    WRAPPER=$(find -L .. -name mdbook_wrapper.sh -perm -u+x | head -n 1)
fi

if [ -z "$WRAPPER" ]; then
    echo "Error: mdbook_wrapper.sh not found."
    exit 1
fi

WRAPPER_ABS=$(realpath "$WRAPPER")
exec "$WRAPPER_ABS" build "$@"
