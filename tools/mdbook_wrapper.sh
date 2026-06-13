#!/bin/bash
set -e

# Find the mdbook binary in the runfiles directory, following symlinks
MDBOOK_BIN=""
if [ -n "$RUNFILES_DIR" ]; then
    MDBOOK_BIN=$(find -L "$RUNFILES_DIR" -name mdbook -perm -u+x | head -n 1)
fi
if [ -z "$MDBOOK_BIN" ]; then
    MDBOOK_BIN=$(find -L . -name mdbook -perm -u+x | head -n 1)
fi
if [ -z "$MDBOOK_BIN" ]; then
    MDBOOK_BIN=$(find -L .. -name mdbook -perm -u+x | head -n 1)
fi

if [ -z "$MDBOOK_BIN" ]; then
    echo "Error: mdbook binary not found in runfiles."
    exit 1
fi

# Make it an absolute path
MDBOOK_BIN_ABS=$(realpath "$MDBOOK_BIN")

# Change directory to the workspace source directory if running under 'bazel run'
if [ -n "$BUILD_WORKSPACE_DIRECTORY" ]; then
    cd "$BUILD_WORKSPACE_DIRECTORY"
fi

# Execute mdbook with the passed arguments
exec "$MDBOOK_BIN_ABS" "$@"
