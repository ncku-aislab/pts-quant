#!/bin/bash

# Usage:
# ./scripts/setup_imagenet.sh /path/to/train /path/to/val

TRAIN_ROOT="$1"
VAL_ROOT="$2"
TARGET_ROOT="data/ImageNet-1k"

if [[ -z "$1" || -z "$2" ]]; then
    echo "Usage: $0 /path/to/train /path/to/val"
    exit 1
fi

function create_symlink() {
    # Convert both to absolute paths to ensure the symlink is robust
    local target=$(realpath -m "$1")
    local symlink=$(realpath -sm "$2")

    # Check if $target exists
    if [[ ! -d "$target" ]]; then
        echo "Error: $target doesn't exists."
        exit 1
    fi

    # Check if $symlink exists and is a "real directory" (not a symbolic link)
    if [[ -d "$symlink" && ! -L "$symlink" ]]; then
        echo "$symlink is an existing directory. Skipping."
        return 0
    fi

    mkdir -p "$(dirname "$symlink")"

    ln -snf "$target" "$symlink"
    echo "link created/updated: $symlink -> $target"
}

create_symlink "$TRAIN_ROOT" "$TARGET_ROOT/train"
create_symlink "$VAL_ROOT" "$TARGET_ROOT/val"
