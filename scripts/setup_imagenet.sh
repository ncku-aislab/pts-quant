#!/bin/bash

# Usage:
# bash scripts/setup_imagenet.sh /path/to/train /path/to/val

TRAIN_ROOT=$1
VAL_ROOT=$2
TARGET_ROOT="data/ImageNet-1k"

if [ -z "$TRAIN_ROOT" ] || [ -z "$VAL_ROOT" ]; then
    echo "Usage: bash scripts/setup_imagenet.sh /path/to/train /path/to/val"
    exit 1
fi

mkdir -p "$TARGET_ROOT"

# Remove old symbolic links
if [ -L "$TARGET_ROOT/train" ]; then
    rm "$TARGET_ROOT/train"
fi

if [ -L "$TARGET_ROOT/val" ]; then
    rm "$TARGET_ROOT/val"
fi

ln -s "$TRAIN_ROOT" "$TARGET_ROOT/train"
ln -s "$VAL_ROOT" "$TARGET_ROOT/val"

echo "Symbolic links created:"
echo "$TARGET_ROOT/train -> $TRAIN_ROOT"
echo "$TARGET_ROOT/val   -> $VAL_ROOT"