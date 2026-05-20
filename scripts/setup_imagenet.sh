#!/bin/bash

# Usage:
# bash scripts/setup_imagenet.sh /path/to/train /path/to/val

TRAIN_ROOT=$(realpath "$1")
VAL_ROOT=$(realpath "$2")

TARGET_ROOT="data/ImageNet-1k"

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: bash scripts/setup_imagenet.sh /path/to/train /path/to/val"
    exit 1
fi

if [ ! -d "$TRAIN_ROOT" ]; then
    echo "Error: training dataset path does not exist:"
    echo "$TRAIN_ROOT"
    exit 1
fi

if [ ! -d "$VAL_ROOT" ]; then
    echo "Error: validation dataset path does not exist:"
    echo "$VAL_ROOT"
    exit 1
fi

mkdir -p "$TARGET_ROOT"

# Handle existing train path
if [ -e "$TARGET_ROOT/train" ]; then
    if [ -L "$TARGET_ROOT/train" ]; then
        rm "$TARGET_ROOT/train"
    else
        echo "Error: $TARGET_ROOT/train already exists and is not a symbolic link."
        exit 1
    fi
fi

# Handle existing val path
if [ -e "$TARGET_ROOT/val" ]; then
    if [ -L "$TARGET_ROOT/val" ]; then
        rm "$TARGET_ROOT/val"
    else
        echo "Error: $TARGET_ROOT/val already exists and is not a symbolic link."
        exit 1
    fi
fi

ln -s "$TRAIN_ROOT" "$TARGET_ROOT/train"
ln -s "$VAL_ROOT" "$TARGET_ROOT/val"

echo "Symbolic links created:"
echo "$TARGET_ROOT/train -> $TRAIN_ROOT"
echo "$TARGET_ROOT/val   -> $VAL_ROOT"