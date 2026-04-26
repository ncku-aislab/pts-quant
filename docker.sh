#!/bin/bash

VER=0.3
NAME=pts-quant
PROJECT_DIR=$(realpath $(dirname $0))
DATASET_DIR=$(realpath $(dirname $1))

build() {
    docker build \
        -t pts-quant:$VER \
        $PROJECT_DIR
}

run() {
    local status=$(docker inspect -f '{{.State.Status}}' $NAME 2> /dev/null)

    case "$status" in
        "")
            docker run \
                -it \
                --name $NAME \
                --hostname docker \
                --gpus all \
                --net=host \
                --mount type=bind,src=$PROJECT_DIR,dst=/workspace \
                --mount type=bind,src=$DATASET_DIR,dst=/workspace/data/ILSVRC2012 \
                $IMAGE
            ;;
        "running")
            docker attach $NAME
            ;;
        "created" | "exited" | "pause")
            docker start -ai $NAME
            ;;
        *)
            echo "Invalid status $status"
            ;;
    esac
}

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 [build | run]"
else
    $@
fi
