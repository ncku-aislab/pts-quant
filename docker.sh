docker run \
    -it \
    --hostname docker \
    --gpus all \
    --mount type=bind,src=$(pwd),dst=/workspace \
    pts-quant:1.0.1 