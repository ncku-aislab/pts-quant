NAME=pts-quant
IMAGE=pts-quant:1.0.2

docker run \
    -it \
    --name $NAME \
    --hostname docker \
    --gpus all \
    --mount type=bind,src=$(pwd),dst=/workspace \
    $IMAGE