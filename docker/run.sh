NAME=pts-quant
IMAGE=pts-quant:1.0.2
ST_ID=n26120626

docker run \
    -it \
    --name $NAME \
    --hostname docker \
    --gpus all \
    --mount type=bind,src=$(pwd),dst=/workspace \
    --mount type=bind,src=/mnt/hdd02-remapped/$ST_ID/ILSVRC2012/image_dir,dst=/workspace/data/ILSVRC2012 \
    $IMAGE