VER=1.0.2
DIR=$(realpath $(dirname $(dirname $0)))

docker build \
    -t pts-quant:$VER \
    $DIR