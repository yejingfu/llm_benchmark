#!/bin/bash

CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/base.sh

IMAGE_TAG="ppinfer_mii:0.1"

function main() {
    LOG INFO "build $IMAGE_TAG"
    name=(${IMAGE_TAG//:/ })
    images=`docker images $IMAGE_TAG`
    if [[ $images =~ "${name[0]}" ]]; then
        LOG ERR "The image already exists: $IMAGE_TAG, no need to create it again"
    fi
    DOCKER_BUILDKIT=1 docker build -t $IMAGE_TAG -f $CUR_DIR/mii/Dockerfile.mii $CUR_DIR/mii
}

main "$@"


