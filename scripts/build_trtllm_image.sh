#!/bin/bash

PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/base.sh

TRTLLM_BACKEND_GIT="https://github.com/triton-inference-server/tensorrtllm_backend.git"
TRTLLM_BACKEND_REV="v0.9.0"
TRTLLM_IMAGE_TAG="ppinfer_triton_trtllm:24.02"
TRTLLM_BACKEND_DIR=
TRTLLM_ENGINE_DIR=

function usage() {
    PRG_NAME=$(basename "${BASH_SOURCE[0]}")
    LOG INFO "$PRG_NAME [options]"
    LOG ""
    LOG INFO "--trtllm_backend \t The path of the Tensorrt-llm backend source code"
    LOG INFO "--revision \t The released revision of trtllm_backend, the default is $TRTLLM_BACKEND_REV"
    LOG INFO "--image_tag \t The tag of the generated docker image, the default is $TRTLLM_IMAGE_TAG"
}

function check_arguments() {
    DOWN_LOADED=0
    if [ x"$TRTLLM_BACKEND_DIR" = x"" ]; then
        TRTLLM_BACKEND_DIR=$CUR_DIR/$RANDOM
        LOG INFO "Set TRTLLM_BACKEND_DIR to random folder under current: $TRTLLM_BACKEND_DIR"
    elif [ ! -d "$TRTLLM_BACKEND_DIR" ]; then
        LOG INFO "The $TRTLLM_BACKEND_DIR does not exist, create it now"
        mkdir $TRTLLM_BACKEND_DIR
    else
        DOWN_LOADED=1
        if [ ! -e "$TRTLLM_BACKEND_DIR/tensorrt_llm/README.md" ] ||
           [ ! -e "$TRTLLM_BACKEND_DIR/tensorrt_llm/3rdparty/cutlass/CMakeLists.txt" ] ||
           [ ! -e "$TRTLLM_BACKEND_DIR/tensorrt_llm/3rdparty/json/CMakeLists.txt" ] ||
           [ ! -e "$TRTLLM_BACKEND_DIR/tensorrt_llm/3rdparty/cxxopts/CMakeLists.txt" ] ||
           [ ! -e "$TRTLLM_BACKEND_DIR/tensorrt_llm/3rdparty/NVTX/CMakeLists.txt" ]; then
            LOG ERR "The $TRTLLM_BACKEND_DIR is not fully downloaded"
        else
            pushd $TRTLLM_BACKEND_DIR
            branches=`git branch`
            popd
            if [[ ! $branches =~ "* $TRTLLM_BACKEND_REV" ]]; then
                LOG ERR "The current revision does not match $TRTLLM_BACKEND_REV"
            fi
        fi
    fi
    if [ $DOWN_LOADED -eq 0 ]; then
        LOG ERR "Downloading trtllm_backend from $TRTLLM_BACKEND_GIT"
        git clone $TRTLLM_BACKEND_GIT -b $TRTLLM_BACKEND_REV $TRTLLM_BACKEND_DIR
        pushd $TRTLLM_BACKEND_DIR
        git submodule update --init --recursive --depth 1
        git lfs install
        git lfs pull
        popd
    fi
    if [ ! -d "$TRTLLM_BACKEND_DIR/$TRTLLM_ENGINE_DIR" ]; then
        LOG ERR "The tensorrt_llm engine dir is not exist in $TRTLLM_BACKEND_DIR, please check it..."
    fi
    name=(${TRTLLM_IMAGE_TAG//:/ })
    images=`docker images $TRTLLM_IMAGE_TAG`
    if [[ $images =~ "${name[0]}" ]]; then
        LOG ERR "The image already exists: $TRTLLM_IMAGE_TAG, no need to create it again"
    fi

}

function build_image() {
    LOG INFO "Build docker image $TRTLLM_IMAGE_TAG from $TRTLLM_BACKEND_DIR"
    pushd $TRTLLM_BACKEND_DIR
    DOCKER_BUILDKIT=1 docker build -t $TRTLLM_IMAGE_TAG --build-arg TRTLLM_ENGINE_DIR=$TRTLLM_ENGINE_DIR -f $CUR_DIR/trtllm/dockerfile/Dockerfile.trt_llm_backend_engine .
    popd
}

function main() {
    if [ x"$1" = x"-h" ] || [ x"$1" = x"--help" ]; then
        usage
        exit
    fi
    while [ "$#" -gt 0 ]; do
    case "$1" in
    --trtllm_backend)
        shift
        TRTLLM_BACKEND_DIR="$1"
        shift
        ;;
    --revision)
        shift
        TRTLLM_BACKEND_REV="$1"
        shift
        ;;
    --image_tag)
        shift
        TRTLLM_IMAGE_TAG="$1"
        shift
        ;;
    --trtllm_engine)
        shift
        TRTLLM_ENGINE_DIR="$1"
        shift
        ;;
    *)
        break
    esac
    done
    check_arguments
    build_image
}

main "$@"

