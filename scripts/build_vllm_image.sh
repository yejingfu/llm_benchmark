#!/bin/bash

PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/base.sh

VLLM_SRC_DIR=
TAG=

function usage() {
    LOG INFO "$PRG_NAME [options]"
    LOG INFO "  --vllm-src  The source code path to the vLLM, ensure its right revision is checked out"
    LOG INFO "  --tag  The docker image tag we want to build from vLLM source code"
    exit
}

function build() {
    if [ ! -d "$VLLM_SRC_DIR" ]; then
        LOG ERR "The VLLM source folder does not exist, please set it by --vllm-src"
    fi
    if [ x"$TAG" = x"" ]; then
        LOG ERR "The tag is null, please set by --tag"
    fi
    arr=(${TAG//:/ })
    len=${#arr[@]}
    if [ $len -ne 2 ]; then
        LOG ERR "Wrong image tag format, should be 'name:revision'"
    fi

    docker images $TAG | grep ${arr[0]}
    if [ $? -eq 0 ]; then
        LOG ERR "The image $TAG already exist, please remove it firstly"
    fi

    LOG INFO "Now build image $TAG from vLLM source: $VLLM_SRC_DIR"
    pushd $VLLM_SRC_DIR
    DOCKER_BUILDKIT=1 docker build --build-arg torch_cuda_arch_list="8.9 9.0+PTX" -t $TAG -f Dockerfile .
    popd
    LOG INFO "Complete to build the vllm image: $TAG"
}

function main() {
    if [ "$#" -eq 0 ];then
        usage
    fi
    if [ x"$1" = x"-h" ] || [ x"$1" = x"--help" ]; then
        usage
    fi
    while [ "$#" -gt 0 ]; do
    case "$1" in
    --vllm-src)
        shift
        VLLM_SRC_DIR="$1"
        shift
        ;;
    --tag)
        shift
        TAG="$1"
        shift
        ;;
    *)
        usage
    esac
    done

    build
}

main "$@"

