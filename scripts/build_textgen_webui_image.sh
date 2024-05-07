#!/bin/bash

PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/base.sh

IMAGE_TAG="ppinfer_textgen_webui:0.1"
WEBUI_REVISION="snapshot-2024-04-28"

function usage() {
    LOG INFO "$PRG_NAME [options]"
    LOG INFO "  --image     The docker image tag, default is $IMAGE_TAG"
    LOG INFO "  --revision  The revision of the text-generation-webui, default is $WEBUI_REVISION"
    exit
}

function build() {
    LOG INFO "Build text-generate-webui image $IMAGE_TAG from souce code with revision $WEBUI_REVISION"
    DOCKER_BUILDKIT=1 docker build -t $IMAGE_TAG --build-arg VERSION_TAG=$WEBUI_REVISION -f $CUR_DIR/textgen_webui/Dockerfile $CUR_DIR
    LOG INFO "DONE"
}

function main() {
    if [ x"$1" = x"-h" ] || [ x"$1" = x"--help" ]; then
        usage
        exit
    fi
    while [ "$#" -gt 0 ]; do
    case "$1" in
    --image)
        shift
        IMAGE_TAG="$1"
        shift
        ;;
    --revision)
        shift
        WEBUI_REVISION="$1"
        shift
        ;;
    *)
        break
    esac
    done
    build
}

main "$@"

