#!/bin/bash
PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/base.sh


### example
### Build a image including llama3-8B
# build_vllm_image.sh --vllm-src </path/to/vllm> --cutlass-src </path/to/cutlass-source> --model-dir </path/to/Meta-Llama-3-8B-Instruct> --model-name llama3_8b --test-data-dir </path/to/dataset> --open-webui-dir=</path/to/webui> --tag ppinfer_vllm_llama3_8b:0.1.0
# for example:
# build_vllm_image.sh --vllm-src /home/ppio/linke/code/vllm --cutlass-src /home/ppio/linke/code/cutlass --model-dir /models/Meta-Llama-3-8B-Instruct --model-name llama3-8b --test-data-dir /models/ShareGPT_Vicuna_unfiltered --open-webui-dir /home/ppio/linke/code/open-webui --tag ppinfer_vllm_llama3_8b:0.1.0

VLLM_SRC_DIR=
CUTLASS_DIR=
MODEL_DIR=()
MODEL_FILE_NAME=
MODEL_NAME=EMPTY
TEST_DATA_DIR=
OPEN_WEBUI_DIR=
BUILD_DEV=prod
TAG=

function usage() {
    LOG INFO "$PRG_NAME [options]"
    LOG INFO "  --tag  The docker image tag we want to build from vLLM source code"
    LOG INFO "  --vllm-src  The source code path to the vLLM, ensure its right revision is checked out"
    LOG INFO "  --cutlass-src(optional)  The source code path to the cutlass, please manually fetch it in advanced from https://github.com/nvidia/cutlass.git"
    LOG INFO "  --model-dir(optional) The path to model folder, if it's set, the model would be built into image"
    LOG INFO "  --model-name(optional) The model name, if it's set, parse from model path"
    LOG INFO "  --test-data-dir(optional) The path to the dataset which is used for benmark testing"
    LOG INFO "  --open-webui-dir(optinal) The path to the source code of Open-WebUI, if not set, the owebui is not built into image. You can downlaod it from https://github.com/open-webui/open-webui"
    LOG INFO "  --dev(optional) Build the image form development, including tests, otherwise build image for production"
    exit
}

function build() {
    if [ ! -d "$VLLM_SRC_DIR" ]; then
        LOG ERR "The VLLM source folder does not exist, please set it by --vllm-src"
    fi
    if [ x"$TAG" = x"" ]; then
        LOG ERR "The tag is null, please set by --tag"
    fi
    local docker_file=Dockerfile
    local use_open_webui=0
    local use_local_cutlass=0

    arr=(${TAG//:/ })
    len=${#arr[@]}
    if [ $len -ne 2 ]; then
        LOG ERR "Wrong image tag format, should be 'name:revision'"
    fi
    docker images $TAG | grep ${arr[0]}
    if [ $? -eq 0 ]; then
        LOG ERR "The image $TAG already exist, please remove it firstly"
    fi

    if [[ x"$CUTLASS_DIR" != x"" ]] && [[ -d "$CUTLASS_DIR" ]]; then
        use_local_cutlass=1
        LOG INFO "Use local cutlass: $CUTLASS_DIR"
        pushd $CUTLASS_DIR
            ret=`git log -1 --oneline | grep 7d49e6c7`
            if [ x"$ret" == x"" ]; then
                LOG ERR "Incorrect cutlass source commit, please checkout 7d49e6c7"
            fi
        popd
    fi

    LOG INFO "Now build image $TAG from vLLM source: $VLLM_SRC_DIR"
    pushd $VLLM_SRC_DIR
        if [[ x"$OPEN_WEBUI_DIR" != x"" ]] && [[ -d "$OPEN_WEBUI_DIR" ]]; then
            LOG INFO "Build Open-WebUI into image: $OPEN_WEBUI_DIR"
            use_open_webui=1
            docker_file=Dockerfile_with_owebui.ppio
            cp -rf $OPEN_WEBUI_DIR open-webui
            cp $CUR_DIR/vllm/start_webui_and_vllm.sh ./
            cp $CUR_DIR/vllm/$docker_file ./
        fi
        mkdir ppio_models
        for path in ${MODEL_DIR[@]}; do
            if [ ! -d "$path" ]; then
                LOG ERR "The model folder does not exist: $path"
            fi
            if [ x"$MODEL_FILE_NAME" = x"" ]; then
                MODEL_FILE_NAME="${path##*/}"
            fi
            cp -r $v ppio_models/
        done
        if [ x"$MODEL_FILE_NAME" != x"" ]; then
            git am $CUR_DIR/vllm/0001-Bake-model-weight-files-into-docker-image.patch
        else
            rm -rf ppio_models
        fi

        if [[ ! x"$TEST_DATA_DIR" = x"" ]] && [[ -d "$TEST_DATA_DIR" ]]; then
            mkdir benchmark_ppio
            mkdir -p benchmark_ppio/scripts benchmark_ppio/test
            cp -rf $TEST_DATA_DIR benchmark_ppio/
            cp $CUR_DIR/base.sh benchmark_ppio/scripts/
            cp $CUR_DIR/../test/launch_benchmark_v2.sh benchmark_ppio/test/
            cp $CUR_DIR/../test/async_request_sender.py benchmark_ppio/test/
            cp $CUR_DIR/../test/benchmark_client.py benchmark_ppio/test/
            cp $CUR_DIR/../test/dataset_sampler.py benchmark_ppio/test/
            cp $CUR_DIR/../test/llm_api_demo.py benchmark_ppio/test/
        fi
        if [ $use_local_cutlass -eq 1 ]; then
            cp -r $CUTLASS_DIR cutlass
            git am $CUR_DIR/vllm/0001-Use-local-cutlass-code.patch
        fi
        DOCKER_BUILDKIT=1 docker build --build-arg torch_cuda_arch_list="8.9 9.0+PTX" --build-arg MODEL_DIR=MODEL_FILE_NAME --build-arg MODEL_NAME=$MODEL_NAME --build-arg ENVIRONMENT=$BUILD_DEV -t $TAG -f $docker_file .
        if [ -d ppio_models ]; then
            rm -rf ppio_models
        fi
        if [ -d open-webui ];then
            rm -rf open-webui
        fi
        if [ -d benchmark_ppio ];then
            rm -rf benchmark_ppio
        fi
        if [ -d cutlass ];then
            rm -rf cutlass
        fi
        if [ -f start_webui_and_vllm.sh ]; then
            rm start_webui_and_vllm.sh
        fi
        if [ -f Dockerfile_with_owebui.ppio ]; then
            rm Dockerfile_with_owebui.ppio
        fi
        if [ $use_local_cutlass -eq 1 ]; then
            git reset --hard HEAD^
        fi
        if [ x"$MODEL_FILE_NAME" != x"" ]; then
            git reset --hard HEAD^
        fi
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
    --cutlass-src)
        shift
        CUTLASS_DIR="$1"
        shift
        ;;
    --model-dir)
        shift
        MODEL_DIR+=("$1")
        shift
        ;;
    --model-name)
        shift
        MODEL_NAME="$1"
        shift
        ;;
    --test-data-dir)
        shift
        TEST_DATA_DIR="$1"
        shift
        ;;
    --open-webui-dir)
        shift
        OPEN_WEBUI_DIR="$1"
        shift
        ;;
    --dev)
        shift
        BUILD_DEV=dev
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

