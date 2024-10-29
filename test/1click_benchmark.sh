#!/bin/bash
PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/util.sh

declare -A DEF_MODEL_HF_NAMES
DEF_MODEL_HF_NAMES["l31-8b"]="yejingfu/Meta-Llama-3.1-8B-Instruct"
DEF_MODEL_HF_NAMES["l31-8b-fp8"]="yejingfu/NousResearch-Meta-Llama-3.1-8B-Instruct-FP8"
DEF_MODEL_HF_NAMES["l31-70b"]="yejingfu/Meta-Llama-3.1-70B-Instruct"
DEF_MODEL_HF_NAMES["l31-70b-fp8"]="yejingfu/Meta-Llama-3.1-70B-Instruct-FP8"
DEF_GPU_TYPES=("4090" "h100" "h20" "h800" "a100" "a800")
DEF_TOKENIZER_HF_NAME="yejingfu/Meta-Llama-3.1-8B-Instruct"
DEF_DS_NAME="ShareGPT_V3_unfiltered_cleaned_split.json"
DEF_DS_HF_PATH="datasets/yejingfu/ShareGPT_V3/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

BM_DOCKER_IMAGE="image.paigpu.com/library/ppinfer_vllm:0.6.2.2"
BM_GPU_TYPE=
BM_GPU_IDS=
BM_MODEL_NAME=
BM_MODEL_DIR=
BM_TOKENIZER_DIR=
BM_TEST_STRENGTH="high"

if [[ x"$HF_ENDPOINT" = x"" ]];then
    HF_ENDPOINT="https://huggingface.co"
fi

function usage() {
    LOG INFO "$PRG_NAME [options]"
    LOG INFO "  --gpu-type The type of GPU, can be ${DEF_GPU_TYPES[@]}"
    LOG INFO "  --gpu-ids The list of GPU IDs, sperated by comma, e.g. 0,1,2,3"
    LOG INFO "  --model-name The model short name, can be ${!DEF_MODEL_HF_NAMES[@]}"
    LOG INFO "  --model-dir (optional) The model folder path, if not set, download from huggingface"
    LOG INFO "  --tokenizer-dir (optional) Tht tokenizer folder path, if not set, download from huggingface: $DEF_TOKENIZER_HF_NAME and save to $CUR_DIR/tokenizer"
    LOG INFO "  --docker-image (optional) The docker image name, if not set, use default image: $BM_DOCKER_IMAGE"
    LOG INFO "  --test-strength The test strength level, can be: low, middle, high, default is high"
    exit
}

function download_model() {
    local hf_name=$1
    local dir=$2
    if dpkg-query -W -f='${Status}' git-lfs 2>/dev/null | grep -q "install ok installed"; then
        LOG INFO "git-lfs is installed"
    else
        LOG INFO "install git-lfs"
        apt-get install -y git git-lfs
    fi
    if [ ! -f "$dir/config.json" ]; then
        if [ -d "$dir" ]; then
            LOG WARN "Delete the folder $dir"
            rm -rf $dir
        fi
        LOG INFO "Create model foler $dir"
        mkdir -p $dir
    fi
    LOG INFO "Download model from huggingface $hf_name and save to $dir"
    GIT_LFS_SKIP_SMUDGE=1 git clone $HF_ENDPOINT/$hf_name $dir
    pushd $dir
    git lfs pull
    popd
    LOG INFO "Completed HF model downloading"
}

function get_hf_model_name() {
    local short_name=$1
    local ret="null"
    for key in "${!DEF_MODEL_HF_NAMES[@]}"; do
        if [[ $short_name == $key ]]; then
            ret=${DEF_MODEL_HF_NAMES[$key]}
            break
        fi
    done
    echo $ret
}

function main() {
    if [ "$#" -eq 0 ]; then
        usage
    fi
    while [ "$#" -gt 0 ]; do
    case "$1" in
    --gpu-type)
        shift
        BM_GPU_TYPE="$1"
        shift
        ;;
    --gpu-ids)
        shift
        BM_GPU_IDS="$1"
        shift
        ;;
    --model-name)
        shift
        BM_MODEL_NAME="$1"
        shift
        ;;
    --model-dir)
        shift
        BM_MODEL_DIR="$1"
        shift
        ;;
    --tokenizer-dir)
        shift
        BM_TOKENIZER_DIR="$1"
        shift
        ;;
    --docker-image)
        shift
        BM_DOCKER_IMAGE="$1"
        shift
        ;;
    --test-strength)
        shift
        BM_TEST_STRENGTH="$1"
        shift
        ;;
    *)
        usage
        break
    esac
    done
    LOG INFO "BM_GPU_TYPE: $BM_GPU_TYPE, BM_GPU_IDS: $BM_GPU_IDS, BM_MODEL_NAME: $BM_MODEL_NAME, BM_DOCKER_IMAGE: $BM_DOCKER_IMAGE, BM_MODEL_DIR: $BM_MODEL_DIR"
    if [ x"$BM_GPU_TYPE" = x"" ];then
        LOG ERR "Empty gpu type, please set by --gpu-type"
    fi
    if ! contains_value "$BM_GPU_TYPE" "${DEF_GPU_TYPES[@]}"; then
        LOG ERR "Invalid GPU type $BM_GPU_TYPE, should set by --gpu-type and its value shoud be in ${DEF_GPU_TYPES[@]}"
    fi
    if [ x"$BM_MODEL_NAME" = x"" ]; then
        LOG ERR "Empty model name, please set by --model-name"
    fi
    if ! contains_value "$BM_MODEL_NAME" "${!DEF_MODEL_HF_NAMES[@]}"; then
        LOG ERR "Invalid model name $BM_MODEL_NAME, should set by --model-name and its value shoud be in ${!DEF_MODEL_HF_NAMES[@]}"
    fi
    if [ x"$BM_GPU_IDS" = x"" ]; then
        LOG ERR "Empty gpu ids, please set by --gpu-ids"
    fi
    if [ x"$BM_MODEL_DIR" = x"" ]; then
        BM_MODEL_DIR="$CUR_DIR/$BM_MODEL_NAME"
        LOG INFO "Set model dir path to $BM_MODEL_DIR"
    fi
    if [[ -f "$BM_MODEL_DIR/config.json" ]]; then
        LOG INFO "Use the existing model folder $BM_MODEL_DIR"
    else
        local hf_model_name=$(get_hf_model_name $BM_MODEL_NAME)
        if [[ x"$hf_model_name" = x"null" ]]; then
            LOG ERR "Invalid model name, the --model-name should be in ${!DEF_MODEL_HF_NAMES[@]}"
        fi
        download_model $hf_model_name $BM_MODEL_DIR
    fi
    if [[ x"$BM_TOKENIZER_DIR" = x"" ]];then
        BM_TOKENIZER_DIR=$CUR_DIR/tokenizer
        LOG INFO "Set tokenizer to $BM_TOKENIZER_DIR"
    fi
    if [[ -f "$BM_TOKENIZER_DIR/config.json" ]]; then
        LOG INFO "Use the existing tokenizer $BM_TOKENIZER_DIR"
    else
        download_model $DEF_TOKENIZER_HF_NAME $BM_TOKENIZER_DIR
    fi
    if [[ ! -f "$CUR_DIR/$DEF_DS_NAME" ]]; then
        LOG INFO "Downloading dataset from: $HF_ENDPOINT/$DEF_DS_HF_PATH"
        wget $HF_ENDPOINT/$DEF_DS_HF_PATH
    fi
    num_gpus=$(count_numbers $BM_GPU_IDS)
    local log_file_path="_gpu_${num_gpus}x${BM_GPU_TYPE}_model_${BM_MODEL_NAME}_$RANDOM.txt"
    local port=$((18000+RANDOM%100))
    server_args="--image-name $BM_DOCKER_IMAGE --model-served-name $BM_MODEL_NAME --model-dir $BM_MODEL_DIR --gpu-ids $BM_GPU_IDS --listen-port $port"
    client_args="--endpoint http://localhost:$port/v1 --tokenizer $BM_TOKENIZER_DIR --dataset $CUR_DIR/$DEF_DS_NAME --log-file $log_file_path --print-raw"
    if [[ x"$BM_TEST_STRENGTH" = x"low" ]];then
        client_args="$client_args --context-lens 1000,3000,5000 --batches 1,2,4,8"
    elif [[ x"$BM_TEST_STRENGTH" = x"middle" ]];then
        client_args="$client_args --context-lens 1000,3000,5000,6000 --batches 1,2,4,8,10"
    else
        client_args="$client_args --context-lens 1000,3000,5000,6000,10000 --batches 1,2,3,4,5,6,7,8,9,10,12,15"
    fi
    LOG INFO "[RUN]: $CUR_DIR/launch_benchmark.sh $server_args $client_args"
    $CUR_DIR/launch_benchmark.sh $server_args $client_args
}

main "$@"

