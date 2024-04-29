#!/bin/bash

CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/../scripts/base.sh

BM_IMAGE_NAME=
BM_CONTAINER_NAME=
BM_MODEL_DIR=
BM_TRT_ENGINE_DIR=
BM_TEST_DATA_DIR=
BM_BACKEND=
BM_CUDA_DEVICES=0,1,2,3,4,5,6,7
BM_PORT=8000
BM_TP=8
BM_PP=1
BM_MEM_FRACTION=0.9
BM_MAX_NUM_SEQ=512
BM_MAX_SEQ_LEN=4096
BM_MAX_BATCHED_TOKENS=4096
BM_MAX_TOKENS_FOR_CUDA_GRAPH=512
BM_DRY_RUN=0
BM_DTYPE=auto

BM_WARMUP_REQS=32
BM_NORM_REQS=512
BM_CONCURRENT_REQS=32
BM_SAMPLING_POLICY="fixed"
BM_INPUT_LEN=1024
BM_OUTPUT_LEN=1024

PRESET_BACKEND=("trtllm" "vllm" "mii" "siliconllm" "tgi")
WORLD_SIZE=$(expr $BM_TP \* $BM_PP)
NUM_GPUS=8

function launch_and_run() {
    LOG INFO "Make sure the docker container is not running: $BM_CONTAINER_NAME"
    if [ $BM_DRY_RUN -eq 0 ];then
        docker stop $BM_CONTAINER_NAME
        docker rm -f $BM_CONTAINER_NAME
    fi

    LOG INFO "launch docker container $BM_CONTAINER_NAME from image $BM_IMAGE_NAME and the backend is $BM_BACKEND"
    LOG INFO ""
    opts="-d --gpus all --privileged --ipc=host --net=host --ulimit stack=67108864 --ulimit memlock=-1 -e HTTPS_PROXY= -e HTTP_PROXY= -e ALL_PROXY= -e https_proxy= -e http_proxy= -e all_proxy= -e CUDA_VISIBLE_DEVICES=$BM_CUDA_DEVICES --name $BM_CONTAINER_NAME "
    cmd=""

    log_file=$(date|tr -d ' ')
    client_cmd="--backend $BM_BACKEND --model $BM_MODEL_DIR --tokenizer $BM_MODEL_DIR --dataset $BM_TEST_DATA_DIR --port $BM_PORT --num-warmup-requests $BM_WARMUP_REQS --num-benchmark-requests $BM_NORM_REQS --max-concurrent-requests $BM_CONCURRENT_REQS "
    client_cmd="$client_cmd --stream --pad-requests --warn-dismatch-output-len --gpus $NUM_GPUS "
    client_cmd="$client_cmd --sampling-policy $BM_SAMPLING_POLICY --fixed_prompt_len $BM_INPUT_LEN --fixed_output_len $BM_OUTPUT_LEN "
    client_cmd="$client_cmd --log-file ${BM_BACKEND}_$log_file.log "

    if [ "$BM_BACKEND" = "trtllm" ]; then
        opts="$opts -v $BM_MODEL_DIR:/model:ro -v $BM_TRT_ENGINE_DIR:/trt-model:ro -w /workspace"
        cmd="$BM_IMAGE_NAME python3 launch_triton_server.py --world_size=$WORLD_SIZE --model_repo=/trt-model"
        client_cmd="$client_cmd --endpoint 'v2/models/ensemble/generate_stream' "
    elif [ "$BM_BACKEND" = "vllm" ]; then
        opts="$opts -v $BM_MODEL_DIR:$BM_MODEL_DIR "
        cmd="$BM_IMAGE_NAME --model $BM_MODEL_DIR --tensor-parallel-size $BM_TP --pipeline-parallel-size $BM_PP --tokenizer-pool-size 2 --block-size 32 --use-v2-block-manager --swap-space 16 --gpu-memory-utilization $BM_MEM_FRACTION"
        cmd="$cmd --max-num-seqs $BM_MAX_NUM_SEQ --max-model-len $BM_MAX_SEQ_LEN --max-context-len-to-capture $BM_MAX_SEQ_LEN --max-num-batched-tokens $BM_MAX_BATCHED_TOKENS --dtype $BM_DTYPE"
        cmd="$cmd --disable-log-stats"
    elif [ "$BM_BACKEND" = "mii" ]; then
        LOG ERR "TODO mii"
    elif [ "$BM_BACKEND" = "siliconllm" ]; then
        opts="$opts -v $BM_MODEL_DIR:$BM_MODEL_DIR -v $CUR_DIR/.triton:/root/.triton"
        cmd="$BM_IMAGE_NAME python -m crossing.server.cli --host 0.0.0.0 --port $BM_PORT --model $BM_MODEL_DIR --max-tokens-for-cuda-graph $BM_MAX_TOKENS_FOR_CUDA_GRAPH --memory-fraction $BM_MEM_FRACTION --max-seq-len $BM_MAX_SEQ_LEN --tensor-parallel-size $BM_TP --pipeline-parallel-size $BM_PP"
        cmd="$cmd --disable-prefix-cache"
        #client_cmd="$client_cmd --add-system-prompt"
    elif [ "$BM_BACKEND" = "tgi" ]; then
        LOG ERR "TODO tgi"
    else
        LOG WARN "Unkown or unsupported backend: $BM_BACKEND"
    fi

    if [ ! x"$cmd" = x"" ]; then
        LOG INFO "docker run $opts $cmd"
        LOG INFO ""
        LOG INFO "python benchmark_serving.py $client_cmd"
        if [ $BM_DRY_RUN -eq 0 ];then
            docker run $opts $cmd
            python $CUR_DIR/benchmark_serving.py $client_cmd
        fi
    fi

    LOG INFO "Complete the tests and stop $BM_CONTAINER_NAME"
    if [ $BM_DRY_RUN -eq 0 ];then
        docker stop $BM_CONTAINER_NAME
        docker rm -f $BM_CONTAINER_NAME
        if [ "$BM_BACKEND" = "siliconllm" ]; then
            rm -f $CUR_DIR/.triton
        fi
    fi
}

function check_params() {
    if [ x"$BM_IMAGE_NAME" = x"" ]; then
        LOG ERR "The BM_IMAGE_NAME is not set"
    fi
    if [ x"$BM_CONTAINER_NAME" = x"" ]; then
        LOG ERR "The BM_CONTAINER_NAME is not set"
    fi
    if [ x"$BM_MODEL_DIR" = x"" ]; then
        LOG ERR "The BM_MODEL_DIR is not set"
    fi
    if [ x"$BM_TEST_DATA_DIR" = x"" ]; then
        LOG ERR "The BM_TEST_DATA_DIR is not set"
    fi
    if [ x"$BM_BACKEND" = x"" ] || [[ ! ${PRESET_BACKEND[@]} =~ $BM_BACKEND ]];then
        LOG ERR "The BM_BACKEND is null or not in: [ ${PRESET_BACKEND[@]} ]"
    fi
    if [ x"$BM_BACKEND" = x"trtllm" ] && [ x"$BM_TRT_ENGINE_DIR" = x"" ];then
        LOG ERR "The BM_TRT_ENGINE_DIR is not set when backend is 'trtllm'"
    fi
    WORLD_SIZE=$(expr $BM_TP \* $BM_PP)
    IFS=',' read -r -a arr <<< "$BM_CUDA_DEVICES"
    NUM_GPUS=${#arr[@]}
    LOG INFO "======= The input parameters for lanching $BM_BACKEND ==============="
    LOG INFO "BM_IMAGE_NAME=$BM_IMAGE_NAME"
    LOG INFO "BM_CONTAINER_NAME=$BM_CONTAINER_NAME"
    LOG INFO "BM_MODEL_DIR=$BM_MODEL_DIR"
    LOG INFO "BM_TEST_DATA_DIR=$BM_TEST_DATA_DIR"
    LOG INFO "BM_BACKEND=$BM_BACKEND"
    LOG INFO "BM_CUDA_DEVICES=$BM_CUDA_DEVICES"
    LOG INFO "BM_PORT=$BM_PORT"
    LOG INFO "BM_TP=$BM_TP"
    LOG INFO "BM_PP=$BM_PP"
    LOG INFO "WORLD_SIZE=$WORLD_SIZE"
    LOG INFO "NUM_GPUS=$NUM_GPUS"
    LOG INFO "BM_MEM_FRACTION=$BM_MEM_FRACTION"
    LOG INFO "BM_MAX_NUM_SEQ=$BM_MAX_NUM_SEQ"
    LOG INFO "BM_MAX_SEQ_LEN=$BM_MAX_SEQ_LEN"
    LOG INFO "BM_MAX_BATCHED_TOKENS=$BM_MAX_BATCHED_TOKENS"
    if [ "$BM_BACKEND" = "trtllm" ];then
        LOG INFO "BM_TRT_ENGINE_DIR=$BM_TRT_ENGINE_DIR"
    fi
    LOG INFO ""

}

function main() {
    while [ "$#" -gt 0 ]; do
    case "$1" in
    --image-name)
        shift
        BM_IMAGE_NAME="$1"
        shift
        ;;
    --container-name)
        shift
        BM_CONTAINER_NAME="$1"
        shift
        ;;
    --model-dir)
        shift
        BM_MODEL_DIR="$1"
        shift
        ;;
    --trt-engine-dir)
        shift
        BM_TRT_ENGINE_DIR="$1"
        shift
        ;;
    --data-dir)
        shift
        BM_TEST_DATA_DIR="$1"
        shift
        ;;
    --backend)
        shift
        BM_BACKEND="$1"
        shift
        ;;
    --cuda-devices)
        shift
        BM_CUDA_DEVICES=$1
        shift
        ;;
    --port)
        shift
        BM_PORT=$1
        shift
        ;;
    --tp)
        shift
        BM_TP=$1
        shift
        ;;
    --pp)
        shift
        BM_PP=$1
        shift
        ;;
    --memory-fraction)
        shift
        BM_MEM_FRACTION=$1
        shift
        ;;
    --max-num-seq)
        shift
        BM_MAX_NUM_SEQ=$1
        shift
        ;;
    --max-seq-len)
        shift
        BM_MAX_SEQ_LEN=$1
        shift
        ;;
    --max-batched-tokens)
        shift
        BM_MAX_BATCHED_TOKENS=$1
        shift
        ;;
    --prompt-policy)
        shift
        BM_SAMPLING_POLICY="$1"
        shift
        ;;
    --warmup-reqs)
        shift
        BM_WARMUP_REQS=$1
        shift
        ;;
    --norm-reqs)
        shift
        BM_NORM_REQS=$1
        shift
        ;;
    --concurrent-reqs)
        shift
        BM_CONCURRENT_REQS=$1
        shift
        ;;
    --prompt-policy)
        shift
        BM_SAMPLING_POLICY="$1"
        shift
        ;;
    --input-len)
        shift
        BM_INPUT_LEN=$1
        shift
        ;;
    --output-len)
        shift
        BM_OUTPUT_LEN=$1
        shift
        ;;
    --dry-run)
        shift
        BM_DRY_RUN=1
        ;;
    *)
        break
    esac
    done

    check_params
    launch_and_run
}

main "$@"


