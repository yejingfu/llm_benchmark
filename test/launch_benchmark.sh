#!/bin/bash

PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/../scripts/base.sh

BM_RPESET_BACKEND=("vllm" "trtllm" "tgi" "siliconllm")
BM_BACKEND="vllm"

#BM_MODEL_LIST=("Mixtral-8x7B-Instruct-v0.1" "Toppy-M-7B mistral_7b" "MythoMax-L2-13b" "lzlv_70b_fp16_hf")
BM_MODEL_LIST=("Meta-Llama-3-8B-Instruct")
BM_CONCUR_LIST=(64 128)
BM_MODEL_DIR=
BM_TEST_DATA_DIR=/models/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json
BM_DRY_RUN=0

## server side
BM_IMAGE_NAME="ppinfer_vllm:latest"
BM_CONTAINER_NAME="benchmark_$BM_BACKEND"
BM_PORT=18000
BM_MEM_FRACTION=0.9
BM_MAX_NUM_SEQ=128
BM_MAX_SEQ_LEN=4096
BM_MAX_BATCHED_TOKENS=4096
BM_MAX_TOKENS_FOR_CUDA_GRAPH=512
BM_DTYPE=auto
BM_CUDA_DEVICES=0,1,2,3,4,5,6,7
IFS=',' read -r -a arr <<< "$BM_CUDA_DEVICES"
BM_NUM_GPUS=${#arr[@]}
BM_TP=8
BM_PP=1
BM_WORLD_SIZE=$(expr $BM_TP \* $BM_PP)
BM_ENABLE_LOG=0
BM_FP8="None"
BM_TRT_ENGINE_DIR=
BM_TRT_IFB_DIR=

BM_WARMUP_REQS=32
BM_NORM_REQS=512
BM_CONCURRENT_REQS=32
BM_SAMPLING_POLICY="fixed"
BM_INPUT_LEN=1024
BM_OUTPUT_LEN=1024

function update_params() {
    BM_CONTAINER_NAME="benchmark_$BM_BACKEND"
    IFS=',' read -r -a arr <<< "$BM_CUDA_DEVICES"
    BM_NUM_GPUS=${#arr[@]}
    BM_WORLD_SIZE=$(expr $BM_TP \* $BM_PP)
    BM_TP=$BM_NUM_GPUS
    if [ "$BM_BACKEND" = "trtllm" ]; then
        BM_IMAGE_NAME="ppinfer_triton_trtllm:24.02"
    elif [ "$BM_BACKEND" = "vllm" ]; then
        BM_IMAGE_NAME="ppinfer_vllm:latest"
    elif [ "$BM_BACKEND" = "mii" ]; then
        BM_IMAGE_NAME="ppinfer_mii:0.1"
    elif [ "$BM_BACKEND" = "tgi" ]; then
        BM_IMAGE_NAME="ppinfer_tgi:0.2"
    elif [ "$BM_BACKEND" = "siliconllm" ]; then
        BM_IMAGE_NAME="crossing:0.9.1"
    else
        LOG ERR "Not support backend $BM_BACKEND"
    fi
}

function launch_inference_server() {
    if [ $BM_DRY_RUN -eq 0 ];then
        LOG INFO "Remove the docker container if it is running: $BM_CONTAINER_NAME"
        remove_docker_container $BM_CONTAINER_NAME
    fi

    LOG INFO "launch docker container $BM_CONTAINER_NAME from image $BM_IMAGE_NAME and the backend is $BM_BACKEND\n"
    opts="-d --gpus all --privileged --ipc=host --net=host --ulimit stack=67108864 --ulimit memlock=-1 -e HTTPS_PROXY= -e HTTP_PROXY= -e ALL_PROXY= -e https_proxy= -e http_proxy= -e all_proxy= -e CUDA_VISIBLE_DEVICES=$BM_CUDA_DEVICES --name $BM_CONTAINER_NAME "
    cmd=""
    if [ "$BM_BACKEND" = "trtllm" ]; then
        opts="$opts -v $BM_MODEL_DIR:$BM_MODEL_DIR:ro -v $BM_TRT_ENGINE_DIR:$BM_TRT_ENGINE_DIR:ro -v $BM_TRT_IFB_DIR:$BM_TRT_IFB_DIR -w /workspace"
        cmd="$python3 $BM_TRT_IFB_DIR/launch_triton_server.py --world_size=$WORLD_SIZE --model_repo=$BM_TRT_IFB_DIR"
        if [ $BM_ENABLE_LOG -eq 1 ]; then
            cmd="$cmd --log --log-file $BM_TRT_IFB_DIR/log.txt"
        fi
    elif [ "$BM_BACKEND" = "vllm" ]; then
        opts="$opts -v $BM_MODEL_DIR:$BM_MODEL_DIR "
        cmd="--host 0.0.0.0 --port $BM_PORT --model $BM_MODEL_DIR --tensor-parallel-size $BM_TP --pipeline-parallel-size $BM_PP --use-v2-block-manager --block-size 32 --swap-space 16 --gpu-memory-utilization $BM_MEM_FRACTION"
        cmd="$cmd --max-num-seqs $BM_MAX_NUM_SEQ --max-model-len $BM_MAX_SEQ_LEN --max-num-batched-tokens $BM_MAX_BATCHED_TOKENS --dtype $BM_DTYPE --served-model-name default"
        if [[ "$BM_IMAGE_NAME" == *"v0.4.0"* ]]; then
            cmd="$cmd --max-context-len-to-capture $BM_MAX_SEQ_LEN"
        else
            cmd="$cmd --max-seq_len-to-capture $BM_MAX_SEQ_LEN"
        fi
        if [ x"$BM_FP8" = x"weight" ] || [ x"$BM_FP8" = x"all" ]; then
            cmd="$cmd --quantization fp8"
        fi
        if [ x"$BM_FP8" = x"kvcache" ] || [ x"$BM_FP8" = x"all" ]; then
            cmd="$cmd --kv-cache-dtype fp8 --quantization-param-path $BM_MODEL_DIR/kv_cache_scales.json"
        fi
        if [ $BM_ENABLE_LOG -eq 0 ]; then
            cmd="$cmd --disable-log-stats"
        fi
    elif [ "$BM_BACKEND" = "siliconllm" ]; then
        opts="$opts -v $BM_MODEL_DIR:$BM_MODEL_DIR -v $CUR_DIR/.triton:/root/.triton"
        cmd="python -m crossing.server.cli --host 0.0.0.0 --port $BM_PORT --model $BM_MODEL_DIR --max-tokens-for-cuda-graph $BM_MAX_TOKENS_FOR_CUDA_GRAPH --memory-fraction $BM_MEM_FRACTION --max-seq-len $BM_MAX_SEQ_LEN --tensor-parallel-size $BM_TP --pipeline-parallel-size $BM_PP"
        cmd="$cmd --disable-prefix-cache"
    elif [ "$BM_BACKEND" = "tgi" ]; then
        LOG ERR "TODO tgi"
    else
        LOG WARN "Unkown or unsupported backend: $BM_BACKEND"
    fi

    if [ ! x"$cmd" = x"" ]; then
        LOG INFO "[RUN]: docker run $opts $BM_IMAGE_NAME $cmd\n"
        if [ $BM_DRY_RUN -eq 0 ];then
            launch_docker_container "$BM_IMAGE_NAME" "$BM_CONTAINER_NAME" "$opts" "$cmd"
        fi
    fi
}

function stop_inference_server() {
    LOG INFO "Complete the tests and stop the container: $BM_CONTAINER_NAME\n\n"
    if [ $BM_DRY_RUN -eq 0 ];then
        remove_docker_container $BM_CONTAINER_NAME
        if [ "$BM_BACKEND" = "siliconllm" ]; then
            rm -f $CUR_DIR/.triton
        fi
    fi
}

function run_benchmark_client() {
    client_cmd="--backend $BM_BACKEND --model $BM_MODEL_DIR --tokenizer $BM_MODEL_DIR --dataset $BM_TEST_DATA_DIR --port $BM_PORT --num-warmup-requests $BM_WARMUP_REQS --num-benchmark-requests $BM_NORM_REQS --max-concurrent-requests $BM_CONCURRENT_REQS "
    client_cmd="$client_cmd --stream --pad-requests --warn-dismatch-output-len --gpus $BM_NUM_GPUS "
    client_cmd="$client_cmd --sampling-policy $BM_SAMPLING_POLICY --fixed_prompt_len $BM_INPUT_LEN --fixed_output_len $BM_OUTPUT_LEN "
    log_file=$(date|tr -d ' ')
    client_cmd="$client_cmd --log-file ${BM_BACKEND}_$log_file.log "

    if [ "$BM_BACKEND" = "trtllm" ]; then
        client_cmd="$client_cmd --endpoint v2/models/ensemble/generate_stream"
    fi

    LOG INFO "[RUN]: python $CUR_DIR/benchmark_client_legacy.py $client_cmd"
    if [ $BM_DRY_RUN -eq 0 ];then
        python $CUR_DIR/benchmark_client_legacy.py $client_cmd
    fi
}


function check_trtllm() {
    local model=$1
    local engine=$2
    local ifb=$3
    LOG INFO "trtllm model: $model, engine: $engine, ifb: $ifb"
    if [ ! -d "$model" ]; then
        LOG ERR "The model does not exist: $model"
    fi
    if [ ! -d "$engine" ]; then
        LOG ERR "The engine does not exist: $engine, please build the engine firstly"
    fi
    if [ ! -d "$ifb" ]; then
        LOG ERR "The inflight-batching folder does not exist: $ifb, run scripts/build_trtllm_config.sh to generate it"
    fi
}

function run() {
    BM_INPUT_LEN=$1
    BM_OUTPUT_LEN=$2
    if [ "$BM_BACKEND" = "trtllm" ]; then
        check_trtllm $BM_MODEL_DIR $BM_TRT_ENGINE_DIR $BM_TRT_IFB_DIR
    fi

    for concur in ${BM_CONCUR_LIST[@]}; do
        BM_CONCURRENT_REQS=$concur
        BM_WARMUP_REQS=32
        BM_NORM_REQS=$(expr $concur \* 16)
        LOG INFO "===== RUN benchmark, model: $BM_MODEL_DIR, concurrency: $BM_CONCURRENT_REQS, requests(warmup/test): $BM_WARMUP_REQS/$BM_NORM_REQS, seq length(in/out): $BM_INPUT_LEN/$BM_OUTPUT_LEN\n"
        launch_inference_server
        run_benchmark_client
        stop_inference_server
    done
}

function usage() {
    LOG INFO "$PRG_NAME [options]"
    LOG INFO "    --dry-run    Print the command details without starting docker"
    LOG INFO "    --backend    Specify the backend from: trtllm, vllm, tgi, siliconllm, mii"
    LOG INFO "    --model      Special model to load, if not set, use preset models"
    LOG INFO "    --fp8        How to config fp8 inference, its value should be: weight, kvcache, all"
    LOG INFO "    --log-stats  Enable vllm to log stats during inference"
    exit
}

function main() {
    if [ "$#" -eq 0 ]; then
        usage
    fi
    while [ "$#" -gt 0 ]; do
    case "$1" in
    --dry-run)
        shift
        BM_DRY_RUN=1
        ;;
    --backend)
        shift
        BM_BACKEND="$1"
        shift
        ;;
    --model)
        shift
        BM_MODEL_DIR="$1"
        shift
        ;;
    --fp8)
        shift
        BM_FP8="$1"
        shift
        ;;
    --log-stats)
        shift
        BM_ENABLE_LOG=1
        ;;
    *)
        usage
        break
    esac
    done

    update_params

    if [ x"$BM_MODEL_DIR" != x"" ]; then
        if [ ! -d "$BM_MODEL_DIR" ]; then
            LOG ERR "The model path does not exist: $BM_MODEL_DIR"
        fi
        LOG INFO "Load the model $BM_MODEL_DIR"
        if [ "$BM_BACKEND" = "trtllm" ];then
            LOG ERR "Not implement"
        fi
        run 1024 1024
        run 3500 500
    else
        for model in ${BM_MODEL_LIST[@]}; do
            BM_MODEL_DIR=/models/${model}
            local trt_engine_path=
            local trt_ifb_path=
            if [ "$BM_BACKEND" = "trtllm" ];then
                BM_TRT_ENGINE_DIR=/models/${model}-trtllm/engine/float16/8-gpu
                BM_TRT_IFB_DIR=/models/${model}-trtllm/ifb
            fi
            run 1024 1024
            run 3500 500
        done
    fi
}

main "$@"

