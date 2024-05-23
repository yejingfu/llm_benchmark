#!/bin/bash

PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/../scripts/base.sh

DEF_MODEL_DIR=/models
DEF_DATA_DIR=/models/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json

DRY_RUN=""
BACKEND="vllm"
FP8_SUPPORT="None"
CUDA_DEVICES=0,1,2,3,4,5,6,7
PORT=8000
TP=8
PP=1
MEM_FRACTION=0.9
MAX_NUM_SEQS=128
MAX_SEQ_LEN=4096
MAX_BATCHED_TOKEN=4096
PROMPT_POLICY="fixed"
ENABLE_LOG_STATS=0

#MODEL_LIST=("Mixtral-8x7B-Instruct-v0.1" "Toppy-M-7B mistral_7b" "MythoMax-L2-13b" "lzlv_70b_fp16_hf")
MODEL_LIST=("Meta-Llama-3-8B-Instruct")
CONCUR_LIST=(64 128)

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
    local model_path="$1"
    local input_len=$2
    local output_len=$3

    local image_tag=""
    local container_name="benchmark_$BACKEND"
    local extra_opts=""
    if [ ! -d "$model_path" ]; then
        LOG ERR "The model does not exist: $model_path"
    fi

    if [ "$BACKEND" = "trtllm" ]; then
        image_tag="ppinfer_triton_trtllm:24.02"
        local trt_engine_path=$4
        local trt_ifb_path=$5
        check_trtllm $model_path $trt_engine_path $trt_ifb_path
        extra_opts="--trt-engine-dir $trt_engine_path --trt-ifb-dir $trt_ifb_path"
    elif [ "$BACKEND" = "vllm" ]; then
        image_tag="ppinfer_vllm:latest"
#        image_tag="ppinfer/vllm-openai:v0.4.2"
#        image_tag="vllm/vllm-openai:v0.4.0"
    elif [ "$BACKEND" = "mii" ]; then
        image_tag="ppinfer_mii:0.1"
    elif [ "$BACKEND" = "tgi" ]; then
        image_tag="ppinfer_tgi:0.2"
    elif [ "$BACKEND" = "siliconllm" ]; then
        image_tag="crossing:0.9.1"
    else
        LOG ERR "Not support backend $BACKEND"
    fi

    for concur in ${CONCUR_LIST[@]}; do
        warmup_reqs=32
        norm_reqs=$(expr $concur \* 16)
        LOG INFO ""
        LOG INFO "===== RUN benchmark, model: $model_path, concurrency: $concur, requests: ($warmup_reqs, $norm_reqs), length($input_len, $output_len)"
        bash $CUR_DIR/benchmark.sh $DRY_RUN --backend $BACKEND --image-name $image_tag --container-name $container_name --port $PORT \
            --model-dir $model_path --data-dir $DEF_DATA_DIR --cuda-devices $CUDA_DEVICES --tp $TP --pp $PP --memory-fraction $MEM_FRACTION \
            --max-num-seq $MAX_NUM_SEQS --max-seq-len $MAX_SEQ_LEN --max-batched-tokens $MAX_BATCHED_TOKEN --prompt-policy $PROMPT_POLICY \
            --warmup-reqs $warmup_reqs --norm-reqs $norm_reqs --concurrent-reqs $concur --prompt-policy fixed \
            --input-len $input_len --output-len $output_len $extra_opts --fp8 $FP8_SUPPORT --log-stats $ENABLE_LOG_STATS
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
        DRY_RUN="--dry-run"
        ;;
    --backend)
        shift
        BACKEND="$1"
        shift
        ;;
    --model)
        shift
        local model_path="$1"
        shift
        ;;
    --fp8)
        shift
        FP8_SUPPORT="$1"
        shift
        ;;
    --log-stats)
        shift
        ENABLE_LOG_STATS=1
        ;;
    *)
        usage
        break
    esac
    done

    if [ x"$model_path" != x"" ]; then
        if [ ! -d "$model_path" ]; then
            LOG ERR "The model path does not exist: $model_path"
        fi
        LOG INFO "Load the model $model_path"
        run $model_path 1024 1024
        run $model_path 3500 500
    else
        for model in ${MODEL_LIST[@]}; do
            local model_path=$DEF_MODEL_DIR/${model}
            local trt_engine_path=
            local trt_ifb_path=
            if [ "$BACKEND" = "trtllm" ];then
                trt_engine_path=$DEF_MODEL_DIR/${model}-trtllm/engine/float16/8-gpu
                trt_ifb_path=$DEF_MODEL_DIR/${model}-trtllm/ifb
            fi
            run $model_path 1024 1024 $trt_engine_path $trt_ifb_path
            run $model_path 3500 500 $trt_engine_path $trt_ifb_path
        done
    fi
}

main "$@"

