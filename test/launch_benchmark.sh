#!/bin/bash

PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/../scripts/base.sh

DEF_MODEL_DIR=/models
DEF_DATA_DIR=/models/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json

DRY_RUN=""
BACKEND="vllm"
CUDA_DEVICES=0,1,2,3,4,5,6,7
PORT=8000
TP=8
PP=1
MEM_FRACTION=0.9
MAX_NUM_SEQS=64
MAX_SEQ_LEN=4096
MAX_BATCHED_TOKEN=4096
PROMPT_POLICY="fixed"

#MODEL_LIST=("Mixtral-8x7B-Instruct-v0.1" "Toppy-M-7B mistral_7b" "MythoMax-L2-13b" "lzlv_70b_fp16_hf")
MODEL_LIST=("Mixtral-8x7B-Instruct-v0.1" "lzlv_70b_fp16_hf")
CONCUR_LIST=(64 128)

function run() {
    model_name="$1"
    input_len=$2
    output_len=$3
    image_tag=""
    container_name="benchmark_$BACKEND"
    model_path=$DEF_MODEL_DIR/$model_name
    trt_engine_path=$model_path
    if [ "$BACKEND" = "trtllm" ]; then
        image_tag="ppinfer_triton_trtllm:24.02"
        trt_engine_path=$DEF_MODEL_DIR/trtllm-$model_name/trt_engines/fp8/8-gpu
    elif [ "$BACKEND" = "vllm" ]; then
        image_tag="vllm/vllm-openai:v0.4.0"
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
        LOG INFO "===== RUN benchmark, model: $model_name, concurrency: $concur, requests: ($warmup_reqs, $norm_reqs), length($input_len, $output_len)"
        bash $CUR_DIR/benchmark.sh $DRY_RUN --backend $BACKEND --image-name $image_tag --container-name $container_name --port $PORT \
            --model-dir $model_path --trt-engine-dir $trt_engine_path --data-dir $DEF_DATA_DIR --cuda-devices $CUDA_DEVICES --tp $TP --pp $PP --memory-fraction $MEM_FRACTION \
            --max-num-seq $MAX_NUM_SEQS --max-seq-len $MAX_SEQ_LEN --max-batched-tokens $MAX_BATCHED_TOKEN --prompt-policy $PROMPT_POLICY \
            --warmup-reqs $warmup_reqs --norm-reqs $norm_reqs --concurrent-reqs $concur --prompt-policy fixed \
            --input-len $input_len --output-len $output_len
    done
}

function usage() {
    LOG INFO "$PRG_NAME [options]"
    LOG INFO "    --dry-run    Print the command details without starting docker"
    LOG INFO "    --beckend    Specify the backend from: trtllm, vllm, tgi, siliconllm, mii"
    exit
}

function main() {
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
    *)
        usage
        break
    esac
    done

    for model in ${MODEL_LIST[@]}; do
        run $model 1024 1024
        run $model 3500 500
    done
}

main "$@"

