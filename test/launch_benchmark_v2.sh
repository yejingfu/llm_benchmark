#!/bin/bash
PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/../scripts/base.sh

## example
## launch_benchmark_v2.sh --model-name llama3-8b --tokenizer /models/Meta-Llama-3-8B-Instruct --dataset /models/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json --backend vllm-local
## launch_benchmark_v2.sh --model-name meta-llama/llama-3-8b-instruct --tokenizer /models/Meta-Llama-3-8B-Instruct --dataset /models/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json --backend novita --base-url https://api.novita.ai --api-key xxxxx

BM_PRESET_BACKEND=("vllm" "vllm-local" "trtllm" "novita", "siliconflow")
BM_BACKEND=
BM_API_KEY=
BM_BASE_URL=
BM_MODEL_NAME=
BM_TOKENIZER_PATH=
BM_DATASET_PATH=

BM_NUM_WARMUP=2
BM_NUM_BENCHMARK=128
BM_FIXED_INPUT_LEN=1024
BM_FIXED_OUTPUT_LEN=1024
BM_MAX_CONCURRENCY=64
BM_CHAT=0
BM_ADD_SYS_PROMPT=0
BM_DISABLE_WARN=0
BM_DRY_RUN=

function usage() {
    LOG INFO "$PRG_NAME [options]"
    LOG INFO "  --dry-run  Print the command without really execute it"
    LOG INFO "  --model-name The model name passed to LLM service in API"
    LOG INFO "  --tokenizer The path to local model folder used for tokenizing prompt or output"
    LOG INFO "  --dataset The path to local dataset used for sampling benchmark requests"
    LOG INFO "  --backend  The backend: ${BM_PRESET_BACKEND[@]}"
    LOG INFO "  --base-url The LLM server URL"
    LOG INFO "  --api-key (optional) The api key used to call commercial service"
    LOG INFO "  --chat (optional) If set, call LLM chat-completions API for testing"
    LOG INFO "  --num-requests (optinal) The total requests send to server for benchmark, default: $BM_NUM_BENCHMARK"
    LOG INFO "  --input-len (optional) The fixed input tokens for every request, default: $BM_FIXED_INPUT_LEN"
    LOG INFO "  --output-len (optional) The fixed output tokens the LLM service should return for every request, default: $BM_FIXED_OUTPUT_LEN"
    LOG INFO "  --disable-warn (optional) Supress the warning message if set"
    exit
}

function run() {
    local path_prefix="/v1"
    local extra_args="$BM_DRY_RUN"
    if [[ ! ${BM_PRESET_BACKEND[@]} =~ $BM_BACKEND ]]; then
        LOG ERR "The backend should be one of (${BM_PRESET_BACKEND[@]})"
    fi
    if [ x"$BM_BACKEND" = x"vllm-local" ]; then
        if [ x"$BM_BASE_URL" = x"" ]; then
            BM_BASE_URL="http://127.0.0.1:18002"
        fi
        BM_BACKEND="vllm"
        BM_NUM_WARMUP=16
    elif [ x"$BM_BACKEND" = x"novita" ]; then
        path_prefix="/v3/openai"
    elif [ x"$BM_BACKEND" = x"siliconflow" ]; then
        extra_args="$extra_args --ignore-check"
    fi
    if [ $BM_NUM_WARMUP -gt $BM_NUM_BENCHMARK ]; then
        BM_NUM_WARMUP=$BM_NUM_BENCHMARK
    fi
    if [ x"$BM_BASE_URL" = x"" ]; then
        LOG ERR "The base-url is not set"
    fi
    if [ x"$BM_MODEL_NAME" = x"" ]; then
        LOG ERR "The model-name is not set"
    fi
    if [ x"$BM_TOKENIZER_PATH" = x"" ]; then
        LOG ERR "The tokenizer path is not set"
    fi
    if [ x"$BM_DATASET_PATH" = x"" ]; then
        LOG ERR "The test dataset path is not set"
    fi
    if [[ $BM_CHAT -eq 1 ]] && [[ $BM_ADD_SYS_PROMPT -eq 1 ]]; then
        LOG ERR "The chat api cannot support adding system prompt"
    fi

    local args="--backend $BM_BACKEND --base-url $BM_BASE_URL --endpoint-models ${path_prefix}/models --endpoint-chat ${path_prefix}/chat/completions --endpoint-completion ${path_prefix}/completions"
    if [ x"$BM_API_KEY" != x"" ]; then
        args="$args --api-key $BM_API_KEY"
    fi
    args="$args --model $BM_MODEL_NAME --num-warmup-requests $BM_NUM_WARMUP --num-benchmark-requests $BM_NUM_BENCHMARK --max-concurrent-requests $BM_MAX_CONCURRENCY --stream "
    args="$args --sampling-policy fixed --fixed_prompt_len $BM_FIXED_INPUT_LEN --fixed_output_len $BM_FIXED_OUTPUT_LEN"
    if [ $BM_CHAT -eq 1 ]; then
        args="$args --api-kind chat"
    fi
    args="$args --tokenizer $BM_TOKENIZER_PATH --dataset $BM_DATASET_PATH --log-file benchmark_${BM_BACKEND}.log $extra_args"

    if [ $BM_DISABLE_WARN -eq 0 ]; then
        args="$args --warn-dismatch-output-len"
    fi
    if [ x"$BM_DRY_RUN" = x"--dry-run" ]; then
        LOG INFO "[RUN]: python $CUR_DIR/benchmark_client.py $args"
    fi
    python $CUR_DIR/benchmark_client.py $args
}

function main() {
    if [ "$#" -eq 0 ]; then
        usage
    fi
    while [ "$#" -gt 0 ]; do
    case "$1" in
    --dry-run)
        shift
        BM_DRY_RUN="--dry-run"
        ;;
    --model-name)
        shift
        BM_MODEL_NAME=$1
        shift
        ;;
    --tokenizer)
        shift
        BM_TOKENIZER_PATH="$1"
        shift
        ;;
    --dataset)
        shift
        BM_DATASET_PATH="$1"
        shift
        ;;
    --backend)
        shift
        BM_BACKEND="$1"
        shift
        ;;
    --base-url)
        shift
        BM_BASE_URL="$1"
        shift
        ;;
    --api-key)
        shift
        BM_API_KEY=$1
        shift
        ;;
    --num-requests)
        shift
        BM_NUM_BENCHMARK=$1
        shift
        ;;
    --input-len)
        shift
        BM_FIXED_INPUT_LEN=$1
        shift
        ;;
    --output-len)
        shift
        BM_FIXED_OUTPUT_LEN=$1
        shift
        ;;
    --parallel)
        shift
        BM_MAX_CONCURRENCY=$1
        shift
        ;;
    --chat)
        shift
        BM_CHAT=1
        ;;
    --add-sys-prompt)
        shift
        BM_ADD_SYS_PROMPT=1
        ;;
    --disable-warn)
        shift
        BM_DISABLE_WARN=1
        ;;
    *)
        usage
        break
    esac
    done
    run
}

main "$@"

