#!/bin/bash
PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/../scripts/base.sh


BM_ENDPOINT=
BM_API_KEY=
BM_TOKENIZER_PATH=
BM_DATASET_PATH=
BM_CHAT=0
BM_ADD_SYS_PROMPT=0
BM_NUM_REQUESTS=120
BM_PRINT_RAW_METRICS=0
BM_LOG_FILE=

## set parallels to 1 to test the single batch, which can get max speed (tps)
parallels=(1 5 10 15 20)
## pair: input-len, output-len
request_len=(1000 100 3000 300 5000 500 10000 1000 20000 2000)

function usage() {
    LOG INFO "$PRG_NAME [options]"
    LOG INFO "  --endpoint The LLM server URL, example: http://localhost:18011/v1"
    LOG INFO "  --api-key (optional) The api key used to call commercial service"
    LOG INFO "  --tokenizer The path to local model folder used for tokenizing prompt or output"
    LOG INFO "  --dataset The path to local dataset used for sampling benchmark requests"
    LOG INFO "  --chat (optional) If set, call LLM chat-completions API for testing"
    LOG INFO "  --num-requests (optional) The total requests send to server for benchmark, default: $BM_NUM_REQUESTS"
    LOG INFO "  --add-sys-prompt (optional) Prepend system prompt prefix, using to test the prefix caching features"
    LOG INFO "  --log-file (optional) Save the output to file if set"
    LOG INFO "  --print-raw (optional) Print the raw metrics data like TTFT or TPOT"
    exit
}

function run() {
    if [ x"$BM_ENDPOINT" = x"" ]; then
        LOG ERR "Please set --endpoint"
    fi
    if [ x"$BM_TOKENIZER_PATH" = x"" ]; then
        LOG ERR "Please set --tokenizer"
    fi
    if [ x"$BM_DATASET_PATH" = x"" ]; then
        LOG ERR "Please set --dataset"
    fi
    local args="--endpoint $BM_ENDPOINT --tokenizer $BM_TOKENIZER_PATH --dataset $BM_DATASET_PATH"
    if [ x"$BM_API_KEY" != x"" ]; then
        args="$args --api-key $BM_API_KEY"
    fi
    if [ $BM_CHAT -eq 1 ]; then
        args="$args --api-kind chat"
    fi
    if [ $BM_ADD_SYS_PROMPT -eq 1 ]; then
        args="$args --add-system-prompt"
    fi
    if [ x"$BM_LOG_FILE" != x"" ]; then
        args="$args --log-file $BM_LOG_FILE"
    fi
    if [ $BM_PRINT_RAW_METRICS -eq 1 ]; then
        args="$args --record-raw-metrics"
    fi

    num_req_len=${#request_len[@]}
    for i in $(seq 0 2 $((num_req_len-2)));do
        for parallel in ${parallels[@]};do
            in_len=${request_len[i]}
            out_len=${request_len[i+1]}
            local args2="$args --sampling-policy normal --prompt-len-mean $in_len --prompt-len-std 10 --output-len-mean $out_len --output-len-std 6 --parallel $parallel"
            if [ $parallel -eq 1 ]; then
                ## single batch, use less requests to save time
                args2="$args2 --num-requests 20"
            else
                args2="$args2 --num-requests $BM_NUM_REQUESTS"
            fi
            echo "==== [Run]: python $CUR_DIR/benchmark_client.py $args2"
            echo ""
            python $CUR_DIR/benchmark_client.py $args2
        done
    done
}

function main() {
    if [ "$#" -eq 0 ]; then
        usage
    fi
    while [ "$#" -gt 0 ]; do
    case "$1" in
    --endpoint)
        shift
        BM_ENDPOINT="$1"
        shift
        ;;
    --api-key)
        shift
        BM_API_KEY=$1
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
    --num-requests)
        shift
        BM_NUM_REQUESTS=$1
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
    --log-file)
        shift
        BM_LOG_FILE=$1
        shift
        ;;
    --print-raw)
        shift
        BM_PRINT_RAW_METRICS=1
        ;;
    *)
        usage
        break
    esac
    done
    run
}

main "$@"

