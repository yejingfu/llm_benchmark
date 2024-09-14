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
BM_NUM_REQUESTS=2000

parallels=(10 20 30)
## pair: input-len, output-len
request_len=(1000 500 1800 200)

function usage() {
    LOG INFO "$PRG_NAME [options]"
    LOG INFO "  --endpoint The LLM server URL, example: http://localhost:18011/v1"
    LOG INFO "  --api-key (optional) The api key used to call commercial service"
    LOG INFO "  --tokenizer The path to local model folder used for tokenizing prompt or output"
    LOG INFO "  --dataset The path to local dataset used for sampling benchmark requests"
    LOG INFO "  --chat (optional) If set, call LLM chat-completions API for testing"
    LOG INFO "  --num-requests (optional) The total requests send to server for benchmark, default: $BM_NUM_REQUESTS"
    LOG INFO "  --add-sys-prompt (optional) Prepend system prompt prefix, using to test the prefix caching features"
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
    local args="--endpoint $BM_ENDPOINT --tokenizer $BM_TOKENIZER_PATH --dataset $BM_DATASET_PATH --num-requests $BM_NUM_REQUESTS"
    if [ x"$BM_API_KEY" != x"" ]; then
        args="$args --api-key $BM_API_KEY"
    fi
    if [ $BM_CHAT -eq 1 ]; then
        args="$args --api-kind chat"
    fi
    if [ $BM_ADD_SYS_PROMPT -eq 1 ]; then
        args="$args --add-system-prompt"
    fi
    dump_file=$(date "+%Y-%m-%d-%H%M%S.txt")
    args="$args --log-file $dump_file"
    echo "Save result to $dump_file"

    num_req_len=${#request_len[@]}
    for i in $(seq 0 2 $((num_req_len-2)));do
        for parallel in ${parallels[@]};do
            in_len=${request_len[i]}
            out_len=${request_len[i+1]}
            local args2="$args --sampling-policy normal --prompt-len-mean $in_len --prompt-len-std 10 --output-len-mean $out_len --output-len-std 6 --parallel $parallel"
            echo "===> [Run]: python $CUR_DIR/benchmark_client.py $args2"
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
    *)
        usage
        break
    esac
    done
    run
}

main "$@"

