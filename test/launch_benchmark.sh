#!/bin/bash
PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/util.sh

# server side
BM_IMAGE="image.paigpu.com/library/ppinfer_vllm:0.6.2.2"
BM_MODEL_DIR=
BM_HF_MODEL=
BM_SERVED_NAME=
BM_GPU_IDS="0"
BM_LISTEN_PORT="18011"
BM_DEF_SERVER_EXTA_ARGS="--swap-space 16 --gpu-memory-utilization 0.92 --dtype auto --max-num-seqs 32 --max-model-len 32768 --disable-log-requests --enable-prefix-caching --enable-chunked-prefill"

if [[ x"$HF_ENDPOINT" = x"" ]]; then
    HF_ENDPOINT="https://huggingface.co"
fi

# client side
BM_ENDPOINT=
BM_API_KEY=
BM_TOKENIZER_PATH=
BM_DATASET_PATH=
BM_CHAT=0
BM_ADD_SYS_PROMPT=0
BM_NUM_REQUESTS=90
BM_NUM_REQUESTS_SINGLE_BATCH=20
BM_PRINT_RAW_METRICS=0
BM_LOG_FILE=
BM_PARALLELS=(1 2 3 4 5 6 7 8 9 10 12 15)
BM_CONTEXT_LEN=(1000 3000 5000 6000 10000)
BM_CTX_LEN_RATIO=10

function usage() {
    LOG INFO "$PRG_NAME [options]"
    LOG INFO " server side: run server only when image-name and model-dir are set"
    LOG INFO "  --image-name (optional) The docker image used to launch server. If not set, do NOT run server"
    LOG INFO "  --model-dir (optional) The path to model folder, loaded by server. If not set, do NOT run server"
    LOG INFO "  --model-hf-name (optional) The huggingface model name, download from it if the local --model-dir does not exist"
    LOG INFO "  --model-served-name (optional) The served model name, which client can query"
    LOG INFO "  --gpu-ids (optional) The list of GPU IDs used to serve LLM, default is 0"
    LOG INFO "  --listen-port (optional) The http listening port, default is $BM_LISTEN_PORT"
    LOG INFO "  --extra-server-args (optional) The extra server argument, default is: $BM_DEF_SERVER_EXTA_ARGS"
    LOG INFO " client side:"
    LOG INFO "  --endpoint The LLM server URL, example: http://localhost:18011/v1"
    LOG INFO "  --api-key (optional) The api key used to call commercial service"
    LOG INFO "  --tokenizer The path to local model folder used for tokenizing prompt or output"
    LOG INFO "  --dataset The path to local dataset used for sampling benchmark requests"
    LOG INFO "  --chat (optional) If set, call LLM chat-completions API for testing"
    LOG INFO "  --num-requests (optional) The total requests send to server for benchmark, default: $BM_NUM_REQUESTS"
    LOG INFO "  --add-sys-prompt (optional) Prepend system prompt prefix, using to test the prefix caching features"
    LOG INFO "  --log-file (optional) Save the output to file if set"
    LOG INFO "  --print-raw (optional) Print the raw metrics data like TTFT or TPOT"
    LOG INFO "  --context-lens (optional) The array of input length, sperated by comma, default is ${BM_CONTEXT_LEN[@]}"
    LOG INFO "  --context-len-ratio (optional) The ratio of input length / output lenght, default is $BM_CTX_LEN_RATIO"
    LOG INFO "  --batches (optional) The list of batch size, sperated by comma, default is $BM_PARALLELS"
    exit
}

function run() {
    ## Server side
    if [ x"$BM_IMAGE" != x"" ] && [ x"$BM_MODEL_DIR" != x"" ]; then
        LOG INFO "Run LLM server"
        if [ x"$BM_MODEL_DIR" = x"" ]; then
            LOG ERR "The --model-dir is not set"
        fi
        install_docker
        if [ $? -ne 1 ]; then
            LOG ERR "Failed to install docker"
        fi
        check_image_exists $BM_IMAGE
        if [ $? -ne 1 ]; then
            LOG INFO "The docker image does not exist, pull it from remote: $BM_IMAGE"
            docker pull $BM_IMAGE
            check_image_exists $BM_IMAGE
            if [ $? -ne 1 ]; then
                LOG INFO "Failed to pull docker image: $BM_IMAGE"
            fi
        fi
        if [ ! -d "$BM_MODEL_DIR" ]; then
            if [ x"$BM_HF_MODEL" = x"" ]; then
                LOG ERR "The local model folder does not exist: $BM_MODEL_DIR , and the --model-hf-name is not set"
            fi
            LOG INFO "Download model from huggingface $BM_HF_MODEL , save into $BM_MODEL_DIR"
            if dpkg-query -W -f='${Status}' git-lfs 2>/dev/null | grep -q "install ok installed"; then
                LOG INFO "git-lfs is installed"
            else
                LOG INFO "install git-lfs"
                apt-get install -y git git-lfs
            fi
            GIT_LFS_SKIP_SMUDGE=1 git clone $HF_ENDPOINT/$BM_HF_MODEL $BM_MODEL_DIR
            pushd $BM_MODEL_DIR
            git lfs pull
            popd
        fi
        num_gpus=$(count_numbers $BM_GPU_IDS)
        docker_name="benchmark_$RANDOM"
        docker_args="-d --gpus all --privileged --ipc=host --net=host -v $BM_MODEL_DIR:/this_model -e CUDA_VISIBLE_DEVICES=$BM_GPU_IDS"
        server_args="--tensor-parallel-size $num_gpus --model /this_model"
        if [ x"$BM_SERVED_NAME" != x"" ]; then
            server_args="$server_args --served-model-name $BM_SERVED_NAME"
        fi
        server_args="$server_args --port $BM_LISTEN_PORT $BM_DEF_SERVER_EXTA_ARGS"
        LOG INFO "docker run $docker_args --name $docker_name $BM_IMAGE $server_args"
        docker run $docker_args --name $docker_name $BM_IMAGE $server_args
        try=0
        while [ $try -lt 20 ]; do
            LOG INFO "Waiting for docker ready ($try): $docker_name..."
            sleep 10
            ret=$(docker logs $docker_name) #> /dev/null 2>&1)
            LOG INFO ">>>[docker logs $docker_name ($try)]: $ret <<<<"
            if [ x"$ret" = x"" ]; then
                docker rm -f $docker_name
                docker_name=
                LOG ERR "No log returned from: $docker_name"
            elif echo "$ret" | grep -q -i "ERROR"; then
                docker rm -f $docker_name
                docker_name=
                LOG ERR "Failed to run docker instance: $docker_name"
            fi
            if echo "$ret" | grep -q "Route: /v1/chat/completions"; then
                LOG INFO "Succeed to run docker $docker_name \n\n"
                break
            fi
            try=$((try + 1))
        done
        if [ $try -eq 10 ];then
            docker rm -f $docker_name
            docker_name=
            LOG ERR "Failed to run docker instance in 200 seconds: $docker_name"
        fi
    fi

    ## Client side
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

    for in_len in ${BM_CONTEXT_LEN[@]}; do
        out_len=$((in_len/$BM_CTX_LEN_RATIO))
        for parallel in ${BM_PARALLELS[@]};do
            local args2="$args --sampling-policy normal --prompt-len-mean $in_len --prompt-len-std 10 --output-len-mean $out_len --output-len-std 6 --parallel $parallel"
            if [ $parallel -eq 1 ]; then
                ## single batch, use less requests to save time
                args2="$args2 --num-requests $BM_NUM_REQUESTS_SINGLE_BATCH"
            else
                args2="$args2 --num-requests $BM_NUM_REQUESTS"
            fi
            echo "==== [Run]: python $CUR_DIR/benchmark_client.py $args2"
            echo ""
            python $CUR_DIR/benchmark_client.py $args2
        done
    done

    LOG INFO "\nThe benchmark test is completed\n"
    if [ x"$docker_name" != x"" ];then
        LOG INFO "Delete the docker instance: $docker_name"
        remove_docker_container $docker_name
    fi
}

function main() {
    if [ "$#" -eq 0 ]; then
        usage
    fi
    while [ "$#" -gt 0 ]; do
    case "$1" in
    ## client side
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
    --context-lens)
        shift
        BM_CONTEXT_LEN=()
        BM_CONTEXT_LEN=$(convert_to_array $1)
        for len in ${BM_CONTEXT_LEN[@]}; do
            if [[ $len -le 0 ]]; then
                LOG ERR "Invalid context len argument: $1"
            fi
        done
        shift
        ;;
    --context-len-ratio)
        shift
        BM_CTX_LEN_RATIO=$1
        if [[ $BM_CTX_LEN_RATIO -le 0 ]]; then
            LOG ERR "Invalid context len ratio: $1"
        fi
        shift
        ;;
    --batches)
        shift
        BM_PARALLELS=()
        BM_PARALLELS=$(convert_to_array $1)
        for p in ${BM_PARALLELS[@]}; do
            if [[ $p -le 0 ]]; then
                LOG ERR "Invalid batches argument: $1"
            fi
        done
        shift
        ;;
    ## server side
    --image-name)
        shift
        BM_IMAGE="$1"
        shift
        ;;
    --model-dir)
        shift
        BM_MODEL_DIR="$1"
        shift
        ;;
    --model-hf-name)
        shift
        BM_HF_MODEL="$1"
        shift
        ;;
    --model-served-name)
        shift
        BM_SERVED_NAME="$1"
        shift
        ;;
    --gpu-ids)
        shift
        BM_GPU_IDS="$1"
        shift
        ;;
    --listen-port)
        shift
        BM_LISTEN_PORT="$1"
        shift
        ;;
    --extra-server-args)
        shift
        BM_DEF_SERVER_EXTA_ARGS="$1"
        shift
        ;;
    *)
        usage
        break
    esac
    done
    run
}

main "$@"

