#!/bin/bash
PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/util.sh

## relaunch the server docker for every test case
BM_RESET_SERVER=0

# server side
BM_IMAGE="image.paigpu.com/library/ppinfer_vllm:0.6.2.2"
BM_MODEL_DIR=
BM_HF_MODEL=
BM_SERVED_NAME=
BM_GPU_IDS="0,1,2,3,4,5,6,7"
BM_GPU_MIG_IDS=
BM_TP=1
BM_MAX_CTX_LEN="32768"
BM_PREFIX_CACHE=1
BM_LISTEN_PORT=
BM_REAL_LISTEN_PORT=
BM_DEF_SERVER_EXTA_ARGS="--swap-space 16 --gpu-memory-utilization 0.92 --dtype auto --max-num-seqs 32 --disable-log-requests --enable-chunked-prefill"
BM_DEF_SERVER_EXTA_ARGS_KVCACHE="--swap-space 16 --gpu-memory-utilization 0.92 --dtype auto --max-num-seqs 32 --disable-log-requests"

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
    LOG INFO "  --gpu-ids (optional) The list of GPU IDs used to serve LLM, default is $BM_GPU_IDS"
    LOG INFO "  --tp (optional) The tensor parallel setting, default is $BM_TP"
    LOG INFO "  --max-ctx-len (optional) The max length of context, default is $BM_MAX_CTX_LEN"
    LOG INFO "  --listen-port (optional) The http listening port"
    LOG INFO "  --disable-prefix-cache (optional) Disable prefix-caching"
    LOG INFO "  --extra-server-args (optional) The extra server argument, default is: $BM_DEF_SERVER_EXTA_ARGS"
    LOG INFO " client side:"
    LOG INFO "  --endpoint (optional) The LLM server URL, if not set, use http://localhost:<port>/v1"
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

function get_avail_docker_name() {
    for i in {0..20}; do
        name=benchmark_$i
        ret=$(check_container_exists $name)
        if [[ $ret -ne 1 ]];then
            echo $name
            break
        fi
    done
}

function run_server() {
    ## Server side
    docker_name=$1
    shift
    local port=$BM_LISTEN_PORT
    if [[ x"$port" == x"" ]];then
        port=$((19000+RANDOM%100))
        port=$(get_available_port $port)
    fi
    BM_REAL_LISTEN_PORT=$port
    LOG INFO "==== [RUN] docker run --name $docker_name $@ --port $port"
    docker run --name $docker_name $@ --port $port
    try=0
    while [ $try -lt 30 ]; do
        LOG INFO "Waiting for docker ready ($try): $docker_name..."
        sleep 10
        ret=$(docker logs $docker_name) #> /dev/null 2>&1)
        LOG INFO ">>>[docker logs $docker_name ($try)]: $ret <<<<"
        if [ x"$ret" = x"" ]; then
            docker rm -f $docker_name
            LOG ERR "No log returned from: $docker_name"
        elif echo "$ret" | grep -q "ERROR"; then
            docker rm -f $docker_name
            LOG ERR "Failed to run docker instance: $docker_name"
        fi
        if echo "$ret" | grep -q "Route: /v1/chat/completions"; then
            LOG INFO "Succeed to run docker $docker_name \n\n"
            break
        fi
        try=$((try + 1))
    done
    if [ $try -eq 30 ];then
        docker rm -f $docker_name
        LOG ERR "Failed to run docker instance in 200 seconds: $docker_name"
    fi
    #return $port ## cannot return the value if its larger than 256
}

function run() {
    BM_REAL_LISTEN_PORT=$BM_LISTEN_PORT
    need_run_server=0
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
        if [ $BM_PREFIX_CACHE -eq 1 ]; then
            BM_DEF_SERVER_EXTA_ARGS="$BM_DEF_SERVER_EXTA_ARGS --enable-prefix-caching"
        fi
        if [[ "$BM_IMAGE" == *_kvcache* ]]; then
            BM_DEF_SERVER_EXTA_ARGS=$BM_DEF_SERVER_EXTA_ARGS_KVCACHE
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
        docker_args="-d --gpus all --privileged --ipc=host --net=host -v $BM_MODEL_DIR:/this_model"
        if [ x"$BM_GPU_MIG_IDS" == x"" ]; then
            docker_args="$docker_args -e CUDA_VISIBLE_DEVICES=$BM_GPU_IDS"
        else
            docker_args="$docker_args -e CUDA_VISIBLE_DEVICES=$BM_GPU_MIG_IDS"
        fi
        if [[ "$BM_MODEL_DIR" == *-888* ]]; then
            docker_args="$docker_args -e VLLM_ATTENTION_BACKEND=FLASHINFER"
            BM_DEF_SERVER_EXTA_ARGS="$BM_DEF_SERVER_EXTA_ARGS --kv_cache_dtype fp8"
        fi

        if [[ "$BM_IMAGE" == *_kvcache* ]]; then
            docker_args="$docker_args -e FULL_KV_LAYERS=6 -e SLIDING_WINDOW_WIDTH=1280"
        fi
        server_args="--tensor-parallel-size $BM_TP --model /this_model"
        if [ x"$BM_SERVED_NAME" != x"" ]; then
            server_args="$server_args --served-model-name $BM_SERVED_NAME"
        fi
        server_args="$server_args $BM_DEF_SERVER_EXTA_ARGS --max-model-len $BM_MAX_CTX_LEN"
        need_run_server=1
    fi

    if [[ $BM_RESET_SERVER -ne 1 ]]; then
        docker_name=$(get_avail_docker_name)
        run_server $docker_name $docker_args $BM_IMAGE $server_args
        if [ x"$BM_LOG_FILE" != x"" ]; then
            echo "docker run --name $docker_name $docker_args $BM_IMAGE $server_args --port $BM_REAL_LISTEN_PORT">>$BM_LOG_FILE
        fi
    fi
    ## Client side
    LOG INFO "Use endpoint: $BM_ENDPOINT"
    if [ x"$BM_TOKENIZER_PATH" = x"" ]; then
        LOG ERR "Please set --tokenizer"
    fi
    if [ x"$BM_DATASET_PATH" = x"" ]; then
        LOG ERR "Please set --dataset"
    fi
    local args="--tokenizer $BM_TOKENIZER_PATH --dataset $BM_DATASET_PATH --model $BM_SERVED_NAME"
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
        if [[ $in_len -eq 6100 ]];then
            out_len=170
        else
            out_len=$((in_len/$BM_CTX_LEN_RATIO))
        fi
        for parallel in ${BM_PARALLELS[@]};do
            ## start server every time
            if [[ $BM_RESET_SERVER -eq 1 ]]; then
                docker_name=$(get_avail_docker_name)
                run_server $docker_name $docker_args $BM_IMAGE $server_args
                if [ x"$BM_LOG_FILE" != x"" ]; then
                    echo "docker run --name $docker_name $docker_args $BM_IMAGE $server_args --port $BM_REAL_LISTEN_PORT">>$BM_LOG_FILE
                fi
            fi
            local args2="--endpoint $BM_ENDPOINT"
            if [ x"$BM_ENDPOINT" = x"" ]; then
                args2="--endpoint http://localhost:$BM_REAL_LISTEN_PORT/v1"
            fi
            args2="$args2 $args --sampling-policy normal --prompt-len-mean $in_len --prompt-len-std 10 --output-len-mean $out_len --output-len-std 6 --parallel $parallel"
            if [ $parallel -eq 1 ]; then
                ## single batch, use less requests to save time
                args2="$args2 --num-requests $BM_NUM_REQUESTS_SINGLE_BATCH"
            else
                args2="$args2 --num-requests $BM_NUM_REQUESTS"
            fi
            echo "==== [Run]: python3 $CUR_DIR/benchmark_client.py $args2"
            echo ""
            python3 $CUR_DIR/benchmark_client.py $args2
            ## stop server every time
            if [[ $BM_RESET_SERVER -eq 1 ]]; then
                LOG INFO "Delete the docker instance: $docker_name"
                remove_docker_container $docker_name
            fi
        done
    done

    if [ x"$docker_name" != x"" ] && [ $BM_RESET_SERVER -ne 1 ];then
        LOG INFO "Delete the docker instance: $docker_name"
        remove_docker_container $docker_name
    fi
    LOG INFO "\nThe benchmark test is completed\n"
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
    --gpu-mig-ids)
        shift
        BM_GPU_MIG_IDS="$1"
        shift
        ;;
    --tp)
        shift
        BM_TP=$1
        shift
        ;;
    --listen-port)
        shift
        BM_LISTEN_PORT=$1
        shift
        ;;
    --disable-prefix-cache)
        shift
        BM_PREFIX_CACHE=0
        ;;
    --extra-server-args)
        shift
        BM_DEF_SERVER_EXTA_ARGS="$1"
        shift
        ;;
    --max-ctx-len)
        shift
        BM_MAX_CTX_LEN="$1"
        shift
        ;;
    *)
        usage
        break
    esac
    done
    run
}

RANDOM=`date +%s`
main "$@"

