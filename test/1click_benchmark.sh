#!/bin/bash
PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/util.sh

declare -A DEF_MODEL_HF_NAMES
DEF_MODEL_HF_NAMES["l31-8b"]="yejingfu/Meta-Llama-3.1-8B-Instruct"
DEF_MODEL_HF_NAMES["l31-8b-fp8"]="yejingfu/nmagic-Meta-Llama-3.1-8B-Instruct-FP8"
DEF_MODEL_HF_NAMES["l31-70b"]="yejingfu/Meta-Llama-3.1-70B-Instruct"
DEF_MODEL_HF_NAMES["l31-70b-fp8"]="yejingfu/nmagic-Meta-Llama-3.1-70B-Instruct-FP8"
DEF_GPU_TYPES=("4090" "h100" "h20" "h800" "a100" "a800" "l20" "l40s" "a6000")
DEF_TOKENIZER_HF_NAME="yejingfu/Meta-Llama-3.1-8B-Instruct"
DEF_DS_NAME="ShareGPT_V3_unfiltered_cleaned_split.json"
DEF_DS_HF_PATH="datasets/yejingfu/ShareGPT_V3/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

BM_DOCKER_IMAGE="image.paigpu.com/library/ppinfer_vllm:0.6.2.2"
BM_GPU_TYPE=
BM_GPU_IDS="0,1,2,3,4,5,6,7"
BM_GPU_MIG_IDS=
BM_MODELS=
BM_SPEC_MODEL=
BM_TPS=
BM_PREFIX_CACHE=0
BM_TOKENIZER_DIR=
BM_TEST_STRENGTH="high"
BM_OUT_DIR="out"
BM_SERVER_ONLY=
BM_CLIENT_ONLY=

if [[ x"$HF_ENDPOINT" = x"" ]];then
    HF_ENDPOINT="https://huggingface.co"
fi

function usage() {
    LOG INFO "$PRG_NAME [options]"
    LOG INFO "  --gpu-type The type of GPU, can be ${DEF_GPU_TYPES[@]}"
    LOG INFO "  --gpu-ids (optional)The list of GPU IDs, sperated by comma, default is 0,1,2,3,4,5,6,7"
    LOG INFO "  --gpu-mig-ids (optional)The list of GPU MIG instance IDs, if it is set, the --gpu-ids is ignored"
    LOG INFO "  --models Can be model short name within ${!DEF_MODEL_HF_NAMES[@]}, or huggingface model name, or local model absolute path. If many, separated by comma"
    LOG INFO "  --tps The tensor parallel setting for each model, seprated by comma"
    LOG INFO "  --tokenizer-dir (optional) Tht tokenizer folder path, if not set, download from huggingface: $DEF_TOKENIZER_HF_NAME and save to $CUR_DIR/tokenizer"
    LOG INFO "  --docker-image (optional) The docker image name, if not set, use default image: $BM_DOCKER_IMAGE"
    LOG INFO "  --spec-model (optional) The speculative decoding draft model"
    LOG INFO "  --test-strength(optional) The test strength level, can be: quick, low, middle, high, very-high, super-high, default is high"
    LOG INFO "  --out-dir(optional) The output folder to save the test results, default is out"
    LOG INFO "  --enable-prefix-cache(optional) Enable prefix caching feature during the tests"
    LOG INFO "  --setup(optional) Setup the testing envrionment, like install docker and git-lfs"
    LOG INFO "  --server-only(optional) Run vLLM engine directly"
    LOG INFO "  --client-only(optional) Run the client only, input server endpoint to connect"
    exit
}

function setup() {
    LOG INFO "Setup environment....."
    LOG INFO "[Install docker]"
    install_docker
    LOG INFO "[Install git-lfs"]
    if dpkg-query -W -f='${Status}' git-lfs 2>/dev/null | grep -q "install ok installed"; then
        LOG INFO "git-lfs is installed"
    else
        LOG INFO "install git-lfs"
        apt-get install -y git git-lfs
    fi
    pip install aiohttp numpy loguru tqdm transformers matplotlib
    LOG INFO "Setup DONE"
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

function guess_served_name() {
    local name=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    local ret=$name
    local sufix=""
    if [[ "$name" == *8b* ]]; then
        sufix="8b"
    elif [[ "$name" == *70b* ]]; then
        sufix="70b"
    elif [[ "$name" == *12b* ]]; then
        sufix="12b"
    elif [[ "$name" == *1b* ]]; then
        sufix="1b"
    fi
    if [[ "$name" == *fp8* ]]; then
        sufix="$sufix-fp8"
    elif [[ "$name" == *awq* ]]; then
        sufix="$sufix-int4"
    elif [[ "$name" == *888* ]]; then
        sufix="$sufix-888"
    fi
    if [[ "$name" == *llama-3.1* ]] || [[ "$name" == *l3.1-* ]]; then
        ret="llama31-$sufix"
    elif [[ "$name" == *llama-3.2* ]] || [[ "$name" == *l3.2-* ]] ; then
        ret="llama32-$sufix"
    elif [[ "$name" == *llama-3.3* ]] || [[ "$name" == *l3.3-* ]] ; then
        ret="llama33-$sufix"
    elif [[ "$name" == *llama-3* ]] || [[ "$name" == *l3-* ]] ; then
        ret="llama3-$sufix"
    elif [[ "$name" == *llama* ]]; then
        ret="llama-$sufix"
    else
        ret="unknown-$sufix"
    fi
    if [ x"$BM_GPU_MIG_IDS" != x"" ]; then
        ret="$ret-mig"
    fi
    echo $ret
}

function get_client_test_strength() {
    if [[ x"$BM_TEST_STRENGTH" = x"quick" ]];then
        ret="--context-lens 1000 --batches 1,2"
    elif [[ x"$BM_TEST_STRENGTH" = x"low" ]];then
        ret="--context-lens 1000,3000,5000 --batches 1,2,4,8"
    elif [[ x"$BM_TEST_STRENGTH" = x"middle" ]];then
        ret="--context-lens 1000,3000,5000,6000 --batches 1,2,4,8,10"
    elif [[ x"$BM_TEST_STRENGTH" = x"high" ]];then
        ret="--context-lens 1000,3000,5000,6000 --batches 1,2,4,6,8,10,12,15"
    elif [[ x"$BM_TEST_STRENGTH" = x"very-high" ]];then
        ret="--context-lens 1000,3000,5000,6000 --batches 1,2,3,4,5,6,7,8,9,10,12,15"
    else
        ret="--context-lens 1000,3000,5000,6000,10000 --batches 1,2,3,4,5,6,7,8,9,10,12,15"
    fi
    echo $ret
}

function run_benchmark() {
    local model_name="$1"
    local served_name=$model_name
    local model_dir=$model_name
    local tp="$2"
    echo ""
    LOG INFO "====== Run benchmark for model: $model_name, with tp: $tp ======"

    if [[ "$model_name" == /* ]]; then
        if [[ ! -d "$model_name" ]] || [[ ! -f "$model_name/config.json" ]] ; then
            LOG ERR "The the model does not exist on local disk: $model_name"
        fi
        served_name=$(guess_served_name $(basename "$model_name"))
        LOG INFO "Load model from local disk, served name: $served_name"
    else
        if contains_value "$model_name" "${!DEF_MODEL_HF_NAMES[@]}"; then
            for key in "${!DEF_MODEL_HF_NAMES[@]}"; do
                if [[ $model_name == $key ]]; then
                    model_name=${DEF_MODEL_HF_NAMES[$key]}
                    served_name=$key
                    break
                fi
            done
        else
            served_name=$(guess_served_name $(basename "$model_name"))
        fi
        model_dir=$CUR_DIR/$served_name
        LOG INFO "Downloading model from huggingface: $model_name and save to $model_dir, and set served_name as $served_name"
        if [[ ! -d "$model_dir" ]] || [ ! -f "$model_dir/config.json" ]; then
            download_model $model_name $model_dir
        else
            LOG INFO "Use model from local disk: $model_dir"
        fi
    fi
    if [[ x"$BM_CLIENT_ONLY" != x"" ]]; then
        if ! python3 -c "import loguru" &> /dev/null; then
            LOG INFO "Install loguru"
            pip install loguru
        fi
        local log_file_path="$BM_OUT_DIR/_gpu_${tp}x${BM_GPU_TYPE}_model_${served_name}_$RANDOM.txt"
        local client_args="--endpoint $BM_CLIENT_ONLY --tokenizer $BM_TOKENIZER_DIR --dataset $CUR_DIR/$DEF_DS_NAME --log-file $log_file_path --print-raw $(get_client_test_strength)"
        LOG INFO "[RUN]: $CUR_DIR/launch_benchmark.sh $client_args"
        $CUR_DIR/launch_benchmark.sh $client_args
        if [ -f "$log_file_path" ]; then
            python3 $CUR_DIR/find_best_throughput.py --log-files $log_file_path --output $log_file_path
        fi
    elif [[ x"$BM_SERVER_ONLY" != x"" ]]; then
        if ! python3 -c "import vllm" &> /dev/null; then
            LOG INFO "Install vllm v0.6.3.post1"
            pip install vllm==0.6.3.post1
            #pip install https://github.com/vllm-project/vllm/releases/download/v0.6.4.post1/vllm-0.6.4.post1+cu118-cp38-abi3-manylinux1_x86_64.whl
        fi
        local port=$((18000+RANDOM%100))
        local server_args="--model $model_dir --tensor-parallel-size $tp --port $port --served-model-name $served_name"
        if [[ "$served_name" == *llama33-* ]]; then
            server_args="$server_args --max-model-len 131072"
        elif [[ "$served_name" == *llama3-* ]]; then
            server_args="$server_args --max-model-len 8192"
        else
            server_args="$server_args --max-model-len 32768"
        fi
        server_args="$server_args --swap-space 16 --gpu-memory-utilization 0.92 --dtype auto --max-num-seqs 32 --disable-log-requests --enable-chunked-prefill"
        if [ $BM_PREFIX_CACHE -eq 1 ]; then
            server_args="$server_args --enable-prefix-caching"
        fi
        LOG INFO "[RUN] python3 -m vllm.entrypoints.openai.api_server $server_args"
        CUDA_VISIBLE_DEVICES=$BM_GPU_IDS python3 -m vllm.entrypoints.openai.api_server $server_args
    else
        local log_file_path="$BM_OUT_DIR/_gpu_${tp}x${BM_GPU_TYPE}_model_${served_name}_$RANDOM.txt"
        local server_args="--image-name $BM_DOCKER_IMAGE --model-served-name $served_name --model-dir $model_dir --gpu-ids $BM_GPU_IDS --tp $tp"
        if [ x"$BM_GPU_MIG_IDS" != x"" ]; then
            server_args="$server_args --gpu-mig-ids $BM_GPU_MIG_IDS"
        fi
        ## server extra args (except: --port, --tp, --model, --model-served-name)
        local server_extra_args="--swap-space 16 --gpu-memory-utilization 0.92 --dtype auto --max-num-seqs 32 --disable-log-requests"
        if [[ "$BM_IMAGE" == *_spec_decode* ]]; then
            if [[ x"$BM_SPEC_MODEL" == x"" ]]; then
                LOG ERR "No speculative draft model, please set with --spec-model"
            fi
            server_extra_args="$server_extra_args --num_speculative_tokens 5 --speculative_disable_by_batch_size 10 --speculative-model $BM_SPEC_MODEL"
        elif [[ "$BM_DOCKER_IMAGE" != *_kvcache* ]]; then
            server_extra_args="$server_extra_args --enable-chunked-prefill"
        fi
        ## set max model length
        if [[ "$served_name" == *llama33-* ]]; then
            server_extra_args="$server_extra_args --max-model-len 131072"
        elif [[ "$served_name" == *llama3-* ]]; then
            server_extra_args="$server_extra_args --max-model-len 8192"
        else
            server_extra_args="$server_extra_args --max-model-len 32768"
        fi
        ## enable prefix caching
        if [ $BM_PREFIX_CACHE -eq 1 ]; then
            server_extra_args="$server_extra_args --enable-prefix-caching"
        fi
        local client_args="--tokenizer $BM_TOKENIZER_DIR --dataset $CUR_DIR/$DEF_DS_NAME --log-file $log_file_path --print-raw $(get_client_test_strength)"
        LOG INFO "[RUN]: $CUR_DIR/launch_benchmark.sh $server_args $client_args --extra-server-args $server_extra_args"
        $CUR_DIR/launch_benchmark.sh $server_args $client_args --extra-server-args $server_extra_args
        if [ -f "$log_file_path" ]; then
            python3 $CUR_DIR/find_best_throughput.py --log-files $log_file_path --output $log_file_path
        fi
    fi
}

function run() {
    LOG INFO "BM_GPU_TYPE: $BM_GPU_TYPE, BM_GPU_IDS: $BM_GPU_IDS, BM_MODEL_NAME: $BM_MODEL_NAME, BM_DOCKER_IMAGE: $BM_DOCKER_IMAGE"
    if [ x"$BM_GPU_TYPE" = x"" ];then
        LOG ERR "Empty gpu type, please set by --gpu-type"
    fi
    if ! contains_value "$BM_GPU_TYPE" "${DEF_GPU_TYPES[@]}"; then
        LOG ERR "Invalid GPU type $BM_GPU_TYPE, should set by --gpu-type and its value shoud be in ${DEF_GPU_TYPES[@]}"
    fi
    if [ x"$BM_MODELS" = x"" ]; then
        LOG ERR "Empty models, please set by --models"
    fi
    if [ x"$BM_TPS" = x"" ]; then
        LOG ERR "The tensor parallel is empty, please set by --tp"
    fi
    if [ x"$BM_GPU_IDS" = x"" ];then
        LOG ERR "Empty gpu id list, please set by --gpu-ids"
    fi
    local model_names=($(split_string $BM_MODELS))
    local tps=($(split_string $BM_TPS))
    local num_models=${#model_names[@]}
    local num_tps=${#tps[@]}
    local num=$((num_models-num_tps))
    if [ $num -gt 0 ] && [ $num_tps -eq 1 ]; then
        for i in $(seq 1 $num); do
            tps+=(${tps[0]})
        done
    fi
    num_tps=${#tps[@]}
    if [[ $num_tps -ne $num_models ]]; then
        LOG ERR "The num of models and tps are mismatched"
    fi

    if [[ x"$BM_TOKENIZER_DIR" = x"" ]];then
        BM_TOKENIZER_DIR=$CUR_DIR/tokenizer
        LOG INFO "Set tokenizer to $BM_TOKENIZER_DIR"
    fi
    if [[ -f "$BM_TOKENIZER_DIR/config.json" ]]; then
        LOG INFO "Use the existing tokenizer $BM_TOKENIZER_DIR"
    else
        LOG INFO "Downloading default tokenizer from huggingface: $DEF_TOKENIZER_HF_NAME and save to $BM_TOKENIZER_DIR"
        download_model $DEF_TOKENIZER_HF_NAME $BM_TOKENIZER_DIR
    fi
    if [[ ! -f "$CUR_DIR/$DEF_DS_NAME" ]]; then
        LOG INFO "Downloading dataset from: $HF_ENDPOINT/$DEF_DS_HF_PATH"
        wget $HF_ENDPOINT/$DEF_DS_HF_PATH
    fi
    if [[ ! -d $BM_OUT_DIR ]]; then
        mkdir $BM_OUT_DIR
    fi
    for i in $(seq 0 $((num_models - 1)));do
        run_benchmark "${model_names[$i]}" ${tps[$i]}
    done
    echo ""
    LOG INFO "[DONE]"
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
    --gpu-mig-ids)
        shift
        BM_GPU_MIG_IDS="$1"
        shift
        ;;
    --models)
        shift
        BM_MODELS="$1"
        shift
        ;;
    --tps)
        shift
        BM_TPS="$1"
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
    --spec-model)
        shift
        BM_SPEC_MODEL="$1"
        shift
        ;;
    --test-strength)
        shift
        BM_TEST_STRENGTH="$1"
        shift
        ;;
    --enable-prefix-cache)
        shift
        BM_PREFIX_CACHE=1
        ;;
    --setup)
        shift
        setup
        ;;
    --server-only)
        shift
        BM_SERVER_ONLY="yes"
        ;;
    --client-only)
        shift
        BM_CLIENT_ONLY="$1"
        shift
        ;;
    --out-dir)
        shift
        BM_OUT_DIR="$1"
        shift
        ;;
    *)
        LOG INFO "Unknown argument $1"
        usage
        break
    esac
    done
    run
}

RANDOM=`date +%s`
main "$@"

