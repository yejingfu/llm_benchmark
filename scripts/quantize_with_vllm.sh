#!/bin/bash
## convert hf model weights from float16 to fp8 which can be loaded in vllm

PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/base.sh

VLLM_IMAGE=ppinfer/vllm-openai:v0.4.2
DOCKER_NAME="vllm_quantization"
ACT_SCHEME="static"
#ACT_SCHEME="dynamic"
MAX_SEQ_LEN=4096

function usage() {
    LOG INFO "$PRG_NAME [options]"
    LOG INFO "  --model    The folder path to float16 model"
    LOG INFO "  --kind     The kind of quantization, its value should be weight or kvcache"
    LOG INFO "  --datasets The path to datasets, make use it contains ultrachat_200k and cnn_dailymail"
    LOG INFO "  --output   The quantized model location, do not need to set if kind if kvcache"
    LOG INFO "  --vllm     The vllm folder path, do not need to set if kind is weight"
    exit
}

function do_quantize() {
    local hf_model="$1"
    local kind="$2"
    local output="$3"
    local datasets="$4"
    local vllm_dir="$5"
    LOG INFO "Do quanization: $hf_model, $kind"
    LOG INFO "datasets: $datasets, vllm: $vllm_dir, output: $output"
    local tmpscript=tmp$RANDOM.sh
    opts="-d --gpus all --privileged --ipc=host --net=host --ulimit stack=67108864 --ulimit memlock=-1 -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --name $DOCKER_NAME"
    opts="$opts -v $hf_model:$hf_model -v $datasets:/root/.cache/huggingface/datasets -v $CUR_DIR:/tmp_scripts"
    local message="Succeed:"
    if [ "$kind" = "weight" ]; then
        opts="$opts -v $output:$output --entrypoint /usr/bin/bash $VLLM_IMAGE /tmp_scripts/$tmpscript"
        echo "python3 /tmp_scripts/vllm/quantize_weight_autofp8.py --model-id $hf_model --save-dir $output --activation-scheme $ACT_SCHEME --max-seq-len $MAX_SEQ_LEN --num-samples 1024">>$CUR_DIR/$tmpscript
        message="$message quantize model from $hf_model to $output"
    else
        opts="$opts -v $vllm_dir:/vllm --entrypoint /usr/bin/bash $VLLM_IMAGE /tmp_scripts/$tmpscript"
        echo "pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com nvidia-ammo~=0.7.3">>$CUR_DIR/$tmpscript
        echo "mkdir /tmp_fp8">>$CUR_DIR/$tmpscript
        echo "python3 /vllm/examples/fp8/quantizer/quantize.py --model_dir $hf_model --dtype float16 --qformat fp8 --kv_cache_dtype fp8 --output_dir /tmp_fp8 --calib_size 512 --tp_size 8">>$CUR_DIR/$tmpscript
        echo "python3 /vllm/examples/fp8/extract_scales.py --quantized_model /tmp_fp8 --tp_size 8">>$CUR_DIR/$tmpscript
        echo "cp /tmp_fp8/kv_cache_scales.json $hf_model/">>$CUR_DIR/$tmpscript
        message="$message create kv cache scale file to $hf_model/kv_cache_scales.json"
    fi

    echo "touch /tmp_scripts/completed">>$CUR_DIR/$tmpscript
    chmod +x $CUR_DIR/$tmpscript
    LOG INFO "docker run $opts"
    cat $CUR_DIR/$tmpscript
    docker run $opts

    while true
    do
        LOG INFO "Wait task in container $DOCKER_NAME"
        sleep 100
        if [ -f "$CUR_DIR/completed" ];then
            LOG INFO "DONE"
            rm -f "$CUR_DIR/completed"
            rm -f "$CUR_DIR/$tmpscript"
            break
        fi
        docker ps | grep $DOCKER_NAME
        if [ $? -eq 1 ]; then
            ## The container is not found
            LOG ERR "The container might be stopped unexpectedly"
        fi
    done
    docker ps -a | grep $DOCKER_NAME
    if [ $? -eq 0 ]; then
        ## remove the container if it still exists
        docker rm -f $DOCKER_NAME
    fi
    LOG INFO "$message"

}

function main() {
    if [ "$#" -eq 0 ]; then
        usage
    fi

    while [ "$#" -gt 0 ]; do
    case "$1" in
    --model)
        shift
        local hf_model="$1"
        shift
        ;;
    --kind)
        shift
        local kind="$1"
        shift
        ;;
    --datasets)
        shift
        local hf_datasets="$1"
        shift
        ;;
    --output)
        shift
        local output="$1"
        shift
        ;;
    --vllm)
        shift
        local vllm_dir="$1"
        shift
        ;;
    *)
        usage
    esac
    done

    if [ x"$hf_model" = x"" ] || [ ! -d "$hf_model" ]; then
        LOG ERR "Invalid model path: $hf_model"
    fi
    if [ x"$kind" != x"weight" ] && [ x"$kind" != x"kvcache" ]; then
        LOG ERR "The kind of quantization should be weight or kvcache"
    fi
    if [ x"$hf_datasets" = x"" ]; then
        LOG ERR "The datasets folder is not set"
    fi
    if [ ! -d "$hf_datasets/ultrachat_200k" ]; then
        LOG ERR "The ultrachat_200k does not exist in $hf_datasets Please manually download it from https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k"
    fi
    if [ ! -d "$hf_datasets/cnn_dailymail" ]; then
        LOG ERR "The cnn_dailymail does not exist in $hf_datasets Please manually download it from https://huggingface.co/datasets/cnn_dailymail"
    fi
    if [ "$kind" = "weight" ]; then
        if [ x"$output" = x"" ]; then
            LOG ERR "The output folder is not set, please set with --output"
        fi
        if [ -d "$output" ]; then
            LOG ERR "The output folder already exists, please delete it firstly"
        fi
    else
        if [ x"$vllm_dir" = x"" ] || [ ! -d "$vllm_dir" ]; then
            LOG ERR "The vllm folder is invalid"
        fi
    fi

    do_quantize $hf_model $kind "$output" "$hf_datasets" "$vllm_dir"
}

main "$@"

