#!/bin/bash
## convert hf model weights from float16 to fp8

PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/base.sh

HF_MODEL_DIR=
HF_DATASETS_DIR=
OUTPUT_DIR=
USE_CPU=0

TRTLLM_IMAGE=ppinfer_triton_trtllm:24.02
QUANTIZE_CONTAINER="quantize"
MAX_SEQ_LEN=4096
MAX_BS=128

function usage() {
    LOG INFO "$PRG_NAME [options]"
    LOG INFO "  --model    The folder path to float16 model"
    LOG INFO "  --datasets The path to datasets, make use it contains cnndailymail"
    LOG INFO "  --output   The quantized model location"
    LOG INFO "  --use-cpu  Use CPU and its memory to quantize model, specially get OOM when using GPU"
    exit
}

function check_arguments() {
    if [ x"$HF_MODEL_DIR" = x"" ] || [ ! -d "$HF_MODEL_DIR" ]; then
        LOG ERR "Invalid model path: $HF_MODEL_DIR"
    fi
    if [ x"$HF_DATASETS_DIR" = x"" ]; then
        LOG ERR "The datasets folder is not set"
    fi
    if [ ! -d "$HF_DATASETS_DIR/cnn_dailymail" ]; then
        LOG ERR "The cnn_dailymail does not exist in $HF_DATASETS_DIR Please manually download it from https://huggingface.co/datasets/cnn_dailymail"
    fi
    if [ x"$OUTPUT_DIR" = x"" ]; then
        LOG ERR "The output folder is not set"
    fi
    if [ -d "$OUTPUT_DIR" ]; then
        LOG ERR "The output folder already exists, please delete it firstly"
    fi
}

function do_quantize() {
    LOG INFO "======== quantize now ==========="
    LOG INFO "HF_MODEL_DIR: $HF_MODEL_DIR"
    LOG INFO "HF_DATASETS_DIR: $HF_DATASETS_DIR"
    LOG INFO "OUTPUT_DIR: $OUTPUT_DIR"
    LOG INFO "USE_CPU: $USE_CPU"

    docker ps -a | grep $QUANTIZE_CONTAINER
    if [ $? -eq 0 ]; then
        LOG INFO "The container $QUANTIZE_CONTAINER is running, delete it now"
        docker rm -f $QUANTIZE_CONTAINER
    fi

    local tmp_script=tmp$RANDOM.sh
    echo "cd /app/tensorrt_llm/examples/quantization">>$CUR_DIR/$tmp_script
    if [ $USE_CPU -eq 1 ];then
        echo "cp /tmp_scripts/trtllm/quantize_by_ammo.py.cpu /usr/local/lib/python3.10/dist-packages/tensorrt_llm/quantization/quantize_by_ammo.py">>$CUR_DIR/$tmp_script
    fi
    echo "python quantize.py --model_dir $HF_MODEL_DIR --qformat fp8 --max_seq_length $MAX_SEQ_LEN --batch_size $MAX_BS --output_dir $OUTPUT_DIR --tp_size 8 --pp_size 1 --kv_cache_dtype fp8" >>$CUR_DIR/$tmp_script
    echo "touch /tmp_scripts/completed">>$CUR_DIR/$tmp_script
    chmod +x $CUR_DIR/$tmp_script

    opts=" -itd --rm --gpus all --ipc=host --ulimit memlock=-1 --name $QUANTIZE_CONTAINER --entrypoint /bin/bash"
    opts="$opts -v $HF_MODEL_DIR:$HF_MODEL_DIR -v $HF_DATASETS_DIR:/root/.cache/huggingface/datasets -v $OUTPUT_DIR:$OUTPUT_DIR -v $CUR_DIR:/tmp_scripts"

    LOG INFO "docker run $opts $TRTLLM_IMAGE /tmp_scripts/$tmp_script"
    cat $CUR_DIR/$tmp_script
    docker run $opts $TRTLLM_IMAGE /tmp_scripts/$tmp_script

    while true
    do
        LOG INFO "Wait task in container $QUANTIZE_CONTAINER "
        sleep 40
        if [ -f "$CUR_DIR/completed" ];then
            LOG INFO "DONE"
            rm -f "$CUR_DIR/completed"
            rm -f "$CUR_DIR/$tmp_script"
            break
        fi
        docker ps | grep $QUANTIZE_CONTAINER
        if [ $? -eq 1 ]; then
            ## The container is not found
            LOG ERR "The container might be stopped unexpectedly"
        fi
    done
    docker ps -a | grep $QUANTIZE_CONTAINER
    if [ $? -eq 0 ]; then
        ## remove the container if it still exists
        docker rm -f $QUANTIZE_CONTAINER
    fi
    LOG INFO "Succeed to convert HF checkpoints to $cp_model"
}

function main() {
    if [ "$#" -eq 0 ]; then
        usage
    fi

    while [ "$#" -gt 0 ]; do
    case "$1" in
    --model)
        shift
        HF_MODEL_DIR="$1"
        shift
        ;;
    --datasets)
        shift
        HF_DATASETS_DIR="$1"
        shift
        ;;
    --output)
        shift
        OUTPUT_DIR="$1"
        shift
        ;;
    --use-cpu)
        shift
        USE_CPU=1
        ;;
    *)
        usage
    esac
    done

    check_arguments
    do_quantize
}

main "$@"


