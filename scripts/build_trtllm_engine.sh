#!/bin/bash
# convert hf check points and call trtllm-build to build trtllm engine

PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/base.sh

TRTLLM_IMG_NAME=ppinfer_triton_trtllm:24.02
TRTLLM_CONTAINER_NAME=trtllm_build
HF_MODEL_DIR=
TP_SIZE=8
PP_SIZE=1
DTYPE=float16
ACTION=both
MAX_INPUT_LEN=4096
MAX_OUTPUT_LEN=3072
MAX_NUM_TOKENS=4096

PRESET_ACTIONS=("both" "convert" "build")
PRESET_DTYPES=("float16" "fp8" "int8")

function usage() {
    LOG INFO "$PRG_NAME [options]"
    LOG INFO "      Convert HF check points and call trtllm-build to build trtllm engine"
    LOG INFO "      It read the HF weights from {model} folder and convert checkponts and build engine into {model}-trtllm"
    LOG INFO "  --model   The path to the source model folder"
    LOG INFO "  --dtype   The converted weight type, can be [float16]"
    LOG INFO "  --action  What to do, can be [both], convert, build"
    exit
}

function convert_checkpoint() {
    local hf_model=$1
    local cp_model=$2
    local dtype=$3
    local tmp_script=tmp$RANDOM.sh
    if [ ! -d "$hf_model" ]; then
        LOG ERR "The HF model folder does not exist"
    fi
    if [ -d "$cp_model" ]; then
        LOG WARN "Remove the checkpoint folder: $cp_model"
        rm -rf $cp_model
    fi
    LOG INFO "Convert HF checkponts($hf_model) to TRTLLM checkpoint ($cp_model)"
    # mkdir -p $cp_model
    local workers=$(expr $TP_SIZE \* $PP_SIZE)
    echo "cd /app/tensorrt_llm/examples/llama">>$CUR_DIR/$tmp_script
    if [[ "$hf_model" =~ "Mixtral" ]];then
        LOG WARN "Fix Mixtral model out of GPU memory issue by using CPU memory"
        echo "cp /tmp_scripts/convert_checkpoint.py.trtllm_llama /app/tensorrt_llm/examples/llama/convert_checkpoint.py">>$CUR_DIR/$tmp_script
    fi
    local args="--model_dir $hf_model --output_dir $cp_model --dtype $dtype --tp_size $TP_SIZE --pp_size $PP_SIZE --workers $workers"
    echo "python convert_checkpoint.py $args">>$CUR_DIR/$tmp_script
    echo "touch /tmp_scripts/completed">>$CUR_DIR/$tmp_script
    chmod +x $CUR_DIR/$tmp_script
    opts="--rm -d --gpus all --privileged --ipc=host --net=host --ulimit stack=67108864 --ulimit memlock=-1 -v $hf_model:$hf_model -v $cp_model:$cp_model -v $CUR_DIR:/tmp_scripts --name $TRTLLM_CONTAINER_NAME --entrypoint /bin/bash $TRTLLM_IMG_NAME /tmp_scripts/$tmp_script"

    LOG INFO "docker run $opts"
    LOG INFO "Run the following command inside container:"
    cat $CUR_DIR/$tmp_script
    docker run $opts

    while true
    do
        LOG INFO "Wait task in container $TRTLLM_CONTAINER_NAME"
        sleep 40
        if [ -f "$CUR_DIR/completed" ];then
            LOG INFO "DONE"
            rm -f "$CUR_DIR/completed"
            rm -f "$CUR_DIR/$tmp_script"
            break
        fi
        docker ps | grep $TRTLLM_CONTAINER_NAME
        if [ $? -eq 1 ]; then
            ## The container is not found
            LOG ERR "The container might be stopped unexpectedly"
        fi
    done
    docker ps -a | grep $TRTLLM_CONTAINER_NAME
    if [ $? -eq 0 ]; then
        ## remove the container if it still exists
        docker rm -f $TRTLLM_CONTAINER_NAME
    fi
    LOG INFO "Succeed to convert HF checkpoints to $cp_model"
}

function trtllm_build() {
    local cp_dir="$1"
    local engine_dir="$2"
    if [ ! -f "$cp_dir/config.json" ];then
        LOG ERR "The trtllm checkpoint folder does not exist under $cp_dir"
    fi
    LOG INFO "Build TRTLLM engine from $cp_dir to $engine_dir"
    local tmp_script=tmp$RANDOM.sh
    local args="--checkpoint_dir $cp_dir --output_dir $engine_dir --gemm_plugin float16 --use_fused_mlp --use_custom_all_reduce disable"
    args="$args --max_input_len $MAX_INPUT_LEN --max_output_len $MAX_OUTPUT_LEN --max_num_tokens $MAX_NUM_TOKENS"
    echo "trtllm-build $args">>$CUR_DIR/$tmp_script
    echo "touch /tmp_scripts/completed_engine">>$CUR_DIR/$tmp_script
    chmod +x $CUR_DIR/$tmp_script
    opts="--rm -d --gpus all --privileged --ipc=host --net=host --ulimit stack=67108864 --ulimit memlock=-1 -v $cp_dir:$cp_dir -v $engine_dir:$engine_dir -v $CUR_DIR:/tmp_scripts --name $TRTLLM_CONTAINER_NAME --entrypoint /bin/bash $TRTLLM_IMG_NAME /tmp_scripts/$tmp_script"

    LOG INFO "docker run $opts"
    LOG INFO "Run the following command inside container:"
    cat $CUR_DIR/$tmp_script
    docker run $opts

    while true
    do
        LOG INFO "Wait task in container $TRTLLM_CONTAINER_NAME"
        sleep 40
        if [ -f "$CUR_DIR/completed_engine" ];then
            LOG INFO "DONE"
            rm -f "$CUR_DIR/completed_engine"
            rm -f "$CUR_DIR/$tmp_script"
            break
        fi
        docker ps | grep $TRTLLM_CONTAINER_NAME
        if [ $? -eq 1 ]; then
            ## The container is not found
            LOG ERR "The container might be stopped unexpectedly"
        fi
    done
    docker ps -a | grep $TRTLLM_CONTAINER_NAME
    if [ $? -eq 0 ]; then
        ## remove the container if it still exists
        docker rm -f $TRTLLM_CONTAINER_NAME
    fi
    LOG INFO "Succeed to build trtllm engine to $engine_dir"
}

function check_arguments() {
    if [ x"$HF_MODEL_DIR" = x"" ]; then
        LOG ERR "The HF_MODEL_DIR is empty, please set it with --model <MODEL_DIRECTORY>"
    fi
    if [ ! -d "$HF_MODEL_DIR" ]; then
        LOG ERR "The model path is not valid folder"
    fi
    # rstrip right special char(/)
    HF_MODEL_DIR=${HF_MODEL_DIR%*/}
    if [[ ! ${PRESET_ACTIONS[@]} =~ $ACTION ]]; then
        LOG ERR "The action value should be in ${PRESET_ACTIONS[@]}"
    fi
    if [[ ! ${PRESET_DTYPES[@]} =~ $DTYPE ]]; then
        LOG ERR "The dtype value should be in ${PRESET_DTYPES[@]}"
    fi
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
    --dtype)
        shift
        DTYPE="$1"
        shift
        ;;
    --action)
        shift
        ACTION="$1"
        shift
        ;;
    *)
        usage
    esac
    done

    check_arguments

    local trtllm_cp_dir="${HF_MODEL_DIR}-trtllm/checkpoint/$DTYPE/8-gpu"
    local trtllm_engine_dir="${HF_MODEL_DIR}-trtllm/engine/$DTYPE/8-gpu"
    if [ "$ACTION" = "both" ] || [ "$ACTION" = "convert" ]; then
        convert_checkpoint $HF_MODEL_DIR $trtllm_cp_dir $DTYPE
    fi
    if [ "$ACTION" = "both" ] || [ "$ACTION" = "build" ]; then
        trtllm_build $trtllm_cp_dir $trtllm_engine_dir
    fi
}

main "$@"

