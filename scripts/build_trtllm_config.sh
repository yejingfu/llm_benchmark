#!/bin/bash
### Build trt ifb configure from <tensorrtllm_backend>/all_models/inflight_batcher_llm

PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/base.sh

## set DECOUPLE_MODE True to support streaming mode
DECOUPLE_MODE=True
BATCHING_STRATEGY=inflight_batching
GPU_MEM_FRAC=0.9
KV_CACHE_REUSE=False
MAX_TOKENS_IN_PAGED_KVCACHE=2560
MAX_ATTN_WINSIZE=2560

MAX_BATCH_SIZE=128
MAX_NUM_SEQS=128
MODEL_DIR=
TRTLLM_BACKEND_DIR=
OUTPUT_DIR=
TRT_ENGINE_DIR=

function usage() {
    LOG INFO "$PRG_NAME [options]"
    LOG INFO "  --model  The path to the target model folder"
    LOG INFO "  --trtllm_backend  The path to the tensorrtllm_backend folder"
    LOG INFO "  --output  The output folder path"
    LOG INFO "  --trt_engine  The path to tensorrtllm engine folder"
    LOG INFO "  --max_bs  The max batch size, default is $MAX_BATCH_SIZE"
    LOG INFO "  --max_num_seqs  The max num of sequences, default is $MAX_NUM_SEQS"
    exit
}

function check_arguments() {
    if [ x"$MODEL_DIR" = x"" ]; then
        LOG ERR "The model folder is empty, please set with --model"
    fi
    if [ x"$TRTLLM_BACKEND_DIR" = x"" ] || [ ! -d "$TRTLLM_BACKEND_DIR" ]; then
        LOG ERR "The folder of TRTLLM_BACKEND_DIR is invalid: $TRTLLM_BACKEND_DIR"
    fi
    if [ ! -d "$TRTLLM_BACKEND_DIR/all_models/inflight_batcher_llm" ];then
        LOG ERR "The ifb folder does not exist: $TRTLLM_BACKEND_DIR/all_models/inflight_batcher_llm"
    fi
    if [ x"$TRT_ENGINE_DIR" = x"" ] || [ ! -d "$TRT_ENGINE_DIR" ]; then
        LOG ERR "The folder of TRT_ENGINE_DIR is invalid: $TRT_ENGINE_DIR"
    fi
    if [ x"$OUTPUT_DIR" = x"" ]; then
        LOG ERR "The output folder is empty, please set with --output"
    fi
    if [ -d "$OUTPUT_DIR" ]; then
        LOG WARN "The output folder already exists, delete it now: $OUTPUT_DIR"
        rm -rf $OUTPUT_DIR
    fi
}

function build_config() {
    LOG INFO "Copy ifb config from $TRTLLM_BACKEND_DIR/all_models/inflight_batcher_llm to $OUTPUT_DIR"
    mkdir $OUTPUT_DIR
    cp -rf $TRTLLM_BACKEND_DIR/all_models/inflight_batcher_llm/* $OUTPUT_DIR/
    cp $CUR_DIR/trtllm/launch_triton_server.py.trtllm_backend $OUTPUT_DIR/launch_triton_server.py

    LOG INFO "Configure"
    cmd_file=$TRTLLM_BACKEND_DIR/tools/fill_template.py
    python3 $cmd_file -i $OUTPUT_DIR/preprocessing/config.pbtxt tokenizer_dir:$MODEL_DIR,tokenizer_type:auto,triton_max_batch_size:$MAX_BATCH_SIZE,preprocessing_instance_count:1
    python3 $cmd_file -i $OUTPUT_DIR/postprocessing/config.pbtxt tokenizer_dir:$MODEL_DIR,tokenizer_type:auto,triton_max_batch_size:$MAX_BATCH_SIZE,postprocessing_instance_count:1
    python3 $cmd_file -i $OUTPUT_DIR/ensemble/config.pbtxt triton_max_batch_size:$MAX_BATCH_SIZE
    python3 $cmd_file -i $OUTPUT_DIR/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:$MAX_BATCH_SIZE,decoupled_mode:$DECOUPLE_MODE,bls_instance_count:1,accumulate_tokens:False
    python3 $cmd_file -i $OUTPUT_DIR/tensorrt_llm/config.pbtxt triton_max_batch_size:$MAX_BATCH_SIZE,decoupled_mode:$DECOUPLE_MODE,max_beam_width:1,engine_dir:$TRT_ENGINE_DIR,max_tokens_in_paged_kv_cache:$MAX_TOKENS_IN_PAGED_KVCACHE,max_attention_window_size:$MAX_ATTN_WINSIZE,kv_cache_free_gpu_mem_fraction:$GPU_MEM_FRAC,exclude_input_in_output:True,enable_kv_cache_reuse:$KV_CACHE_REUSE,batching_strategy:$BATCHING_STRATEGY,max_queue_delay_microseconds:600,enable_trt_overlap:False,max_num_sequences:$MAX_NUM_SEQS
}

function main() {
    if [ "$#" -eq 0 ];then
        usage
    fi
    if [ x"$1" = x"-h" ] || [ x"$1" = x"--help" ]; then
        usage
    fi
    while [ "$#" -gt 0 ]; do
    case "$1" in
    --model)
        shift
        MODEL_DIR="$1"
        shift
        ;;
    --trtllm_backend)
        shift
        TRTLLM_BACKEND_DIR="$1"
        shift
        ;;
    --output)
        shift
        OUTPUT_DIR="$1"
        shift
        ;;
    --trt_engine)
        shift
        TRT_ENGINE_DIR="$1"
        shift
        ;;
    --max_bs)
        shift
        MAX_BATCH_SIZE="$1"
        shift
        ;;
    --max_num_seqs)
        shift
        MAX_NUM_SEQS="$1"
        shift
        ;;
    *)
        usage
        break
    esac
    done
    check_arguments
    build_config
}

main "$@"


