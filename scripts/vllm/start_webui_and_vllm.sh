#!/bin/bash
KEY_FILE=.webui_secret_key

if [ x"$RUN_OPEN_WEBUI" = x"1" ]; then

pushd /owebui/backend
OW_PORT="${OW_PORT:-8080}"
OW_HOST="${OW_HOST:-0.0.0.0}"

KEY_FILE=.webui_secret_key
if ! [ -e "$KEY_FILE" ]; then
    echo "Generating WEBUI_SECRET_KEY"
    echo $(head -c 12 /dev/random | base64) > "$KEY_FILE"
fi
WEBUI_SECRET_KEY=$(cat "$KEY_FILE")
echo "[RUN]: WEBUI_SECRET_KEY=$WEBUI_SECRET_KEY exec uvicorn main:app --host $OW_HOST --port $OW_PORT --forwarded-allow-ips '*' > /owebui/backend/logs.txt 2>&1 &"
WEBUI_SECRET_KEY="$WEBUI_SECRET_KEY" exec uvicorn main:app --host "$OW_HOST" --port "$OW_PORT" --forwarded-allow-ips '*' > /owebui/backend/logs.txt 2>&1 &
popd

fi

## run vllm
FP8_ARGS=""
if [ x"$FP8_SUPPORT" = x"1" ]; then
    FP8_ARGS="--quantization fp8"
fi
VLLM_PORT=$(expr $OW_PORT \+ 1)
args=" --host $OW_HOST --port $VLLM_PORT --model /root/.cache/huggingface/hub/$MODEL_DIR --tensor-parallel-size $VLLM_TP --pipeline-parallel-size $VLLM_PP  --use-v2-block-manager --block-size 32 --swap-space 16 --gpu-memory-utilization 0.9 --dtype auto --served-model-name $MODEL_NAME --max-num-seqs $VLLM_MAX_NUM_SEQS --max-model-len $VLLM_MAX_MODEL_LEN --max-num-batched-tokens $VLLM_MAX_MODEL_LEN --max-seq-len-to-capture $VLLM_MAX_MODEL_LEN $FP8_ARGS"
ENV="${ENV:-prod}"
if [ x"$ENV" = x"prod" ]; then
    ## --disable-log-stats
    args="$args --disable-log-requests"
fi
if [ x"$RUN_DEFAULT_MODEL" = x"1" ]; then
    args="$args $@"
else
    args="$@"
fi
echo "[RUN]: python3 -m vllm.entrypoints.openai.api_server $args"

python3 -m vllm.entrypoints.openai.api_server $args

