#!/bin/bash
PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/../scripts/base.sh

python $CUR_DIR/benchmark_client.py --backend novita --base-url https://api.novita.ai --api-key f872e830-2651-4db7-baec-12130efb5882 \
    --model meta-llama/llama-3-8b-instruct --endpoint-models /v3/openai/models --endpoint-chat /v3/openai/chat/completions --endpoint-completion /v3/openai/completions \
    --num-warmup-requests 1 --num-benchmark-requests 10  --stream \
    --tokenizer /models/Meta-Llama-3-8B-Instruct --dataset /models/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json \
    --log-file benchmark_novita.log
   # --dry-run

