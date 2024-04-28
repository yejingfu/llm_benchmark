# PPINFER mlops

## 1. TensorRT-LLM

- Build TensorRT-LLM docker images
```bash
./script/build_trtllm_image.sh
```

- Llama2 model conversion and run inside rtllm docker
```bash
TRTLLM_BECKEND=/path/to/tensorrtllm_backend
MODELS_DIR=/path/to/models
IMAGE_TAG=ppinfer_triton_trtllm:24.02
CONTAINER_NAME=triton_test

docker run -itd --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v $TRTLLM_BECKEND:/tensorrtllm_backend -v $MODELS_DIR:/models --entrypoint /bin/bash --name $CONTAINER_NAME $IMAGE_TAG

docker exec -it $CONTAINER_NAME bash
## install requirements
> cd /app/tensorrt_llm/examples/llama
> pip install -r requirements.txt
## converting
> cd /app/tensorrt_llm/examples/llama
> python convert_checkpoint.py --model_dir /models/MythoMax-L2-13b --output_dir /models/trtllm-MythoMax-L2-13b/trt_checkpoint/fp16/8-gpu --dtype float16  --tp_size 8 --workers 8
## build engine
> cd /models/trtllm-MythoMax-L2-13b
> trtllm-build --checkpoint_dir ./trt_checkpoint/fp16/8-gpu --output_dir ./trt_engines/fp16/8-gpu --gemm_plugin float16 --use_fused_mlp --use_custom_all_reduce disable
## run
> cd /app/tensorrt_llm/examples
> mpirun -n 8 --allow-run-as-root python run.py --max_output_len 128 --tokenizer_dir /models/MythoMax-L2-13b --engine_dir /models/trtllm-MythoMax-L2-13b/trt_engines/fp16/8-gpu --input_text "long long time ago, 
```

