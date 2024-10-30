import os
import asyncio
import aiohttp
import argparse
import time
from loguru import logger
import util

EVAL_MODEL="local-completions"
## reference: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
EVAL_TASKS="mmlu,gsm8k,gpqa,bbh,mathqa"
EVAL_CONCURRENT=10
EVAL_LOG_SAMPLE=True
EVAL_CHECK_MODEL=False
EVAL_FEWSHOT=None
EVAL_BS=None
EVAL_LIMIT=10

try:
    import lm_eval
    from lm_eval.tasks import TaskManager
    from lm_eval.loggers import EvaluationTracker
    from lm_eval.utils import handle_non_serializable, make_table
except ImportError:
    print("Please install lm_eval by 'pip install lm-eval' and 'pip install lm-eval[api]'")
    exit(0)

async def check_logprobs(url, model):
    logger.info(f"Checking logprobs with url: {url}, model: {model}")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(3600)) as session:
        payload = {
            "model": model,
            "prompt": "The following are multiple choice questions (with answers) about philosophy.\n\nFor Socrates, the soul is harmed by lack of _____.\nA. knowledge\nB. wealth\nC. community\nD. courage\nAnswer: B",
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": 1,
            "seed": 1234,
            "echo": True
        }
        headers = {"Authorization": "Bearer "}
        for i in range(2):
            async with session.post(url=url, json=payload, headers=headers) as response:
                if response.status == 200:
                    ret = await response.json()
                    logger.info(f"Succeed to check logprobs: {ret}")
                else:
                    raise RuntimeError(f"Failed to check logprobs: {response.status}")

def run_eval(args: argparse.Namespace):
    model_args = (f"base_url={args.endpoint}/completions"
        f",tokenizer={args.tokenizer}"
        f",model={args.model}"
        f",num_concurrent={EVAL_CONCURRENT},tokenized_requests=False"
        f",device=cpu"
    )
    logger.info(f"Run llm evaluation with model_args: {model_args}")

    # prepare tasks
    tasks = args.tasks if args.tasks is not None else EVAL_TASKS
    tasks = tasks.split(",")
    tasks2 = []
    for t in tasks:
        if t == "mmlu":
            #tasks2.append("mmlu_stem")
            #tasks2.append("mmlu_social_sciences")
            tasks2.append("mmlu_humanities")
            #tasks2.append("mmlu_other")
        else:
            tasks2.append(t)
    tasks = tasks2
    task_manager = TaskManager()
    task_names = task_manager.match_tasks(tasks)
    task_missed = [t for t in tasks if t not in task_names]
    logger.info(f"Evaluation tasks: {task_names}, missed tasks: {task_missed}")

    # run
    gen_args = None
    start_time = time.perf_counter()
    tracker = EvaluationTracker()
    results = lm_eval.simple_evaluate(
        model=EVAL_MODEL,
        model_args=model_args,
        tasks=task_names,
        num_fewshot=EVAL_FEWSHOT,
        limit=EVAL_LIMIT,
        batch_size=EVAL_BS,
        evaluation_tracker=tracker,
        log_samples=EVAL_LOG_SAMPLE,
        gen_kwargs=gen_args,
    )
    duration = time.perf_counter() - start_time
    samples = results.pop("samples") if EVAL_LOG_SAMPLE else None
    tracker.save_results_aggregated(results=results, samples=samples)
    print(f"\n========== Result: tasks: {task_names}, duration: {duration} =========")
    print(make_table(results))
    print("\n\n")

def main(args: argparse.Namespace):
    logger.info(args)
    logger.info("\n\n")
    if args.tokenizer is None or not os.path.exists(args.tokenizer):
        raise ValueError("Invalid tokenizer")
    if args.endpoint is None:
        args.endpoint = "http://localhost:18011/v1"
        logger.warning(f"The --endpoint is set to default value: {args.endpoint}")
    if args.model is None:
        server_model = util.get_model(args.endpoint + "/models")
        if server_model is None:
            raise RuntimeError("Failed to query model name from server")
        args.model = server_model
    logger.info(f"Model name: {args.model}")
    if args.check_logprobs:
        asyncio.run(check_logprobs(args.endpoint+"/completions", args.model))
    run_eval(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation of online LLM server."
    )
    parser.add_argument("--endpoint", type=str, help="The LLM server endpoint, default is http://localhost:18011/v1")
    parser.add_argument("--model", type=str, help="The model name, if not set, get from server")
    parser.add_argument("--tasks", type=str, help=f"The evaluation tasks seperated by comma, default is {EVAL_TASKS}")
    parser.add_argument("--tokenizer", type=str, help="The local folder path to the model data for token decoding and encoding")
    parser.add_argument("--check-logprobs", action="store_true", help="Check the logprobs with correct request")
    parser.add_argument("--verbose", action="store_true", help="print in verbose mode")
    args = parser.parse_args()
    main(args)

