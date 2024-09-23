import os
import argparse
import time
from loguru import logger

import util

EVAL_MODEL="local-completions"
## reference: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
EVAL_TASKS="mmlu,gsm8k,gpqa,bbh,mathqa"
EVAL_CONCURRENT=20
EVAL_LOG_SAMPLE=True
EVAL_CHECK_MODEL=False

try:
    import lm_eval
    from lm_eval.tasks import TaskManager
    from lm_eval.loggers import EvaluationTracker
    from lm_eval.utils import handle_non_serializable, make_table
except ImportError:
    print("Please install lm_eval by 'pip install lm-eval' and 'pip install lm-eval[api]'")
    exit(0)

def run_eval(provider: util.LlmProvider, args: argparse.Namespace):
    if args.tokenizer is None or not os.path.exists(args.tokenizer):
        raise ValueError("Invalid tokenizer")
    model_args = (f"base_url={provider.endpoint}/completions,"
        f"tokenizer={args.tokenizer},"
        f"model={provider.model},"
        f"num_concurrent={EVAL_CONCURRENT},tokenized_requests=False"
    )
    logger.info(f"Run evaluation: {provider} with model_args: {model_args}")

    # check
    if EVAL_CHECK_MODEL:
        headers = {"Content-Type": "application/json"}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        real_models = util.get_model_list(provider.endpoint+"/models", headers)
        #print(f"models: {real_models}")
        if provider.model not in real_models:
            logger.error(f"Dismatched model name {provider.model}, candicates: {real_models}")
            return

    # prepare tasks
    task_manager = TaskManager()
    tasks = args.tasks if args.tasks is not None else EVAL_TASKS
    tasks = tasks.split(",")
    task_names = task_manager.match_tasks(tasks)
    task_missed = [t for t in tasks if t not in task_names]
    logger.info(f"Evaluation tasks: {task_names}, missed tasks: {task_missed}")

    # run
    start_time = time.perf_counter()
    tracker = EvaluationTracker()
    results = lm_eval.simple_evaluate(model=EVAL_MODEL, model_args=model_args, tasks=task_names, evaluation_tracker=tracker, log_samples=EVAL_LOG_SAMPLE)
    duration = time.perf_counter() - start_time
    samples = results.pop("samples") if EVAL_LOG_SAMPLE else None
    tracker.save_results_aggregated(results=results, samples=samples)
    print(f"\n========== Result: {provider.provider}.{provider.model}, tasks: {task_names}, duration: {duration} =========")
    print(make_table(results))
    print("\n\n")

def main(args: argparse.Namespace):
    logger.info(args)
    logger.info("\n\n")

    providers = util.get_llm_provider(os.path.dirname(os.path.abspath(__file__)) + "/llm_providers.json")
    assert len(providers) > 0, "Empty providers"
    for provider in providers:
        run_eval(provider, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation of online LLM server."
    )
    parser.add_argument("--tokenizer", type=str, help="The local folder path to the model data for token decoding and encoding")
    parser.add_argument("--tasks", type=str, help=f"The evaluation tasks seperated by comma, default is {EVAL_TASKS}")
    parser.add_argument("--verbose", action="store_true", help="print in verbose mode")
    args = parser.parse_args()
    main(args)

