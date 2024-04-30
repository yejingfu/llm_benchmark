import requests
from collections.abc import Callable, Iterable, Mapping
from threading import Thread
import sys
import os
import time
from typing import Any
import numpy as np
from queue import Queue
import argparse
import time
import json
from pydantic import BaseModel, validator, ValidationError
from base_types import Client, OaClient, SilClient
from transformers import AutoTokenizer

#SERVER_NAME = "180.166.208.13:18000"

class CalRealToken:
    def __init__(self, host, model_path) -> None:
        self.host = host
        # workaround
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def get_real_token_num(self, msg):
        encoded_input = self.tokenizer.encode(msg)
        return len(encoded_input)

    def make_msg(self, base_msg, need_len):
        cur_size = self.get_real_token_num(base_msg)
        ratio = float(len(base_msg)) / cur_size
        #print(f"base_size: {cur_size}, need: {need_len}, ratio: {ratio}")
        ret_msg = base_msg
        if cur_size > need_len:
            ret_msg = ret_msg[:-int((cur_size - need_len) * ratio)]
            cur_size = self.get_real_token_num(ret_msg)
        else:
            while need_len - cur_size > 5:
                added = (need_len - cur_size) * ratio
                ret_msg = base_msg[:int(added)] + ret_msg
                cur_size = self.get_real_token_num(ret_msg)
        return cur_size, ret_msg

def print_table(heads, data, float_format="{:.3f}"):
    print_list = []
    max_len = 0
    for head in heads:
        max_len = max(max_len, len(head))
    print_list.append([""] + heads)
    for k, v_list in data.items():
        one_list = []
        max_len = max(max_len, len(k))
        one_list.append(k)
        for v in v_list:
            if isinstance(v, float):
                str_v = float_format.format(v)
            else:
                str_v = str(v)
            max_len = max(max_len, len(str_v))
            one_list.append(str_v)
        print_list.append(one_list)
    max_len += 2
    row_format = "{:<" + str(max_len) + "}"
    row_format *= (len(heads) + 1)
    for v_list in print_list:
        print(row_format.format(*v_list))


def show_info(name, stat_info):
    def get_p(info):
        return (np.mean(info), np.percentile(info, 50), np.percentile(info, 90), np.percentile(info, 99))
    fisrt_tokeninfo=None
    pp_dict = {}
    v_dict = {}
    for k, v in stat_info.items():
        if len(v) > 1:
            if k == "latency_first":
                fisrt_tokeninfo=get_p(np.array(v))
            pp_dict[k] = get_p(np.array(v))
        elif len(v) > 0:
            v_dict[k] = v
    print(name)
    print_table(["mean", "p50", "p90", "p99"], pp_dict)
    print_table(["value"], v_dict)
    return fisrt_tokeninfo

class ClientThread(Thread):
    def __init__(self, host, wk_index, req_list_q, gen_params, enter_q, exit_q, print_ret = False, backend = "silicon") -> None:
        super().__init__()
        self.req_list_q = req_list_q
        self.stat_info = {}
        self.wk_index = wk_index
        self.print_ret = print_ret
        self.gen_params = gen_params
        self.enter_q = enter_q
        self.exit_q = exit_q
        self.backend = backend
        if self.backend == "tgi":
            self.client = Client(f'http://{host}', timeout=360)
        elif self.backend == "silicon":
            self.client = OaClient(f"http://{host}/v1/completions", timeout=360)
        else:
            raise ValidationError(f"Invalid backend: {self.backend}")
    
    def generate(self, msg):
        if self.backend == "tgi":
            return self.client.generate_stream(msg, **self.gen_params)
        elif self.backend == "silicon":
            return self.client.generate_stream(msg, **self.gen_params)

    def run(self):
        # print(f"start worker {self.wk_index}")

        self.stat_info = {
            "latency": [],
            "latency_first": [],
            "ret_tokens_count": [],
            "latency_per_token": [],
        }
        res = []
        enter_q_flag = False
        time_start_f = None
        time_end_f = None
        while True:
            if time_start_f is None:
                if self.enter_q.qsize() == 0:  #目的是让线程同时运行时，第一轮一瞬间，取完enter_q队列里面的元素，然后第二轮开始正式计时
                    time_start_f = time.time()
            try:
                msg = self.req_list_q.get_nowait()
            except:
                break

            #第一波和最后一波 不是满batch，不参与计算耗时
            if not enter_q_flag:
                self.enter_q.get()
                enter_q_flag = True
            generated_tokens = 0
            text = ''
            time_start = time.time()
            first_duration = None
            for response in self.generate(msg):
                if not response.token.special:
                    if self.print_ret:
                        text += response.token.text
                if response.details is not None and response.details.generated_tokens is not None:
                    generated_tokens += response.details.generated_tokens
                if first_duration is None:
                    time_end = time.time()
                    first_duration = time_end - time_start  #首token时延
                    self.stat_info["latency_first"].append(first_duration * 1000)
                    time_start = time_end
            
            if time_start_f is None:  #为了让第一轮所有线程跑着的时候，直接进行第二轮，第二轮时间就不是None了，保持满batch
                continue

            if self.exit_q.qsize() == 0:
                time_end_f = time.time() #每次线程处理任务都会被更新，这个时间是该线程处理所有任务的结束时间（1线程有多个任务）
            else:
                continue
            time_end = time.time()
            duration = time_end - time_start  #每次请求任务的输出token的时延
            self.stat_info["latency"].append(duration * 1000)  #每请求输出token时延
            self.stat_info["latency_per_token"].append(duration * 1000/max(generated_tokens, 1))  #每请求每token耗时
            self.stat_info["ret_tokens_count"].append(generated_tokens)  #每请求生成的token数
            if self.print_ret:
                print(f"worker {self.wk_index} prompt:{msg} \n =========> gen: {text}")
            res.append((msg, text))
        self.exit_q.put(1)  #任务都完成，每个线程都会退出队列放入一个1
        if time_end_f is None:
            print(f"worker {self.wk_index} not meet time_end_f")
            return
        if time_start_f is None:
            print(f"worker {self.wk_index} not meet time_start_f")
            return
        duration = time_end_f - time_start_f #线程执行完线程上所有任务的耗时
        count = len(res)
        qps =  count / duration #线程的qps
        self.stat_info["qps"] = [qps]
        self.stat_info["cost"] = [duration]
        self.stat_info["count"] = [count]
        print(f"end worker {self.wk_index} count: {count}, qps:{qps:.2f} cost: {duration:.2f}")

    def run2(self):
        self.stat_info = {
            "latency": [],
            "latency_first": [],
            "ret_tokens_count": [],
            "latency_per_token": [],
        }
        res = []
        enter_q_flag = False
        time_start_f = None
        time_end_f = None
        while True:
            if time_start_f is None:
                if self.enter_q.qsize() == 0:
                    time_start_f = time.time()
            try:
                msg = self.req_list_q.get_nowait()
            except:
                break
            if not enter_q_flag:
                self.enter_q.get()
                enter_q_flag = True
            generated_tokens = 0
            text = ''
            time_start = time.time()
            first_duration = None
            for response in self.client.generate_stream(msg, **self.gen_params):
                if not response.token.special:
                    if self.print_ret:
                        text += response.token.text
                if response.details is not None and response.details.generated_tokens is not None:
                    generated_tokens += response.details.generated_tokens
                if first_duration is None:
                    time_end = time.time()
                    first_duration = time_end - time_start
                    self.stat_info["latency_first"].append(first_duration * 1000)
                    time_start = time_end
            
            if time_start_f is None:
                continue
            if self.exit_q.qsize() == 0:
                time_end_f = time.time() 
            else:
                continue
            time_end = time.time()
            duration = time_end - time_start
            self.stat_info["latency"].append(duration * 1000)
            self.stat_info["latency_per_token"].append(duration * 1000/max(generated_tokens, 1))
            self.stat_info["ret_tokens_count"].append(generated_tokens)
            if self.print_ret:
                print(f"worker {self.wk_index} prompt:{msg} gen: {text}")
            res.append((msg, text))
        self.exit_q.put(1)
        if time_end_f is None:
            print(f"worker {self.wk_index} not meet time_end_f")
            return
        if time_start_f is None:
            print(f"worker {self.wk_index} not meet time_start_f")
            return
        duration = time_end_f - time_start_f
        count = len(res)
        qps =  count / duration
        self.stat_info["qps"] = [qps]
        self.stat_info["cost"] = [duration]
        self.stat_info["count"] = [count]
        print(f"end worker {self.wk_index} count: {count}, qps:{qps:.2f} cost: {duration:.2f}")

    def get_result(self):
        return self.stat_info


def check_health(host, backend):
    if backend == "tgi":
        ret = requests.get(f'http://{host}/health')
    elif backend == "trt":
        ret = requests.get(f"http://{host}/v2/health/ready")
    elif backend == "vllm":
        ret = requests.get(f"http://{host}/health")
    else:
        ret = requests.get(f'http://{host}/v1/ready')
    #ret = requests.get(f'http://{host}/v1/models')
    print(f"response status: {ret.status_code}, text: {ret.text}")

def generate(host, backend, stream = False):
    do_sample = True
    time_start = time.time()
    prompt = "解释一下什么是反向传播算法(Backpropagation Algorithm)"
    if backend == "openai":
        params = {
            "model": "llama",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 512,
            "frequency_penalty": 0.01,
            "top_k":2,
            "top_p":0.1,
        }
        #ret = requests.post(f"http://{host}/v1/chat/completions", json=params)
        client = OaClient(f"http://{host}/v1/chat/completions", timeout=360)
        ret = client.generate(**params).generated_text[0]
    elif backend == "silicon":
        params = {
            "model": "llama",
            "max_tokens": 512,
            "temperature": 0.8,
            "frequency_penalty": 0.01,
            "top_k":2,
            "top_p":0.1,
        }
        #ret = requests.post(f"http://{host}/v1/completions", json=body)
        client = OaClient(f"http://{host}/v1/completions", timeout=360)
        if stream == False:
            res = client.generate(prompt, **params)
        else:
            res = client.generate_stream(prompt, **params)
        #print(f"response: {res.json()}")
        ret = res.generated_text[0]
    elif backend == "tgi":
        params = {
            "repetition_penalty": 0.01,
            # "temperature":0.01,​
            "top_k":2,
            "top_p":0.1,
            "do_sample": do_sample,
            "max_new_tokens": 512,
            "seed":2023
        }
        client = Client(f'http://{host}', timeout=360)
        ret = client.generate(prompt, **params).generated_text
    elif backend == "vllm":
        params = {
            "model": "/models/Mixtral-8x7B-Instruct-v0.1",
            "prompt": prompt,
            "best_of": 1,
            "temperature": 0.01,
            "top_p": 0.1,
            "top_k": 2,
            "max_tokens": 512,
            "ignore_eos": True,
            "stream": stream,
        }
        ret = requests.post(f"http://{host}/v1/completions", json=params).json()
        ret = ret["choices"][0]["text"]
    elif backend == "trt":
        params = {
            "text_input": prompt,
            "max_tokens": 512,
            "bad_words": "",
            "stop_words": "",
            "stream": stream,
        }
        if do_sample:
            params["temperature"] = 0.01
            params["top_p"] = 0.1
            params["top_k"] = 2
        if stream:
            ret = requests.post(f"http://{host}/v2/models/ensemble/generate_stream", json=params).text
            #print(f"response: {ret}")
        else:
            ret = requests.post(f"http://{host}/v2/models/ensemble/generate", json=params).json()
            print(f"response: {ret}")
            ret = ret["text_output"]
    else:
        raise ValidationError(f"Invalid backend {backend}")
    time_delay_sec = time.time() - time_start
    if stream:
        print(f"\n\ntime_delay_sec:{time_delay_sec}, len:{len(ret)}, stream output: >>> ")
        chunks = ret.split("\n\n")
        final_result = ""
        for i in range(len(chunks)):
            data = chunks[i].lstrip("data:").rstrip("\n\n").strip()
            if len(data) > 0:
                obj = json.loads(data)
                print(f"text_output[{i}]: {obj['text_output']}")
                final_result += obj["text_output"]
        print(f"final text output: {final_result}")
    else:
        print(f"\n\ntime_delay_sec:{time_delay_sec}, len:{len(ret)}, output: {ret}")

def stress(args, cal_token = True):
    prompts = []
    with open("long_prompts.txt", "r", encoding= 'utf-8') as f:
        content = f.read()
        chunks = content.split("\n\n\n")
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) > 0:
                prompts.append(chunk)

    if args.input_len == 16:
        msg = prompts[0]
    elif args.input_len == 32:
        msg = prompts[1]
    elif args.input_len == 64:
        msg = prompts[2]
    elif args.input_len == 128:
        msg = prompts[3]
    elif args.input_len == 256:
        msg = prompts[4]
    elif args.input_len == 512:
        msg = prompts[5]
    elif args.input_len == 1024:
        msg = prompts[6]
    elif args.input_len == 2048:
        msg = prompts[7]
    elif args.input_len > 2048:
        msg_2048 = prompts[7]
        msg_1024 = prompts[6]
        msg = ""
        for _ in range(args.input_len // 2048):
            msg += msg_2048
        if args.input_len % 2048 != 0:
            msg += msg_1024
    else:
        raise ValueError('invalid input-tokens')
    if msg is None or msg == "":
        raise ValueError("invalid input message")
    print(f"input message: {msg}")

    if cal_token:
        token_client = CalRealToken(args.host, args.model)
        msg_size, msg = token_client.make_msg(msg, args.input_len)
        print(f"input message({msg_size}: {msg}")

    if args.backend == "tgi":
        params = {
            "repetition_penalty": 0.01 if args.input_len != 0 else 1.0,
            # "temperature":0.01,
            "top_k":2,
            "top_p":0.1,
            "do_sample": True,
            "max_new_tokens": args.output_len,
            "seed":2023
        }
        time_start = time.time()
        client = Client(f'http://{args.host}', timeout=360)
        text = client.generate(msg, **params).generated_text
        time_end = time.time()
        time_delay_sec = time_end - time_start
        print(f"[warmup] time_delay_sec:{time_delay_sec} len:{len(text)}, {text}")
    elif args.backend == "silicon":
        params = {
            "model": "llama",
            "max_tokens": args.output_len,
            "temperature": 0.8,
            "frequency_penalty": 0.01,
            "top_k":2,
            "top_p":0.1,
        }
        time_start = time.time()
        client = OaClient(f"http://{args.host}/v1/completions", timeout=360)
        text = client.generate(msg, **params).generated_text
        time_end = time.time()
        time_delay_sec = time_end - time_start
        print(f"[warmup] time_delay_sec:{time_delay_sec} len:{len(text)}, {text}")
    else:
        raise ValidationError(f"Not support {args.backend}")

    ## stress now
    thread_list = []
    args.number_of_request += 2
    args.number_of_request *= args.concurrency
    req_list = [msg] * args.number_of_request

    req_list_q = Queue(args.number_of_request * 2)
    for idx, req in enumerate(req_list):
        req_list_q.put(req)      
    thread_enter_q = Queue(args.concurrency * 2)
    for _ in range(args.concurrency):
        thread_enter_q.put(1)
    thread_exist_q = Queue(args.concurrency * 2)

    time_start = time.time()
    print(f"req_list:{len(req_list)}")
    for i in range(args.concurrency):
        time.sleep(float(time_delay_sec) / args.concurrency)
        t = ClientThread(args.host, i, req_list_q, gen_params=params, enter_q=thread_enter_q, exit_q=thread_exist_q, print_ret=args.print_gen, backend=args.backend)
        t.daemon = True
        t.start()
        thread_list.append(t)

    stat_info_all = {}
    for i, t in enumerate(thread_list):
        t.join()
        for k, v in t.get_result().items():
            if k not in stat_info_all:
                stat_info_all[k] = v
            else:
                stat_info_all[k].extend(v)  
    time_end = time.time()
    cost = time_end - time_start    
    qps = np.sum(np.array(stat_info_all["count"])) / np.mean(np.array(stat_info_all["cost"]))   #np.mean,因为线程是并行的，所以总共耗时就大约为单个线程的耗时​
    tokens=np.sum(np.array(stat_info_all["ret_tokens_count"])) / np.mean(np.array(stat_info_all["cost"])) #总生成token个数/每线程的平均耗时​
    sintokens=tokens/args.concurrency
    print("==========================")
    fisrt_tokeninfo=show_info(f'all down task cost: {cost:.2f}, qps:{qps:.3f},batch_token:{sintokens:.3f},total_tokens/s:{tokens:.3f}\n', stat_info_all)

    result_str="need_data: {\"batch\":%s,\"首tokenp90\":%s,\"首tokenp99\":%s,\"单路吞吐\":%s,\"总吞吐\":%s,\"qps\":%s,\"total_time\":%s}"
    print(result_str % (args.concurrency, fisrt_tokeninfo[2],fisrt_tokeninfo[3], sintokens, tokens, qps,cost))

def generate_from_prompt_file(host, prompt_file, backend):
    if not os.path.isfile(prompt_file):
        raise IOError(f"file does not exist: {prompt_file}")
    base_name = os.path.splitext(os.path.basename(prompt_file))[0]
    parts = base_name.split('_')
    if len(parts) < 2:
        raise ValidationError("The prompt file name should be following the pattern: 'name_len'")
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_content = f.read()
    #input_len = int(parts[1])
    #print(f"The input len of tokens is: {input_len}: {prompt_content}({len(prompt_content)})")
    if backend == "tgi":
        params = {
            "repetition_penalty": 0.01 if args.input_len != 0 else 1.0,
            # "temperature":0.01,
            "top_k":2,
            "top_p":0.1,
            "do_sample": True,
            "max_new_tokens": 500,
            "seed":2023
        }
        time_start = time.time()
        client = Client(f'http://{host}', timeout=360)
        text = client.generate(prompt_content, **params).generated_text
        time_end = time.time()
        time_delay_sec = time_end - time_start
        print(f"[generate] time_delay_sec:{time_delay_sec} len:{len(text)}, {text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", "--host", default="127.0.0.1:8000", help="host name, IP and port")
    parser.add_argument("-m", "--model", type=str, help="The absolute path to the model, used for tokenizer")
    parser.add_argument("-a", "--action", type=str, default="check", choices=["check", "generate", "stress", "prompt"], help="action to execute")
    parser.add_argument("--backend", type=str, default="silicon", choices=["silicon", "openai", "tgi", "trt", "vllm"], help="the backend style")
    parser.add_argument("-S", '--stream', default=False, type=bool, help='stream')
    parser.add_argument("-C", '--concurrency', default=4, type=int, help='concurrency')
    parser.add_argument("-N", '--number_of_request', default=4, type=int, help='number_of_request per thread')
    parser.add_argument("-PR", '--print_gen', default=False, type=int, help='print_gen')
    parser.add_argument("-I", '--input_len', default=1024, type=int, help='input_len')
    parser.add_argument("-O", '--output_len', default=1024, type=int, help='output_len')
    parser.add_argument("-PF", "--prompt-file", default="", type=str, help="the prompt file path")
    args = parser.parse_args()
    print(f"Connecting to server {args.host}")
    if args.action == "check":
        check_health(args.host, args.backend)
    elif args.action == "generate":
        print("Generate tokens")
        generate(args.host, args.backend, args.stream)
    elif args.action == "stress":
        print("Stress testing")
        stress(args, False)
    elif args.action == "prompt":
        generate_from_prompt_file(args.host, args.prompt_file, args.backend)
    else:
        print("Internal error")


