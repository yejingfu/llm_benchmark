import argparse
import json
import base64
import wave
import time
import os
import aiohttp
import asyncio
import ormsgpack
import pyaudio
import requests
import numpy as np
from tabulate import tabulate
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
from dataclasses import dataclass, field
from fish_speech_utils import audio_to_bytes, read_ref_text, ServeReferenceAudio, ServeTTSRequest


DEF_AUDIO_CHANNELS=1
DEF_AUDIO_RATE=44100

DEF_SPEECH_TEXT = """
A large language model (LLM) is a type of computational model designed for natural language processing tasks such as language generation. As language models, LLMs acquire these abilities by learning statistical relationships from vast amounts of text during a self-supervised and semi-supervised training process.
The largest and most capable LLMs are artificial neural networks built with a decoder-only transformer-based architecture, enabling efficient processing and generation of large-scale text data. Modern models can be fine-tuned for specific tasks or guided by prompt engineering. These models acquire predictive power regarding syntax, semantics, and ontologies[3] inherent in human language corpora, but they also inherit inaccuracies and biases present in the data they are trained in.
Before 2017, there were a few language models that were large as compared to capacities then available. In the 1990s, the IBM alignment models pioneered statistical language modelling. A smoothed n-gram model in 2001 trained on 0.3 billion words achieved state-of-the-art perplexity at the time. In the 2000s, as Internet use became prevalent, some researchers constructed Internet-scale language datasets ("web as corpus"), upon which they trained statistical language models. In 2009, in most language processing tasks, statistical language models dominated over symbolic language models, as they can usefully ingest large datasets.
After neural networks became dominant in image processing around 2012, they were applied to language modelling as well. Google converted its translation service to Neural Machine Translation in 2016. As it was before transformers, it was done by seq2seq deep LSTM networks.
An illustration of main components of the transformer model from the original paper, where layers were normalized after (instead of before) multiheaded attention
At the 2017 NeurIPS conference, Google researchers introduced the transformer architecture in their landmark paper "Attention Is All You Need". This paper's goal was to improve upon 2014 seq2seq technology,[11] and was based mainly on the attention mechanism developed by Bahdanau et al. in 2014. The following year in 2018, BERT was introduced and quickly became "ubiquitous". Though the original transformer has both encoder and decoder blocks, BERT is an encoder-only model. Academic and research usage of BERT began to decline in 2023, following rapid improvements in the abilities of decoder-only models (such as GPT) to solve tasks via prompting.

Although decoder-only GPT-1 was introduced in 2018, it was GPT-2 in 2019 that caught widespread attention because OpenAI at first deemed it too powerful to release publicly, out of fear of malicious use.[15] GPT-3 in 2020 went a step further and as of 2024 is available only via API with no offering of downloading the model to execute locally. But it was the 2022 consumer-facing browser-based ChatGPT that captured the imaginations of the general population and caused some media hype and online buzz. The 2023 GPT-4 was praised for its increased accuracy and as a "holy grail" for its multimodal capabilities. OpenAI did not reveal the high-level architecture and the number of parameters of GPT-4. The release of ChatGPT led to an uptick in LLM usage across several research subfields of computer science, including robotics, software engineering, and societal impact work.
Competing language models have for the most part been attempting to equal the GPT series, at least in terms of number of parameters.
Since 2022, source-available models have been gaining popularity, especially at first with BLOOM and LLaMA, though both have restrictions on the field of use. Mistral AI's models Mistral 7B and Mixtral 8x7b have the more permissive Apache License. As of June 2024, The Instruction fine tuned variant of the Llama 3 70 billion parameter model is the most powerful open LLM according to the LMSYS Chatbot Arena Leaderboard, being more powerful than GPT-3.5 but not as powerful as GPT-4.
As of 2024, the largest and most capable models are all based on the Transformer architecture. Some recent implementations are based on other architectures, such as recurrent neural network variants and Mamba (a state space model).
Because machine learning algorithms process numbers rather than text, the text must be converted to numbers. In the first step, a vocabulary is decided upon, then integer indices are arbitrarily but uniquely assigned to each vocabulary entry, and finally, an embedding is associated to the integer index. Algorithms include byte-pair encoding (BPE) and WordPiece. There are also special tokens serving as control characters, such as [MASK] for masked-out token (as used in BERT), and [UNK] ("unknown") for characters not appearing in the vocabulary. Also, some special symbols are used to denote special text formatting. For example, "Ġ" denotes a preceding whitespace in RoBERTa and GPT. "##" denotes continuation of a preceding word in BERT.

Large language models, also known as LLMs, are very large deep learning models that are pre-trained on vast amounts of data. The underlying transformer is a set of neural networks that consist of an encoder and a decoder with self-attention capabilities. The encoder and decoder extract meanings from a sequence of text and understand the relationships between words and phrases in it.
Transformer LLMs are capable of unsupervised training, although a more precise explanation is that transformers perform self-learning. It is through this process that transformers learn to understand basic grammar, languages, and knowledge.
Unlike earlier recurrent neural networks (RNN) that sequentially process inputs, transformers process entire sequences in parallel. This allows the data scientists to use GPUs for training transformer-based LLMs, significantly reducing the training time.
Transformer neural network architecture allows the use of very large models, often with hundreds of billions of parameters. Such large-scale models can ingest massive amounts of data, often from the internet, but also from sources such as the Common Crawl, which comprises more than 50 billion web pages, and Wikipedia, which has approximately 57 million pages.
A key factor in how LLMs work is the way they represent words. Earlier forms of machine learning used a numerical table to represent each word. But, this form of representation could not recognize relationships between words such as words with similar meanings. This limitation was overcome by using multi-dimensional vectors, commonly referred to as word embeddings, to represent words so that words with similar contextual meanings or other relationships are close to each other in the vector space.
Using word embeddings, transformers can pre-process text as numerical representations through the encoder and understand the context of words and phrases with similar meanings as well as other relationships between words such as parts of speech. It is then possible for LLMs to apply this knowledge of the language through the decoder to produce a unique output.
Transformer-based neural networks are very large. These networks contain multiple nodes and layers. Each node in a layer has connections to all nodes in the subsequent layer, each of which has a weight and a bias. Weights and biases along with embeddings are known as model parameters. Large transformer-based neural networks can have billions and billions of parameters. The size of the model is generally determined by an empirical relationship between the model size, the number of parameters, and the size of the training data.
Training is performed using a large corpus of high-quality data. During training, the model iteratively adjusts parameter values until the model correctly predicts the next token from an the previous squence of input tokens. It does this through self-learning techniques which teach the model to adjust parameters to maximize the likelihood of the next tokens in the training examples.
Once trained, LLMs can be readily adapted to perform multiple tasks using relatively small sets of supervised data, a process known as fine tuning.
Three common learning models exist:
Zero-shot learning; Base LLMs can respond to a broad range of requests without explicit training, often through prompts, although answer accuracy varies.
Few-shot learning: By providing a few relevant training examples, base model performance significantly improves in that specific area.
Fine-tuning: This is an extension of few-shot learning in that data scientists train a base model to adjust its parameters with additional data relevant to the specific application.
"""

@dataclass
class Context:
    index: int = field(default=0)
    text: str = field(default="") ## input
    output: str = field(default="") ## output path to mp3 file
    e2e: float = field(default=0) ## end to end latency
    audio_duration: float = field(default=0) ## the duration(seconds) of audio
    audio_size: int = field(default=0) ## the bytes of the audio content
    audio_channels: int = field(default=0) ## the audio channel
    audio_rate: int = field(default=0) ## the audio rate

def prepare_context(args):
    text = DEF_SPEECH_TEXT
    total = len(text)
    offset = 0
    ctxs = []
    for i in range(args.num_requests):
        num = args.num_char
        if offset + num + 20 > total:
            offset = 0
        while text[offset] in [' ', '\t', '\n']:
            offset += 1
        while text[offset+num] not in [' ', '\t', '\n']:
            num += 1
        ctx = Context(index=i, text=text[offset:offset+num])
        ctxs.append(ctx)
        offset = offset + num
    return ctxs

async def send_one_request(ctx, args):
    payload = {
        "text": ctx.text,
        "references": [],
        "reference_id": None,
        "normalize": True,
        "format": "mp3",
        "max_new_tokens": 1024,
        "chunk_length": 200,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "temperature": 0.7,
        "streaming": args.streaming,
        "use_memory_cache": "off",
        "seed": None,
    }

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6 * 60 * 60)) as session:
        pydantic_data = ServeTTSRequest(**payload)
        st_start = time.perf_counter()
        async with session.post(
            args.endpoint,
            data=ormsgpack.packb(pydantic_data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
            #stream=args.streaming,
            headers={
                "authorization": "Bearer YOUR_API_KEY",
                "content-type": "application/msgpack",
            },
        ) as response:
            if response.status == 200:
                #st_end = time.perf_counter()
                audio_content = await response.read()
                st_end = time.perf_counter()
                if args.out_dir:
                    audio_path = f"{args.out_dir}/generated_{ctx.index}.mp3"
                    with open(audio_path, "wb") as audio_file:
                        audio_file.write(audio_content)
                    print(f"Audio has been saved to '{audio_path}'.")
                #if args.play:
                #    audio = AudioSegment.from_file(audio_path, format="mp3")
                #    play(audio)
                audio = AudioSegment.from_file(BytesIO(audio_content), format="mp3")
                ctx.audio_duration = audio.duration_seconds
                ctx.audio_size = len(audio.raw_data)
                ctx.audio_channels = audio.channels
                ctx.audio_rate = audio.frame_rate
                ctx.e2e = st_end - st_start
            else:
                print(f"Request failed with status code {response.status_code}")
                print(response.json())


async def send_batch_requests(ctxs, args):
    num = len(ctxs)
    print(f"Begin send {num} requests in parallel")
    tasks: List[asyncio.Task] = []
    t1 = time.perf_counter()
    for i in range(num):
        print(f"Send[{ctxs[i].index}]: {ctxs[i].text}({len(ctxs[i].text)})")
        tasks.append(asyncio.create_task(send_one_request(ctxs[i], args)))
    await asyncio.gather(*tasks)
    t2 = time.perf_counter()
    print(f"End send {num} requests, time: {(t2 - t1):.2f}\n")
    ## start_index, stop_index, text_len, e2e_latency, duration, size, time, [e2e_latency]
    s = [ctxs[0].index, ctxs[-1].index, 0, 0, 0, 0, 0, []]
    for i in range(num):
        s[2] += len(ctxs[i].text)
        s[3] += ctxs[i].e2e
        s[4] += ctxs[i].audio_duration
        s[5] += ctxs[i].audio_size
        s[7].append(ctxs[i].e2e)
    s[2] /= num
    s[3] /= num
    s[4] /= num
    s[5] /= num
    s[6] = t2-t1
    return s

def main(args: argparse.Namespace):
    if args.num_char > 1000 or args.num_char < 10:
        raise ValueError(f"Invalid num-char: {args.num_char}")
    if args.num_requests < 1:
        raise ValueError(f"Invalid num-requests: {args.num_requests}")
    if args.parallel < 1:
        raise ValueError(f"Invalid parallel: {args.parallel}")
    if args.num_requests < args.parallel:
        raise ValueError(f"The num of requests should be larger than parallel")
    if args.streaming:
        raise ValueError("Currently not support streaming mode")
    if args.out_dir:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
    contexts = prepare_context(args)
    total_reqs = len(contexts)
    print(f"num of requests: {total_reqs}")
    total_seconds = 0
    column = ["index", "input-chars", "e2e-latency(sec)", "audio-duration(sec)", "audio-size(KB)"]
    data = []
    e2e_raw = []
    sum_value = [0,0,0,0]
    num = 0
    i = 0
    while i < total_reqs:
        j = i + args.parallel
        if j > total_reqs:
            j = total_reqs
        ret = asyncio.run(send_batch_requests(contexts[i:j], args))
        i += args.parallel
        data.append([f"{ret[0]}-{ret[1]}", f"{int(ret[2])}", f"{ret[3]:.2f}", f"{ret[4]:.2f}", f"{(ret[5]/1024):.1f}"])
        sum_value[0] += ret[2]
        sum_value[1] += ret[3]
        sum_value[2] += ret[4]
        sum_value[3] += ret[5]
        total_seconds += ret[6]
        e2e_raw.extend(ret[7])
        num += 1
    #e2e_raw_simple = [round(x, 2) for x in e2e_raw]
    #print(f"E2E latency: {e2e_raw_simple}")
    data.insert(0, ["avg", f"{int(sum_value[0]/num)}", f"{(sum_value[1]/num):.2f}", f"{(sum_value[2]/num):.2f}", f"{(sum_value[3]/num/1024):.1f}"])
    percentile = [50, 90, 99]
    e2e_p = np.percentile(e2e_raw, percentile)
    e2e_avg = np.mean(e2e_raw)
    log = f"Total requests: {total_reqs}, Parallel: {args.parallel}\n"
    log += f"RPS: {(total_reqs/total_seconds):.2f}, E2E latency: {e2e_avg:.2f},{e2e_p[0]:.2f},{e2e_p[1]:.2f},{e2e_p[2]:.2f}\n"
    log += tabulate(data, headers=column, tablefmt="grid")
    log += "\n\n"
    print(log)
    if args.log_file:
        with open(args.log_file, "a") as f:
            f.write(log)
        print(f"Benchmark metrics are save into {args.log_file}")
    print("DONE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark fish speech throughput.")
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:8080/v1/tts", help="The fish-speech serving endpoint, default is: http://127.0.0.1:8080/v1/tts")
    parser.add_argument("--num-char", type=int, default=100, help="Number of characters for every request.")
    parser.add_argument("--num-requests", type=int, default=100, help="Number of prompts for benckmark.")
    parser.add_argument("--parallel", type=int, default=10)
    parser.add_argument("--streaming", type=bool, default=False, help="Enable streaming response")
    parser.add_argument("--out-dir", type=str, help="The output folder to save the generated audio files(mp3)")
    parser.add_argument("--log-file", type=str, help="The log file to save the metrics data")

    args = parser.parse_args()
    main(args)

