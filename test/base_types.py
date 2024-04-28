## https://github.com/huggingface/text-generation-inference/tree/main/clients/python/text_generation

import json
import requests

from aiohttp import ClientSession, ClientTimeout
from pydantic import BaseModel, validator, ValidationError
from typing import Dict, Optional, List, AsyncIterator, Iterator, Union
from enum import Enum

####################################### errors #########################
# Text Generation Inference Errors
class ValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class GenerationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class OverloadedError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class IncompleteGenerationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


# API Inference Errors
class BadRequestError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class ShardNotReadyError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class ShardTimeoutError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class NotFoundError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class RateLimitExceededError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class NotSupportedError(Exception):
    def __init__(self, model_id: str):
        message = (
            f"Model `{model_id}` is not available for inference with this client. \n"
            "Use `huggingface_hub.inference_api.InferenceApi` instead."
        )
        super(NotSupportedError, self).__init__(message)


# Unknown error
class UnknownError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def parse_error(status_code: int, payload: Dict[str, str]) -> Exception:
    """
    Parse error given an HTTP status code and a json payload

    Args:
        status_code (`int`):
            HTTP status code
        payload (`Dict[str, str]`):
            Json payload

    Returns:
        Exception: parsed exception

    """
    # Try to parse a Text Generation Inference error
    message = payload["error"]
    if "error_type" in payload:
        error_type = payload["error_type"]
        if error_type == "generation":
            return GenerationError(message)
        if error_type == "incomplete_generation":
            return IncompleteGenerationError(message)
        if error_type == "overloaded":
            return OverloadedError(message)
        if error_type == "validation":
            return ValidationError(message)

    # Try to parse a APIInference error
    if status_code == 400:
        return BadRequestError(message)
    if status_code == 403 or status_code == 424:
        return ShardNotReadyError(message)
    if status_code == 504:
        return ShardTimeoutError(message)
    if status_code == 404:
        return NotFoundError(message)
    if status_code == 429:
        return RateLimitExceededError(message)

    # Fallback to an unknown error
    return UnknownError(message)


#################################### types ################################

# enum for grammar type
class GrammarType(str, Enum):
    Json = "json"
    Regex = "regex"


# Grammar type and value
class Grammar(BaseModel):
    # Grammar type
    type: GrammarType
    # Grammar value
    value: Union[str, dict]


class Parameters(BaseModel):
    # Activate logits sampling
    do_sample: bool = False
    # Maximum number of generated tokens
    max_new_tokens: int = 20
    # The parameter for repetition penalty. 1.0 means no penalty.
    # See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    repetition_penalty: Optional[float] = None
    # Whether to prepend the prompt to the generated text
    return_full_text: bool = False
    # Stop generating tokens if a member of `stop_sequences` is generated
    stop: List[str] = []
    # Random sampling seed
    seed: Optional[int] = None
    # The value used to module the logits distribution.
    temperature: Optional[float] = None
    # The number of highest probability vocabulary tokens to keep for top-k-filtering.
    top_k: Optional[int] = None
    # If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
    # higher are kept for generation.
    top_p: Optional[float] = None
    # truncate inputs tokens to the given size
    truncate: Optional[int] = None
    # Typical Decoding mass
    # See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
    typical_p: Optional[float] = None
    # Generate best_of sequences and return the one if the highest token logprobs
    best_of: Optional[int] = None
    # Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
    watermark: bool = False
    # Get generation details
    details: bool = False
    # Get decoder input token logprobs and ids
    decoder_input_details: bool = False
    # Return the N most likely tokens at each step
    top_n_tokens: Optional[int] = None
    # grammar to use for generation
    grammar: Optional[Grammar] = None

    @validator("best_of")
    def valid_best_of(cls, field_value, values):
        if field_value is not None:
            if field_value <= 0:
                raise ValidationError("`best_of` must be strictly positive")
            if field_value > 1 and values["seed"] is not None:
                raise ValidationError("`seed` must not be set when `best_of` is > 1")
            sampling = (
                values["do_sample"]
                | (values["temperature"] is not None)
                | (values["top_k"] is not None)
                | (values["top_p"] is not None)
                | (values["typical_p"] is not None)
            )
            if field_value > 1 and not sampling:
                raise ValidationError("you must use sampling when `best_of` is > 1")

        return field_value

    @validator("repetition_penalty")
    def valid_repetition_penalty(cls, v):
        if v is not None and v <= 0:
            raise ValidationError("`repetition_penalty` must be strictly positive")
        return v

    @validator("seed")
    def valid_seed(cls, v):
        if v is not None and v < 0:
            raise ValidationError("`seed` must be positive")
        return v

    @validator("temperature")
    def valid_temp(cls, v):
        if v is not None and v <= 0:
            raise ValidationError("`temperature` must be strictly positive")
        return v

    @validator("top_k")
    def valid_top_k(cls, v):
        if v is not None and v <= 0:
            raise ValidationError("`top_k` must be strictly positive")
        return v

    @validator("top_p")
    def valid_top_p(cls, v):
        if v is not None and (v <= 0 or v >= 1.0):
            raise ValidationError("`top_p` must be > 0.0 and < 1.0")
        return v

    @validator("truncate")
    def valid_truncate(cls, v):
        if v is not None and v <= 0:
            raise ValidationError("`truncate` must be strictly positive")
        return v

    @validator("typical_p")
    def valid_typical_p(cls, v):
        if v is not None and (v <= 0 or v >= 1.0):
            raise ValidationError("`typical_p` must be > 0.0 and < 1.0")
        return v

    @validator("top_n_tokens")
    def valid_top_n_tokens(cls, v):
        if v is not None and v <= 0:
            raise ValidationError("`top_n_tokens` must be strictly positive")
        return v

    @validator("grammar")
    def valid_grammar(cls, v):
        if v is not None:
            if v.type == GrammarType.Regex and not v.value:
                raise ValidationError("`value` cannot be empty for `regex` grammar")
            if v.type == GrammarType.Json and not v.value:
                raise ValidationError("`value` cannot be empty for `json` grammar")
        return v


class Request(BaseModel):
    # Prompt
    inputs: str
    # Generation parameters
    parameters: Optional[Parameters] = None
    # Whether to stream output tokens
    stream: bool = False

    @validator("inputs")
    def valid_input(cls, v):
        if not v:
            raise ValidationError("`inputs` cannot be empty")
        return v

    @validator("stream")
    def valid_best_of_stream(cls, field_value, values):
        parameters = values["parameters"]
        if (
            parameters is not None
            and parameters.best_of is not None
            and parameters.best_of > 1
            and field_value
        ):
            raise ValidationError(
                "`best_of` != 1 is not supported when `stream` == True"
            )
        return field_value


# Decoder input tokens
class InputToken(BaseModel):
    # Token ID from the model tokenizer
    id: int
    # Token text
    text: str
    # Logprob
    # Optional since the logprob of the first token cannot be computed
    logprob: Optional[float] = None


# Generated tokens
class Token(BaseModel):
    # Token ID from the model tokenizer
    id: int
    # Token text
    text: str
    # Logprob
    logprob: Optional[float] = None
    # Is the token a special token
    # Can be used to ignore tokens when concatenating
    special: bool


# Generation finish reason
class FinishReason(str, Enum):
    # number of generated tokens == `max_new_tokens`
    Length = "length"
    # the model generated its end of sequence token
    EndOfSequenceToken = "eos_token"
    # the model generated a text included in `stop_sequences`
    StopSequence = "stop_sequence"


# Additional sequences when using the `best_of` parameter
class BestOfSequence(BaseModel):
    # Generated text
    generated_text: str
    # Generation finish reason
    finish_reason: FinishReason
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int] = None
    # Decoder input tokens, empty if decoder_input_details is False
    prefill: List[InputToken]
    # Generated tokens
    tokens: List[Token]
    # Most likely tokens
    top_tokens: Optional[List[List[Token]]] = None


# `generate` details
class Details(BaseModel):
    # Generation finish reason
    finish_reason: FinishReason
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int] = None
    # Decoder input tokens, empty if decoder_input_details is False
    prefill: List[InputToken]
    # Generated tokens
    tokens: List[Token]
    # Most likely tokens
    top_tokens: Optional[List[List[Token]]] = None
    # Additional sequences when using the `best_of` parameter
    best_of_sequences: Optional[List[BestOfSequence]] = None


# `generate` return value
class Response(BaseModel):
    # Generated text
    generated_text: str
    # Generation details
    details: Details


# `generate_stream` details
class StreamDetails(BaseModel):
    # Generation finish reason
    finish_reason: FinishReason
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int] = None


# `generate_stream` return value
class StreamResponse(BaseModel):
    # Generated token
    token: Token
    # Most likely tokens
    top_tokens: Optional[List[Token]] = None
    # Complete generated text
    # Only available when the generation is finished
    generated_text: Optional[str] = None
    # Generation details
    # Only available when the generation is finished
    details: Optional[StreamDetails] = None


##################### Client #######################################

class Client:
    """Client to make calls to a text-generation-inference instance

     Example:

     ```python
     >>> from text_generation import Client

     >>> client = Client("https://api-inference.huggingface.co/models/bigscience/bloomz")
     >>> client.generate("Why is the sky blue?").generated_text
     ' Rayleigh scattering'

     >>> result = ""
     >>> for response in client.generate_stream("Why is the sky blue?"):
     >>>     if not response.token.special:
     >>>         result += response.token.text
     >>> result
    ' Rayleigh scattering'
     ```
    """

    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: int = 10,
    ):
        """
        Args:
            base_url (`str`):
                text-generation-inference instance base url
            headers (`Optional[Dict[str, str]]`):
                Additional headers
            cookies (`Optional[Dict[str, str]]`):
                Cookies to include in the requests
            timeout (`int`):
                Timeout in seconds
        """
        self.base_url = base_url
        self.headers = headers
        self.cookies = cookies
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        do_sample: bool = False,
        max_new_tokens: int = 20,
        best_of: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
        decoder_input_details: bool = False,
        top_n_tokens: Optional[int] = None,
        grammar: Optional[Grammar] = None,
    ) -> Response:
        """
        Given a prompt, generate the following text

        Args:
            prompt (`str`):
                Input text
            do_sample (`bool`):
                Activate logits sampling
            max_new_tokens (`int`):
                Maximum number of generated tokens
            best_of (`int`):
                Generate best_of sequences and return the one if the highest token logprobs
            repetition_penalty (`float`):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            return_full_text (`bool`):
                Whether to prepend the prompt to the generated text
            seed (`int`):
                Random sampling seed
            stop_sequences (`List[str]`):
                Stop generating tokens if a member of `stop_sequences` is generated
            temperature (`float`):
                The value used to module the logits distribution.
            top_k (`int`):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`):
                Truncate inputs tokens to the given size
            typical_p (`float`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
            decoder_input_details (`bool`):
                Return the decoder input token logprobs and ids
            top_n_tokens (`int`):
                Return the `n` most likely tokens at each step

        Returns:
            Response: generated response
        """
        # Validate parameters
        parameters = Parameters(
            best_of=best_of,
            details=True,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            return_full_text=return_full_text,
            seed=seed,
            stop=stop_sequences if stop_sequences is not None else [],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            truncate=truncate,
            typical_p=typical_p,
            watermark=watermark,
            decoder_input_details=decoder_input_details,
            top_n_tokens=top_n_tokens,
            grammar=grammar,
        )
        request = Request(inputs=prompt, stream=False, parameters=parameters)

        resp = requests.post(
            self.base_url,
            json=request.dict(),
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.timeout,
        )
        payload = resp.json()
        if resp.status_code != 200:
            raise parse_error(resp.status_code, payload)
        return Response(**payload[0])

    def generate_stream(
        self,
        prompt: str,
        do_sample: bool = False,
        max_new_tokens: int = 20,
        repetition_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
        top_n_tokens: Optional[int] = None,
        grammar: Optional[Grammar] = None,
    ) -> Iterator[StreamResponse]:
        """
        Given a prompt, generate the following stream of tokens

        Args:
            prompt (`str`):
                Input text
            do_sample (`bool`):
                Activate logits sampling
            max_new_tokens (`int`):
                Maximum number of generated tokens
            repetition_penalty (`float`):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            return_full_text (`bool`):
                Whether to prepend the prompt to the generated text
            seed (`int`):
                Random sampling seed
            stop_sequences (`List[str]`):
                Stop generating tokens if a member of `stop_sequences` is generated
            temperature (`float`):
                The value used to module the logits distribution.
            top_k (`int`):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`):
                Truncate inputs tokens to the given size
            typical_p (`float`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
            top_n_tokens (`int`):
                Return the `n` most likely tokens at each step

        Returns:
            Iterator[StreamResponse]: stream of generated tokens
        """
        # Validate parameters
        parameters = Parameters(
            best_of=None,
            details=True,
            decoder_input_details=False,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            return_full_text=return_full_text,
            seed=seed,
            stop=stop_sequences if stop_sequences is not None else [],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            truncate=truncate,
            typical_p=typical_p,
            watermark=watermark,
            top_n_tokens=top_n_tokens,
            grammar=grammar,
        )
        request = Request(inputs=prompt, stream=True, parameters=parameters)

        resp = requests.post(
            self.base_url,
            json=request.dict(),
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.timeout,
            stream=True,
        )

        if resp.status_code != 200:
            raise parse_error(resp.status_code, resp.json())

        # Parse ServerSentEvents
        for byte_payload in resp.iter_lines():
            # Skip line
            if byte_payload == b"\n":
                continue

            payload = byte_payload.decode("utf-8")

            # Event data
            if payload.startswith("data:"):
                # Decode payload
                json_payload = json.loads(payload.lstrip("data:").rstrip("/n"))
                # Parse payload
                try:
                    response = StreamResponse(**json_payload)
                except ValidationError:
                    # If we failed to parse the payload, then it is an error payload
                    raise parse_error(resp.status_code, json_payload)
                yield response


class AsyncClient:
    """Asynchronous Client to make calls to a text-generation-inference instance

     Example:

     ```python
     >>> from text_generation import AsyncClient

     >>> client = AsyncClient("https://api-inference.huggingface.co/models/bigscience/bloomz")
     >>> response = await client.generate("Why is the sky blue?")
     >>> response.generated_text
     ' Rayleigh scattering'

     >>> result = ""
     >>> async for response in client.generate_stream("Why is the sky blue?"):
     >>>     if not response.token.special:
     >>>         result += response.token.text
     >>> result
    ' Rayleigh scattering'
     ```
    """

    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: int = 10,
    ):
        """
        Args:
            base_url (`str`):
                text-generation-inference instance base url
            headers (`Optional[Dict[str, str]]`):
                Additional headers
            cookies (`Optional[Dict[str, str]]`):
                Cookies to include in the requests
            timeout (`int`):
                Timeout in seconds
        """
        self.base_url = base_url
        self.headers = headers
        self.cookies = cookies
        self.timeout = ClientTimeout(timeout * 60)

    async def generate(
        self,
        prompt: str,
        do_sample: bool = False,
        max_new_tokens: int = 20,
        best_of: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
        decoder_input_details: bool = False,
        top_n_tokens: Optional[int] = None,
        grammar: Optional[Grammar] = None,
    ) -> Response:
        """
        Given a prompt, generate the following text asynchronously

        Args:
            prompt (`str`):
                Input text
            do_sample (`bool`):
                Activate logits sampling
            max_new_tokens (`int`):
                Maximum number of generated tokens
            best_of (`int`):
                Generate best_of sequences and return the one if the highest token logprobs
            repetition_penalty (`float`):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            return_full_text (`bool`):
                Whether to prepend the prompt to the generated text
            seed (`int`):
                Random sampling seed
            stop_sequences (`List[str]`):
                Stop generating tokens if a member of `stop_sequences` is generated
            temperature (`float`):
                The value used to module the logits distribution.
            top_k (`int`):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`):
                Truncate inputs tokens to the given size
            typical_p (`float`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
            decoder_input_details (`bool`):
                Return the decoder input token logprobs and ids
            top_n_tokens (`int`):
                Return the `n` most likely tokens at each step

        Returns:
            Response: generated response
        """

        # Validate parameters
        parameters = Parameters(
            best_of=best_of,
            details=True,
            decoder_input_details=decoder_input_details,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            return_full_text=return_full_text,
            seed=seed,
            stop=stop_sequences if stop_sequences is not None else [],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            truncate=truncate,
            typical_p=typical_p,
            watermark=watermark,
            top_n_tokens=top_n_tokens,
            grammar=grammar,
        )
        request = Request(inputs=prompt, stream=False, parameters=parameters)

        async with ClientSession(
            headers=self.headers, cookies=self.cookies, timeout=self.timeout
        ) as session:
            async with session.post(self.base_url, json=request.dict()) as resp:
                payload = await resp.json()

                if resp.status != 200:
                    raise parse_error(resp.status, payload)
                return Response(**payload[0])

    async def generate_stream(
        self,
        prompt: str,
        do_sample: bool = False,
        max_new_tokens: int = 20,
        repetition_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
        top_n_tokens: Optional[int] = None,
        grammar: Optional[Grammar] = None,
    ) -> AsyncIterator[StreamResponse]:
        """
        Given a prompt, generate the following stream of tokens asynchronously

        Args:
            prompt (`str`):
                Input text
            do_sample (`bool`):
                Activate logits sampling
            max_new_tokens (`int`):
                Maximum number of generated tokens
            repetition_penalty (`float`):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            return_full_text (`bool`):
                Whether to prepend the prompt to the generated text
            seed (`int`):
                Random sampling seed
            stop_sequences (`List[str]`):
                Stop generating tokens if a member of `stop_sequences` is generated
            temperature (`float`):
                The value used to module the logits distribution.
            top_k (`int`):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`):
                Truncate inputs tokens to the given size
            typical_p (`float`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
            top_n_tokens (`int`):
                Return the `n` most likely tokens at each step

        Returns:
            AsyncIterator[StreamResponse]: stream of generated tokens
        """
        # Validate parameters
        parameters = Parameters(
            best_of=None,
            details=True,
            decoder_input_details=False,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            return_full_text=return_full_text,
            seed=seed,
            stop=stop_sequences if stop_sequences is not None else [],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            truncate=truncate,
            typical_p=typical_p,
            watermark=watermark,
            top_n_tokens=top_n_tokens,
            grammar=grammar,
        )
        request = Request(inputs=prompt, stream=True, parameters=parameters)

        async with ClientSession(
            headers=self.headers, cookies=self.cookies, timeout=self.timeout
        ) as session:
            async with session.post(self.base_url, json=request.dict()) as resp:
                if resp.status != 200:
                    raise parse_error(resp.status, await resp.json())

                # Parse ServerSentEvents
                async for byte_payload in resp.content:
                    # Skip line
                    if byte_payload == b"\n":
                        continue

                    payload = byte_payload.decode("utf-8")

                    # Event data
                    if payload.startswith("data:"):
                        # Decode payload
                        json_payload = json.loads(payload.lstrip("data:").rstrip("/n"))
                        # Parse payload
                        try:
                            response = StreamResponse(**json_payload)
                        except ValidationError:
                            # If we failed to parse the payload, then it is an error payload
                            raise parse_error(resp.status, json_payload)
                        yield response


## OpenAI compatible Client
class OaRequest(BaseModel):
    model: str = "llama"
    stream: bool = False
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    max_tokens: Optional[int] = None #16
    temperature: Optional[float] = None # 1.0
    top_p: Optional[float] = None # 1.0
    top_k: Optional[int] = None # -1
    n: Optional[int] = None #1
    use_beam_search: Optional[bool] = None # False
    logprobs: Optional[int] = None
    #top_logprobs: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None #0
    frequency_penalty: Optional[float] = None # 0
    best_of: Optional[int] = None # 1
    logit_bias: Optional[Dict[int, float]] = None
    early_stopping: Optional[Union[bool, str]] = None
    #response_format: Optional[Dict] = None
    #seed: Optional[int] = None
    #tools: Optional[List] = None
    #tool_choice: Optional[str] = None
    #user: Optional[str] = None

    def to_json(self):
        kv = {"model": self.model, "stream": self.stream}
        if self.prompt is not None:
            kv["prompt"] = self.prompt
        if self.messages is not None:
            kv["messages"] = self.messages
        if self.max_tokens is not None:
            kv["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kv["temperature"] = self.temperature
        if self.top_p is not None:
            kv["top_p"] = self.top_p
        if self.top_k is not None:
            kv["top_k"] = self.top_k
        if self.n is not None:
            kv["n"] = self.n
        if self.use_beam_search is not None:
            kv["use_beam_search"] = self.use_beam_search
        if self.logprobs is not None:
            kv["logprobs"] = self.logprobs
        if self.stop is not None:
            kv["stop"] = self.stop
        if self.presence_penalty is not None:
            kv["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            kv["frequency_penalty"] = self.frequency_penalty
        if self.best_of is not None:
            kv["best_of"] = self.best_of
        if self.logit_bias is not None:
            kv["logit_bias"] = self.logit_bias
        if self.early_stopping is not None:
            kv["early_stopping"] = self.early_stopping  
        return kv
        
class OaResponse(BaseModel):
    generated_text: List[str] = []
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int = 0

    def from_json(self, obj):
        if "choices" in obj:
            for e in obj["choices"]:
                self.generated_text.append(e["text"])
        if "usage" in obj:
            self.prompt_tokens = obj["usage"]["prompt_tokens"]
            self.total_tokens = obj["usage"]["total_tokens"]
            self.completion_tokens = obj["usage"]["completion_tokens"]

class OaStreamResponse(BaseModel):
    token: Token = Token(id=0, text="", special=False)
    generated_text: Optional[str] = None
    details: Optional[StreamDetails] = None

    #def __init__(self):
    #    self.token = Token(0, "", special=False)
    def from_json(self, obj):
        if "choices" in obj:
            self.generated_text = ""
            for e in obj["choices"]:
                self.generated_text += e["text"]
            self.token.text = self.generated_text
        if "usage" in obj:
            self.details = StreamDetails()
            self.details.generated_tokens = obj["usage"]["completion_tokens"]


class OaClient:
    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: int = 10,
    ):
        self.base_url = base_url
        self.headers = headers
        self.cookies = cookies
        self.timeout = timeout

    ## Stream == False
    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        n: Optional[int] = None,
        use_beam_search: Optional[bool] = None,
        logprobs: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        best_of: Optional[int] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        early_stopping: Optional[Union[bool, str]] = None,
    ) -> OaResponse:
        # Validate parameters
        request = OaRequest(
            prompt=prompt,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            n=n,
            use_beam_search=use_beam_search,
            logprobs=logprobs,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            best_of=best_of,
            logit_bias=logit_bias,
            early_stopping=early_stopping,
            stream=False
        )
        #print(f"request: {request.dict(exclude_none=True)}\n\n")
        resp = requests.post(
            self.base_url,
            #json=request.to_json(),
            json=request.dict(exclude_none=True),
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise parse_error(resp.status_code, resp.json())
        #print(f"resp: {resp.json()}\n\n")
        ret = OaResponse()
        ret.from_json(resp.json())
        return ret
    
    ## Stream == True
    def generate_stream(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        n: Optional[int] = None,
        use_beam_search: Optional[bool] = None,
        logprobs: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        best_of: Optional[int] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        early_stopping: Optional[Union[bool, str]] = None,
    ) -> Iterator[OaResponse]:
        # Validate parameters
        request = OaRequest(
            prompt=prompt,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            n=n,
            use_beam_search=use_beam_search,
            logprobs=logprobs,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            best_of=best_of,
            logit_bias=logit_bias,
            early_stopping=early_stopping,
            stream=True
        )

        resp = requests.post(
            self.base_url,
            json=request.dict(exclude_none=True),
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.timeout,
            stream=True,
        )

        if resp.status_code != 200:
            raise parse_error(resp.status_code, resp.json())
        # Parse ServerSentEvents
        for byte_payload in resp.iter_lines():
            # Skip line
            if byte_payload == b"\n":
                continue

            payload = byte_payload.decode("utf-8")

            # Event data
            if payload.startswith("data:"):
                # Decode payload
                #print(f"raw payload: {payload}")
                try:
                    json_payload = json.loads(payload.lstrip("data:").rstrip("/n"))
                except ValueError as e:
                    json_payload = None
                if json_payload is not None:
                    #print(f"json_payload: {json_payload}")
                    response = OaStreamResponse()
                    response.from_json(json_payload)
                    yield response


class SilRequest(BaseModel):
    model: str = "llama"
    prompt: str = ""
    max_tokens: int = 32
    temperature: float = 0.0
    stream: bool = False

class SilClient:
    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: int = 10,
    ):
        self.base_url = base_url
        self.headers = headers
        self.cookies = cookies
        self.timeout = timeout