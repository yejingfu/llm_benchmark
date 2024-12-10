import base64
import os
import queue
import torch
import json
import logging
from loguru import logger
from natsort import natsorted
from pathlib import Path
from dataclasses import dataclass, field
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field, conint, conlist
from pydantic.functional_validators import SkipValidation

import tiktoken

logger = logging.getLogger(__name__)

############### Tokenizer #######################
# This is a modified version of the default pattern from GPT-4o, that better handles punctuations.
FISH_TIKTOKEN_PATTERN = "|".join(
    [
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
        r"\p{P}",
        r"[^\r\n\p{L}\p{N}]?\p{L}+",
        r"\p{N}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(\?!\S)",
        r"\s+",
    ]
)
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

BOS_TOKEN = "<|begin_of_text|>"
EOS_TOKEN = "<|end_of_text|>"
PAD_TOKEN = "<|pad|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"

MODALITY_TEXT_TOKEN = "<|text|>"
MODALITY_VOICE_TOKEN = "<|voice|>"
MODALITY_INTERLEAVE_TOKEN = "<|interleave|>"
MODALITY_TOKENS = {
    "text": MODALITY_TEXT_TOKEN,
    "voice": MODALITY_VOICE_TOKEN,
    "interleave": MODALITY_INTERLEAVE_TOKEN,
}

PLACEHOLDER_TOKEN = [""] * 4
for i in range(4):
    PLACEHOLDER_TOKEN[i] = f"<|placeholder:{i}|>"

SEMANTIC_TOKEN_TEMPLATE = "<|semantic:{i}|>"
SEMANTIC_TOKENS = [SEMANTIC_TOKEN_TEMPLATE.format(i=i) for i in range(1024)]

# Warning: when you add a new special token, you should only add it to the end of the list.
ALL_SPECIAL_TOKENS = [
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    IM_START_TOKEN,
    IM_END_TOKEN,
    PLACEHOLDER_TOKEN[0],
    PLACEHOLDER_TOKEN[1],
    PLACEHOLDER_TOKEN[2],
    PLACEHOLDER_TOKEN[3],
    MODALITY_TEXT_TOKEN,
    MODALITY_VOICE_TOKEN,
    MODALITY_INTERLEAVE_TOKEN,
    *SEMANTIC_TOKENS,
]


class FishTokenizer:
    def __init__(self, model_path: str) -> None:
        mergeable_ranks = self.load_tiktoken_bpe(model_path)
        special_token_begin = len(mergeable_ranks)
        self.all_special_tokens_with_ids = {
            token: special_token_begin + i for i, token in enumerate(ALL_SPECIAL_TOKENS)
        }
        self.semantic_id_to_token_id = {
            i: self.all_special_tokens_with_ids[token]
            for i, token in enumerate(SEMANTIC_TOKENS)
        }
        self.semantic_begin_id = self.all_special_tokens_with_ids[SEMANTIC_TOKENS[0]]
        self.semantic_end_id = self.all_special_tokens_with_ids[SEMANTIC_TOKENS[-1]]

        self.tkt_model = tiktoken.core.Encoding(
            name=Path(model_path).stem,
            pat_str=FISH_TIKTOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.all_special_tokens_with_ids,
        )

    @staticmethod
    def load_tiktoken_bpe(tiktoken_bpe_file: str) -> dict[bytes, int]:
        data = {}
        for line in open(tiktoken_bpe_file).read().splitlines():
            if not line:
                continue
            token, rank = line.split()
            data[base64.b64decode(token)] = int(rank)
        return data

    def get_token_id(self, token: str) -> int:
        return self.all_special_tokens_with_ids[token]

    def encode(self, s: str, allowed_special: bool | set[str] = True) -> list[int]:
        assert isinstance(s, str)

        subs = []
        for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS):
            subs.append(s[i : i + TIKTOKEN_MAX_ENCODE_CHARS])

        if allowed_special is True:
            allowed_special = self.tkt_model.special_tokens_set
        elif allowed_special is False:
            allowed_special = set()

        return sum(
            self.tkt_model.encode_batch(
                subs, allowed_special=allowed_special, disallowed_special=set()
            ),
            start=[],
        )

    def decode(self, tokens: list[int]) -> str:
        return self.tkt_model.decode(tokens)

    def save_pretrained(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "tokenizer.tiktoken", "w") as f:
            for token, rank in self.tkt_model._mergeable_ranks.items():
                f.write(f"{base64.b64encode(token).decode()} {rank}\n")

        with open(path / "special_tokens.json", "w") as f:
            json.dump(
                self.all_special_tokens_with_ids,
                f,
                indent=2,
                ensure_ascii=False,
            )

    @staticmethod
    def from_pretrained(path: str):
        return FishTokenizer(Path(path) / "tokenizer.tiktoken")


if __name__ == "__main__":
    tokenizer = FishTokenizer("data/mpacks/v1.4-pretrain/tokenizer.all.tiktoken")
    tokenizer.save_pretrained("checkpoints/fish-speech-0.5B")
    tokenizer = FishTokenizer.from_pretrained("checkpoints/fish-speech-0.5B")

    print(
        [
            tokenizer.decode([i])
            for i in tokenizer.encode(f"{BOS_TOKEN}你好，世界！{EOS_TOKEN}")
        ]
    )

############## conversion ###################

CODEBOOK_PAD_TOKEN_ID = 0

@dataclass(kw_only=True)
class BasePart:
    pass


@dataclass(kw_only=True)
class VQPart(BasePart):
    codes: torch.Tensor


@dataclass(kw_only=True)
class TextPart(BasePart):
    text: str


@dataclass(kw_only=True)
class EncodedMessage:
    tokens: torch.Tensor
    labels: torch.Tensor
    vq_mask_tokens: torch.Tensor | None = None
    vq_mask_labels: torch.Tensor | None = None
    vq_parts: list[torch.Tensor]
    vq_require_losses: torch.Tensor | None = None


@dataclass(kw_only=True)
class Message:
    role: Literal["system", "user", "assistant"]
    parts: list[VQPart | TextPart] = field(default_factory=list)
    add_im_start: bool = True
    add_im_end: bool = True
    cal_loss: bool = False
    modality: Literal["text", "voice", "interleave"] | None = None

    # By default, ignore the loss of the auto-generated im_start token
    ignore_im_start_loss: bool = True

    def encode(
        self: "Message",
        tokenizer: FishTokenizer,
    ) -> EncodedMessage:
        all_tokens = []
        all_labels = []

        # Multi-modal tokens
        vq_parts = []
        vq_masks = []

        parts = self.parts.copy()
        if self.add_im_start:
            modality_token = MODALITY_TOKENS[self.modality] if self.modality else ""
            parts.insert(0, TextPart(text=f"<|im_start|>{self.role}\n{modality_token}"))

        if self.add_im_end:
            parts.append(TextPart(text="<|im_end|>"))

        for part in parts:
            if isinstance(part, TextPart):
                tokens = torch.tensor(
                    tokenizer.encode(part.text),
                    dtype=torch.int,
                )
            elif isinstance(part, VQPart):
                curr_codes = part.codes.clone()
                tokens = torch.tensor(
                    [
                        tokenizer.semantic_id_to_token_id[i.item()]
                        for i in curr_codes[0].int()
                    ],
                    dtype=torch.int,
                )
                vq_parts.append(curr_codes)
            else:
                raise ValueError(f"Unsupported part type: {type(part)}")

            all_tokens.append(tokens)
            if isinstance(part, VQPart):
                vq_masks.append(torch.ones_like(tokens, dtype=torch.bool))
            else:
                vq_masks.append(torch.zeros_like(tokens, dtype=torch.bool))

            if self.cal_loss:
                all_labels.append(tokens.clone())
            else:
                all_labels.append(torch.full_like(tokens, -100))

        tokens = torch.cat(all_tokens, dim=0)
        labels = torch.cat(all_labels, dim=0)
        vq_masks = torch.cat(vq_masks, dim=0)

        assert tokens.shape == labels.shape == vq_masks.shape

        if self.ignore_im_start_loss and self.add_im_start:
            labels[: len(all_tokens[0])] = -100

        return EncodedMessage(
            tokens=tokens,
            labels=labels,
            vq_parts=vq_parts,
            vq_mask_tokens=vq_masks,
            vq_mask_labels=vq_masks,
        )


@dataclass
class Conversation:
    messages: list[Message]

    def __init__(self: "Conversation", messages: list[Message] | None = None):
        self.messages = messages or []

    def encode(
        self: "Conversation",
        tokenizer: FishTokenizer,
        add_shift: bool = True,
        ignore_loss_tokens: list[str] = [],
    ) -> EncodedMessage:
        # Build the input_ids and labels
        tokens = []
        labels = []
        vq_parts = []
        vq_mask_tokens = []
        vq_mask_labels = []
        vq_require_losses = []
        ignore_loss_token_ids = [tokenizer.get_token_id(i) for i in ignore_loss_tokens]

        for message in self.messages:
            encoded = message.encode(
                tokenizer,
            )
            tokens.append(encoded.tokens)
            labels.append(encoded.labels)
            vq_parts.extend(encoded.vq_parts)
            vq_mask_tokens.append(encoded.vq_mask_tokens)
            vq_mask_labels.append(encoded.vq_mask_labels)
            vq_require_losses.extend([message.cal_loss] * len(encoded.vq_parts))

        tokens = torch.cat(tokens, dim=0)
        labels = torch.cat(labels, dim=0)
        vq_mask_tokens = torch.cat(vq_mask_tokens, dim=0)
        vq_mask_labels = torch.cat(vq_mask_labels, dim=0)
        vq_require_losses = torch.tensor(vq_require_losses, dtype=torch.bool)

        if add_shift:
            tokens = tokens[:-1]
            labels = labels[1:]
            vq_mask_tokens = vq_mask_tokens[:-1]
            vq_mask_labels = vq_mask_labels[1:]

        for i in ignore_loss_token_ids:
            assert i != -100 and i is not None
            labels[labels == i] = -100

        assert tokens.dtype in [
            torch.int,
            torch.long,
        ], f"Invalid dtype: {tokens.dtype}, conv: {conversation}"

        return EncodedMessage(
            tokens=tokens,
            labels=labels,
            vq_parts=vq_parts,
            vq_mask_tokens=vq_mask_tokens,
            vq_mask_labels=vq_mask_labels,
            vq_require_losses=vq_require_losses,
        )

    def encode_for_inference(
        self: "Conversation",
        tokenizer: FishTokenizer,
        num_codebooks: int,
    ) -> EncodedMessage:
        # self.visualize(tokenizer)

        encoded = self.encode(tokenizer, add_shift=False)
        tokens = encoded.tokens
        values = torch.zeros((num_codebooks + 1, len(tokens)), dtype=torch.int)
        values[0] = tokens

        if encoded.vq_parts is None or len(encoded.vq_parts) == 0:
            return values

        vq_parts = encoded.vq_parts
        vq_parts = [part.to(values.device) for part in vq_parts]
        vq_parts = torch.cat(vq_parts, dim=1)
        values[0, encoded.vq_mask_tokens] = vq_parts[0] + tokenizer.semantic_begin_id
        values[1:, encoded.vq_mask_tokens] = vq_parts

        return values

    def visualize(
        self: "Conversation",
        tokenizer: FishTokenizer,
        ignore_loss_tokens: list[str] = [],
    ):
        encoded = self.encode(
            tokenizer, add_shift=False, ignore_loss_tokens=ignore_loss_tokens
        )

        # Colors for alternating tokens
        colors = {
            "blue": "\033[94m",  # Light blue
            "cyan": "\033[96m",  # Cyan
            "green": "\033[92m",  # Light green
            "dark_green": "\033[32m",  # Dark green
        }
        blue_idx = 0
        green_idx = 0

        def print_in_blue(x):
            nonlocal blue_idx
            color = colors["blue"] if blue_idx % 2 == 0 else colors["cyan"]
            print(f"{color}{x}\033[0m", end="")
            blue_idx += 1

        def print_in_green(x):
            nonlocal green_idx
            color = colors["green"] if green_idx % 2 == 0 else colors["dark_green"]
            print(f"{color}{x}\033[0m", end="")
            green_idx += 1

        for tok, lab in zip(encoded.tokens, encoded.labels):
            val = tokenizer.decode([tok])

            if lab == -100:
                print_in_green(val)
            else:
                print_in_blue(val)

        print()

    def append(self: "Conversation", message: Message):
        self.messages.append(message)

###################### file & schema ####################

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".wma",
    ".aac",
    ".aiff",
    ".aif",
    ".aifc",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
}


def audio_to_bytes(file_path):
    if not file_path or not Path(file_path).exists():
        return None
    with open(file_path, "rb") as wav_file:
        wav = wav_file.read()
    return wav


def read_ref_text(ref_text):
    path = Path(ref_text)
    if path.exists() and path.is_file():
        with path.open("r", encoding="utf-8") as file:
            return file.read()
    return ref_text


def list_files(
    path: Union[Path, str],
    extensions: set[str] = None,
    recursive: bool = False,
    sort: bool = True,
) -> list[Path]:
    """List files in a directory.

    Args:
        path (Path): Path to the directory.
        extensions (set, optional): Extensions to filter. Defaults to None.
        recursive (bool, optional): Whether to search recursively. Defaults to False.
        sort (bool, optional): Whether to sort the files. Defaults to True.

    Returns:
        list: List of files.
    """

    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Directory {path} does not exist.")

    files = [file for ext in extensions for file in path.rglob(f"*{ext}")]

    if sort:
        files = natsorted(files)

    return files


def load_filelist(path: Path | str) -> list[tuple[Path, str, str, str]]:
    """
    Load a Bert-VITS2 style filelist.
    """

    files = set()
    results = []
    count_duplicated, count_not_found = 0, 0

    LANGUAGE_TO_LANGUAGES = {
        "zh": ["zh", "en"],
        "jp": ["jp", "en"],
        "en": ["en"],
    }

    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            splits = line.strip().split("|", maxsplit=3)
            if len(splits) != 4:
                logger.warning(f"Invalid line: {line}")
                continue

            filename, speaker, language, text = splits
            file = Path(filename)
            language = language.strip().lower()

            if language == "ja":
                language = "jp"

            assert language in ["zh", "jp", "en"], f"Invalid language {language}"
            languages = LANGUAGE_TO_LANGUAGES[language]

            if file in files:
                logger.warning(f"Duplicated file: {file}")
                count_duplicated += 1
                continue

            if not file.exists():
                logger.warning(f"File not found: {file}")
                count_not_found += 1
                continue

            results.append((file, speaker, languages, text))

    if count_duplicated > 0:
        logger.warning(f"Total duplicated files: {count_duplicated}")

    if count_not_found > 0:
        logger.warning(f"Total files not found: {count_not_found}")

    return results

#### schema
class ServeVQPart(BaseModel):
    type: Literal["vq"] = "vq"
    codes: SkipValidation[list[list[int]]]


class ServeTextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ServeAudioPart(BaseModel):
    type: Literal["audio"] = "audio"
    audio: bytes


@dataclass
class ASRPackRequest:
    audio: torch.Tensor
    result_queue: queue.Queue
    language: str


class ServeASRRequest(BaseModel):
    # The audio should be an uncompressed PCM float16 audio
    audios: list[bytes]
    sample_rate: int = 44100
    language: Literal["zh", "en", "ja", "auto"] = "auto"


class ServeASRTranscription(BaseModel):
    text: str
    duration: float
    huge_gap: bool


class ServeASRSegment(BaseModel):
    text: str
    start: float
    end: float


class ServeTimedASRResponse(BaseModel):
    text: str
    segments: list[ServeASRSegment]
    duration: float


class ServeASRResponse(BaseModel):
    transcriptions: list[ServeASRTranscription]


class ServeMessage(BaseModel):
    role: Literal["system", "assistant", "user"]
    parts: list[ServeVQPart | ServeTextPart]

    def to_conversation_message(self):
        new_message = Message(role=self.role, parts=[])
        if self.role == "assistant":
            new_message.modality = "voice"

        for part in self.parts:
            if isinstance(part, ServeTextPart):
                new_message.parts.append(TextPart(text=part.text))
            elif isinstance(part, ServeVQPart):
                new_message.parts.append(
                    VQPart(codes=torch.tensor(part.codes, dtype=torch.int))
                )
            else:
                raise ValueError(f"Unsupported part type: {part}")

        return new_message


class ServeChatRequest(BaseModel):
    messages: Annotated[list[ServeMessage], conlist(ServeMessage, min_length=1)]
    max_new_tokens: int = 1024
    top_p: float = 0.7
    repetition_penalty: float = 1.2
    temperature: float = 0.7
    streaming: bool = False
    num_samples: int = 1
    early_stop_threshold: float = 1.0


class ServeVQGANEncodeRequest(BaseModel):
    # The audio here should be in wav, mp3, etc
    audios: list[bytes]


class ServeVQGANEncodeResponse(BaseModel):
    tokens: SkipValidation[list[list[list[int]]]]


class ServeVQGANDecodeRequest(BaseModel):
    tokens: SkipValidation[list[list[list[int]]]]


class ServeVQGANDecodeResponse(BaseModel):
    # The audio here should be in PCM float16 format
    audios: list[bytes]


class ServeForwardMessage(BaseModel):
    role: str
    content: str


class ServeResponse(BaseModel):
    messages: list[ServeMessage]
    finish_reason: Literal["stop", "error"] | None = None
    stats: dict[str, int | float | str] = {}


class ServeStreamDelta(BaseModel):
    role: Literal["system", "assistant", "user"] | None = None
    part: ServeVQPart | ServeTextPart | None = None


class ServeStreamResponse(BaseModel):
    sample_id: int = 0
    delta: ServeStreamDelta | None = None
    finish_reason: Literal["stop", "error"] | None = None
    stats: dict[str, int | float | str] | None = None


class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str

    def __repr__(self) -> str:
        return f"ServeReferenceAudio(text={self.text!r}, audio_size={len(self.audio)})"


class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    # Audio format
    format: Literal["wav", "pcm", "mp3"] = "wav"
    # References audios for in-context learning
    references: list[ServeReferenceAudio] = []
    # Reference id
    # For example, if you want use https://fish.audio/m/7f92f8afb8ec43bf81429cc1c9199cb1/
    # Just pass 7f92f8afb8ec43bf81429cc1c9199cb1
    reference_id: str | None = None
    seed: int | None = None
    use_memory_cache: Literal["on", "off"] = "off"
    # Normalize text for en & zh, this increase stability for numbers
    normalize: bool = True
    # not usually used below
    streaming: bool = False
    max_new_tokens: int = 1024
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.2
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7

    class Config:
        # Allow arbitrary types for pytorch related types
        arbitrary_types_allowed = True
