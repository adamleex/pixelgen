import os
import time
import fcntl
import torch
import torch.nn as nn
from src.models.conditioner.base import BaseConditioner

from transformers import Qwen3Model, Qwen2Tokenizer


def _ensure_downloaded(weight_path: str):
    """Use a file lock so only one process downloads to shared EFS; others wait."""
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    os.makedirs(cache_dir, exist_ok=True)
    lock_path = os.path.join(cache_dir, "download.lock")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    with open(lock_path, "w") as lock_file:
        print(f"[rank {local_rank}] acquiring download lock for {weight_path}...", flush=True)
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            print(f"[rank {local_rank}] lock acquired, loading {weight_path}", flush=True)
            tokenizer = Qwen2Tokenizer.from_pretrained(weight_path)
            model = Qwen3Model.from_pretrained(weight_path)
            print(f"[rank {local_rank}] model loaded successfully", flush=True)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    return tokenizer, model


class Qwen3TextEncoder(BaseConditioner):
    def __init__(self, weight_path: str, embed_dim:int=None, max_length=128):
        super().__init__()
        tokenizer, model = _ensure_downloaded(weight_path)
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = max_length
        self.tokenizer.padding_side = "right"
        self.model = model.to(torch.bfloat16)
        self.model.compile()
        self.uncondition_embedding = None
        self.embed_dim = embed_dim
        self.max_length = max_length

    def _impl_condition(self, y, metadata:dict={}):
        tokenized = self.tokenizer(y, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        input_ids = tokenized.input_ids.cuda()
        attention_mask = tokenized.attention_mask.cuda()
        metadata["valid_length_y"] = torch.sum(attention_mask, dim=-1)
        y = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        if y.shape[2] < self.embed_dim:
            y = torch.cat([y, torch.zeros(y.shape[0], y.shape[1], self.embed_dim - y.shape[2]).to(y.device, y.dtype)], dim=-1)
        if y.shape[2] > self.embed_dim:
            y = y[:, :, :self.embed_dim]
        return y

    def _impl_uncondition(self, y, metadata:dict=None):
        if self.uncondition_embedding is not None and "negative_prompt" not in metadata:
            return self.uncondition_embedding.repeat(len(y), 1, 1)
        negative_prompt = "" if "negative_prompt" not in metadata else metadata['negative_prompt']
        self.uncondition_embedding = self._impl_condition([negative_prompt,])
        return self.uncondition_embedding.repeat(len(y), 1, 1)