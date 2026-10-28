"""A small decoder-only Transformer written for HW1."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 1024
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    use_position_embedding: bool = True
    norm_type: str = "pre"


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.output = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, past_kv=None):
        batch_size, sequence_length, embedding_size = x.shape
        q, k, v = self.qkv(x).split(embedding_size, dim=2)

        q = q.view(batch_size, sequence_length, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.n_head, self.head_size).transpose(1, 2)

        past_length = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_length = past_k.size(2)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_size)

        query_positions = past_length + torch.arange(sequence_length, device=x.device)
        key_positions = torch.arange(k.size(2), device=x.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        result = weights @ v
        result = result.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_size)
        return self.output(result), (k, v)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.expand = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.contract = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.expand(x)
        x = F.gelu(x, approximate="tanh")
        x = self.contract(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm_type = config.norm_type
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None):
        if self.norm_type == "pre":
            attention_output, new_kv = self.attention(self.ln1(x), past_kv)
            x = x + attention_output
            x = x + self.mlp(self.ln2(x))
        elif self.norm_type == "post":
            attention_output, new_kv = self.attention(x, past_kv)
            x = self.ln1(x + attention_output)
            x = self.ln2(x + self.mlp(x))
        else:
            raise ValueError("norm_type must be 'pre' or 'post'")
        return x, new_kv


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = (
            nn.Embedding(config.block_size, config.n_embd)
            if config.use_position_embedding
            else None
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = []
        for layer_number in range(config.n_layer):
            block = Block(config)
            self.add_module(f"block_{layer_number}", block)
            self.blocks.append(block)
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, past_kv=None, use_cache=False):
        batch_size, sequence_length = idx.shape
        past_length = 0 if past_kv is None else past_kv[0][0].size(2)
        if past_length + sequence_length > self.config.block_size:
            raise ValueError("sequence is longer than block_size")

        positions = torch.arange(
            past_length,
            past_length + sequence_length,
            device=idx.device,
        )
        x = self.token_embedding(idx)
        if self.position_embedding is not None:
            x = x + self.position_embedding(positions)
        x = self.dropout(x)

        new_cache = []
        for layer_number, block in enumerate(self.blocks):
            layer_cache = None if past_kv is None else past_kv[layer_number]
            x, new_kv = block(x, layer_cache)
            new_cache.append(new_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )

        if use_cache:
            return logits, loss, new_cache
        return logits, loss

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

