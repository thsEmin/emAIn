"""Text generation utilities for the GPT model."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _sample_next_token(logits, temperature, top_k, top_p, generator):
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if temperature < 0:
        raise ValueError("temperature cannot be negative")

    logits = logits / temperature

    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        top_k = min(top_k, logits.size(-1))
        cutoff = torch.topk(logits, top_k).values[:, -1].unsqueeze(1)
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probabilities = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(logits, float("-inf"))
        logits = filtered.scatter(1, sorted_indices, sorted_logits)

    probabilities = F.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator)


def generate(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool = True,
    seed: int | None = None,
) -> list[int]:
    """Return the prompt followed by generated token IDs."""
    if not prompt_ids:
        raise ValueError("prompt_ids cannot be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if len(prompt_ids) + max_new_tokens > model.config.block_size:
        raise ValueError("prompt and generated tokens exceed block_size")

    device = next(model.parameters()).device
    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    was_training = model.training
    model.eval()
    cache = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if use_cache:
                model_input = token_ids if cache is None else token_ids[:, -1:]
                logits, _, cache = model(model_input, past_kv=cache, use_cache=True)
            else:
                logits, _ = model(token_ids)

            next_token = _sample_next_token(
                logits[:, -1, :], temperature, top_k, top_p, generator
            )
            token_ids = torch.cat((token_ids, next_token), dim=1)

    if was_training:
        model.train()
    return token_ids[0].tolist()

