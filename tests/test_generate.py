import unittest

import torch

from generate import _sample_next_token, generate
from model import GPT, GPTConfig


class GenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        config = GPTConfig(
            vocab_size=24,
            block_size=12,
            n_layer=2,
            n_head=2,
            n_embd=8,
            dropout=0.0,
        )
        self.model = GPT(config)

    def test_cached_and_uncached_greedy_match(self) -> None:
        cached = generate(self.model, [1, 2, 3], 6, temperature=0, use_cache=True)
        uncached = generate(self.model, [1, 2, 3], 6, temperature=0, use_cache=False)
        self.assertEqual(cached, uncached)

    def test_seed_makes_sampling_repeatable(self) -> None:
        first = generate(self.model, [1, 2], 5, top_k=5, seed=123)
        second = generate(self.model, [1, 2], 5, top_k=5, seed=123)
        self.assertEqual(first, second)

    def test_top_k_keeps_exactly_k_tokens_when_logits_tie(self) -> None:
        logits = torch.zeros((1, 4))
        allowed = set(torch.topk(logits, 2).indices[0].tolist())

        for seed in range(20):
            generator = torch.Generator().manual_seed(seed)
            token = _sample_next_token(logits, 1.0, 2, None, generator).item()
            self.assertIn(token, allowed)

    def test_top_p_keeps_the_threshold_crossing_token(self) -> None:
        probabilities = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
        logits = probabilities.log()

        for seed in range(20):
            generator = torch.Generator().manual_seed(seed)
            token = _sample_next_token(logits, 1.0, None, 0.6, generator).item()
            self.assertIn(token, {0, 1})


if __name__ == "__main__":
    unittest.main()

