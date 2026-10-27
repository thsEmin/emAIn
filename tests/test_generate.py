import unittest

import torch

from generate import generate
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


if __name__ == "__main__":
    unittest.main()

