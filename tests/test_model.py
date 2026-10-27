import unittest

import torch

from model import GPT, GPTConfig


class GPTTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = GPTConfig(
            vocab_size=32,
            block_size=8,
            n_layer=2,
            n_head=2,
            n_embd=8,
            dropout=0.0,
        )
        self.model = GPT(self.config).eval()

    def test_output_shape_and_loss(self) -> None:
        tokens = torch.randint(0, self.config.vocab_size, (3, 5))
        logits, loss = self.model(tokens, tokens)
        self.assertEqual(logits.shape, (3, 5, self.config.vocab_size))
        self.assertEqual(loss.ndim, 0)

    def test_future_token_does_not_change_earlier_logits(self) -> None:
        first = torch.tensor([[1, 2, 3, 4]])
        second = torch.tensor([[1, 2, 9, 4]])
        first_logits, _ = self.model(first)
        second_logits, _ = self.model(second)
        torch.testing.assert_close(first_logits[:, :2], second_logits[:, :2])

    def test_parameter_formula_and_weight_tying(self) -> None:
        c = self.config
        expected = (
            c.vocab_size * c.n_embd
            + c.block_size * c.n_embd
            + c.n_layer * (12 * c.n_embd**2 + 4 * c.n_embd)
            + 2 * c.n_embd
        )
        self.assertEqual(self.model.num_params(), expected)
        self.assertIs(self.model.lm_head.weight, self.model.token_embedding.weight)


if __name__ == "__main__":
    unittest.main()

