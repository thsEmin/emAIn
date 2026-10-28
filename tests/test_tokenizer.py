import unittest
from pathlib import Path

from tokenizer import BPETokenizer


class BPETokenizerTests(unittest.TestCase):
    def test_tie_uses_the_smallest_pair(self) -> None:
        tokenizer = BPETokenizer()
        tokenizer.train("baac", vocab_size=257)
        self.assertEqual(tokenizer.merges, [(97, 97)])

    def test_overlapping_pair_is_merged_left_to_right(self) -> None:
        tokenizer = BPETokenizer()
        tokenizer.train("aaa", vocab_size=257)
        self.assertEqual(tokenizer.encode("aaa"), [256, 97])

    def test_unicode_text_round_trips(self) -> None:
        text = "Salam, Bakı! 🎭"
        tokenizer = BPETokenizer()
        tokenizer.train(text, vocab_size=265)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_saved_tokenizer_loads_with_the_same_behavior(self) -> None:
        text = "to be or not to be"
        tokenizer = BPETokenizer()
        tokenizer.train(text, vocab_size=264)

        path = Path(__file__).parent / "_temporary_tokenizer.json"
        try:
            tokenizer.save(str(path))
            loaded = BPETokenizer.load(str(path))
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(loaded.merges, tokenizer.merges)
        self.assertEqual(loaded.encode(text), tokenizer.encode(text))
        self.assertEqual(loaded.decode(loaded.encode(text)), text)


if __name__ == "__main__":
    unittest.main()

