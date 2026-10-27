"""A small byte-level BPE tokenizer for HW1."""

from __future__ import annotations

import json
from pathlib import Path


def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace non-overlapping copies of one pair from left to right."""
    result: list[int] = []
    i = 0

    while i < len(ids):
        pair_matches = i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair
        if pair_matches:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1

    return result


class BPETokenizer:
    """Byte-pair encoding without word splitting or tokenizer libraries."""

    def __init__(self) -> None:
        self.merges: list[tuple[int, int]] = []

    def train(self, text: str, vocab_size: int) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        ids = list(text.encode("utf-8"))
        self.merges = []

        for merge_number in range(vocab_size - 256):
            counts: dict[tuple[int, int], int] = {}
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1

            if not counts:
                break

            # Highest count wins. For equal counts, the smaller pair wins.
            best_pair = min(counts, key=lambda pair: (-counts[pair], pair))
            new_id = 256 + merge_number
            self.merges.append(best_pair)
            ids = _merge_pair(ids, best_pair, new_id)

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))

        for merge_number, pair in enumerate(self.merges):
            ids = _merge_pair(ids, pair, 256 + merge_number)

        return ids

    def decode(self, ids: list[int]) -> str:
        vocabulary = {byte: bytes([byte]) for byte in range(256)}

        for merge_number, (left, right) in enumerate(self.merges):
            vocabulary[256 + merge_number] = vocabulary[left] + vocabulary[right]

        try:
            data = b"".join(vocabulary[token_id] for token_id in ids)
        except KeyError as error:
            raise ValueError(f"unknown token id: {error.args[0]}") from error

        return data.decode("utf-8")

    def save(self, path: str) -> None:
        data = {"merges": [list(pair) for pair in self.merges]}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls()
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]
        return tokenizer

