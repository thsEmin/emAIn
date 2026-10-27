# Report evidence notes

Only record values produced by the project. Do not fill missing results by guessing.

## Tokenizer

- Training characters: 1,003,854
- Validation characters/bytes: 111,540
- Vocabulary size: 1,024
- Learned merges: 768
- Training tokens: 397,994
- Validation tokens: 46,683
- Validation bytes per token: 2.3893065998329157
- Validation round-trip check: passed
- First ten merges as text: `e `, `th`, `t `, `s `, `d `, `, `, `ou`, `er`, `in`, `y `
- Corresponding merge pairs: `(101, 32)`, `(116, 104)`, `(116, 32)`, `(115, 32)`, `(100, 32)`, `(44, 32)`, `(111, 117)`, `(101, 114)`, `(105, 110)`, `(121, 32)`

## Baseline

- Bigram validation loss: 4.926122873366513 nats/token
- Required GPT target: at most 4.433510586029862 nats/token (10% lower)

## Verified before full training

- All unit tests passed.
- S parameter count: 172,672.
- M parameter count: 936,192.
- L parameter count: 2,880,384.
- Cached and uncached greedy generation matched on the small test model.
- A two-step end-to-end training smoke test completed, including evaluation, CSV logging, and checkpoint creation. The temporary smoke outputs were removed because they are not assignment experiments.

## Evidence still required

- S, M, L and A1-A5 training logs and `runs.csv` rows.
- Proof that the best GPT beats the bigram baseline by at least 10%.
- M cached/uncached speed benchmark.
- Three L samples at temperature 0.8.
- Two figures created from the completed logs.
- R1-R5 answers citing the real run IDs, log rows, and commits.
- Full name and final four-page report.

