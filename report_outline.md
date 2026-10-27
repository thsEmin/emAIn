# HW1 report outline

The final PDF must be at most four pages, excluding an appendix. Replace every placeholder only with evidence from the project logs, `runs.csv`, `benchmark.csv`, `samples.txt`, and Git history.

## Full name

Write the student's full name on the first line.

## 1. What I built

Briefly explain the byte-level BPE tokenizer, decoder-only GPT, training pipeline, KV cache, and experiment design. Mention the verified parameter counts and tokenizer statistics from `report_notes.md`.

## 2. Results

Include:

- `figures/validation_curves.png`
- `figures/scaling.png`
- the bigram result and the best GPT result from `runs.csv`
- model M cached and uncached generation speeds from `benchmark.csv`

Every number must match its source file exactly.

## 3. Reflections

### R1 - Most expensive bug

Describe a real symptom, how it was diagnosed, and the Git commit that fixed it.

### R2 - Most surprising experiment

Compare real run IDs and quote their values from `runs.csv`.

### R3 - Largest validation-loss effect

Name the two comparable run IDs and their best validation losses.

### R4 - Generation failure

Discuss one pattern visible in the three samples in `samples.txt` and connect it to a design choice.

### R5 - Tokenizer

Use the first ten text merges and measured bytes/token from `report_notes.md`. Explain why common letter-space and letter-pair patterns appear early.

## AI disclosure

Summarize the complete and truthful `ai_disclosure.md`. Do not claim that the work was completed without AI assistance.

