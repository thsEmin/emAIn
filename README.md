# emAIn

This project builds a small GPT-style language model from scratch for HW1.

The work is divided into the same parts as the assignment:

1. `tokenizer.py` - byte-level BPE tokenizer
2. `model.py` - decoder-only Transformer
3. `train.py` and `bigram.py` - training pipeline and baseline
4. `generate.py` - text generation with a KV cache
5. `configs/` and `logs/` - controlled experiments
6. `report.pdf` and `ai_disclosure.md` - results and reflection

The notebooks are learning aids. The grader imports the Python files directly, so the required implementation stays in those files.

## Dataset

Tiny Shakespeare is stored locally as `data/input.txt`. The `data/` directory is ignored by Git and must not be included in the final submission.

Source: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

## Part A checks

Run the tokenizer tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```

Open `notebooks/01_tokenizer_walkthrough.ipynb` to step through a small example.

