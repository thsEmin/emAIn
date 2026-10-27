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

## Development checks

Run all tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```

Open `notebooks/01_tokenizer_walkthrough.ipynb` to step through a small example.

`notebooks/03_colab_training.ipynb` provides separate cells for the baseline and each required run. Upload the repository folder to Google Drive before using it.

## Training workflow

Run these commands from the repository root on a machine with PyTorch installed:

```powershell
python bigram.py
python train.py --config configs/S.json
python train.py --config configs/M.json
python train.py --config configs/L.json
python train.py --config configs/A1.json
python train.py --config configs/A2.json
python train.py --config configs/A3.json
python train.py --config configs/A4.json
python train.py --config configs/A5.json
```

Training creates `tokenizer.json`, cached token IDs, logs, checkpoints, and rows in `runs.csv`. The first tokenizer build can take time because all 768 BPE merges are learned from the training text.

After the runs finish:

```powershell
python make_figures.py
python benchmark.py
python make_samples.py
```

Checkpoints and cached data stay local. CSV logs, figures, `samples.txt`, and `tokenizer.json` are evidence for the report and should be committed after verifying them.

