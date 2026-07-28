# reasoning_llm — a from-scratch Transformer, NumPy only

A small GPT-style language model, its training loop, a custom automatic
differentiation engine, a synthetic reasoning benchmark, an evaluation
pipeline that compares it against a baseline model, and a chat website —
all implemented from scratch with **no ML framework, only NumPy** (Flask is
used only for the web server, not for any model computation).

## Honest scope, read this first

This is a real, working, from-scratch research/education project — not a
GPT-4 competitor. It's a ~64k-parameter transformer trained for a few
minutes on a CPU, on a synthetic dataset of arithmetic and simple logic
questions. On the included benchmark it currently scores **~50% exact-match
accuracy** (vs. 0% for a simple n-gram baseline) — genuinely better than the
baseline, genuinely imperfect at harder arithmetic. That gap between "beats
a naive baseline" and "reliable arithmetic" is real, is reported honestly
by `evaluate.py`, and is exactly the kind of result you can write up
truthfully in a course project, blog post, or workshop-style report. See
`DOCUMENTATION.md` for a full written account of the method and results,
including limitations, that you can use as the basis for a paper draft.

If you want to go further, `DOCUMENTATION.md` has a "next steps" section
(bigger model, more training, harder tasks, real held-out generalization
tests) that would meaningfully strengthen a publication-quality claim.

## What's in here

```
engine/autograd.py       from-scratch reverse-mode autodiff (the "PyTorch-lite")
model/gpt.py              TinyGPT: decoder-only transformer built on the engine
tokenizer.py               character-level tokenizer
optimizer.py                Adam optimizer + gradient clipping, in NumPy
data/generate_dataset.py    synthetic reasoning dataset generator
train.py                     training loop (CPU, checkpointing, resumable)
grad_check.py                 verifies the autodiff engine against numerical gradients
generate.py                    autoregressive sampling / text generation
chat_cli.py                     terminal chat
app.py + templates/index.html    Flask chat website
benchmark/reasoning_eval.py       held-out benchmark (arithmetic/comparison/logic/counting)
baseline_ngram.py                  simple n-gram baseline model, for comparison
evaluate.py                         full evaluation pipeline + comparison table
checkpoint.npz                       trained weights (included, ready to use)
tokenizer.json                        the vocabulary the checkpoint was trained with
results.json                           benchmark results already produced by evaluate.py
```

## Your machine (HP Victus, 8GB RAM, 512GB disk / 60GB free, 4GB NVIDIA GPU)

This project **does not use the GPU at all** — everything runs on NumPy on
CPU. That's a deliberate design choice, not a limitation you need to work
around:

- The 4GB GPU can't be used anyway without CUDA-enabled frameworks
  (PyTorch/TensorFlow), which this project intentionally avoids.
- At this model size (~64k–110k parameters), CPU NumPy is fast enough:
  training the included checkpoint took a few minutes total on a
  comparable CPU.
- Memory footprint is tiny — well under 1GB RAM for training or inference
  at the shipped model size. Disk footprint of the whole repo, including
  the dataset and checkpoint, is a few MB — 60GB free is far more than
  you'll need.
- If you scale the model up (see "Scaling up" below), watch RAM: an
  n_layers × d_model × d_ff transformer's activations scale with
  `batch_size × block_size × d_model`, and Adam keeps two extra copies of
  every parameter (m and v). At 8GB RAM you have comfortable headroom up
  to a few million parameters trained with modest batch sizes.

## Setup

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. Install the only two dependencies
pip install -r requirements.txt
```

That's it — no CUDA toolkit, no PyTorch, no GPU drivers needed.

## Quickstart (using the included pre-trained checkpoint)

A trained `checkpoint.npz` and `tokenizer.json` are already included, so you
can chat with the model immediately without training anything:

```bash
# Terminal chat
python3 chat_cli.py

# Web chat (open http://localhost:5000)
python3 app.py
```

## Training from scratch yourself

```bash
# 1. Regenerate the dataset (or edit data/generate_dataset.py for your own tasks)
python3 data/generate_dataset.py

# 2. Verify the autodiff engine is correct on your machine (~5 seconds)
python3 grad_check.py
# should print "Max relative error: ... (PASS)"

# 3. Train. This config matches the shipped checkpoint and takes roughly
#    3-6 minutes total on a modern laptop CPU (it prints progress as it goes).
python3 train.py --steps 5000 --eval_every 250 \
    --block_size 48 --batch_size 16 --d_model 48 --n_layers 2 --n_heads 4 --d_ff 192 \
    --out checkpoint.npz

# Training is resumable - to keep improving an existing checkpoint:
python3 train.py --steps 2000 --resume checkpoint.npz --out checkpoint.npz \
    --block_size 48 --batch_size 16 --d_model 48 --n_layers 2 --n_heads 4 --d_ff 192
```

### Scaling up (if you want a stronger model and have patience)

The default `train.py` arguments (no flags) use a larger config
(`d_model=64, block_size=64`, ~110k params) that trains a bit slower but
should still finish in well under an hour on your CPU:

```bash
python3 train.py --steps 5000
```

Bigger still (e.g. `d_model=128, n_layers=4`) will take proportionally
longer and use more RAM, but 8GB RAM comfortably supports this — just budget
more wall-clock time since there's no GPU acceleration in this project.

## Evaluating / benchmarking

```bash
python3 evaluate.py
```

This runs both TinyGPT and the n-gram baseline against
`benchmark/reasoning_eval.py`'s held-out prompt set, computes validation
perplexity, per-category exact-match accuracy, and latency, prints a
comparison table, and writes full results (including every prompt/answer
pair) to `results.json`.

To compare against a *different* existing model (e.g. a small pretrained
Hugging Face model, if you have internet access and are willing to
`pip install transformers torch`), implement a small adapter class in
`evaluate.py` with `.perplexity(text)` and `.answer(prompt)` methods,
following the `NgramAdapter` example, and add it to the `models` dict in
`evaluate.py`'s `main()`.

## Deploying the website beyond localhost

`app.py` is a plain Flask app — for anything beyond local testing, put it
behind a production WSGI server (e.g. `gunicorn app:app`) and a reverse
proxy, and consider adding rate limiting since generation is CPU-bound.

## License / originality note

All code here is written from scratch for this project. The dataset is
synthetically generated by `data/generate_dataset.py` (no scraped or
copyrighted text). You're free to use, modify, and publish results based on
this code.
