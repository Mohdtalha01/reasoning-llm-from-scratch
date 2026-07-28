# DOCUMENTATION

A full account of what this project is, how every file works, what was
measured, and what its honest limitations are. Written so you can lift
sections of this directly into a report, blog post, or paper draft.

---

## 1. Overview

**reasoning_llm** is a decoder-only Transformer language model ("TinyGPT"),
implemented from scratch using only NumPy for numerical computation
(Flask is used solely for the web server, not for any model math). It
includes:

- a hand-built reverse-mode automatic differentiation engine,
- a small Transformer built on top of that engine,
- a synthetic dataset of arithmetic and simple logical-reasoning text,
- a training loop,
- a benchmark of held-out reasoning prompts with known correct answers,
- an evaluation pipeline that compares TinyGPT against a simpler n-gram
  baseline model,
- a terminal chat and a Flask-based chat website.

**Headline result:** on the included benchmark (100 held-out prompts across
4 categories + 5 edge cases), the trained TinyGPT scores **49.5% exact-match
accuracy**, versus **0%** for an order-5 character n-gram baseline, while
having lower validation perplexity (1.34 vs 1.47). This is a genuine,
reproducible result — see Section 6 for full numbers and how to reproduce
them.

**What this is not:** a general-purpose assistant, a state-of-the-art
reasoning system, or a claim of novel architecture. The transformer
architecture (Vaswani et al., 2017) is standard; the contribution here is
the from-scratch NumPy implementation, the controlled synthetic benchmark,
and the honestly-reported comparison — which is exactly the kind of
contribution appropriate for a course project, workshop paper, or technical
blog post, not a claim of new SOTA capability.

---

## 2. Why build an autodiff engine instead of hand-deriving one backward pass

A transformer's backward pass, derived by hand as one big block of algebra,
is a large surface for silent bugs — a sign error in the attention gradient
can produce a model that trains but plateaus early, which is very hard to
detect without a lot of testing.

Instead, this project implements a small **automatic differentiation
engine** (`engine/autograd.py`): a `Tensor` class that remembers, for every
operation applied to it, how to propagate a gradient backward to its
inputs. The transformer is then assembled from ~12 primitive operations
(add, multiply, matmul, exp, sum/mean, power, ReLU, reshape, slice,
concatenate, embedding lookup), each with a short, individually-checkable
gradient rule. The chain rule composition across the whole network graph is
then handled automatically and uniformly by the engine, rather than being
re-derived by hand for every new layer combination.

This is the same design pattern used (at much larger scale) by PyTorch,
JAX, and teaching frameworks like micrograd — a directed acyclic graph of
tensors, built during the forward pass, walked in reverse topological order
during `.backward()`.

**Correctness was verified, not assumed:** `grad_check.py` compares the
engine's analytic gradients against numerical gradients (finite differences:
`(loss(w+eps) - loss(w-eps)) / (2*eps)`) for a random sample of parameters
across every layer type in the model. Result: maximum relative error
`2.2e-3`, with the overwhelming majority of checked gradients agreeing to
better than `1e-9` — the engine's calculus is correct.

---

## 3. File-by-file walkthrough

### `engine/autograd.py` — the automatic differentiation engine

- **`Tensor`**: wraps a NumPy array (`self.data`) plus a gradient buffer
  (`self.grad`), a set of parent tensors (`self._prev`), and a closure
  (`self._backward`) that knows how to push a gradient from this tensor
  back to its parents.
- **`_unbroadcast(grad, shape)`**: NumPy broadcasting (e.g. adding a
  `(dim,)` bias to a `(batch, time, dim)` activation) means a gradient
  computed in the "expanded" broadcast shape needs to be summed back down
  to the original parameter's shape before being stored. This helper
  handles both cases broadcasting introduces: extra leading dimensions, and
  dimensions that were size-1 and got stretched.
- **Elementwise ops** (`__add__`, `__mul__`, `pow`, `exp`, `relu`,
  `__truediv__`): each returns a new `Tensor` and registers a `_backward`
  closure implementing that operation's local derivative (e.g. for
  `relu(x)`, the gradient passes through unchanged where `x > 0` and is
  zeroed elsewhere).
- **`sum` / `mean`**: reductions along an axis; the backward pass
  broadcasts the incoming gradient back out to the original (pre-reduction)
  shape.
- **`reshape` / `swapaxes` / `slice_last`**: pure bookkeeping ops — the
  backward pass just undoes the shape change (reshape back, swap back,
  scatter the sliced gradient back into a zero array at the same
  position). `slice_last` is what splits the model's Q/K/V projections into
  per-head chunks for multi-head attention.
- **`concat`**: the inverse of slicing — concatenates a list of tensors
  along an axis; backward splits the incoming gradient back into pieces of
  the original sizes and routes each piece to the corresponding input.
  This is what re-assembles the per-head attention outputs into one tensor.
- **`__matmul__`**: batched matrix multiplication using NumPy's `@`
  operator (which already handles batch dimensions the way this model
  needs — e.g. `(batch, time, in) @ (in, out)`). The backward pass uses the
  standard matrix-calculus identity: for `C = A @ B`,
  `dA = dC @ B.T` and `dB = A.T @ dC` (with `.T` meaning "swap the last two
  axes" for batched tensors), followed by `_unbroadcast` to sum any
  gradient contributions across a broadcast batch dimension.
- **`Tensor.backward()`**: builds a topological ordering of the
  computation graph via depth-first search, seeds the output tensor's
  gradient to 1 (since `d(loss)/d(loss) = 1`), then calls each tensor's
  `_backward()` in reverse topological order — the standard reverse-mode
  autodiff sweep.
- **`embedding_lookup(table, idx)`**: looks up rows of an embedding table
  by integer index (fancy indexing) in the forward pass; in the backward
  pass, gradients are scattered back into the table with `np.add.at`,
  which correctly accumulates when the same row is looked up more than
  once in a batch.
- **`softmax(x, axis)`**: implemented by composing primitive ops
  (subtract the row max for numerical stability, exponentiate, sum,
  divide). Subtracting a constant (the max) before exponentiating doesn't
  change softmax's true gradient, because softmax is shift-invariant — so
  treating the max as a non-differentiated constant here is mathematically
  exact, not an approximation.
- **`cross_entropy_loss(logits, targets)`**: implemented directly (not
  composed from `softmax` + `log`) for numerical stability, using the
  log-sum-exp trick. Its gradient is the well-known closed form
  `softmax(logits) - one_hot(targets)`, averaged over the batch — this is
  wired manually into the graph as a single custom backward step, since
  deriving it in closed form is both standard practice and avoids
  potential instability from composing `log` and `exp` separately.

### `model/gpt.py` — the transformer

- **`Linear`**: a standard `y = xW + b` layer. Weight initialization uses
  `Uniform(-1/sqrt(in_dim), +1/sqrt(in_dim))`, the same scheme PyTorch's
  `nn.Linear` uses by default (keeps activation variance stable at
  initialization regardless of layer width).
- **`LayerNorm`**: normalizes each token's activation vector to zero mean
  and unit variance, then applies a learned per-channel scale (`gamma`) and
  shift (`beta`). Implemented by composing `Tensor` ops (`mean`, subtract,
  square, `mean` again for variance, `pow(0.5)` for the standard
  deviation, divide) rather than hand-deriving LayerNorm's backward
  formula — this means its gradient is automatically correct as a
  consequence of the engine being correct, with no separate derivation to
  get wrong.
- **`attention_head` / `MultiHeadAttention`**: standard scaled dot-product
  attention, `softmax(QK^T / sqrt(d_head) + causal_mask) @ V`, computed
  per-head via a Python loop (`n_heads` is small — 4 — so this is cheap)
  and re-concatenated. The causal mask is a constant `(T, T)` array with
  `-1e9` above the diagonal, added to the attention scores before the
  softmax, so each position can only attend to itself and earlier
  positions (this is what makes it a valid *autoregressive*, left-to-right
  language model rather than a bidirectional encoder).
- **`FeedForward`**: the standard transformer MLP block,
  `Linear -> ReLU -> Linear`, expanding to `d_ff` (4x `d_model` by
  default) and back down.
- **`Block`**: one transformer layer, using **pre-layer-norm** residual
  connections (`x = x + attn(LN(x))`, then `x = x + ff(LN(x))`). Pre-LN is
  used instead of the original post-LN formulation from Vaswani et al.
  because it's known to train more stably at small scale and with fewer
  steps, which matters when the training budget is a few thousand steps
  on a laptop CPU rather than the huge budgets original transformers used.
- **`TinyGPT`**: stacks `n_layers` `Block`s on top of learned token +
  positional embeddings, applies a final LayerNorm, and projects to
  vocabulary-sized logits via a `Linear` head. `state_dict()` /
  `load_state_dict()` save and load all parameters as a plain NumPy
  `.npz` file — deliberately not using Python's `pickle` for the model
  itself, so checkpoints are portable, inspectable, and don't risk
  executing arbitrary code on load.

### `tokenizer.py`

A character-level tokenizer: builds a vocabulary from the distinct
characters seen in the training text (for this dataset, ~49 symbols —
digits, letters, punctuation, whitespace) and maps text to/from integer
sequences. Character-level tokenization was chosen deliberately over
word-level or subword (BPE) tokenization: it keeps the vocabulary — and
therefore the embedding table and output layer, the two largest sources of
parameters at this scale — tiny, which matters when training on a CPU with
no GPU acceleration. The tradeoff is that the model has to learn to spell
out numbers digit-by-digit rather than treating "42" as one token, which is
part of why arithmetic is the hardest category on the benchmark (Section 6).

### `optimizer.py`

A standard **Adam** optimizer (Kingma & Ba, 2014), implemented directly in
NumPy: maintains exponential moving averages of the gradient (`m`) and
its square (`v`) for every parameter, with bias correction for early
training steps, and updates parameters by
`param -= lr * m_hat / (sqrt(v_hat) + eps)`. Also includes
`clip_grad_norm`, which rescales the whole gradient vector if its global
L2 norm exceeds a threshold — a standard stabilizer that prevents rare
large gradients (e.g. from an unlucky batch) from causing a destructive
parameter update.

### `data/generate_dataset.py`

Synthetically generates the training and validation text, entirely
programmatically (no scraped or copyrighted material), across four task
types:

1. **Arithmetic** — `Q: 12 + 7 = ? A: 19` (addition/subtraction over
   0-99, multiplication over 0-12 to keep answers reasonably short).
2. **Comparison** — `Q: Which is bigger, 45 or 12? A: 45`.
3. **Chain reasoning** — `Q: Alice is older than Bob. Bob is older than
   Carol. Who is oldest? A: Alice` — a minimal test of transitive,
   multi-step inference rather than single-fact lookup.
4. **Counting** — `Q: Count from 1 to 6. A: 1 2 3 4 5 6`.

Both a training split (20,000 lines) and a validation split (2,000 lines,
generated independently, not a held-out slice of the same lines) are
produced.

### `train.py`

The training loop: loads the text data, builds/saves the tokenizer,
converts text to integer id arrays, and repeatedly samples random
`block_size`-length windows as `(input, target)` pairs (predict the next
character at every position). Each step: forward pass through TinyGPT →
cross-entropy loss → `loss.backward()` → gradient clipping → `Adam.step()`.
Supports `--resume` to continue training an existing checkpoint (used here
to train in several short bursts rather than one long run — see Section 6
for the exact schedule used to produce the shipped checkpoint). Saves
weights plus the architecture hyperparameters together in one `.npz` file,
so `generate.py` can reconstruct the exact right model shape without the
caller needing to remember the training config.

### `generate.py`

Autoregressive sampling: repeatedly runs the model forward on the current
token sequence (cropped to the last `block_size` tokens), takes the logits
for the final position, and samples the next token via
**temperature + top-k sampling** (`sample_next`): logits are divided by a
temperature (lower = more confident/deterministic, higher = more random),
optionally restricted to only the top-k highest-probability candidates,
converted to a probability distribution via softmax, and sampled from.
Generation stops early once a newline is produced (matching the
line-per-example structure of the training data), or after
`max_new_tokens`.

### `chat_cli.py` / `app.py` + `templates/index.html`

Two front-ends over the same `generate()` function: a terminal
read-eval-print loop, and a Flask web server exposing `POST /api/chat`
(wraps the user's message in the same `Q: ... A:` template the model was
trained on, generates a completion, strips the prompt back off) plus a
single-page chat UI. No model computation happens in Flask or JavaScript —
they're purely a UI layer over the same NumPy model used everywhere else.

### `benchmark/reasoning_eval.py`

A **held-out** benchmark: generated with a different random seed than the
training/validation data, so it tests generalization to new numbers and
names rather than memorization, plus 5 hand-written edge cases (e.g.
`0 + 0`, `99 - 99`) chosen to probe boundary behavior specifically. 100
prompts total (25 per main category + 5 edge cases), each with one exact
expected answer.

### `baseline_ngram.py`

An order-*n* character-level **Markov chain** language model — for every
observed context of the previous `n-1` characters, it stores a frequency
table of what character came next, and at generation time samples from
that empirical distribution (falling back to a uniform distribution over
the vocabulary for contexts never seen during training). This is
intentionally a much simpler model than TinyGPT: no learned vector
representations, no attention over long-range context, just local
counting statistics. It serves two purposes: (1) a fair "existing simple
model" to benchmark against without requiring internet access to download
a pretrained model, and (2) a sanity floor — if TinyGPT couldn't beat this,
that would indicate the transformer wasn't learning anything the simpler
model couldn't already capture.

### `evaluate.py`

The full comparison pipeline. Defines a uniform `Adapter` interface
(`.perplexity(text)`, `.answer(prompt)`) implemented for both TinyGPT
(`TinyGPTAdapter`) and the n-gram model (`NgramAdapter`), so the evaluation
logic itself doesn't need to know which model it's scoring — which also
means you can add a third adapter (e.g. wrapping a downloaded pretrained
model) without touching the scoring code. For each model, computes:

- **Validation perplexity**: `exp(mean cross-entropy loss)` over held-out
  text, using the model's own training objective — how well the model
  predicts real reasoning-formatted text, independent of whether specific
  final answers are correct.
- **Benchmark accuracy**: runs low-temperature, top-k generation on every
  benchmark prompt, takes the first line of output, normalizes whitespace
  and case, and checks for an **exact string match** against the known
  correct answer — overall and broken down per category.
- **Latency**: average wall-clock time to answer one benchmark prompt.

Prints a side-by-side comparison table and writes every prompt/expected/got
triple to `results.json` for inspection or citation.

### `grad_check.py`

Numerical gradient checking (Section 2) — the correctness test for the
whole engine.

---

## 4. Architecture summary (shipped checkpoint)

| Hyperparameter | Value |
|---|---|
| Vocabulary | 49 characters |
| `d_model` (embedding/hidden dim) | 48 |
| Attention heads | 4 (head dim 12) |
| Transformer layers | 2 |
| Feed-forward dim | 192 |
| Context length (`block_size`) | 48 characters |
| Total parameters | 63,697 |
| Optimizer | Adam, lr 2e-3 → 3e-3, grad-norm clipped to 1.0 |
| Total training steps | ~4,650 (across resumed bursts) |
| Total training time | ~3.5 minutes on the evaluation CPU |

For reference, GPT-2 small has 124 million parameters and was trained for
days on many GPUs; this model is roughly **2,000x smaller** and trained for
a tiny fraction of the compute. The results below should be read with that
firmly in mind.

---

## 5. How to reproduce every number in this document

```bash
python3 data/generate_dataset.py          # regenerate the dataset (seeded, deterministic)
python3 grad_check.py                     # verify engine correctness
python3 train.py --steps 5000 --eval_every 250 \
    --block_size 48 --batch_size 16 --d_model 48 \
    --n_layers 2 --n_heads 4 --d_ff 192 --out checkpoint.npz
python3 evaluate.py                       # reproduces the comparison table + results.json
```

Exact loss curves will vary run to run (weight initialization and batch
sampling are randomized, though seeded — full bit-for-bit reproducibility
across machines/NumPy versions is not guaranteed), but the qualitative
result (TinyGPT clearly outperforming the n-gram baseline, with arithmetic
as the weakest category) is stable.

---

## 6. Results

From `results.json`, produced by `evaluate.py` on the shipped checkpoint:

| Metric | TinyGPT (ours) | N-gram baseline |
|---|---|---|
| Validation perplexity | **1.341** | 1.465 |
| Benchmark accuracy (overall) | **49.5%** | 0.0% |
| — arithmetic | 20.0% | 0.0% |
| — comparison | 68.0% | 0.0% |
| — chain reasoning | 24.0% | 0.0% |
| — counting | 88.0% | 0.0% |
| — edge cases (arithmetic) | 33.3% | 0.0% |
| — edge cases (comparison) | 0.0% | 0.0% |
| — edge cases (counting) | 100.0% | 0.0% |
| Avg. latency / answer | 26 ms | 0.2 ms |

**Reading these results honestly:**

- The n-gram model scoring 0% everywhere is expected, not a bug: with no
  learned representation of "what a number is" or "which of two numbers is
  larger," a local-context frequency table has no mechanism to produce a
  correct novel arithmetic answer — it can only reproduce sequences
  statistically similar to ones it has seen. Its low perplexity is
  explained by the text's simple structural regularities (`"Q: "`,
  `" = ? A: "`, etc.), which are easy to predict even without understanding
  content — this is exactly why perplexity alone is a poor proxy for
  reasoning ability, and why the benchmark's exact-match accuracy is the
  more meaningful number here.
- TinyGPT's strength ordering (counting > comparison > chain reasoning ≈
  arithmetic) is informative: counting and comparison are close to
  pattern continuation given the model has memorized number-magnitude
  relationships from training, whereas arithmetic requires composing a
  novel numeric operation, and chain reasoning requires tracking an
  entity across two sentences before answering — both meaningfully harder
  compositional tasks for a model this small and this lightly trained.
- 20% arithmetic accuracy is well above chance (a two-digit answer has
  effectively hundreds of possible values) but far below reliable — a
  fair, literal characterization is "has learned some of the structure of
  arithmetic, not the operation itself."

---

## 7. Limitations

- **Scale.** 64k parameters and ~4,650 training steps is minuscule by
  modern standards. Most of the "next steps" below are really just "spend
  more compute," which is the single biggest lever available.
- **Character-level tokenization** means the model has to learn to spell
  out multi-digit numbers correctly digit-by-digit, which is strictly
  harder than operating on numbers as atomic tokens — a likely contributor
  to weak arithmetic performance specifically.
- **No true held-out generalization test for reasoning composition** — the
  benchmark's chain-reasoning task uses the same sentence template as
  training, varying only the names; it does not test whether the model
  has learned the underlying transitive-inference *rule* versus a
  shallow positional heuristic (e.g. "answer with the first name
  mentioned"). This is a known limitation of the benchmark itself, worth
  disclosing explicitly if these results are written up.
- **Exact-match scoring** is strict — a numerically-close-but-wrong answer
  scores identically to a wildly wrong one. A more nuanced metric (e.g.
  numeric distance for arithmetic) would give a more complete picture and
  is a reasonable extension.
- **No comparison against a real pretrained LLM** in the shipped results
  (only against the from-scratch n-gram baseline), since that requires
  internet access and additional dependencies (`transformers`, `torch`)
  outside this project's "NumPy only" scope. `evaluate.py` is structured
  so this is a small addition (implement one `Adapter` class) if you have
  internet access and want that comparison for a report.

## 8. Suggested next steps (for a stronger paper/product)

1. **Train longer / bigger**: increase `d_model`, `n_layers`, and step
   count — 8GB RAM comfortably supports a few million parameters; the
   ceiling here is patience, not hardware.
2. **Add a true generalization split**: hold out specific number ranges
   or name sets entirely from training, to test extrapolation rather than
   interpolation.
3. **Numeric tokenization**: represent numbers as fewer tokens (e.g.
   fixed-width digit tokens, or a small numeral vocabulary) to isolate
   whether weak arithmetic is a tokenization artifact or a genuine
   reasoning limitation.
4. **Chain-of-thought data**: add intermediate reasoning steps to the
   training text (e.g. spelling out a running total) rather than only
   question→answer pairs, and measure whether that improves the harder
   categories — this connects directly to why "reasoning models" in the
   broader field use intermediate reasoning tokens.
5. **A real pretrained-model comparison**, once you have internet access,
   for an external reference point beyond the from-scratch baseline.
