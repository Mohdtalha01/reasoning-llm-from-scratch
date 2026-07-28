"""
End-to-end evaluation pipeline. Computes, for TinyGPT and for the n-gram
baseline (and optionally any other model you plug in - see `ModelAdapter`
below):

  1. Validation-set perplexity (how well the model predicts held-out text)
  2. Benchmark exact-match accuracy, overall and broken down by category
     (arithmetic / comparison / chain reasoning / counting)
  3. Wall-clock generation latency

...and prints a comparison table, plus writes results to results.json so you
can cite exact numbers in a report.

To compare against a *different* pretrained model (e.g. a small HF model),
implement a ModelAdapter with .perplexity(text) and .answer(prompt) methods
and add it to the `models` dict in main().
"""
import time
import json
import re
import numpy as np

from generate import load_model, generate
from baseline_ngram import NgramModel
from benchmark.reasoning_eval import build_benchmark
from engine.autograd import cross_entropy_loss


# ---------------------------------------------------------------------------
# Adapters: a uniform interface so evaluate() doesn't care which model it's
# scoring. Add a new adapter class to compare against any other model.
# ---------------------------------------------------------------------------
class TinyGPTAdapter:
    name = "TinyGPT (ours, from-scratch transformer)"

    def __init__(self, checkpoint="checkpoint.npz", tokenizer="tokenizer.json"):
        self.model, self.tok = load_model(checkpoint, tokenizer)

    def perplexity(self, text, chunk=48):
        """Average cross-entropy -> perplexity over held-out text, using the
        model's own training objective for a fair, literal comparison."""
        ids = np.array(self.tok.encode(text), dtype=np.int64)
        block = self.model.max_seq
        losses = []
        for i in range(0, len(ids) - block - 1, block):
            x = ids[i:i + block][None, :]
            y = ids[i + 1:i + block + 1][None, :]
            if x.shape[1] != block or y.shape[1] != block:
                continue
            logits = self.model(x)
            _, loss = cross_entropy_loss(logits, y)
            losses.append(loss)
        return float(np.exp(np.mean(losses))) if losses else float("nan")

    def answer(self, prompt):
        t0 = time.time()
        out = generate(self.model, self.tok, prompt, max_new_tokens=40,
                        temperature=0.2, top_k=3, seed=0)
        dt = time.time() - t0
        reply = out[len(prompt):].strip().split("\n")[0].strip()
        return reply, dt


class NgramAdapter:
    name = "N-gram baseline (order-5 Markov chain)"

    def __init__(self, train_path="data/train.txt", n=5):
        self.model = NgramModel(n=n)
        with open(train_path) as f:
            self.model.train(f.read())

    def perplexity(self, text, chunk=48):
        return self.model.perplexity(text[:8000])  # capped for speed

    def answer(self, prompt):
        t0 = time.time()
        out = self.model.generate(prompt, max_new_tokens=40, temperature=0.3)
        dt = time.time() - t0
        reply = out[len(prompt):].strip().split("\n")[0].strip()
        return reply, dt


# ---------------------------------------------------------------------------
def normalize(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def evaluate(adapter, benchmark_items):
    correct_by_cat = {}
    total_by_cat = {}
    latencies = []
    rows = []
    for prompt, expected, category in benchmark_items:
        reply, dt = adapter.answer(prompt)
        latencies.append(dt)
        is_correct = normalize(reply) == normalize(expected)
        total_by_cat[category] = total_by_cat.get(category, 0) + 1
        correct_by_cat[category] = correct_by_cat.get(category, 0) + int(is_correct)
        rows.append({
            "prompt": prompt, "expected": expected, "got": reply,
            "category": category, "correct": is_correct,
        })

    per_cat_acc = {c: correct_by_cat[c] / total_by_cat[c] for c in total_by_cat}
    overall_acc = sum(correct_by_cat.values()) / sum(total_by_cat.values())
    return {
        "overall_accuracy": overall_acc,
        "per_category_accuracy": per_cat_acc,
        "avg_latency_sec": float(np.mean(latencies)),
        "rows": rows,
    }


def main():
    with open("data/val.txt") as f:
        val_text = f.read()
    benchmark_items = build_benchmark(n_per_category=25)

    models = {
        "tinygpt": TinyGPTAdapter(),
        "ngram": NgramAdapter(),
    }

    results = {}
    for key, adapter in models.items():
        print(f"\n=== Evaluating: {adapter.name} ===")
        ppl = adapter.perplexity(val_text)
        bench = evaluate(adapter, benchmark_items)
        results[key] = {
            "name": adapter.name,
            "val_perplexity": ppl,
            **{k: v for k, v in bench.items() if k != "rows"},
        }
        results[key]["_rows"] = bench["rows"]
        print(f"val perplexity        : {ppl:.3f}")
        print(f"benchmark accuracy     : {bench['overall_accuracy']*100:.1f}%")
        for cat, acc in bench["per_category_accuracy"].items():
            print(f"  - {cat:<18s}: {acc*100:5.1f}%")
        print(f"avg latency/answer     : {bench['avg_latency_sec']*1000:.1f} ms")

    # ---- comparison table ----
    print("\n" + "=" * 72)
    print(f"{'metric':<26s}" + "".join(f"{k:>20s}" for k in models))
    print("-" * 72)
    print(f"{'val perplexity':<26s}" +
          "".join(f"{results[k]['val_perplexity']:>20.3f}" for k in models))
    print(f"{'benchmark accuracy':<26s}" +
          "".join(f"{results[k]['overall_accuracy']*100:>19.1f}%" for k in models))
    print(f"{'avg latency (ms)':<26s}" +
          "".join(f"{results[k]['avg_latency_sec']*1000:>20.1f}" for k in models))
    print("=" * 72)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote results.json")


if __name__ == "__main__":
    main()
