"""
A simple character-level n-gram (Markov chain) language model, used as a
baseline to compare TinyGPT against. This is intentionally a *much* simpler
model (no attention, no learned representations, just conditional counting
frequencies) — it establishes a lower bound: if TinyGPT can't beat this, the
transformer isn't earning its complexity.

Also useful as a second thing to plug into evaluate.py if you don't have
internet access to download a real pretrained model to compare against.
"""
import random
from collections import defaultdict, Counter


class NgramModel:
    def __init__(self, n=4):
        self.n = n
        self.counts = defaultdict(Counter)
        self.vocab = set()

    def train(self, text):
        self.vocab = set(text)
        pad = "\x00" * (self.n - 1)
        text = pad + text
        for i in range(len(text) - self.n + 1):
            context = text[i:i + self.n - 1]
            nxt = text[i + self.n - 1]
            self.counts[context][nxt] += 1

    def next_char_probs(self, context):
        context = context[-(self.n - 1):]
        context = context.rjust(self.n - 1, "\x00")
        counter = self.counts.get(context)
        if not counter:
            # back off to uniform over observed vocab
            return {c: 1.0 / len(self.vocab) for c in self.vocab}
        total = sum(counter.values())
        return {c: v / total for c, v in counter.items()}

    def generate(self, prompt, max_new_tokens=60, temperature=1.0, rng=None, stop_at="\n"):
        rng = rng or random.Random(0)
        text = prompt
        for _ in range(max_new_tokens):
            probs = self.next_char_probs(text)
            chars, weights = zip(*probs.items())
            if temperature != 1.0:
                weights = [w ** (1.0 / max(temperature, 1e-6)) for w in weights]
                s = sum(weights)
                weights = [w / s for w in weights]
            nxt = rng.choices(chars, weights=weights, k=1)[0]
            text += nxt
            if len(text) > len(prompt) + 1 and nxt == stop_at:
                break
        return text

    def perplexity(self, text):
        pad = "\x00" * (self.n - 1)
        padded = pad + text
        log_prob_sum = 0.0
        count = 0
        for i in range(len(padded) - self.n + 1):
            context = padded[i:i + self.n - 1]
            nxt = padded[i + self.n - 1]
            probs = self.next_char_probs(context)
            p = probs.get(nxt, 1e-6)
            log_prob_sum += -1.0 * __import__("math").log(max(p, 1e-9))
            count += 1
        import math
        return math.exp(log_prob_sum / max(count, 1))


if __name__ == "__main__":
    with open("data/train.txt") as f:
        train_text = f.read()
    with open("data/val.txt") as f:
        val_text = f.read()

    model = NgramModel(n=5)
    model.train(train_text)
    print("val perplexity:", model.perplexity(val_text[:5000]))
    print(model.generate("Q: 7 + 5 = ? A:", max_new_tokens=30))
