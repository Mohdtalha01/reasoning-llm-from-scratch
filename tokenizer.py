"""Character-level tokenizer. Keeps vocab tiny (<100 symbols) which keeps
the whole model tiny and trainable on CPU."""
import json


class CharTokenizer:
    def __init__(self, text=None, vocab=None):
        if vocab is not None:
            self.stoi = vocab
        else:
            chars = sorted(set(text))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.stoi, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            stoi = json.load(f)
        return cls(vocab=stoi)
