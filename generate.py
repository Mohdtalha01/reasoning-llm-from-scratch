"""
Autoregressive sampling from a trained TinyGPT checkpoint.
No frameworks - just repeated forward passes + numpy sampling.
"""
import numpy as np
from model.gpt import TinyGPT
from tokenizer import CharTokenizer


def load_model(checkpoint_path="checkpoint.npz", tokenizer_path="tokenizer.json"):
    d = np.load(checkpoint_path)
    model = TinyGPT(
        vocab_size=int(d["vocab_size"]),
        d_model=int(d["d_model"]),
        n_heads=int(d["n_heads"]),
        n_layers=int(d["n_layers"]),
        d_ff=int(d["d_ff"]),
        max_seq=int(d["block_size"]),
    )
    model.load_state_dict(d)
    tok = CharTokenizer.load(tokenizer_path)
    return model, tok


def sample_next(logits_row, temperature=0.8, top_k=20, rng=None):
    """logits_row: 1D numpy array of logits for the next token."""
    rng = rng or np.random
    logits_row = logits_row / max(temperature, 1e-6)
    if top_k is not None and top_k < len(logits_row):
        idx = np.argpartition(logits_row, -top_k)[-top_k:]
        mask = np.full_like(logits_row, -np.inf)
        mask[idx] = logits_row[idx]
        logits_row = mask
    logits_row = logits_row - np.max(logits_row)
    probs = np.exp(logits_row)
    probs = probs / probs.sum()
    return rng.choice(len(probs), p=probs)


def generate(model, tok, prompt, max_new_tokens=120, temperature=0.8, top_k=20, seed=None):
    rng = np.random.default_rng(seed)
    ids = tok.encode(prompt)
    if len(ids) == 0:
        ids = [0]
    block_size = model.max_seq
    for _ in range(max_new_tokens):
        context = ids[-block_size:]
        x = np.array([context], dtype=np.int64)
        logits = model(x)  # (1, T, V) - Tensor
        last_logits = logits.data[0, -1, :]
        next_id = sample_next(last_logits, temperature, top_k, rng)
        ids.append(int(next_id))
        # simple stop condition: stop after a couple of newlines once we've
        # generated past the prompt, so chat replies don't ramble forever
        if len(ids) > len(tok.encode(prompt)) + 2 and tok.itos.get(int(next_id)) == "\n":
            break
    return tok.decode(ids)


if __name__ == "__main__":
    model, tok = load_model()
    out = generate(model, tok, "Q: 7 + 5 = ? A:", max_new_tokens=40, temperature=0.5, top_k=10)
    print(out)
