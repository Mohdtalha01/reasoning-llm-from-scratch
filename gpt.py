"""
TinyGPT: a small decoder-only Transformer, built entirely out of the
engine.Tensor ops in engine/autograd.py. No PyTorch/TensorFlow/JAX anywhere.
"""
import numpy as np
from engine.autograd import Tensor, embedding_lookup, softmax


class Linear:
    def __init__(self, in_dim, out_dim):
        limit = 1.0 / np.sqrt(in_dim)
        self.W = Tensor(np.random.uniform(-limit, limit, (in_dim, out_dim)))
        self.b = Tensor(np.zeros(out_dim))

    def params(self):
        return [self.W, self.b]

    def __call__(self, x):
        return x @ self.W + self.b


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = Tensor(np.ones(dim))
        self.beta = Tensor(np.zeros(dim))
        self.eps = eps

    def params(self):
        return [self.gamma, self.beta]

    def __call__(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        std = (var + self.eps).pow(0.5)
        norm = xc / std
        return norm * self.gamma + self.beta


def attention_head(q, k, v, mask):
    dh = q.shape[-1]
    scores = (q @ k.swapaxes(-1, -2)) * (1.0 / np.sqrt(dh))  # (B,T,T)
    scores = scores + mask                                    # causal mask
    attn = softmax(scores, axis=-1)
    out = attn @ v                                             # (B,T,dh)
    return out


class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)

    def params(self):
        p = []
        for l in (self.Wq, self.Wk, self.Wv, self.Wo):
            p += l.params()
        return p

    def __call__(self, x, mask):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        heads = []
        for h in range(self.n_heads):
            s, e = h * self.dh, (h + 1) * self.dh
            qh, kh, vh = Q.slice_last(s, e), K.slice_last(s, e), V.slice_last(s, e)
            heads.append(attention_head(qh, kh, vh, mask))
        concat = Tensor.concat(heads, axis=-1)
        return self.Wo(concat)


class FeedForward:
    def __init__(self, d_model, d_ff):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)

    def params(self):
        return self.fc1.params() + self.fc2.params()

    def __call__(self, x):
        return self.fc2(self.fc1(x).relu())


class Block:
    """Pre-LayerNorm transformer block (more stable to train at small scale)."""
    def __init__(self, d_model, n_heads, d_ff):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def params(self):
        return self.ln1.params() + self.attn.params() + self.ln2.params() + self.ff.params()

    def __call__(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT:
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, d_ff=256, max_seq=64):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq = max_seq
        self.tok_emb = Tensor(np.random.randn(vocab_size, d_model) * 0.02)
        self.pos_emb = Tensor(np.random.randn(max_seq, d_model) * 0.02)
        self.blocks = [Block(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size)
        self._mask_np = np.triu(np.ones((max_seq, max_seq)), k=1) * -1e9

    def params(self):
        p = [self.tok_emb, self.pos_emb]
        for b in self.blocks:
            p += b.params()
        p += self.ln_f.params() + self.head.params()
        return p

    def __call__(self, idx: np.ndarray):
        """idx: int array (B,T) of token ids. Returns logits Tensor (B,T,vocab)."""
        B, T = idx.shape
        tok = embedding_lookup(self.tok_emb, idx)            # (B,T,D)
        pos = embedding_lookup(self.pos_emb, np.arange(T))    # (T,D)
        x = tok + pos
        mask = Tensor(self._mask_np[:T, :T], requires_grad=False)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        return self.head(x)

    # ---- save / load (plain numpy .npz, no pickling of framework objects) ----
    def state_dict(self):
        return {f"p{i}": p.data for i, p in enumerate(self.params())}

    def load_state_dict(self, d):
        for i, p in enumerate(self.params()):
            p.data = d[f"p{i}"].astype(np.float64)
