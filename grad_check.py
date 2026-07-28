"""
Numerical gradient check: perturbs each parameter by +-eps, measures the
change in loss, and compares to the analytic gradient from backward().
Run this first after setup - if it passes, the autodiff engine and model
are wired correctly.
"""
import numpy as np
from model.gpt import TinyGPT
from engine.autograd import cross_entropy_loss

np.random.seed(0)
vocab_size, T, B = 12, 5, 2
model = TinyGPT(vocab_size=vocab_size, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq=T)

x = np.random.randint(0, vocab_size, size=(B, T))
y = np.random.randint(0, vocab_size, size=(B, T))


def loss_fn():
    logits = model(x)
    _, val = cross_entropy_loss(logits, y)
    return val


params = model.params()
logits = model(x)
loss_t, _ = cross_entropy_loss(logits, y)
for p in params:
    p.zero_grad()
loss_t.backward()

eps = 1e-5
max_rel_err = 0.0
for p in params:
    flat = p.data.reshape(-1)
    grad_flat = p.grad.reshape(-1)
    idxs = np.random.choice(len(flat), size=min(3, len(flat)), replace=False)
    for i in idxs:
        orig = flat[i]
        flat[i] = orig + eps
        l1 = loss_fn()
        flat[i] = orig - eps
        l2 = loss_fn()
        flat[i] = orig
        numeric = (l1 - l2) / (2 * eps)
        analytic = grad_flat[i]
        rel_err = abs(numeric - analytic) / (abs(numeric) + abs(analytic) + 1e-8)
        max_rel_err = max(max_rel_err, rel_err)
        print(f"numeric={numeric: .6f} analytic={analytic: .6f} rel_err={rel_err:.2e}")

print(f"\nMax relative error: {max_rel_err:.2e}  "
      f"({'PASS' if max_rel_err < 1e-2 else 'FAIL - check code'})")
