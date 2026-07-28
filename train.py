"""
Trains TinyGPT on the synthetic reasoning dataset, from scratch, on CPU,
using only NumPy. Designed to fit comfortably in 8GB RAM.
"""
import time
import argparse
import numpy as np
from model.gpt import TinyGPT
from tokenizer import CharTokenizer
from optimizer import Adam, clip_grad_norm
from engine.autograd import cross_entropy_loss


def load_data(path):
    with open(path) as f:
        return f.read()


def get_batch(data_ids, block_size, batch_size, rng):
    n = len(data_ids)
    ix = rng.integers(0, n - block_size - 1, size=batch_size)
    x = np.stack([data_ids[i:i + block_size] for i in ix])
    y = np.stack([data_ids[i + 1:i + block_size + 1] for i in ix])
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_path", type=str, default="data/train.txt")
    ap.add_argument("--val_path", type=str, default="data/val.txt")
    ap.add_argument("--out", type=str, default="checkpoint.npz")
    ap.add_argument("--resume", type=str, default=None,
                     help="path to an existing checkpoint.npz to continue training from")
    args = ap.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_text = load_data(args.train_path)
    val_text = load_data(args.val_path)

    tok = CharTokenizer(text=train_text)
    tok.save("tokenizer.json")

    train_ids = np.array(tok.encode(train_text), dtype=np.int64)
    val_ids = np.array(tok.encode(val_text), dtype=np.int64)

    model = TinyGPT(vocab_size=tok.vocab_size, d_model=args.d_model, n_heads=args.n_heads,
                     n_layers=args.n_layers, d_ff=args.d_ff, max_seq=args.block_size)
    if args.resume:
        d = np.load(args.resume)
        model.load_state_dict(d)
        print(f"Resumed weights from {args.resume}")
    opt = Adam(model.params(), lr=args.lr)

    n_params = sum(p.data.size for p in model.params())
    print(f"vocab_size={tok.vocab_size}  params={n_params:,}")

    t0 = time.time()
    history = []
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_ids, args.block_size, args.batch_size, rng)
        logits = model(x)
        loss_t, loss_val = cross_entropy_loss(logits, y)

        opt.zero_grad()
        loss_t.backward()
        clip_grad_norm(model.params(), max_norm=1.0)
        opt.step()

        if step % args.eval_every == 0 or step == 1:
            vx, vy = get_batch(val_ids, args.block_size, args.batch_size, rng)
            vlogits = model(vx)
            _, vloss = cross_entropy_loss(vlogits, vy)
            dt = time.time() - t0
            print(f"step {step:5d} | train_loss {loss_val:.3f} | val_loss {vloss:.3f} "
                  f"| {dt:.1f}s elapsed")
            history.append((step, loss_val, vloss))

    save_dict = model.state_dict()
    save_dict.update(vocab_size=tok.vocab_size, d_model=args.d_model, n_heads=args.n_heads,
                      n_layers=args.n_layers, d_ff=args.d_ff, block_size=args.block_size)
    np.savez(args.out, **save_dict)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
