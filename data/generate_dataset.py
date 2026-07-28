"""
Generates a synthetic 'reasoning' dataset: arithmetic and simple deductive
chains, as plain text. This keeps everything from-scratch and licence-free,
and lets us control task difficulty precisely for evaluation.
"""
import random
import os

random.seed(0)

NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"]


def gen_arith(n):
    lines = []
    for _ in range(n):
        op = random.choice(["+", "-", "*"])
        if op == "*":
            a, b = random.randint(0, 12), random.randint(0, 12)
            ans = a * b
        else:
            a, b = random.randint(0, 99), random.randint(0, 99)
            ans = a + b if op == "+" else a - b
        lines.append(f"Q: {a} {op} {b} = ? A: {ans}\n")
    return lines


def gen_compare(n):
    lines = []
    for _ in range(n):
        a, b = random.sample(range(1, 100), 2)
        lines.append(f"Q: Which is bigger, {a} or {b}? A: {max(a, b)}\n")
    return lines


def gen_chain(n):
    """Simple transitive-order reasoning: A>B, B>C -> who is oldest?"""
    lines = []
    for _ in range(n):
        p1, p2, p3 = random.sample(NAMES, 3)
        lines.append(
            f"Q: {p1} is older than {p2}. {p2} is older than {p3}. "
            f"Who is oldest? A: {p1}\n"
        )
    return lines


def gen_count(n):
    lines = []
    for _ in range(n):
        k = random.randint(1, 20)
        lines.append(f"Q: Count from 1 to {k}. A: " +
                      " ".join(str(i) for i in range(1, k + 1)) + "\n")
    return lines


def build_dataset(path_train, path_val, n_train=20000, n_val=2000):
    os.makedirs(os.path.dirname(path_train), exist_ok=True)
    gens = [gen_arith, gen_compare, gen_chain, gen_count]
    train_lines, val_lines = [], []
    for g in gens:
        train_lines += g(n_train // len(gens))
        val_lines += g(n_val // len(gens))
    random.shuffle(train_lines)
    random.shuffle(val_lines)
    with open(path_train, "w") as f:
        f.writelines(train_lines)
    with open(path_val, "w") as f:
        f.writelines(val_lines)
    print(f"Wrote {len(train_lines)} train lines -> {path_train}")
    print(f"Wrote {len(val_lines)} val lines -> {path_val}")


if __name__ == "__main__":
    build_dataset("data/train.txt", "data/val.txt")
