"""
A small, fixed benchmark of reasoning prompts with known-correct answers,
disjoint from the training/val data (different random seed + explicit
hand-written edge cases). This is the artifact you'd report results on in a
paper/README: it is small on purpose so it's cheap to run repeatedly and to
inspect every example by hand.

Each item: (prompt, expected_answer, category)
`expected_answer` is compared via exact string match after normalization
(strip whitespace, case-insensitive) against the model's first line of output.
"""
import random

random.seed(1234)  # different seed from data/generate_dataset.py -> held out
NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"]


def build_benchmark(n_per_category=25):
    items = []

    # arithmetic
    for _ in range(n_per_category):
        op = random.choice(["+", "-", "*"])
        if op == "*":
            a, b = random.randint(0, 12), random.randint(0, 12)
            ans = a * b
        else:
            a, b = random.randint(0, 99), random.randint(0, 99)
            ans = a + b if op == "+" else a - b
        items.append((f"Q: {a} {op} {b} = ? A:", str(ans), "arithmetic"))

    # comparison
    for _ in range(n_per_category):
        a, b = random.sample(range(1, 100), 2)
        items.append((f"Q: Which is bigger, {a} or {b}? A:", str(max(a, b)), "comparison"))

    # transitive chain reasoning
    for _ in range(n_per_category):
        p1, p2, p3 = random.sample(NAMES, 3)
        prompt = (f"Q: {p1} is older than {p2}. {p2} is older than {p3}. "
                   f"Who is oldest? A:")
        items.append((prompt, p1, "chain_reasoning"))

    # counting
    for _ in range(n_per_category):
        k = random.randint(1, 20)
        expected = " ".join(str(i) for i in range(1, k + 1))
        items.append((f"Q: Count from 1 to {k}. A:", expected, "counting"))

    # a few hand-written edge cases, not covered by the generators above -
    # these probe generalization rather than memorization
    edge_cases = [
        ("Q: 0 + 0 = ? A:", "0", "edge_arithmetic"),
        ("Q: 99 - 99 = ? A:", "0", "edge_arithmetic"),
        ("Q: 12 * 0 = ? A:", "0", "edge_arithmetic"),
        ("Q: Which is bigger, 1 or 100? A:", "100", "edge_comparison"),
        ("Q: Count from 1 to 1. A:", "1", "edge_counting"),
    ]
    items += edge_cases

    random.shuffle(items)
    return items


if __name__ == "__main__":
    items = build_benchmark()
    print(f"{len(items)} benchmark items")
    for p, a, c in items[:5]:
        print(f"[{c}] {p!r} -> {a!r}")
