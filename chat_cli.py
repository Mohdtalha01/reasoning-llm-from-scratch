"""Simple terminal chat loop against the trained TinyGPT model."""
from generate import load_model, generate

if __name__ == "__main__":
    model, tok = load_model()
    print("TinyGPT chat (type 'quit' to exit)")
    print("Tip: this model was trained on Q/A-style reasoning text, so phrase")
    print("things like: Q: 12 + 8 = ? A:")
    while True:
        try:
            user = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break
        if user.strip().lower() in ("quit", "exit"):
            break
        prompt = f"Q: {user.strip()} A:"
        reply = generate(model, tok, prompt, max_new_tokens=60, temperature=0.6, top_k=10)
        reply = reply[len(prompt):].strip()
        print(f"Model: {reply}")
