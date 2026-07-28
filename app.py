"""
Flask web app: serves a simple chat UI backed by the from-scratch TinyGPT
model. Run with: python3 app.py, then open http://localhost:5000
"""
from flask import Flask, request, jsonify, render_template
from generate import load_model, generate

app = Flask(__name__)
MODEL, TOK = load_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_msg = (data.get("message") or "").strip()
    temperature = float(data.get("temperature", 0.6))
    top_k = int(data.get("top_k", 10))
    if not user_msg:
        return jsonify({"reply": ""})

    prompt = f"Q: {user_msg} A:"
    out = generate(MODEL, TOK, prompt, max_new_tokens=60,
                    temperature=temperature, top_k=top_k)
    reply = out[len(prompt):].strip()
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
