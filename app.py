import os
from flask import Flask, render_template, request, jsonify
from model import generate_code
from classifier import IntentClassifier, CodeQualityScorer

app = Flask(__name__)

# ── Load both Random Forest models ONCE at startup ───────────────────────────
# Training on synthetic data takes < 1 s; doing it here means every request
# is served without re-training overhead.
print("Training IntentClassifier...")
intent_clf = IntentClassifier()
print("Training CodeQualityScorer...")
quality_scorer = CodeQualityScorer()
print("Both RF models ready.")


@app.route("/", methods=["GET"])
def index():
    """
    Render the main UI page.
    Serves index.html from the templates/ directory.
    """
    return render_template("index.html")


@app.route("/classify_intent", methods=["POST"])
def classify_intent():
    """
    POST /classify_intent — Run the Intent Classifier on a user prompt.

    Expected JSON body:
        { "prompt": "<partial code or comment>" }

    Returns JSON:
        {
          "intent":        str,            e.g. "algorithm"
          "confidence":    float,          0-1
          "probabilities": dict[str,float],
          "recommended":   { "temperature": float, "max_new_tokens": int }
        }
    """
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "No prompt provided"}), 400

    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is empty"}), 400

    try:
        result = intent_clf.predict(prompt)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/generate", methods=["POST"])
def generate():
    """
    POST /generate — Generate code completion + quality score.

    Expected JSON body:
        {
          "prompt":      "<partial code>",
          "temperature": 0.2,           (optional, default 0.2)
          "max_tokens":  150            (optional, default 150)
        }

    Returns JSON:
        {
          "result":  "<generated code>",
          "quality": { "score": float, "label": str, "breakdown": dict }
        }
    """
    data = request.get_json()

    if not data or "prompt" not in data:
        return jsonify({"error": "No prompt provided"}), 400

    prompt = data.get("prompt", "")

    # Clamp temperature to a safe range [0.01, 1.5]
    raw_temp = data.get("temperature", 0.2)
    try:
        temperature = max(0.01, min(float(raw_temp), 1.5))
    except (ValueError, TypeError):
        temperature = 0.2

    # Accept max_tokens from frontend (RF-recommended or user-chosen)
    raw_tokens = data.get("max_tokens", 150)
    try:
        max_new_tokens = max(30, min(int(raw_tokens), 300))
    except (ValueError, TypeError):
        max_new_tokens = 150

    try:
        completion = generate_code(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

        # Score the generated output with the Code Quality Scorer
        quality = quality_scorer.score(completion)

        return jsonify({
            "result": completion,
            "quality": quality,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)
