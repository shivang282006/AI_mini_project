import os
from flask import Flask, render_template, request, jsonify
from model import generate_code

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    """
    Render the main UI page.
    Serves index.html from the templates/ directory — the user interface for
    typing partial code and viewing AI-generated completions.
    """
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """
    POST /generate — API endpoint to generate a code completion.

    Expected JSON body:
        { "prompt": "<partial code>", "temperature": 0.2 }

    Returns JSON:
        { "result": "<generated completion>" }   on success
        { "error": "<message>" }                 on failure
    """
    data = request.get_json()

    # Validate that a prompt was actually sent in the request body
    if not data or "prompt" not in data:
        return jsonify({"error": "No prompt provided"}), 400

    prompt = data.get("prompt", "")

    # FIX: Clamp temperature to a safe range [0.01, 1.5].
    # This prevents a crash if the frontend sends a malformed or extreme value.
    raw_temp = data.get("temperature", 0.2)
    try:
        temperature = max(0.01, min(float(raw_temp), 1.5))
    except (ValueError, TypeError):
        temperature = 0.2  # Fall back to a safe default if parsing fails

    try:
        # Call generate_code() from model.py with the user's prompt and temperature
        completion = generate_code(prompt, max_new_tokens=150, temperature=temperature)

        # Return the AI-generated completion as a JSON response to the frontend
        return jsonify({"result": completion})

    except Exception as e:
        # Catch any unexpected inference error and return a structured JSON error
        # so the frontend can display a sensible message to the user
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # FIX: Read debug mode from an environment variable.
    # Default is False so debug info is never accidentally shown during a demo.
    # To enable: set FLASK_DEBUG=true in your terminal before running.
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)
