# AI Code Assistant Mini-Project

A local, lightweight AI code generation tool built using Python, Flask, and Hugging Face Transformers.

## Overview
This application accepts partial Python code or natural language comments and uses a pretrained local transformer model (`bigcode/tiny_starcoder_py`) to generate context-aware code completions. It features a responsive, modern web UI with syntax highlighting.

## Features
- **Local Inference**: Runs entirely on your machine — no external API calls, no internet required after setup.
- **Transformer Powered**: Uses a state-of-the-art Causal Language Model fine-tuned on Python code.
- **Syntax Highlighted Output**: Generated Python code is displayed with Prism.js colour formatting.
- **Adjustable Temperature**: Slider to control how creative vs. precise the model's output is.
- **Copy to Clipboard**: One-click copy of the generated code.
- **Simple Architecture**: Easy to understand for a college mini-project or viva.

## Project Structure
```
ai-code-assistant/
├── app.py             # Flask application & API routes
├── model.py           # Machine learning model loading & inference logic
├── requirements.txt   # Pinned Python dependencies
├── README.md          # Project documentation
└── templates/
    └── index.html     # Frontend UI with editor, controls and output panel
```

## How It Works (Architecture)

This project implements a classic **3-layer AI pipeline**:

```
User Input (Partial Code)
        ↓
[1] TOKENIZATION  (model.py — AutoTokenizer)
    "def is_prime(n):" → [7109, 318, 62, ...] (numerical token IDs)
        ↓
[2] TRANSFORMER INFERENCE  (model.py — AutoModelForCausalLM)
    Self-Attention layers compute relationships between all input tokens.
    The model predicts the next token, appends it, and repeats (auto-regression).
    Temperature controls the randomness of each token selection.
        ↓
[3] DECODING  (model.py — tokenizer.decode)
    [7109, 318, 62, 994, 11, ...] → "    if n < 2: return False ..."
        ↓
Flask API  (app.py — POST /generate)
        ↓
Frontend Display  (index.html — Prism.js highlighted output)
```

### About the Model: `bigcode/tiny_starcoder_py`
- **Model type**: Causal Language Model (GPT-style decoder-only transformer)
- **Training data**: Python source code from The Stack dataset (open-source code repositories)
- **Why this model**: It is tiny (~160M parameters), runs well on CPU, and is purpose-built for Python completion tasks
- **Context window**: 2048 tokens. We cap input at 900 tokens to always leave room for generation.

### Key Concepts for Viva
| Term | Plain English Explanation |
|---|---|
| **Token** | A chunk of text (word piece or symbol) the model reads/writes at one time |
| **Tokenization** | Converting raw text into a list of numeric IDs |
| **Self-Attention** | The mechanism that lets the model understand how each word relates to all others |
| **Temperature** | A number that controls randomness: low = safe/predictable, high = creative/risky |
| **Autoregressive** | Generating one token at a time using all previous tokens as context |
| **`max_new_tokens`** | The hard limit on how many tokens the model is allowed to generate |

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Step 1 — Navigate to the project folder
```shell
cd ai-code-assistant
```

### Step 2 — Create a virtual environment (recommended)
```shell
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```shell
pip install -r requirements.txt
```
> **Note:** This downloads PyTorch, Transformers, and the AI model weights from Hugging Face (~200 MB). Only happens on the first run.

### Step 4 — Run the application
```shell
python app.py
```
Open your browser at: **http://127.0.0.1:5000/**

## Usage
1. Open the web interface.
2. Type a comment or partial Python code in the input box. Example:
   ```python
   # function to check if a number is prime
   def is_prime(n):
   ```
3. Adjust the **Temperature slider** if desired.
4. Click **Generate Completion** or press **Ctrl + Enter**.
5. The AI-generated code completion appears highlighted in the output panel.
6. Click **Copy** to copy it to your clipboard.

## Environment Variables
| Variable | Default | Description |
|---|---|---|
| `FLASK_DEBUG` | `false` | Set to `true` to enable Flask debug mode during development |

```shell
# Example: enable debug mode
set FLASK_DEBUG=true   # Windows
export FLASK_DEBUG=true  # Mac/Linux
python app.py
```
