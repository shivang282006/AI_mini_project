import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# We use "bigcode/tiny_starcoder_py" — a small, pretrained transformer model
# specifically fine-tuned for Python code generation and completion.
# It runs efficiently on CPUs making it ideal for a demo/mini-project.
MODEL_NAME = "bigcode/tiny_starcoder_py"

# The model has a maximum context window of 2048 tokens.
# We cap input at 900 tokens to leave ample room for generation
# and prevent a silent crash if the user pastes in a very large block of code.
MAX_INPUT_TOKENS = 900

print(f"Loading {MODEL_NAME}...")

# HOW THE MODEL WORKS:
# 1. Tokenization: The text (partial code) is split into "tokens" (subwords/characters).
#    The tokenizer converts these tokens into numerical IDs the AI model can process.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 2. Transformer Model Inference: The pretrained model reads the input token IDs,
#    computes self-attention (contextual relationships between all tokens), and
#    predicts the most likely next token — repeating this process iteratively.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
).to(device)

print(f"Model loaded successfully on '{device}'!")


def _clean_and_trim(text: str) -> str:
    """
    Two-stage post-processing to clean up raw model output.

    STAGE 1 — Strip leading garbage:
        The model sometimes emits a stray character (e.g. 's', ',') right at the
        start before the actual code begins. We skip any leading lines that look
        like output artifacts (single tokens with no real code content) and start
        from the first line that mreally looks like code or a comment.

    STAGE 2 — Trim to the first complete top-level block:
        Causal LMs are trained to keep generating text indefinitely. When a prompt
        is vague (e.g. "# code for palindrome"), the model has no inherent stopping
        point and will generate a second function, a third, a fourth etc.

        Strategy: we parse the output line-by-line, tracking whether we are
        "inside" a top-level function/class body. The moment we see a SECOND
        top-level definition (def/class at column 0) OR a top-level comment '#'
        after we have already captured a complete function body, we stop.

        This is structurally aware and works even when every function has a unique
        name (so simple duplicate-line checks fail).

    Example of what this trims:

        # stray leading 's' character      ← Stage 1 removes this
        def isPalindrome(s):               ← Stage 2 keeps this (first def)
            return s == s[::-1]
                                           ← blank line after body: function done
        # code for reverse_palindrome      ← Stage 2 STOPS here (new top-level)
        def reverse_palindrome(s): ...
    """
    if not text or not text.strip():
        return text

    lines = text.split('\n')

    # ── STAGE 1: drop leading garbage (single stray tokens) ──────────────────
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        # A "real" line has more than 1 meaningful character, or is blank
        # (blank lines between a comment and function are fine to keep)
        if not stripped or len(stripped) > 1:
            start_idx = i
            break
    lines = lines[start_idx:]

    # ── STAGE 2: stop after the first complete top-level block ───────────────
    result = []
    inside_function = False   # True once we have seen the opening "def" line
    function_body_seen = False  # True once we have seen at least one body line

    for line in lines:
        stripped = line.strip()

        # Detect whether this line is at the top (column-0) level
        is_toplevel_nonempty = bool(line) and not line[0].isspace()

        # A new top-level def/class while we are already inside a function →
        # the previous function is complete; stop before this new one begins.
        is_new_def = is_toplevel_nonempty and (
            stripped.startswith('def ')
            or stripped.startswith('async def ')
            or stripped.startswith('class ')
        )

        if is_new_def:
            if inside_function:
                # Second top-level definition: we are done.
                break
            else:
                # First definition: enter it.
                inside_function = True

        # A top-level comment '#' that arrives AFTER we already have a function
        # body signals the model is starting a new "section" — stop there.
        elif inside_function and function_body_seen and is_toplevel_nonempty and stripped.startswith('#'):
            break

        # Any non-empty, non-indented line that is not a comment appearing after
        # the function body has started means a new top-level statement → stop.
        elif inside_function and function_body_seen and is_toplevel_nonempty and stripped:
            break

        result.append(line)

        # Mark that we have now seen at least one line of the function body
        if inside_function and not is_new_def and stripped:
            function_body_seen = True

    return '\n'.join(result).rstrip()


def generate_code(prompt: str, max_new_tokens: int = 150, temperature: float = 0.2) -> str:
    """
    Generates a Python code completion based on the input prompt.

    Args:
        prompt:         Partial Python code or natural language comment to complete.
        max_new_tokens: Maximum number of new tokens the model should generate.
        temperature:    Randomness control. Low (0.1) = precise, High (1.0) = creative.

    Returns:
        The AI-generated completion string — prompt prefix stripped, leading
        garbage removed, and trimmed to exactly the first complete function.
    """
    if not prompt or not prompt.strip():
        return ""

    # Truncate input to stay within the model's context window (2048 tokens).
    # truncation=True + max_length=MAX_INPUT_TOKENS prevents a silent crash when
    # the user pastes in a very large snippet.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    ).to(device)

    # Disable gradient tracking — only needed during training, not inference.
    # This saves significant memory and speeds up the forward pass.
    with torch.no_grad():
        # Generate new tokens autoregressively: at each step the model sees all
        # previously generated tokens and predicts the single most-likely next one.
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(0.01, temperature),  # clamp: avoid deterministic 0.0
            do_sample=True,
            top_p=0.95,              # nucleus sampling: cut the lowest-prob tokens
            repetition_penalty=1.5,  # penalise already-seen tokens to break loops
            no_repeat_ngram_size=5,  # hard-block any 5-token sequence from repeating
            eos_token_id=tokenizer.eos_token_id,   # stop at end-of-sequence
            pad_token_id=tokenizer.eos_token_id
        )

    # 3. Decoding: convert predicted token IDs back into human-readable text.
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Strip the input prompt from the front — return ONLY the newly generated part.
    if full_text.startswith(prompt):
        completion = full_text[len(prompt):]
    else:
        completion = full_text

    # 4. Post-process: remove leading garbage tokens AND trim to the first complete
    #    function body. This is the definitive guard against multi-function looping.
    completion = _clean_and_trim(completion)

    return completion
