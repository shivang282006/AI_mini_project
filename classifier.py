"""
classifier.py — Random Forest ML layer for the AI Code Assistant
=================================================================
Contains two classes:

  IntentClassifier    — RandomForestClassifier that reads a user's prompt and
                        returns one of 5 intent labels plus recommended
                        generation parameters (temperature, max_new_tokens).

  CodeQualityScorer   — RandomForestRegressor that scores a generated code
                        snippet 0-100 and returns a quality label + feature
                        breakdown.

Both models train on synthetic data at import time (< 1 s) so there is no
separate training step, and no model files need to be saved to disk.
"""

import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ─────────────────────────────────────────────────────────────────────────────
# INTENT CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

# 5 intent categories
INTENT_LABELS = ["function", "class", "algorithm", "data_struct", "general"]

# Recommended generation params per intent
INTENT_PARAMS = {
    "function":    {"temperature": 0.2, "max_new_tokens": 120},
    "class":       {"temperature": 0.25, "max_new_tokens": 180},
    "algorithm":   {"temperature": 0.15, "max_new_tokens": 200},
    "data_struct": {"temperature": 0.2,  "max_new_tokens": 160},
    "general":     {"temperature": 0.4,  "max_new_tokens": 100},
}

# Keywords for feature extraction
_ALGO_KEYWORDS = [
    "sort", "search", "binary", "bubble", "merge", "quick", "dfs", "bfs",
    "dijkstra", "dynamic", "dp", "greedy", "recursion", "recursive",
    "palindrome", "fibonacci", "prime", "factorial", "gcd", "lcm",
    "path", "graph", "tree", "matrix", "reverse", "anagram",
]
_DS_KEYWORDS = [
    "stack", "queue", "linked", "list", "dict", "dictionary", "set",
    "heap", "trie", "deque", "array", "hash", "map", "node", "pointer",
    "insert", "delete", "push", "pop", "append",
]


def _extract_intent_features(prompt: str) -> np.ndarray:
    """
    Convert a raw prompt string into a 12-dimensional numeric feature vector.

    Features
    --------
    0  has_def          — contains the word 'def'
    1  has_class        — contains the word 'class'
    2  has_hash_comment — starts with '#' (natural-language comment)
    3  has_algo_kw      — contains ≥1 algorithm keyword
    4  has_ds_kw        — contains ≥1 data-structure keyword
    5  word_count       — total word count (normalised /100)
    6  line_count       — number of lines (normalised /10)
    7  has_return       — contains 'return'
    8  has_self         — contains 'self' (implies class method)
    9  has_init         — contains '__init__'
    10 has_for_while    — contains 'for' or 'while' (loop-heavy = algorithm)
    11 avg_word_len     — average word length (normalised /10)
    """
    p = prompt.lower()
    words = re.findall(r'\w+', p)
    avg_wl = (sum(len(w) for w in words) / len(words) / 10) if words else 0

    return np.array([
        1 if re.search(r'\bdef\b', p) else 0,
        1 if re.search(r'\bclass\b', p) else 0,
        1 if prompt.lstrip().startswith('#') else 0,
        1 if any(kw in p for kw in _ALGO_KEYWORDS) else 0,
        1 if any(kw in p for kw in _DS_KEYWORDS) else 0,
        len(words) / 100,
        len(prompt.split('\n')) / 10,
        1 if 'return' in p else 0,
        1 if 'self' in p else 0,
        1 if '__init__' in p else 0,
        1 if re.search(r'\bfor\b|\bwhile\b', p) else 0,
        avg_wl,
    ], dtype=float)


# ── Synthetic training data ──────────────────────────────────────────────────

_SYNTHETIC_PROMPTS = [
    # function (label 0)
    ("def add(a, b):", 0),
    ("# function to reverse a string\ndef reverse_string(s):", 0),
    ("def calculate_area(radius):", 0),
    ("# write a function that returns the square of a number", 0),
    ("def greet(name):\n    '''say hello'''", 0),
    ("# function to check if a number is even\ndef is_even(n):", 0),
    ("def multiply(x, y):", 0),
    ("# function to compute power\ndef power(base, exp):", 0),
    ("def convert_to_celsius(f):", 0),
    ("# helper function to split a string\ndef split_words(text):", 0),

    # class (label 1)
    ("class Animal:", 1),
    ("class BankAccount:\n    def __init__(self, balance):", 1),
    ("# create a class for a student\nclass Student:", 1),
    ("class Rectangle:\n    def __init__(self, w, h):\n        self.w = w", 1),
    ("class Node:\n    def __init__(self, val):\n        self.val = val", 1),
    ("class Stack:\n    def __init__(self):\n        self.items = []", 1),
    ("# design a class for a car\nclass Car:", 1),
    ("class Employee:\n    def __init__(self, name, salary):", 1),
    ("class Shape:\n    def area(self):", 1),
    ("class Matrix:\n    def __init__(self, rows, cols):", 1),

    # algorithm (label 2)
    ("# bubble sort algorithm\ndef bubble_sort(arr):", 2),
    ("# binary search\ndef binary_search(arr, target):", 2),
    ("# check if a string is palindrome\ndef is_palindrome(s):", 2),
    ("# fibonacci sequence\ndef fibonacci(n):", 2),
    ("# find all prime numbers up to n using sieve", 2),
    ("# merge sort implementation\ndef merge_sort(arr):", 2),
    ("# quicksort\ndef quicksort(arr):", 2),
    ("# depth first search\ndef dfs(graph, node, visited):", 2),
    ("# gcd of two numbers\ndef gcd(a, b):", 2),
    ("# dynamic programming — longest common subsequence\ndef lcs(s1, s2):", 2),

    # data_struct (label 3)
    ("# implement a stack using a list\nclass Stack:", 3),
    ("# linked list implementation\nclass LinkedList:", 3),
    ("# implement a queue\nclass Queue:", 3),
    ("# binary search tree\nclass BST:\n    def insert(self, val):", 3),
    ("# min heap\nimport heapq", 3),
    ("# implement a hash map from scratch\nclass HashMap:", 3),
    ("# trie data structure\nclass TrieNode:\n    def __init__(self):", 3),
    ("# deque using doubly linked list\nclass Deque:", 3),
    ("# graph using adjacency list\ngraph = {}", 3),
    ("# implement a priority queue\nimport heapq\npq = []", 3),

    # general (label 4)
    ("# hello world", 4),
    ("print('hello')", 4),
    ("# write some python code", 4),
    ("x = 10\ny = 20", 4),
    ("# example code", 4),
    ("import os\nimport sys", 4),
    ("# read a file\nwith open('file.txt') as f:", 4),
    ("# connect to a database", 4),
    ("# send an email", 4),
    ("# parse a JSON file\nimport json", 4),
]


class IntentClassifier:
    """Classify a user prompt into one of 5 intent categories."""

    def __init__(self):
        X = np.array([_extract_intent_features(p) for p, _ in _SYNTHETIC_PROMPTS])
        y = np.array([label for _, label in _SYNTHETIC_PROMPTS])
        self._model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            class_weight="balanced",
        )
        self._model.fit(X, y)

    def predict(self, prompt: str) -> dict:
        """
        Returns
        -------
        {
          "intent":          str,           # e.g. "algorithm"
          "confidence":      float,         # 0-1 probability of top class
          "probabilities":   dict[str,float],# per-class probabilities
          "recommended":     {"temperature": float, "max_new_tokens": int}
        }
        """
        features = _extract_intent_features(prompt).reshape(1, -1)
        proba = self._model.predict_proba(features)[0]

        # Build full probability dict (model may not have seen all classes)
        prob_dict = {}
        for i, cls_idx in enumerate(self._model.classes_):
            prob_dict[INTENT_LABELS[cls_idx]] = round(float(proba[i]), 4)
        # Ensure all 5 labels are present
        for label in INTENT_LABELS:
            prob_dict.setdefault(label, 0.0)

        top_idx = int(np.argmax(proba))
        intent = INTENT_LABELS[self._model.classes_[top_idx]]
        confidence = float(proba[top_idx])

        return {
            "intent": intent,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "recommended": INTENT_PARAMS[intent],
        }


# ─────────────────────────────────────────────────────────────────────────────
# CODE QUALITY SCORER
# ─────────────────────────────────────────────────────────────────────────────

def _bracket_balance(code: str) -> float:
    """
    Returns a 0-1 balance score.
    1.0 = perfectly balanced brackets/parens/braces.
    0.0 = every bracket is mismatched.
    """
    opens  = code.count('(') + code.count('[') + code.count('{')
    closes = code.count(')') + code.count(']') + code.count('}')
    if opens == 0 and closes == 0:
        return 1.0
    return 1.0 - abs(opens - closes) / max(opens + closes, 1)


def _indentation_consistency(lines: list) -> float:
    """
    Fraction of non-empty lines whose indentation is a clean multiple of 4.
    """
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return 1.0
    ok = sum(1 for l in non_empty if (len(l) - len(l.lstrip())) % 4 == 0)
    return ok / len(non_empty)


def _extract_quality_features(code: str) -> dict:
    """
    Extract 10 numerical quality-proxy features from a code snippet.

    Features
    --------
    line_count          — number of lines
    has_return          — 1 if 'return' keyword present
    bracket_balance     — 0-1 score (1 = perfectly balanced)
    indentation_ok      — fraction of lines with indent % 4 == 0
    has_docstring       — 1 if triple-quoted string present
    has_comments        — 1 if any '#' comment line present
    avg_line_length     — mean characters per non-empty line (normalised /80)
    keyword_density     — Python keyword count / word count
    has_def             — 1 if at least one 'def' statement
    has_type_hints      — 1 if '->' or ':' type annotation patterns exist
    """
    import keyword as _kw
    lines = code.split('\n')
    non_empty = [l for l in lines if l.strip()]
    words = re.findall(r'\w+', code)

    avg_ll = (sum(len(l) for l in non_empty) / len(non_empty) / 80) if non_empty else 0
    kw_density = (sum(1 for w in words if _kw.iskeyword(w)) / len(words)) if words else 0

    features = {
        "line_count":       len(lines),
        "has_return":       1 if re.search(r'\breturn\b', code) else 0,
        "bracket_balance":  round(_bracket_balance(code), 4),
        "indentation_ok":   round(_indentation_consistency(lines), 4),
        "has_docstring":    1 if '"""' in code or "'''" in code else 0,
        "has_comments":     1 if re.search(r'^\s*#', code, re.M) else 0,
        "avg_line_length":  round(avg_ll, 4),
        "keyword_density":  round(kw_density, 4),
        "has_def":          1 if re.search(r'\bdef\b', code) else 0,
        "has_type_hints":   1 if re.search(r'->\s*\w+|:\s*(int|str|float|bool|list|dict|None)', code) else 0,
    }
    return features


def _features_to_vector(feat: dict) -> np.ndarray:
    return np.array([
        feat["line_count"],
        feat["has_return"],
        feat["bracket_balance"],
        feat["indentation_ok"],
        feat["has_docstring"],
        feat["has_comments"],
        feat["avg_line_length"],
        feat["keyword_density"],
        feat["has_def"],
        feat["has_type_hints"],
    ], dtype=float)


# ── Score thresholds ─────────────────────────────────────────────────────────

def _score_to_label(score: float) -> str:
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 50:
        return "Fair"
    else:
        return "Needs Improvement"


# ── Synthetic training data for quality scorer ───────────────────────────────
# (code_snippet, quality_score)  — quality is a subjective 0-100 target

_QUALITY_TRAINING = [
    # Excellent (90-100)
    ('def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n    return a + b\n', 97),
    ('def is_prime(n: int) -> bool:\n    """Check if n is prime."""\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        # factor found\n        if n % i == 0:\n            return False\n    return True\n', 95),
    ('class Stack:\n    """A simple stack using a list."""\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop()\n', 93),
    ('def binary_search(arr: list, target: int) -> int:\n    """Return index of target in sorted arr, or -1."""\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n', 96),

    # Good (75-89)
    ('def reverse(s):\n    # reverse a string\n    return s[::-1]\n', 82),
    ('def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)\n', 80),
    ('def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(n - i - 1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\n', 78),
    ('def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n', 76),

    # Fair (50-74)
    ('def add(a,b):\n return a+b\n', 55),
    ('def check(x):\n    if x>0:\n     return True\n    return False\n', 52),
    ('result = []\nfor i in range(10):\n    result.append(i*2)\n', 60),
    ('x=1\ny=2\nz=x+y\nprint(z)\n', 50),

    # Needs improvement (0-49)
    ('x\n', 10),
    ('def f(x):\nreturn x\n', 20),
    ('print(\n', 5),
    ('if True\n    pass\n', 15),
    ('a=1;b=2;c=3;d=4;e=5\n', 35),
    ('def foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n', 45),
]


class CodeQualityScorer:
    """Score generated code quality from 0 to 100."""

    def __init__(self):
        X = np.array([_features_to_vector(_extract_quality_features(code))
                      for code, _ in _QUALITY_TRAINING])
        y = np.array([score for _, score in _QUALITY_TRAINING], dtype=float)
        self._model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=42,
        )
        self._model.fit(X, y)

    def score(self, code: str) -> dict:
        """
        Returns
        -------
        {
          "score":     float,   # 0-100
          "label":     str,     # "Excellent" / "Good" / "Fair" / "Needs Improvement"
          "breakdown": dict     # individual feature values
        }
        """
        if not code or not code.strip():
            return {"score": 0.0, "label": "Needs Improvement", "breakdown": {}}

        features = _extract_quality_features(code)
        vec = _features_to_vector(features).reshape(1, -1)
        raw = float(self._model.predict(vec)[0])
        final_score = round(max(0.0, min(100.0, raw)), 1)

        return {
            "score": final_score,
            "label": _score_to_label(final_score),
            "breakdown": features,
        }
