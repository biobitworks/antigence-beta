#!/usr/bin/env python3
"""
Test CodeBERT as a vulnerability detection antibody.
Uses mrm8488/codebert-base-finetuned-detect-insecure-code
"""

import json
from pathlib import Path
from datetime import datetime

MODELS_DIR = Path.home() / ".antigence" / "models"

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_codebert_prediction(text, model_name="mrm8488_codebert-base-finetuned-detect-insecure-code"):
    """Get vulnerability prediction from CodeBERT model."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
        import torch

        model_path = MODELS_DIR / model_name

        # Use pipeline for simplicity
        if not hasattr(get_codebert_prediction, 'classifiers'):
            get_codebert_prediction.classifiers = {}

        if model_name not in get_codebert_prediction.classifiers:
            log(f"  Loading {model_name}...")
            classifier = pipeline(
                "text-classification",
                model=str(model_path),
                tokenizer=str(model_path),
                truncation=True,
                max_length=512
            )
            get_codebert_prediction.classifiers[model_name] = classifier

        classifier = get_codebert_prediction.classifiers[model_name]
        result = classifier(text[:512])[0]

        # Return probability of being vulnerable
        # Label mapping depends on training: usually LABEL_1 = vulnerable
        if result['label'] in ['LABEL_1', 'vulnerable', '1', 'insecure']:
            return result['score']
        else:
            return 1 - result['score']

    except Exception as e:
        log(f"    Error: {e}")
        return None


def main():
    log("="*60)
    log("TESTING CODEBERT ANTIBODIES")
    log("="*60)

    TESTS = [
        # Vulnerable
        ("SQL Injection", 'query = "SELECT * FROM users WHERE id = " + user_id', "vulnerable"),
        ("Command Injection", 'os.system("ping " + hostname)', "vulnerable"),
        ("Buffer Overflow", 'char buf[10]; strcpy(buf, src);', "vulnerable"),
        ("XSS", 'document.innerHTML = "<div>" + userData + "</div>"', "vulnerable"),
        ("Path Traversal", 'fopen(basePath + userFile, "r")', "vulnerable"),
        ("Format String", 'printf(userInput);', "vulnerable"),
        ("Use After Free", 'free(ptr); use(ptr);', "vulnerable"),
        ("Integer Overflow", 'int result = a * b;', "vulnerable"),
        # Safe
        ("Safe SQL", 'cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))', "safe"),
        ("Safe Copy", 'strncpy(buf, src, sizeof(buf)-1); buf[sizeof(buf)-1] = 0;', "safe"),
        ("Safe Print", 'printf("%s", userInput);', "safe"),
        ("Safe Alloc", 'if (ptr) { use(ptr); free(ptr); ptr = NULL; }', "safe"),
    ]

    models_to_test = [
        "mrm8488_codebert-base-finetuned-detect-insecure-code",
        "mahdin70_codebert-devign-code-vulnerability-detector",
    ]

    for model in models_to_test:
        log(f"\n{'='*50}")
        log(f"Model: {model}")
        log(f"{'='*50}")

        correct = 0
        for name, code, expected in TESTS:
            score = get_codebert_prediction(code, model)
            if score is None:
                print(f"[SKIP] {name}")
                continue

            pred = "vulnerable" if score > 0.5 else "safe"
            ok = pred == expected
            correct += int(ok)
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {pred} (score={score:.2f})")

        log(f"\nAccuracy: {correct}/{len(TESTS)} ({correct*100//len(TESTS)}%)")


if __name__ == "__main__":
    main()
