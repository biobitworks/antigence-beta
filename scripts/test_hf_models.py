#!/usr/bin/env python3
"""
Test HuggingFace model integration with Antigence.

Usage:
    python scripts/test_hf_models.py
"""

import os
from pathlib import Path

def test_codebert_insecure():
    """Test CodeBERT insecure code detection."""
    print("\n=== Testing CodeBERT Insecure Code Detection ===\n")

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model_path = Path.home() / ".antigence/models/mrm8488_codebert-base-finetuned-detect-insecure-code"

        if not model_path.exists():
            print(f"[SKIP] Model not found at {model_path}")
            return False

        print(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

        # Test samples
        samples = [
            ("eval(user_input)", "vulnerable"),
            ("print('hello world')", "safe"),
            ("os.system(cmd)", "vulnerable"),
            ("x = 1 + 2", "safe"),
            ("exec(request.GET['code'])", "vulnerable"),
        ]

        print("\nResults:")
        correct = 0
        for code, expected in samples:
            inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            label = "vulnerable" if prediction == 1 else "safe"
            match = "✓" if label == expected else "✗"
            if label == expected:
                correct += 1
            print(f"  {match} '{code[:30]}...' → {label} (expected: {expected})")

        print(f"\nAccuracy: {correct}/{len(samples)} ({100*correct/len(samples):.0f}%)")
        return True

    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_ollama_embeddings():
    """Test Ollama embedding generation."""
    print("\n=== Testing Ollama Embeddings ===\n")

    try:
        import requests

        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": "Test embedding"},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            embedding = data.get("embedding", [])
            print(f"  [OK] nomic-embed-text: {len(embedding)} dimensions")
            return True
        else:
            print(f"  [ERROR] Status {response.status_code}")
            return False

    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_graphcodebert_cwe():
    """Test GraphCodeBERT CWE classification."""
    print("\n=== Testing GraphCodeBERT CWE Classification ===\n")

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model_path = Path.home() / ".antigence/models/mahdin70_GraphCodeBERT-VulnCWE"

        if not model_path.exists():
            print(f"[SKIP] Model not found at {model_path}")
            return False

        print(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

        # Test sample
        code = "char buffer[10]; strcpy(buffer, user_input);"
        inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

        print(f"  Code: {code}")
        print(f"  Prediction class: {prediction}")
        print(f"  [OK] Model loaded and inference working")
        return True

    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def main():
    print("=" * 60)
    print("Antigence HuggingFace Model Integration Test")
    print("=" * 60)

    results = {
        "Ollama Embeddings": test_ollama_embeddings(),
        "CodeBERT Insecure": test_codebert_insecure(),
        "GraphCodeBERT CWE": test_graphcodebert_cwe(),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
