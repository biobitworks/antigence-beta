#!/usr/bin/env python3
"""
Smoke test for trained Antigence agents.

Tests B Cell and NK Cell against known vulnerable and safe code samples.
"""

import json
import sys
from pathlib import Path

# Test samples - real CVE examples from Big-Vul
VULNERABLE_SAMPLES = [
    {
        "name": "CVE-2009-1194 - Integer Overflow",
        "cwe": "CWE-189",
        "code": """
void pango_glyph_string_set_size(PangoGlyphString *string, gint new_len) {
    while (new_len > string->space) {
        if (string->space == 0)
            string->space = 1;
        else
            string->space *= 2;  // Integer overflow vulnerability

        if (string->space < 0) {
            g_warning("glyph string length overflows");
            new_len = string->space = G_MAXINT - 8;
        }
    }
}
""",
        "expected": "vulnerable"
    },
    {
        "name": "SQL Injection",
        "cwe": "CWE-89",
        "code": """
def get_user(username):
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    cursor.execute(query)
    return cursor.fetchone()
""",
        "expected": "vulnerable"
    },
    {
        "name": "Buffer Overflow",
        "cwe": "CWE-120",
        "code": """
void copy_string(char *dest, char *src) {
    strcpy(dest, src);  // No bounds checking
}

int main() {
    char buffer[10];
    copy_string(buffer, argv[1]);
}
""",
        "expected": "vulnerable"
    },
    {
        "name": "Command Injection",
        "cwe": "CWE-78",
        "code": """
import os
def run_command(user_input):
    os.system("echo " + user_input)
""",
        "expected": "vulnerable"
    }
]

SAFE_SAMPLES = [
    {
        "name": "Parameterized Query",
        "code": """
def get_user(username):
    query = "SELECT * FROM users WHERE name = %s"
    cursor.execute(query, (username,))
    return cursor.fetchone()
""",
        "expected": "safe"
    },
    {
        "name": "Safe String Copy",
        "code": """
void copy_string(char *dest, size_t dest_size, const char *src) {
    strncpy(dest, src, dest_size - 1);
    dest[dest_size - 1] = '\\0';
}
""",
        "expected": "safe"
    },
    {
        "name": "Safe Command Execution",
        "code": """
import subprocess
def run_command(args):
    subprocess.run(args, shell=False, check=True)
""",
        "expected": "safe"
    }
]

HALLUCINATION_SAMPLES = [
    {
        "name": "Fake Citation",
        "text": "Q: What is quantum consciousness?\nA: According to Dr. John Smith's 2024 paper in Nature Neuroscience, quantum effects in microtubules create consciousness through orchestrated objective reduction.",
        "expected": "hallucinated"
    },
    {
        "name": "Real Fact",
        "text": "Q: What is the capital of France?\nA: Paris is the capital of France.",
        "expected": "truthful"
    }
]


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    """Get embedding from Ollama."""
    import requests
    try:
        resp = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": text[:2000]},
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json().get("embedding", [])
    except Exception:
        pass
    return []


def cosine_similarity(a: list, b: list) -> float:
    """Calculate cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def load_patterns(model_file: str) -> list:
    """Load trained patterns from JSON file."""
    models_dir = Path.home() / ".antigence" / "trained"
    filepath = models_dir / model_file

    if not filepath.exists():
        return []

    with open(filepath) as f:
        data = json.load(f)

    return data.get("patterns", data.get("self_patterns", []))


def run_bcell_check(code: str, patterns: list, threshold: float = 0.7) -> dict:
    """Test code against B Cell patterns."""
    if not patterns:
        return {"error": "No patterns loaded"}

    embedding = get_embedding(code)
    if not embedding:
        return {"error": "Failed to get embedding"}

    # Find most similar patterns
    similarities = []
    for p in patterns:
        p_emb = p.get("embedding", [])
        if p_emb:
            sim = cosine_similarity(embedding, p_emb)
            similarities.append({
                "similarity": sim,
                "label": p.get("label", "unknown"),
                "cwe": p.get("cwe", ""),
            })

    if not similarities:
        return {"error": "No valid patterns"}

    # Sort by similarity
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top_matches = similarities[:5]

    # Voting: count vulnerable vs safe in top matches
    vuln_score = sum(1 for m in top_matches if m["label"] == "vulnerable")
    safe_score = sum(1 for m in top_matches if m["label"] == "safe")

    prediction = "vulnerable" if vuln_score > safe_score else "safe"
    confidence = max(top_matches[0]["similarity"], 0.5)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "top_match": top_matches[0],
        "vuln_votes": vuln_score,
        "safe_votes": safe_score,
    }


def run_nkcell_check(text: str, self_patterns: list, threshold: float = 0.6) -> dict:
    """Test text against NK Cell self-patterns (negative selection)."""
    if not self_patterns:
        return {"error": "No self-patterns loaded"}

    embedding = get_embedding(text)
    if not embedding:
        return {"error": "Failed to get embedding"}

    # Find maximum similarity to any self pattern
    max_sim = 0.0
    for p in self_patterns:
        p_emb = p.get("embedding", [])
        if p_emb:
            sim = cosine_similarity(embedding, p_emb)
            max_sim = max(max_sim, sim)

    # NK Cell logic: if NOT similar to self, it's foreign (hallucinated)
    is_self = max_sim >= threshold

    return {
        "prediction": "truthful" if is_self else "hallucinated",
        "max_self_similarity": max_sim,
        "threshold": threshold,
        "confidence": abs(max_sim - threshold) + 0.5,
    }


def run_smoke_test():
    """Run smoke test on all samples."""
    print("=" * 60)
    print("ANTIGENCE SMOKE TEST")
    print("=" * 60)

    # Load patterns
    print("\nLoading trained patterns...")
    bcell_patterns = load_patterns("bcell_overnight_nomic-embed-text.json")
    nkcell_patterns = load_patterns("nkcell_overnight_nomic-embed-text.json")

    print(f"  B Cell patterns: {len(bcell_patterns)}")
    print(f"  NK Cell patterns: {len(nkcell_patterns)}")

    if not bcell_patterns:
        print("ERROR: No B Cell patterns found!")
        return False

    # Test B Cell on vulnerable code
    print("\n" + "-" * 60)
    print("B CELL TEST: Vulnerable Code Samples")
    print("-" * 60)

    bcell_correct = 0
    bcell_total = 0

    for sample in VULNERABLE_SAMPLES:
        result = run_bcell_check(sample["code"], bcell_patterns)
        correct = result.get("prediction") == sample["expected"]
        bcell_correct += int(correct)
        bcell_total += 1

        status = "✓" if correct else "✗"
        print(f"\n{status} {sample['name']} ({sample.get('cwe', 'N/A')})")
        print(f"  Expected: {sample['expected']}")
        print(f"  Predicted: {result.get('prediction', 'error')}")
        print(f"  Confidence: {result.get('confidence', 0):.2f}")
        print(f"  Votes: {result.get('vuln_votes', 0)} vuln / {result.get('safe_votes', 0)} safe")

    # Test B Cell on safe code
    print("\n" + "-" * 60)
    print("B CELL TEST: Safe Code Samples")
    print("-" * 60)

    for sample in SAFE_SAMPLES:
        result = run_bcell_check(sample["code"], bcell_patterns)
        correct = result.get("prediction") == sample["expected"]
        bcell_correct += int(correct)
        bcell_total += 1

        status = "✓" if correct else "✗"
        print(f"\n{status} {sample['name']}")
        print(f"  Expected: {sample['expected']}")
        print(f"  Predicted: {result.get('prediction', 'error')}")
        print(f"  Confidence: {result.get('confidence', 0):.2f}")

    # Test NK Cell
    print("\n" + "-" * 60)
    print("NK CELL TEST: Hallucination Detection")
    print("-" * 60)

    nkcell_correct = 0
    nkcell_total = 0

    if nkcell_patterns:
        for sample in HALLUCINATION_SAMPLES:
            result = run_nkcell_check(sample["text"], nkcell_patterns)
            correct = result.get("prediction") == sample["expected"]
            nkcell_correct += int(correct)
            nkcell_total += 1

            status = "✓" if correct else "✗"
            print(f"\n{status} {sample['name']}")
            print(f"  Expected: {sample['expected']}")
            print(f"  Predicted: {result.get('prediction', 'error')}")
            print(f"  Self-similarity: {result.get('max_self_similarity', 0):.2f}")
    else:
        print("  Skipped (no NK Cell patterns)")

    # Summary
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)

    bcell_acc = bcell_correct / bcell_total * 100 if bcell_total > 0 else 0
    nkcell_acc = nkcell_correct / nkcell_total * 100 if nkcell_total > 0 else 0

    print(f"\nB Cell Accuracy: {bcell_correct}/{bcell_total} ({bcell_acc:.1f}%)")
    print(f"NK Cell Accuracy: {nkcell_correct}/{nkcell_total} ({nkcell_acc:.1f}%)")

    overall = (bcell_correct + nkcell_correct) / (bcell_total + nkcell_total) * 100
    print(f"\nOverall: {overall:.1f}%")

    return overall >= 50  # Pass if at least 50% correct


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
