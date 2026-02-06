#!/usr/bin/env python3
"""Quick smoke test using existing overnight models with larger embeddings."""

import json
import sys
from pathlib import Path

MODELS_DIR = Path.home() / ".antigence" / "trained"

def get_embedding(text: str, model: str) -> list:
    import requests
    try:
        resp = requests.post("http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": text[:2000]}, timeout=60)
        if resp.status_code == 200:
            return resp.json().get("embedding", [])
    except: pass
    return []

def cosine_sim(a, b):
    if not a or not b or len(a) != len(b): return 0.0
    dot = sum(x*y for x,y in zip(a,b))
    na = sum(x*x for x in a)**0.5
    nb = sum(x*x for x in b)**0.5
    return dot/(na*nb) if na and nb else 0.0

# Test cases
VULN_TESTS = [
    ("SQL Injection", 'query = "SELECT * FROM users WHERE id = " + user_id', "vulnerable"),
    ("Command Injection", 'os.system("ping " + hostname)', "vulnerable"),
    ("Buffer Overflow", 'strcpy(dest, src);', "vulnerable"),
    ("XSS", 'innerHTML = "<div>" + userData + "</div>";', "vulnerable"),
    ("Path Traversal", 'open("/var/data/" + filename).read()', "vulnerable"),
    ("Safe Param SQL", 'cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))', "safe"),
    ("Safe Subprocess", 'subprocess.run(["ping", hostname], shell=False)', "safe"),
]

TRUTH_TESTS = [
    ("Fake Citation", "According to Smith (2024) in Nature, quantum computing achieved consciousness.", "hallucinated"),
    ("Factual Error", "Q: Capital of Australia?\nA: Sydney is the capital.", "hallucinated"),
    ("Real Fact", "Q: Capital of France?\nA: Paris is the capital of France.", "truthful"),
    ("Science Fact", "The speed of light is approximately 299,792 km/s.", "truthful"),
]

def test_model(model_name: str):
    print(f"\n{'='*50}")
    print(f"Testing with: {model_name}")
    print(f"{'='*50}")

    # Load patterns
    bcell_file = MODELS_DIR / f"bcell_overnight_{model_name}.json"
    nkcell_file = MODELS_DIR / f"nkcell_overnight_{model_name}.json"

    if not bcell_file.exists():
        print(f"  B Cell model not found: {bcell_file}")
        return None

    with open(bcell_file) as f:
        bcell = json.load(f).get("patterns", [])
    print(f"  B Cell patterns: {len(bcell)}")

    with open(nkcell_file) as f:
        nkcell = json.load(f).get("self_patterns", [])
    print(f"  NK Cell patterns: {len(nkcell)}")

    # B Cell tests
    print("\n--- B Cell (Vulnerability Detection) ---")
    bcell_correct = 0
    for name, code, expected in VULN_TESTS:
        emb = get_embedding(code, model_name)
        if not emb:
            print(f"  [SKIP] {name}: No embedding")
            continue

        # Vote among top 5 matches
        scores = [(cosine_sim(emb, p["embedding"]), p["label"]) for p in bcell if p.get("embedding")]
        scores.sort(reverse=True)
        top5 = scores[:5]
        vuln = sum(1 for _, l in top5 if l == "vulnerable")
        safe = sum(1 for _, l in top5 if l == "safe")
        pred = "vulnerable" if vuln > safe else "safe"

        ok = pred == expected
        bcell_correct += int(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {pred} ({vuln}v/{safe}s) sim={top5[0][0]:.2f}")

    # NK Cell tests
    print("\n--- NK Cell (Hallucination Detection) ---")
    nkcell_correct = 0
    for name, text, expected in TRUTH_TESTS:
        emb = get_embedding(text, model_name)
        if not emb:
            print(f"  [SKIP] {name}: No embedding")
            continue

        # Max similarity to self patterns
        max_self = max((cosine_sim(emb, p["embedding"]) for p in nkcell if p.get("embedding")), default=0)

        # Higher threshold for larger models (they're more discriminative)
        threshold = 0.75
        pred = "truthful" if max_self >= threshold else "hallucinated"

        ok = pred == expected
        nkcell_correct += int(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {pred} (self_sim={max_self:.2f}, thresh={threshold})")

    # Results
    bcell_acc = bcell_correct / len(VULN_TESTS) * 100
    nkcell_acc = nkcell_correct / len(TRUTH_TESTS) * 100
    overall = (bcell_correct + nkcell_correct) / (len(VULN_TESTS) + len(TRUTH_TESTS)) * 100

    print(f"\n--- Results ---")
    print(f"B Cell: {bcell_correct}/{len(VULN_TESTS)} ({bcell_acc:.0f}%)")
    print(f"NK Cell: {nkcell_correct}/{len(TRUTH_TESTS)} ({nkcell_acc:.0f}%)")
    print(f"Overall: {overall:.0f}%")

    return {"bcell": bcell_acc, "nkcell": nkcell_acc, "overall": overall}

def main():
    print("ANTIGENCE QUICK SMOKE TEST")
    print("Testing overnight models with larger embeddings")

    results = {}
    for model in ["nomic-embed-text", "mxbai-embed-large", "bge-m3"]:
        r = test_model(model)
        if r:
            results[model] = r

    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    for m, r in results.items():
        print(f"{m:25s} B={r['bcell']:3.0f}% NK={r['nkcell']:3.0f}% Total={r['overall']:3.0f}%")

if __name__ == "__main__":
    main()
