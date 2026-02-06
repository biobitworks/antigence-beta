#!/usr/bin/env python3
"""Test balanced models."""
import json
from pathlib import Path

MODELS_DIR = Path.home() / ".antigence" / "trained"
MODEL = "mxbai-embed-large"

def get_emb(text):
    import requests
    try:
        r = requests.post("http://localhost:11434/api/embeddings", json={"model": MODEL, "prompt": text[:2000]}, timeout=60)
        if r.status_code == 200: return r.json().get("embedding", [])
    except: pass
    return []

def cosine(a, b):
    if not a or not b or len(a) != len(b): return 0
    dot = sum(x*y for x,y in zip(a,b))
    return dot / ((sum(x*x for x in a)**0.5) * (sum(x*x for x in b)**0.5))

VULN_TESTS = [
    ("SQL Injection", 'query = "SELECT * FROM users WHERE id = " + user_id', "vulnerable"),
    ("Command Injection", 'os.system("ping " + hostname)', "vulnerable"),
    ("Buffer Overflow", 'strcpy(dest, src);', "vulnerable"),
    ("XSS", 'innerHTML = "<div>" + userData + "</div>";', "vulnerable"),
    ("Parameterized SQL", 'cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))', "safe"),
    ("Safe Subprocess", 'subprocess.run(["ping", hostname], shell=False)', "safe"),
]

TRUTH_TESTS = [
    ("Fake Citation", "According to Smith (2024) in Nature, quantum computing achieved consciousness.", "hallucinated"),
    ("Factual Error", "Q: Capital of Australia?\nA: Sydney is the capital.", "hallucinated"),
    ("Real Fact", "Q: Capital of France?\nA: Paris is the capital of France.", "truthful"),
    ("Science Fact", "DNA stands for deoxyribonucleic acid.", "truthful"),
]

def main():
    print("="*50)
    print("TESTING BALANCED MODELS")
    print("="*50)

    # Load models
    with open(MODELS_DIR / "bcell_balanced.json") as f:
        bcell = json.load(f)["patterns"]
    with open(MODELS_DIR / "nkcell_balanced.json") as f:
        nk = json.load(f)
        self_p = nk["self_patterns"]
        nonself_p = nk["nonself_patterns"]

    print(f"B Cell: {len(bcell)} patterns")
    print(f"NK Cell: {len(self_p)} self, {len(nonself_p)} non-self\n")

    # B Cell tests
    print("--- B CELL (Vulnerability Detection) ---")
    bc = 0
    for name, code, expected in VULN_TESTS:
        emb = get_emb(code)
        if not emb:
            print(f"[SKIP] {name}")
            continue
        scores = [(cosine(emb, p["embedding"]), p["label"]) for p in bcell]
        scores.sort(reverse=True)
        top5 = scores[:5]
        v = sum(1 for _, l in top5 if l == "vulnerable")
        s = sum(1 for _, l in top5 if l == "safe")
        pred = "vulnerable" if v > s else "safe"
        ok = pred == expected
        bc += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {pred} ({v}v/{s}s)")

    # NK Cell tests - compare self vs non-self similarity
    print("\n--- NK CELL (Hallucination Detection) ---")
    nkc = 0
    for name, text, expected in TRUTH_TESTS:
        emb = get_emb(text)
        if not emb:
            print(f"[SKIP] {name}")
            continue
        max_self = max((cosine(emb, p["embedding"]) for p in self_p), default=0)
        max_nonself = max((cosine(emb, p["embedding"]) for p in nonself_p), default=0)
        # If more similar to non-self, it's hallucinated
        pred = "hallucinated" if max_nonself > max_self else "truthful"
        ok = pred == expected
        nkc += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {pred} (self={max_self:.2f} vs nonself={max_nonself:.2f})")

    # Results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"B Cell: {bc}/{len(VULN_TESTS)} ({bc*100//len(VULN_TESTS)}%)")
    print(f"NK Cell: {nkc}/{len(TRUTH_TESTS)} ({nkc*100//len(TRUTH_TESTS)}%)")
    total = (bc + nkc) / (len(VULN_TESTS) + len(TRUTH_TESTS)) * 100
    print(f"Overall: {total:.0f}%")

if __name__ == "__main__":
    main()
