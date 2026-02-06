#!/usr/bin/env python3
"""Final training with all datasets and comprehensive smoke test."""

import json, random
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.home() / ".antigence" / "data"
MODELS_DIR = Path.home() / ".antigence" / "trained"
MODEL = "mxbai-embed-large"

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

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

def load_json_dataset(name):
    p = DATA_DIR / name / "samples.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    return []

def main():
    log("="*60)
    log("FINAL TRAINING WITH ALL DATASETS")
    log("="*60)

    # Load all code samples
    code = []
    for ds in ["juliet", "sard", "python_vulns", "cwe_samples"]:
        samples = load_json_dataset(ds)
        code.extend(samples)
        log(f"  {ds}: {len(samples)} samples")

    vuln = [s for s in code if s.get("label") == "vulnerable"]
    safe = [s for s in code if s.get("label") == "safe"]
    log(f"\nTotal: {len(vuln)} vulnerable, {len(safe)} safe")

    # Balance
    n = min(len(vuln), len(safe))
    balanced = random.sample(vuln, n) + random.sample(safe, n)
    random.shuffle(balanced)
    log(f"Balanced: {len(balanced)} samples")

    # Train B Cell
    log("\nTraining B Cell...")
    patterns = []
    for i, s in enumerate(balanced):
        if i % 30 == 0: log(f"  {i}/{len(balanced)}")
        emb = get_emb(s.get("code", ""))
        if emb:
            patterns.append({"label": s["label"], "cwe": s.get("cwe", ""), "lang": s.get("lang", ""), "embedding": emb})

    bcell_file = MODELS_DIR / "bcell_final.json"
    with open(bcell_file, "w") as f:
        json.dump({"model": MODEL, "patterns": patterns, "created": datetime.now().isoformat()}, f)
    log(f"  Saved {len(patterns)} B Cell patterns")

    # Load truth samples
    truth = load_json_dataset("negative_examples")
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(DATA_DIR / "truthfulqa"))
        for split in ds:
            for item in ds[split]:
                q = item.get("question", "")
                mc1 = item.get("mc1_targets", {})
                for choice, label in zip(mc1.get("choices", []), mc1.get("labels", [])):
                    truth.append({"text": f"Q: {q}\nA: {choice}", "label": "truthful" if label == 1 else "hallucinated"})
    except: pass

    truthful = [s for s in truth if s.get("label") in ["truthful", "supported"]]
    halluc = [s for s in truth if s.get("label") == "hallucinated"]
    log(f"\nTruth samples: {len(truthful)} truthful, {len(halluc)} hallucinated")

    # Train NK Cell
    log("\nTraining NK Cell...")
    self_p, nonself_p = [], []
    for i, s in enumerate(truthful[:300]):
        if i % 100 == 0: log(f"  Self: {i}/300")
        emb = get_emb(s.get("text", ""))
        if emb: self_p.append({"embedding": emb})

    for i, s in enumerate(halluc[:300]):
        if i % 100 == 0: log(f"  Non-self: {i}/300")
        emb = get_emb(s.get("text", ""))
        if emb: nonself_p.append({"embedding": emb})

    nkcell_file = MODELS_DIR / "nkcell_final.json"
    with open(nkcell_file, "w") as f:
        json.dump({"model": MODEL, "self_patterns": self_p, "nonself_patterns": nonself_p, "created": datetime.now().isoformat()}, f)
    log(f"  Saved {len(self_p)} self + {len(nonself_p)} non-self patterns")

    # ========== SMOKE TEST ==========
    log("\n" + "="*60)
    log("COMPREHENSIVE SMOKE TEST")
    log("="*60)

    TESTS = [
        # Vulnerable
        ("SQL Injection (Python)", 'query = "SELECT * FROM users WHERE id = " + user_id', "vulnerable"),
        ("SQL Injection (Java)", 'stmt.executeQuery("SELECT * FROM users WHERE id=" + userId);', "vulnerable"),
        ("Command Injection (C)", 'system(argv[1]);', "vulnerable"),
        ("Command Injection (Python)", 'os.system("ping " + hostname)', "vulnerable"),
        ("Buffer Overflow", 'strcpy(dest, src);', "vulnerable"),
        ("XSS", 'innerHTML = "<div>" + userData + "</div>";', "vulnerable"),
        ("Path Traversal", 'open("/var/data/" + filename).read()', "vulnerable"),
        ("Format String", 'printf(userInput);', "vulnerable"),
        # Safe
        ("Safe SQL (Python)", 'cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))', "safe"),
        ("Safe SQL (Java)", 'pstmt.setInt(1, userId);', "safe"),
        ("Safe Subprocess", 'subprocess.run(["ping", hostname], shell=False)', "safe"),
        ("Safe Copy", 'strncpy(buf, src, sizeof(buf)-1);', "safe"),
    ]

    TRUTH_TESTS = [
        ("Fake Citation", "According to Smith (2024) in Nature, quantum achieved consciousness.", "hallucinated"),
        ("Factual Error", "Q: Capital of Australia?\nA: Sydney is the capital.", "hallucinated"),
        ("Fake Stat", "Studies show 99% of scientists agree on this unproven claim.", "hallucinated"),
        ("Real Fact", "Q: Capital of France?\nA: Paris is the capital of France.", "truthful"),
        ("Science Fact", "DNA stands for deoxyribonucleic acid.", "truthful"),
        ("Math Fact", "The square root of 16 is 4.", "truthful"),
    ]

    # Test B Cell
    log("\n--- B CELL TESTS ---")
    bc = 0
    for name, code, expected in TESTS:
        emb = get_emb(code)
        if not emb:
            print(f"[SKIP] {name}")
            continue
        scores = [(cosine(emb, p["embedding"]), p["label"]) for p in patterns]
        scores.sort(reverse=True)
        v = sum(1 for _, l in scores[:5] if l == "vulnerable")
        s = sum(1 for _, l in scores[:5] if l == "safe")
        pred = "vulnerable" if v > s else "safe"
        ok = pred == expected
        bc += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {pred} ({v}v/{s}s)")

    # Test NK Cell
    log("\n--- NK CELL TESTS ---")
    nkc = 0
    for name, text, expected in TRUTH_TESTS:
        emb = get_emb(text)
        if not emb:
            print(f"[SKIP] {name}")
            continue
        max_self = max((cosine(emb, p["embedding"]) for p in self_p), default=0)
        max_nonself = max((cosine(emb, p["embedding"]) for p in nonself_p), default=0)
        pred = "hallucinated" if max_nonself > max_self else "truthful"
        ok = pred == expected
        nkc += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {pred} (self={max_self:.2f} vs nonself={max_nonself:.2f})")

    # Results
    log("\n" + "="*60)
    log("FINAL RESULTS")
    log("="*60)
    log(f"B Cell: {bc}/{len(TESTS)} ({bc*100//len(TESTS)}%)")
    log(f"NK Cell: {nkc}/{len(TRUTH_TESTS)} ({nkc*100//len(TRUTH_TESTS)}%)")
    total = (bc + nkc) / (len(TESTS) + len(TRUTH_TESTS)) * 100
    log(f"Overall: {total:.0f}%")


if __name__ == "__main__":
    main()
