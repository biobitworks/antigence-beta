#!/usr/bin/env python3
"""Retrain with balanced data + negative examples. Uses mxbai-embed-large (best performer)."""

import json, sys, random
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

# Load Python vulns
def load_python():
    p = DATA_DIR / "python_vulns" / "samples.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    return []

# Load CWE samples
def load_cwe():
    p = DATA_DIR / "cwe_samples" / "samples.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    return []

# Load negative examples
def load_negatives():
    p = DATA_DIR / "negative_examples" / "samples.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    return []

# Load TruthfulQA
def load_truthful():
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(DATA_DIR / "truthfulqa"))
        samples = []
        for split in ds:
            for item in ds[split]:
                q = item.get("question", "")
                mc1 = item.get("mc1_targets", {})
                for choice, label in zip(mc1.get("choices", []), mc1.get("labels", [])):
                    samples.append({"text": f"Q: {q}\nA: {choice}", "label": "truthful" if label == 1 else "hallucinated"})
        return samples
    except: return []

def main():
    log("="*50)
    log("BALANCED RETRAINING")
    log(f"Model: {MODEL}")
    log("="*50)

    # Load code samples
    log("\nLoading code samples...")
    code = load_python() + load_cwe()
    vuln = [s for s in code if s.get("label") == "vulnerable"]
    safe = [s for s in code if s.get("label") == "safe"]
    log(f"  Vulnerable: {len(vuln)}, Safe: {len(safe)}")

    # Balance: use min of both
    n = min(len(vuln), len(safe))
    balanced_code = random.sample(vuln, n) + random.sample(safe, n)
    random.shuffle(balanced_code)
    log(f"  Balanced: {len(balanced_code)} samples")

    # Train B Cell
    log("\nTraining B Cell...")
    patterns = []
    for i, s in enumerate(balanced_code):
        if i % 20 == 0: log(f"  {i}/{len(balanced_code)}")
        emb = get_emb(s.get("code", ""))
        if emb:
            patterns.append({"label": s["label"], "cwe": s.get("cwe", ""), "embedding": emb})

    bcell_file = MODELS_DIR / "bcell_balanced.json"
    with open(bcell_file, "w") as f:
        json.dump({"model": MODEL, "patterns": patterns, "created": datetime.now().isoformat()}, f)
    log(f"  Saved {len(patterns)} patterns to {bcell_file.name}")

    # Load truth samples
    log("\nLoading truth samples...")
    truth = load_truthful() + load_negatives()
    truthful = [s for s in truth if s.get("label") == "truthful"]
    halluc = [s for s in truth if s.get("label") == "hallucinated"]
    log(f"  Truthful: {len(truthful)}, Hallucinated: {len(halluc)}")

    # Train NK Cell with BOTH self and non-self
    log("\nTraining NK Cell (self + non-self)...")
    self_patterns = []
    for i, s in enumerate(truthful[:200]):
        if i % 50 == 0: log(f"  Self: {i}/200")
        emb = get_emb(s.get("text", ""))
        if emb: self_patterns.append({"embedding": emb})

    nonself_patterns = []
    for i, s in enumerate(halluc[:200]):
        if i % 50 == 0: log(f"  Non-self: {i}/200")
        emb = get_emb(s.get("text", ""))
        if emb: nonself_patterns.append({"embedding": emb})

    nkcell_file = MODELS_DIR / "nkcell_balanced.json"
    with open(nkcell_file, "w") as f:
        json.dump({"model": MODEL, "self_patterns": self_patterns, "nonself_patterns": nonself_patterns, "created": datetime.now().isoformat()}, f)
    log(f"  Saved {len(self_patterns)} self + {len(nonself_patterns)} non-self to {nkcell_file.name}")

    log("\n" + "="*50)
    log("RETRAINING COMPLETE")
    log("="*50)

if __name__ == "__main__":
    main()
