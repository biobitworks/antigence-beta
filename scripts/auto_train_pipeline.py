#!/usr/bin/env python3
"""
Automated training pipeline - monitors for downloads and trains when ready.
Offloads embedding generation to Ollama.

Usage:
    python scripts/auto_train_pipeline.py          # Run full pipeline
    python scripts/auto_train_pipeline.py --check  # Just check status
"""

import json
import csv
import sys
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.home() / ".antigence" / "data"
MODELS_DIR = Path.home() / ".antigence" / "trained"
MODEL = "mxbai-embed-large"

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def get_embedding(text):
    import requests
    try:
        r = requests.post("http://localhost:11434/api/embeddings",
            json={"model": MODEL, "prompt": text[:2000]}, timeout=60)
        if r.status_code == 200:
            return r.json().get("embedding", [])
    except: pass
    return []

def cosine(a, b):
    if not a or not b or len(a) != len(b): return 0
    dot = sum(x*y for x,y in zip(a,b))
    return dot / ((sum(x*x for x in a)**0.5) * (sum(x*x for x in b)**0.5))

# ============ DATA LOADING ============

def load_primevul():
    """Load PrimeVul from JSONL files."""
    primevul_dir = DATA_DIR / "primevul"
    samples = []

    for jsonl in primevul_dir.glob("**/*.jsonl"):
        log(f"  Loading {jsonl.name}...")
        with open(jsonl, 'r', errors='ignore') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    func = item.get('func', '') or item.get('code', '')
                    target = item.get('target', 0)
                    if func and len(func) > 50:
                        samples.append({
                            "code": func[:3000],
                            "label": "vulnerable" if str(target) == "1" else "safe",
                            "cwe": item.get('cwe', ''),
                            "source": "primevul"
                        })
                except: pass
    return samples

def load_diversevul():
    """Load DiverseVul from JSONL (one JSON per line)."""
    diversevul_dir = DATA_DIR / "diversevul"
    samples = []

    for jf in diversevul_dir.glob("**/*.json"):
        if jf.name in ["samples.json", "README.md"]:
            continue
        log(f"  Loading {jf.name} (JSONL format)...")
        try:
            with open(jf, 'r', errors='ignore') as f:
                for line_num, line in enumerate(f):
                    if line_num % 5000 == 0 and line_num > 0:
                        log(f"    Progress: {line_num} lines, {len(samples)} samples")
                    try:
                        item = json.loads(line.strip())
                        func = item.get('func', '') or item.get('code', '')
                        if func and len(func) > 50:
                            samples.append({
                                "code": func[:3000],
                                "label": "vulnerable",  # DiverseVul is all vulnerable
                                "cwe": item.get('cwe', ''),
                                "source": "diversevul"
                            })
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            log(f"    Error: {e}")
    log(f"  Loaded {len(samples)} DiverseVul samples")
    return samples

def load_existing():
    """Load existing datasets."""
    samples = []

    # CodeXGLUE
    codexglue = DATA_DIR / "codexglue"
    if codexglue.exists():
        try:
            from datasets import load_from_disk
            ds = load_from_disk(str(codexglue))
            for split in ds:
                for item in ds[split]:
                    func = item.get('func', '')
                    if func and len(func) > 50:
                        samples.append({
                            "code": func[:3000],
                            "label": "vulnerable" if item.get('target') == 1 else "safe",
                            "source": "codexglue"
                        })
            log(f"  CodeXGLUE: {len([s for s in samples if s['source']=='codexglue'])} samples")
        except: pass

    # Curated datasets
    for name in ["juliet", "sard", "python_vulns", "cwe_samples"]:
        f = DATA_DIR / name / "samples.json"
        if f.exists():
            with open(f) as fp:
                data = json.load(fp)
                for s in data:
                    samples.append({
                        "code": s.get("code", ""),
                        "label": s.get("label", ""),
                        "cwe": s.get("cwe", ""),
                        "source": name
                    })
            log(f"  {name}: {len(data)} samples")

    return samples

def check_downloads():
    """Check if manual downloads are available."""
    primevul_ready = any((DATA_DIR / "primevul").glob("**/*.jsonl"))
    diversevul_ready = any((DATA_DIR / "diversevul").glob("**/*.json")) and \
                       not all(f.name in ["README.md"] for f in (DATA_DIR / "diversevul").glob("*"))

    return primevul_ready, diversevul_ready

# ============ TRAINING ============

def train_bcell(samples, max_samples=5000):
    """Train B Cell with Ollama embeddings."""
    log(f"\nTraining B Cell on {len(samples)} samples (max {max_samples})...")

    # Balance
    vuln = [s for s in samples if s['label'] == 'vulnerable']
    safe = [s for s in samples if s['label'] == 'safe']
    n = min(len(vuln), len(safe), max_samples // 2)
    balanced = random.sample(vuln, n) + random.sample(safe, n)
    random.shuffle(balanced)
    log(f"  Balanced: {len(balanced)} samples ({n} each)")

    # Generate embeddings via Ollama
    patterns = []
    for i, s in enumerate(balanced):
        if i % 200 == 0:
            log(f"  Progress: {i}/{len(balanced)}")
        emb = get_embedding(s['code'])
        if emb:
            patterns.append({
                "label": s['label'],
                "cwe": s.get('cwe', ''),
                "source": s.get('source', ''),
                "embedding": emb
            })

    # Save
    output = MODELS_DIR / "bcell_full_pipeline.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump({
            "model": MODEL,
            "patterns": patterns,
            "created": datetime.now().isoformat(),
            "sources": list(set(s.get('source', '') for s in balanced))
        }, f)

    log(f"  Saved {len(patterns)} patterns to {output.name}")
    return patterns

# ============ TESTING ============

TESTS = [
    ("SQL Injection (Py)", 'query = "SELECT * FROM users WHERE id = " + user_id', "vulnerable"),
    ("SQL Injection (Java)", 'stmt.executeQuery("SELECT * FROM users WHERE id=" + userId);', "vulnerable"),
    ("Command Injection", 'system(argv[1]);', "vulnerable"),
    ("Buffer Overflow", 'strcpy(dest, src);', "vulnerable"),
    ("Format String", 'printf(userInput);', "vulnerable"),
    ("Use After Free", 'free(ptr); ptr->data = 0;', "vulnerable"),
    ("Path Traversal", 'fopen(basePath + userFile, "r");', "vulnerable"),
    ("XSS", 'document.innerHTML = userInput;', "vulnerable"),
    ("Safe SQL", 'cursor.execute("SELECT * FROM users WHERE id=?", (uid,))', "safe"),
    ("Safe Copy", 'strncpy(buf, src, sizeof(buf)-1);', "safe"),
    ("Safe Print", 'printf("%s", userInput);', "safe"),
    ("Safe Free", 'free(ptr); ptr = NULL;', "safe"),
]

def run_smoke_test(patterns):
    """Run smoke test on trained patterns."""
    log("\n" + "="*50)
    log("SMOKE TEST")
    log("="*50)

    correct = 0
    for name, code, expected in TESTS:
        emb = get_embedding(code)
        if not emb:
            print(f"[SKIP] {name}")
            continue

        scores = [(cosine(emb, p["embedding"]), p["label"]) for p in patterns]
        scores.sort(reverse=True)
        v = sum(1 for _, l in scores[:5] if l == "vulnerable")
        s = sum(1 for _, l in scores[:5] if l == "safe")
        pred = "vulnerable" if v > s else "safe"

        ok = pred == expected
        correct += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {pred} ({v}v/{s}s)")

    acc = correct * 100 // len(TESTS)
    log(f"\nAccuracy: {correct}/{len(TESTS)} ({acc}%)")
    return acc

# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Just check status")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max training samples")
    args = parser.parse_args()

    log("="*60)
    log("ANTIGENCE AUTO-TRAINING PIPELINE")
    log("="*60)

    # Check download status
    primevul_ready, diversevul_ready = check_downloads()
    log(f"\nDownload Status:")
    log(f"  PrimeVul: {'✓ Ready' if primevul_ready else '○ Waiting'}")
    log(f"  DiverseVul: {'✓ Ready' if diversevul_ready else '○ Waiting'}")

    if args.check:
        return

    # Load all available data
    log("\nLoading datasets...")
    all_samples = []

    all_samples.extend(load_existing())

    if primevul_ready:
        all_samples.extend(load_primevul())

    if diversevul_ready:
        all_samples.extend(load_diversevul())

    # Stats
    vuln = sum(1 for s in all_samples if s['label'] == 'vulnerable')
    safe = sum(1 for s in all_samples if s['label'] == 'safe')
    log(f"\nTotal: {len(all_samples)} samples ({vuln} vuln, {safe} safe)")

    by_source = {}
    for s in all_samples:
        src = s.get('source', 'unknown')
        by_source[src] = by_source.get(src, 0) + 1
    for src, cnt in sorted(by_source.items(), key=lambda x: -x[1]):
        log(f"  {src}: {cnt}")

    if len(all_samples) < 100:
        log("\nNot enough samples. Waiting for downloads...")
        return

    # Train
    patterns = train_bcell(all_samples, args.max_samples)

    # Test
    acc = run_smoke_test(patterns)

    log("\n" + "="*60)
    log("PIPELINE COMPLETE")
    log(f"Final Accuracy: {acc}%")
    log("="*60)


if __name__ == "__main__":
    main()
