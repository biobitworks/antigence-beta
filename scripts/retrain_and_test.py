#!/usr/bin/env python3
"""
Retrain agents with new data and run smoke tests.

Orchestrates:
1. Load all datasets (including new Python vulns and negative examples)
2. Retrain B Cell and NK Cell with larger embedding models
3. Run comprehensive smoke tests
"""

import json
import csv
import sys
import time
import random
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.home() / ".antigence" / "data"
MODELS_DIR = Path.home() / ".antigence" / "trained"

# Use larger embedding models
EMBEDDING_MODELS = ["mxbai-embed-large", "bge-m3"]


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def get_embedding(text: str, model: str) -> list:
    import requests
    try:
        resp = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": text[:2000]},
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json().get("embedding", [])
    except:
        pass
    return []


def cosine_similarity(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def load_all_code_samples() -> list:
    """Load all code vulnerability samples."""
    samples = []

    # Big-Vul
    csv_path = DATA_DIR / "bigvul" / "all_c_cpp_release2.0.csv"
    if csv_path.exists():
        log("Loading Big-Vul...")
        csv.field_size_limit(sys.maxsize)
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 5000:
                    break
                code = row.get("func_before", "") or row.get("files_changed", "")
                if code and len(code) > 50:
                    samples.append({"code": code[:2000], "label": "vulnerable", "cwe": row.get("cwe_id", ""), "source": "bigvul"})
                safe = row.get("func_after", "")
                if safe and len(safe) > 50:
                    samples.append({"code": safe[:2000], "label": "safe", "source": "bigvul"})
        log(f"  Loaded {len(samples)} Big-Vul samples")

    # CodeXGLUE
    codexglue_path = DATA_DIR / "codexglue"
    if codexglue_path.exists():
        log("Loading CodeXGLUE...")
        try:
            from datasets import load_from_disk
            ds = load_from_disk(str(codexglue_path))
            count = 0
            for split in ds:
                for item in ds[split]:
                    func = item.get("func", "")
                    if len(func) > 50:
                        samples.append({"code": func[:2000], "label": "vulnerable" if item.get("target") == 1 else "safe", "source": "codexglue"})
                        count += 1
            log(f"  Loaded {count} CodeXGLUE samples")
        except Exception as e:
            log(f"  Error: {e}")

    # Python vulns
    pyvul_path = DATA_DIR / "python_vulns" / "samples.json"
    if pyvul_path.exists():
        log("Loading Python vulns...")
        with open(pyvul_path) as f:
            pyvul = json.load(f)
            for s in pyvul:
                samples.append({"code": s["code"], "label": s["label"], "cwe": s.get("cwe", ""), "source": "python_vulns"})
            log(f"  Loaded {len(pyvul)} Python vuln samples")

    # CWE samples
    cwe_path = DATA_DIR / "cwe_samples" / "samples.json"
    if cwe_path.exists():
        log("Loading CWE samples...")
        with open(cwe_path) as f:
            cwe = json.load(f)
            for s in cwe:
                samples.append({"code": s["code"], "label": s["label"], "cwe": s.get("cwe", ""), "source": "cwe_samples"})
            log(f"  Loaded {len(cwe)} CWE samples")

    return samples


def load_all_truth_samples() -> list:
    """Load all truthfulness samples including negatives."""
    samples = []

    # TruthfulQA
    tqa_path = DATA_DIR / "truthfulqa"
    if tqa_path.exists():
        log("Loading TruthfulQA...")
        try:
            from datasets import load_from_disk
            ds = load_from_disk(str(tqa_path))
            for split in ds:
                for item in ds[split]:
                    q = item.get("question", "")
                    mc1 = item.get("mc1_targets", {})
                    for choice, label in zip(mc1.get("choices", []), mc1.get("labels", [])):
                        samples.append({"text": f"Q: {q}\nA: {choice}", "label": "truthful" if label == 1 else "hallucinated", "source": "truthfulqa"})
            log(f"  Loaded {len(samples)} TruthfulQA samples")
        except Exception as e:
            log(f"  Error: {e}")

    # Negative examples
    neg_path = DATA_DIR / "negative_examples" / "samples.json"
    if neg_path.exists():
        log("Loading negative examples...")
        with open(neg_path) as f:
            negs = json.load(f)
            for s in negs:
                samples.append({"text": s["text"], "label": s["label"], "source": "negative_examples"})
            log(f"  Loaded {len(negs)} negative examples")

    return samples


def train_bcell(samples: list, model: str) -> str:
    """Train B Cell with specified embedding model."""
    log(f"\nTraining B Cell with {model}...")

    random.shuffle(samples)
    train = samples[:int(len(samples) * 0.8)]

    patterns = []
    for i, s in enumerate(train):
        if i % 500 == 0:
            log(f"  Progress: {i}/{len(train)}")
        code = s.get("code", "")
        if not code:
            continue
        emb = get_embedding(code, model)
        if emb:
            patterns.append({"label": s["label"], "cwe": s.get("cwe", ""), "embedding": emb, "source": s.get("source", "")})

    output = MODELS_DIR / f"bcell_v2_{model.replace(':', '-')}.json"
    with open(output, "w") as f:
        json.dump({"model": model, "patterns": patterns, "created": datetime.now().isoformat()}, f)

    log(f"  Saved {len(patterns)} patterns to {output.name}")
    return str(output)


def train_nkcell(samples: list, model: str) -> str:
    """Train NK Cell with self AND non-self patterns."""
    log(f"\nTraining NK Cell with {model}...")

    self_samples = [s for s in samples if s["label"] in ["truthful", "safe", "supported"]]
    nonself_samples = [s for s in samples if s["label"] in ["hallucinated", "unsupported"]]

    log(f"  Self: {len(self_samples)}, Non-self: {len(nonself_samples)}")

    self_patterns = []
    for i, s in enumerate(self_samples[:500]):
        if i % 100 == 0:
            log(f"  Self progress: {i}/500")
        text = s.get("text", "")
        if not text:
            continue
        emb = get_embedding(text, model)
        if emb:
            self_patterns.append({"embedding": emb, "source": s.get("source", "")})

    nonself_patterns = []
    for i, s in enumerate(nonself_samples[:500]):
        if i % 100 == 0:
            log(f"  Non-self progress: {i}/500")
        text = s.get("text", "")
        if not text:
            continue
        emb = get_embedding(text, model)
        if emb:
            nonself_patterns.append({"embedding": emb, "source": s.get("source", "")})

    output = MODELS_DIR / f"nkcell_v2_{model.replace(':', '-')}.json"
    with open(output, "w") as f:
        json.dump({
            "model": model,
            "self_patterns": self_patterns,
            "nonself_patterns": nonself_patterns,
            "created": datetime.now().isoformat()
        }, f)

    log(f"  Saved {len(self_patterns)} self + {len(nonself_patterns)} non-self patterns")
    return str(output)


# ============== SMOKE TEST ==============

VULN_TESTS = [
    {"name": "SQL Injection (Python)", "code": 'query = "SELECT * FROM users WHERE id = " + user_id', "expected": "vulnerable", "cwe": "CWE-89"},
    {"name": "Command Injection", "code": 'os.system("ping " + hostname)', "expected": "vulnerable", "cwe": "CWE-78"},
    {"name": "Buffer Overflow", "code": 'strcpy(dest, src);', "expected": "vulnerable", "cwe": "CWE-120"},
    {"name": "XSS", "code": 'innerHTML = "<div>" + userData + "</div>";', "expected": "vulnerable", "cwe": "CWE-79"},
    {"name": "Path Traversal", "code": 'open("/var/data/" + filename).read()', "expected": "vulnerable", "cwe": "CWE-22"},
    {"name": "Safe Parameterized", "code": 'cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))', "expected": "safe"},
    {"name": "Safe Subprocess", "code": 'subprocess.run(["ping", hostname], shell=False)', "expected": "safe"},
]

TRUTH_TESTS = [
    {"name": "Fake Citation", "text": "According to Smith et al. (2024) in Nature, quantum computing achieved consciousness.", "expected": "hallucinated"},
    {"name": "Factual Error", "text": "Q: What is the capital of Australia?\nA: Sydney is the capital.", "expected": "hallucinated"},
    {"name": "Real Fact", "text": "Q: What is the capital of France?\nA: Paris is the capital of France.", "expected": "truthful"},
    {"name": "Science Fact", "text": "The speed of light is approximately 299,792 km/s.", "expected": "truthful"},
]


def run_smoke_test(bcell_file: str, nkcell_file: str, model: str):
    """Run smoke test with trained models."""
    log(f"\n{'='*60}")
    log(f"SMOKE TEST: {model}")
    log(f"{'='*60}")

    # Load patterns
    with open(bcell_file) as f:
        bcell_data = json.load(f)
    bcell_patterns = bcell_data.get("patterns", [])

    with open(nkcell_file) as f:
        nkcell_data = json.load(f)
    self_patterns = nkcell_data.get("self_patterns", [])
    nonself_patterns = nkcell_data.get("nonself_patterns", [])

    log(f"B Cell: {len(bcell_patterns)} patterns")
    log(f"NK Cell: {len(self_patterns)} self, {len(nonself_patterns)} non-self")

    # B Cell tests
    log("\n--- B Cell Tests ---")
    bcell_correct = 0
    for test in VULN_TESTS:
        emb = get_embedding(test["code"], model)
        if not emb:
            continue

        # Find top matches
        scores = []
        for p in bcell_patterns:
            sim = cosine_similarity(emb, p["embedding"])
            scores.append((sim, p["label"]))
        scores.sort(reverse=True)
        top5 = scores[:5]

        vuln_votes = sum(1 for _, l in top5 if l == "vulnerable")
        safe_votes = sum(1 for _, l in top5 if l == "safe")
        pred = "vulnerable" if vuln_votes > safe_votes else "safe"

        correct = pred == test["expected"]
        bcell_correct += int(correct)
        status = "PASS" if correct else "FAIL"
        log(f"  [{status}] {test['name']}: {pred} (votes: {vuln_votes}v/{safe_votes}s)")

    # NK Cell tests
    log("\n--- NK Cell Tests ---")
    nkcell_correct = 0
    for test in TRUTH_TESTS:
        emb = get_embedding(test["text"], model)
        if not emb:
            continue

        # Check similarity to self patterns
        max_self = max((cosine_similarity(emb, p["embedding"]) for p in self_patterns), default=0)
        # Check similarity to non-self patterns
        max_nonself = max((cosine_similarity(emb, p["embedding"]) for p in nonself_patterns), default=0)

        # Prediction: if more similar to non-self, it's hallucinated
        pred = "hallucinated" if max_nonself > max_self else "truthful"

        correct = pred == test["expected"]
        nkcell_correct += int(correct)
        status = "PASS" if correct else "FAIL"
        log(f"  [{status}] {test['name']}: {pred} (self:{max_self:.2f} vs nonself:{max_nonself:.2f})")

    # Summary
    bcell_acc = bcell_correct / len(VULN_TESTS) * 100
    nkcell_acc = nkcell_correct / len(TRUTH_TESTS) * 100
    overall = (bcell_correct + nkcell_correct) / (len(VULN_TESTS) + len(TRUTH_TESTS)) * 100

    log(f"\n--- Results for {model} ---")
    log(f"B Cell: {bcell_correct}/{len(VULN_TESTS)} ({bcell_acc:.0f}%)")
    log(f"NK Cell: {nkcell_correct}/{len(TRUTH_TESTS)} ({nkcell_acc:.0f}%)")
    log(f"Overall: {overall:.0f}%")

    return {"bcell": bcell_acc, "nkcell": nkcell_acc, "overall": overall}


def main():
    log("=" * 60)
    log("ANTIGENCE RETRAIN AND TEST")
    log("=" * 60)

    # Load data
    code_samples = load_all_code_samples()
    truth_samples = load_all_truth_samples()

    log(f"\nTotal code samples: {len(code_samples)}")
    log(f"Total truth samples: {len(truth_samples)}")

    results = {}

    for model in EMBEDDING_MODELS:
        log(f"\n{'#'*60}")
        log(f"# MODEL: {model}")
        log(f"{'#'*60}")

        # Train
        bcell_file = train_bcell(code_samples, model)
        nkcell_file = train_nkcell(truth_samples, model)

        # Test
        results[model] = run_smoke_test(bcell_file, nkcell_file, model)

    # Final summary
    log("\n" + "=" * 60)
    log("FINAL RESULTS")
    log("=" * 60)
    for model, r in results.items():
        log(f"{model}: B={r['bcell']:.0f}% NK={r['nkcell']:.0f}% Overall={r['overall']:.0f}%")


if __name__ == "__main__":
    main()
