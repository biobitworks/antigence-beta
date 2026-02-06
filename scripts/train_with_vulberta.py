#!/usr/bin/env python3
"""
Train B Cell using VulBERTa as a specialized antibody.

VulBERTa is pre-trained on C/C++ vulnerabilities - use it as an expert antibody.
Combine with general embeddings for ensemble detection.
"""

import json
import random
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.home() / ".antigence" / "data"
MODELS_DIR = Path.home() / ".antigence" / "trained"
VULBERTA_DIR = Path.home() / ".antigence" / "models" / "vulberta"

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_ollama_embedding(text, model="mxbai-embed-large"):
    """Get embedding from Ollama."""
    import requests
    try:
        r = requests.post("http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": text[:2000]}, timeout=60)
        if r.status_code == 200:
            return r.json().get("embedding", [])
    except:
        pass
    return []


def get_vulberta_prediction(text):
    """Get vulnerability prediction from VulBERTa model."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        # Load model (cache it)
        if not hasattr(get_vulberta_prediction, 'model'):
            log("  Loading VulBERTa model...")
            get_vulberta_prediction.tokenizer = AutoTokenizer.from_pretrained(str(VULBERTA_DIR))
            get_vulberta_prediction.model = AutoModelForSequenceClassification.from_pretrained(str(VULBERTA_DIR))
            get_vulberta_prediction.model.eval()

        tokenizer = get_vulberta_prediction.tokenizer
        model = get_vulberta_prediction.model

        inputs = tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            # Class 1 = vulnerable
            vuln_prob = probs[0][1].item()
            return vuln_prob
    except Exception as e:
        log(f"    VulBERTa error: {e}")
        return None


def cosine(a, b):
    if not a or not b or len(a) != len(b): return 0
    dot = sum(x*y for x,y in zip(a,b))
    return dot / ((sum(x*x for x in a)**0.5) * (sum(x*x for x in b)**0.5))


def main():
    log("="*60)
    log("TRAINING WITH VULBERTA ANTIBODY")
    log("="*60)

    # Load combined training data
    samples_file = DATA_DIR / "combined_training" / "samples.json"
    if not samples_file.exists():
        log("No combined training data found!")
        return

    with open(samples_file) as f:
        all_samples = json.load(f)

    log(f"Loaded {len(all_samples)} samples")

    # Balance the dataset
    vuln = [s for s in all_samples if s['label'] == 'vulnerable']
    safe = [s for s in all_samples if s['label'] == 'safe']
    n = min(len(vuln), len(safe), 2000)  # Limit for speed
    balanced = random.sample(vuln, n) + random.sample(safe, n)
    random.shuffle(balanced)
    log(f"Balanced: {len(balanced)} samples ({n} each)")

    # Create patterns with both Ollama embeddings AND VulBERTa scores
    log("\nGenerating embeddings + VulBERTa scores...")
    patterns = []
    for i, s in enumerate(balanced):
        if i % 100 == 0:
            log(f"  {i}/{len(balanced)}")

        code = s.get('code', '')
        if not code or len(code) < 20:
            continue

        # Get Ollama embedding
        emb = get_ollama_embedding(code)
        if not emb:
            continue

        # Get VulBERTa prediction (expert antibody)
        vulberta_score = get_vulberta_prediction(code)

        patterns.append({
            "label": s['label'],
            "cwe": s.get('cwe', ''),
            "lang": s.get('lang', ''),
            "source": s.get('source', ''),
            "embedding": emb,
            "vulberta_score": vulberta_score
        })

    # Save patterns
    output_file = MODELS_DIR / "bcell_vulberta_ensemble.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model": "mxbai-embed-large + vulberta",
            "patterns": patterns,
            "created": datetime.now().isoformat()
        }, f)
    log(f"\nSaved {len(patterns)} patterns with VulBERTa scores")

    # ========== SMOKE TEST ==========
    log("\n" + "="*60)
    log("ENSEMBLE SMOKE TEST")
    log("="*60)

    TESTS = [
        ("SQL Injection (Python)", 'query = "SELECT * FROM users WHERE id = " + user_id', "vulnerable"),
        ("SQL Injection (Java)", 'stmt.executeQuery("SELECT * FROM users WHERE id=" + userId);', "vulnerable"),
        ("Command Injection", 'os.system("ping " + hostname)', "vulnerable"),
        ("Buffer Overflow", 'strcpy(dest, src);', "vulnerable"),
        ("XSS", 'innerHTML = "<div>" + userData + "</div>";', "vulnerable"),
        ("Path Traversal", 'open("/var/data/" + filename).read()', "vulnerable"),
        ("Format String", 'printf(userInput);', "vulnerable"),
        ("Use After Free", 'free(ptr); ptr->value = 0;', "vulnerable"),
        ("Safe SQL", 'cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))', "safe"),
        ("Safe Copy", 'strncpy(buf, src, sizeof(buf)-1);', "safe"),
        ("Safe Subprocess", 'subprocess.run(["ping", hostname], shell=False)', "safe"),
        ("Safe Print", 'printf("%s", userInput);', "safe"),
    ]

    correct = 0
    for name, code, expected in TESTS:
        # Get test embedding
        emb = get_ollama_embedding(code)
        vulberta = get_vulberta_prediction(code)

        if not emb:
            print(f"[SKIP] {name}")
            continue

        # Embedding-based voting
        scores = [(cosine(emb, p["embedding"]), p["label"]) for p in patterns]
        scores.sort(reverse=True)
        top5 = scores[:5]
        emb_vuln = sum(1 for _, l in top5 if l == "vulnerable")
        emb_safe = sum(1 for _, l in top5 if l == "safe")
        emb_pred = "vulnerable" if emb_vuln > emb_safe else "safe"

        # VulBERTa prediction (threshold 0.5)
        vulberta_pred = "vulnerable" if vulberta and vulberta > 0.5 else "safe"

        # Ensemble: agree = high confidence, disagree = use VulBERTa (expert)
        if emb_pred == vulberta_pred:
            final_pred = emb_pred
            method = "agree"
        else:
            # Trust VulBERTa for C/C++ code, embedding for others
            if any(kw in code for kw in ['strcpy', 'malloc', 'free', 'printf', 'scanf']):
                final_pred = vulberta_pred
                method = "vulberta"
            else:
                final_pred = emb_pred
                method = "embed"

        ok = final_pred == expected
        correct += int(ok)
        vb_str = f"VB={vulberta:.2f}" if vulberta else "VB=N/A"
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {final_pred} ({emb_vuln}v/{emb_safe}s, {vb_str}, {method})")

    log(f"\nAccuracy: {correct}/{len(TESTS)} ({correct*100//len(TESTS)}%)")


if __name__ == "__main__":
    main()
