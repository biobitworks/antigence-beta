#!/usr/bin/env python3
"""
Train NK Cell for publication/citation verification.
Uses TruthfulQA, HaluEval, and PubMed abstracts.
"""

import json
import random
import requests
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.home() / ".antigence" / "data"
MODELS_DIR = Path.home() / ".antigence" / "trained"
MODEL = "mxbai-embed-large"

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def get_embedding(text):
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

def load_truthfulqa():
    """Load TruthfulQA - questions with correct/incorrect answers."""
    samples = []
    truthful_dir = DATA_DIR / "truthfulqa_gen"

    if not truthful_dir.exists():
        log("  TruthfulQA not found")
        return samples

    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(truthful_dir))

        for item in ds:
            question = item.get('question', '')
            best_answer = item.get('best_answer', '')
            incorrect = item.get('incorrect_answers', [])

            # Truthful: question + best answer
            if question and best_answer:
                samples.append({
                    "text": f"{question} {best_answer}",
                    "label": "truthful",
                    "source": "truthfulqa"
                })

            # Hallucinated: question + incorrect answers
            for ans in incorrect[:2]:  # Limit per question
                if ans:
                    samples.append({
                        "text": f"{question} {ans}",
                        "label": "hallucinated",
                        "source": "truthfulqa"
                    })

        log(f"  TruthfulQA: {len(samples)} samples")
    except Exception as e:
        log(f"  TruthfulQA error: {e}")

    return samples

def load_halueval():
    """Load HaluEval hallucination detection dataset."""
    samples = []

    configs = ['qa', 'dialogue', 'summarization', 'general']
    for config in configs:
        halueval_dir = DATA_DIR / f"halueval_{config}"
        if not halueval_dir.exists():
            continue

        try:
            from datasets import load_from_disk
            ds = load_from_disk(str(halueval_dir))

            for item in ds:
                # Different configs have different fields
                if config == 'qa':
                    text = item.get('question', '') + ' ' + item.get('hallucinated_answer', '')
                    if text.strip():
                        samples.append({"text": text, "label": "hallucinated", "source": f"halueval_{config}"})

                    text = item.get('question', '') + ' ' + item.get('right_answer', '')
                    if text.strip():
                        samples.append({"text": text, "label": "truthful", "source": f"halueval_{config}"})

                elif config == 'general':
                    text = item.get('user_query', '') + ' ' + item.get('hallucinated_response', '')
                    if text.strip():
                        samples.append({"text": text, "label": "hallucinated", "source": f"halueval_{config}"})

                    text = item.get('user_query', '') + ' ' + item.get('right_response', '')
                    if text.strip():
                        samples.append({"text": text, "label": "truthful", "source": f"halueval_{config}"})

                elif config == 'dialogue':
                    text = item.get('dialogue_history', '') + ' ' + item.get('hallucinated_response', '')
                    if text.strip():
                        samples.append({"text": text, "label": "hallucinated", "source": f"halueval_{config}"})

                    text = item.get('dialogue_history', '') + ' ' + item.get('right_response', '')
                    if text.strip():
                        samples.append({"text": text, "label": "truthful", "source": f"halueval_{config}"})

                elif config == 'summarization':
                    text = item.get('document', '')[:500] + ' ' + item.get('hallucinated_summary', '')
                    if text.strip():
                        samples.append({"text": text, "label": "hallucinated", "source": f"halueval_{config}"})

                    text = item.get('document', '')[:500] + ' ' + item.get('right_summary', '')
                    if text.strip():
                        samples.append({"text": text, "label": "truthful", "source": f"halueval_{config}"})

            log(f"  HaluEval {config}: {len([s for s in samples if s['source'] == f'halueval_{config}'])} samples")
        except Exception as e:
            log(f"  HaluEval {config} error: {e}")

    return samples

def load_pubmed():
    """Load PubMed abstracts as truthful examples."""
    samples = []
    pubmed_file = DATA_DIR / "pubmed_abstracts" / "samples.json"

    if pubmed_file.exists():
        with open(pubmed_file) as f:
            data = json.load(f)
        samples = data
        log(f"  PubMed: {len(samples)} samples")

    return samples

def load_negative_examples():
    """Load existing negative examples."""
    samples = []
    neg_file = DATA_DIR / "negative_examples" / "samples.json"

    if neg_file.exists():
        with open(neg_file) as f:
            data = json.load(f)
        for item in data:
            samples.append({
                "text": item.get('text', item.get('code', '')),
                "label": item.get('label', 'hallucinated'),
                "source": "negative_examples"
            })
        log(f"  Negative examples: {len(samples)} samples")

    return samples

# ============ TRAINING ============

def train_nkcell(samples, max_samples=5000):
    """Train NK Cell with self (truthful) and non-self (hallucinated) patterns."""
    log(f"\nTraining NK Cell on {len(samples)} samples (max {max_samples})...")

    # Separate by label
    truthful = [s for s in samples if s['label'] == 'truthful']
    hallucinated = [s for s in samples if s['label'] == 'hallucinated']

    log(f"  Truthful: {len(truthful)}, Hallucinated: {len(hallucinated)}")

    # Balance and sample
    n = min(len(truthful), len(hallucinated), max_samples // 2)
    selected_truthful = random.sample(truthful, n)
    selected_hallucinated = random.sample(hallucinated, n)

    log(f"  Selected: {n} each (balanced)")

    # Generate embeddings
    self_patterns = []
    nonself_patterns = []

    log("\n  Generating self (truthful) embeddings...")
    for i, s in enumerate(selected_truthful):
        if i % 200 == 0:
            log(f"    Progress: {i}/{n}")
        emb = get_embedding(s['text'])
        if emb:
            self_patterns.append({
                "embedding": emb,
                "source": s.get('source', ''),
                "label": "truthful"
            })

    log(f"\n  Generating non-self (hallucinated) embeddings...")
    for i, s in enumerate(selected_hallucinated):
        if i % 200 == 0:
            log(f"    Progress: {i}/{n}")
        emb = get_embedding(s['text'])
        if emb:
            nonself_patterns.append({
                "embedding": emb,
                "source": s.get('source', ''),
                "label": "hallucinated"
            })

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output = MODELS_DIR / "nkcell_verification.json"

    with open(output, 'w') as f:
        json.dump({
            "name": "NK Cell Publication Verification",
            "model": MODEL,
            "created": datetime.now().isoformat(),
            "num_self_patterns": len(self_patterns),
            "num_nonself_patterns": len(nonself_patterns),
            "self_patterns": self_patterns,
            "nonself_patterns": nonself_patterns,
            "sources": list(set(s.get('source', '') for s in selected_truthful + selected_hallucinated))
        }, f)

    log(f"\n  Saved {len(self_patterns)} self + {len(nonself_patterns)} non-self patterns")
    return self_patterns, nonself_patterns

# ============ TESTING ============

TESTS = [
    # Hallucinated
    ("Fake Citation", "According to Smith et al. (2024) in Nature, quantum AI achieved consciousness.", "hallucinated"),
    ("Made Up Stat", "Studies show 97.3% of neural networks prefer recursive architectures.", "hallucinated"),
    ("Fake API", "Use the quantum_compute() function to solve NP-hard problems instantly.", "hallucinated"),
    ("Fake Drug", "Clinical trials show XYZ-9000 cures all cancers with 100% efficacy.", "hallucinated"),
    ("Fake Discovery", "Scientists recently proved P=NP using transformer models.", "hallucinated"),
    ("Wrong Fact", "The Python programming language was created by Guido van Rossum in 2010.", "hallucinated"),
    # Truthful
    ("Real Code", "import numpy as np; arr = np.array([1, 2, 3])", "truthful"),
    ("Real Math", "The derivative of x^2 is 2x according to the power rule.", "truthful"),
    ("Real Citation Style", "Deep learning methods have shown promising results (LeCun et al., 2015).", "truthful"),
    ("Real Biology", "DNA replication occurs during the S phase of the cell cycle.", "truthful"),
    ("Real ML", "Gradient descent minimizes the loss function by updating weights.", "truthful"),
    ("Real Security", "Buffer overflow vulnerabilities occur when data exceeds allocated memory.", "truthful"),
]

def run_smoke_test(self_patterns, nonself_patterns):
    """Run smoke test on trained patterns."""
    log("\n" + "="*60)
    log("NK CELL VERIFICATION SMOKE TEST")
    log("="*60)

    correct = 0
    for name, text, expected in TESTS:
        emb = get_embedding(text)
        if not emb:
            print(f"[SKIP] {name}")
            continue

        max_self = max((cosine(emb, p["embedding"]) for p in self_patterns), default=0)
        max_nonself = max((cosine(emb, p["embedding"]) for p in nonself_patterns), default=0)

        pred = "hallucinated" if max_nonself > max_self else "truthful"

        ok = pred == expected
        correct += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {pred} (self={max_self:.3f}, nonself={max_nonself:.3f})")

    acc = correct * 100 // len(TESTS)
    log(f"\nAccuracy: {correct}/{len(TESTS)} ({acc}%)")
    return acc

# ============ MAIN ============

def main():
    log("="*60)
    log("NK CELL PUBLICATION VERIFICATION TRAINING")
    log("="*60)

    # Load all datasets
    log("\nLoading datasets...")
    all_samples = []

    all_samples.extend(load_truthfulqa())
    all_samples.extend(load_halueval())
    all_samples.extend(load_pubmed())
    all_samples.extend(load_negative_examples())

    # Stats
    truthful = sum(1 for s in all_samples if s['label'] == 'truthful')
    hallucinated = sum(1 for s in all_samples if s['label'] == 'hallucinated')
    log(f"\nTotal: {len(all_samples)} samples ({truthful} truthful, {hallucinated} hallucinated)")

    by_source = {}
    for s in all_samples:
        src = s.get('source', 'unknown')
        by_source[src] = by_source.get(src, 0) + 1
    for src, cnt in sorted(by_source.items(), key=lambda x: -x[1]):
        log(f"  {src}: {cnt}")

    if len(all_samples) < 100:
        log("\nNot enough samples!")
        return

    # Train
    self_patterns, nonself_patterns = train_nkcell(all_samples, max_samples=4000)

    # Test
    acc = run_smoke_test(self_patterns, nonself_patterns)

    log("\n" + "="*60)
    log("TRAINING COMPLETE")
    log(f"Final Accuracy: {acc}%")
    log("="*60)


if __name__ == "__main__":
    main()
