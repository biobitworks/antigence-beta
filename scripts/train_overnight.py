#!/usr/bin/env python3
"""
Overnight training for Antigence agents using Ollama embeddings.

This script trains agents on all available datasets with maximum samples.
Run in background for extended training sessions.

Usage:
    python scripts/train_overnight.py              # Full overnight training
    python scripts/train_overnight.py --quick      # Quick test run
"""

import argparse
import json
import csv
import random
import time
import sys
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.home() / ".antigence" / "data"
MODELS_DIR = Path.home() / ".antigence" / "trained"
LOG_FILE = MODELS_DIR / "overnight_training.log"

# Ollama embedding models to use
EMBEDDING_MODELS = [
    "nomic-embed-text",
    "mxbai-embed-large",
    "bge-m3",
]


def log(msg: str, also_print: bool = True):
    """Log message to file and optionally print."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {msg}"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_msg + "\n")

    if also_print:
        print(log_msg)


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    """Get embedding from Ollama."""
    import requests
    try:
        resp = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": text[:2000]},
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json().get("embedding", [])
    except Exception as e:
        pass
    return []


def load_bigvul(max_samples: int = 50000) -> list:
    """Load Big-Vul CVE dataset with full parsing."""
    csv_path = DATA_DIR / "bigvul" / "all_c_cpp_release2.0.csv"
    if not csv_path.exists():
        log(f"Big-Vul not found at {csv_path}")
        return []

    log(f"Loading Big-Vul (max {max_samples} samples)...")
    csv.field_size_limit(sys.maxsize)

    samples = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if len(samples) >= max_samples:
                break

            # Get vulnerable code
            vul_code = row.get("func_before", "") or row.get("files_changed", "")
            safe_code = row.get("func_after", "")
            cwe = row.get("cwe_id", "")
            cve = row.get("cve_id", "")

            if vul_code and len(vul_code) > 50:
                samples.append({
                    "code": vul_code[:3000],
                    "label": "vulnerable",
                    "cwe": cwe,
                    "cve": cve,
                    "source": "bigvul",
                })

            if safe_code and len(safe_code) > 50:
                samples.append({
                    "code": safe_code[:3000],
                    "label": "safe",
                    "cwe": cwe,
                    "source": "bigvul_fixed",
                })

    random.shuffle(samples)
    log(f"  Loaded {len(samples)} Big-Vul samples")
    return samples


def load_codexglue() -> list:
    """Load CodeXGLUE defect detection dataset."""
    ds_path = DATA_DIR / "codexglue"
    if not ds_path.exists():
        log(f"CodeXGLUE not found")
        return []

    log("Loading CodeXGLUE...")
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(ds_path))

        samples = []
        for split in ["train", "validation", "test"]:
            if split in ds:
                for item in ds[split]:
                    func = item.get("func", "")
                    if len(func) > 50:
                        samples.append({
                            "code": func[:3000],
                            "label": "vulnerable" if item.get("target", 0) == 1 else "safe",
                            "source": "codexglue",
                        })

        log(f"  Loaded {len(samples)} CodeXGLUE samples")
        return samples
    except Exception as e:
        log(f"  Error loading CodeXGLUE: {e}")
        return []


def load_truthfulqa() -> list:
    """Load TruthfulQA for hallucination detection."""
    ds_path = DATA_DIR / "truthfulqa"
    if not ds_path.exists():
        log("TruthfulQA not found")
        return []

    log("Loading TruthfulQA...")
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(ds_path))

        samples = []
        for split in ds:
            for item in ds[split]:
                question = item.get("question", "")
                mc1 = item.get("mc1_targets", {})
                choices = mc1.get("choices", [])
                labels = mc1.get("labels", [])

                for choice, label in zip(choices, labels):
                    samples.append({
                        "text": f"Q: {question}\nA: {choice}",
                        "label": "truthful" if label == 1 else "hallucinated",
                        "source": "truthfulqa",
                    })

        log(f"  Loaded {len(samples)} TruthfulQA samples")
        return samples
    except Exception as e:
        log(f"  Error: {e}")
        return []


def load_pubmedqa() -> list:
    """Load PubMedQA for scientific reasoning."""
    ds_path = DATA_DIR / "pubmedqa"
    if not ds_path.exists():
        log("PubMedQA not found")
        return []

    log("Loading PubMedQA...")
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(ds_path))

        samples = []
        for split in ds:
            for item in ds[split]:
                question = item.get("question", "")
                context = " ".join(item.get("context", {}).get("contexts", []))[:1000]
                answer = item.get("final_decision", "")

                samples.append({
                    "text": f"Context: {context}\nQ: {question}\nA: {answer}",
                    "label": "supported" if answer == "yes" else "unsupported",
                    "source": "pubmedqa",
                })

        log(f"  Loaded {len(samples)} PubMedQA samples")
        return samples
    except Exception as e:
        log(f"  Error: {e}")
        return []


def load_scifact() -> list:
    """Load SciFact claims."""
    scifact_dir = DATA_DIR / "scifact"
    claims_file = scifact_dir / "claims_train.jsonl"

    if not claims_file.exists():
        claims_file = scifact_dir / "data" / "claims_train.jsonl"

    if not claims_file.exists():
        log("SciFact not found")
        return []

    log("Loading SciFact...")
    samples = []
    try:
        with open(claims_file, "r") as f:
            for line in f:
                item = json.loads(line)
                samples.append({
                    "text": item.get("claim", ""),
                    "label": "supported" if item.get("evidence") else "unsupported",
                    "source": "scifact",
                })
        log(f"  Loaded {len(samples)} SciFact samples")
        return samples
    except Exception as e:
        log(f"  Error: {e}")
        return []


def train_bcell_overnight(samples: list, batch_size: int = 100, embedding_model: str = "nomic-embed-text"):
    """Extended B Cell training with full embedding generation."""
    log(f"\n{'='*60}")
    log(f"OVERNIGHT B CELL TRAINING")
    log(f"Model: {embedding_model}")
    log(f"Samples: {len(samples)}")
    log(f"{'='*60}")

    if not samples:
        log("No samples!")
        return None

    random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_data = samples[:split]
    test_data = samples[split:]

    # Label distribution
    label_counts = {}
    for s in train_data:
        label = s.get("label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
    log(f"Labels: {label_counts}")

    # Generate embeddings for all training data
    patterns = []
    start_time = time.time()

    for i, sample in enumerate(train_data):
        if i % batch_size == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(train_data) - i) / rate if rate > 0 else 0
            log(f"  Progress: {i}/{len(train_data)} ({i*100//len(train_data)}%) - ETA: {remaining/60:.1f} min")

        text = sample.get("code") or sample.get("text", "")
        if not text or len(text) < 20:
            continue

        embedding = get_embedding(text, embedding_model)
        if embedding:
            patterns.append({
                "label": sample.get("label"),
                "embedding": embedding,
                "cwe": sample.get("cwe", ""),
                "source": sample.get("source", ""),
            })

    total_time = time.time() - start_time
    log(f"Generated {len(patterns)} embeddings in {total_time/60:.1f} minutes")

    # Save patterns
    output_name = f"bcell_overnight_{embedding_model.replace(':', '-')}"
    patterns_path = MODELS_DIR / f"{output_name}.json"

    with open(patterns_path, "w") as f:
        json.dump({
            "name": output_name,
            "model": embedding_model,
            "created": datetime.now().isoformat(),
            "training_time_minutes": total_time / 60,
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "num_patterns": len(patterns),
            "label_counts": label_counts,
            "patterns": patterns,
        }, f)

    log(f"Saved to {patterns_path}")
    return patterns_path


def train_nkcell_overnight(samples: list, embedding_model: str = "nomic-embed-text"):
    """Extended NK Cell self-pattern training."""
    log(f"\n{'='*60}")
    log(f"OVERNIGHT NK CELL TRAINING")
    log(f"Model: {embedding_model}")
    log(f"{'='*60}")

    # NK Cell only learns "self" (truthful/safe/supported)
    self_samples = [s for s in samples if s.get("label") in ["truthful", "safe", "supported"]]
    log(f"Self samples: {len(self_samples)} (from {len(samples)} total)")

    if not self_samples:
        log("No self samples!")
        return None

    self_patterns = []
    start_time = time.time()

    for i, sample in enumerate(self_samples):
        if i % 50 == 0:
            log(f"  Progress: {i}/{len(self_samples)}")

        text = sample.get("text") or sample.get("code", "")
        if not text:
            continue

        embedding = get_embedding(text, embedding_model)
        if embedding:
            self_patterns.append({
                "embedding": embedding,
                "source": sample.get("source", ""),
            })

    total_time = time.time() - start_time
    output_name = f"nkcell_overnight_{embedding_model.replace(':', '-')}"
    output_path = MODELS_DIR / f"{output_name}.json"

    with open(output_path, "w") as f:
        json.dump({
            "name": output_name,
            "model": embedding_model,
            "created": datetime.now().isoformat(),
            "training_time_minutes": total_time / 60,
            "num_self_patterns": len(self_patterns),
            "self_patterns": self_patterns,
        }, f)

    log(f"Saved {len(self_patterns)} self-patterns to {output_path}")
    return output_path


def run_overnight_training(quick: bool = False):
    """Run full overnight training pipeline."""
    log("\n" + "="*60)
    log("ANTIGENCE OVERNIGHT TRAINING STARTED")
    log("="*60)
    start_time = time.time()

    # Load all datasets
    max_samples = 1000 if quick else 50000

    code_samples = []
    code_samples.extend(load_bigvul(max_samples))
    code_samples.extend(load_codexglue())

    truth_samples = []
    truth_samples.extend(load_truthfulqa())
    truth_samples.extend(load_pubmedqa())
    truth_samples.extend(load_scifact())

    log(f"\nTotal code samples: {len(code_samples)}")
    log(f"Total truth samples: {len(truth_samples)}")

    # Train with each embedding model
    models_to_use = EMBEDDING_MODELS[:1] if quick else EMBEDDING_MODELS

    for model in models_to_use:
        log(f"\n>>> Using embedding model: {model}")

        # Train B Cell
        if code_samples:
            train_bcell_overnight(code_samples, embedding_model=model)

        # Train NK Cell
        if truth_samples:
            train_nkcell_overnight(truth_samples, embedding_model=model)

    total_time = time.time() - start_time
    log("\n" + "="*60)
    log(f"OVERNIGHT TRAINING COMPLETE")
    log(f"Total time: {total_time/3600:.2f} hours")
    log("="*60)

    # Show what was created
    log("\nTrained models:")
    for f in sorted(MODELS_DIR.glob("*.json")):
        size_mb = f.stat().st_size / (1024 * 1024)
        log(f"  {f.name} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Overnight Antigence training")
    parser.add_argument("--quick", action="store_true", help="Quick test run (1000 samples)")
    args = parser.parse_args()

    run_overnight_training(quick=args.quick)


if __name__ == "__main__":
    main()
