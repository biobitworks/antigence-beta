#!/usr/bin/env python3
"""Download vulnerability datasets directly from HuggingFace."""

import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.home() / ".antigence" / "data"

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def download_dataset(name, hf_path, split_map=None):
    """Download a HuggingFace dataset."""
    log(f"Downloading {name}...")
    output_dir = DATA_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        ds = load_dataset(hf_path)
        ds.save_to_disk(str(output_dir))
        log(f"  ✓ Downloaded {name}")
        return True
    except Exception as e:
        log(f"  Error: {e}")
        return False


def convert_to_samples():
    """Convert all downloaded datasets to unified training samples."""
    log("\nConverting datasets to samples...")
    all_samples = []

    # CodeXGLUE Defect Detection (already have)
    codexglue_dir = DATA_DIR / "codexglue"
    if codexglue_dir.exists():
        log("  Processing CodeXGLUE...")
        try:
            from datasets import load_from_disk
            ds = load_from_disk(str(codexglue_dir))
            for split in ds:
                for item in ds[split]:
                    func = item.get('func', '')
                    if func and len(func) > 50:
                        all_samples.append({
                            "code": func[:3000],
                            "label": "vulnerable" if item.get('target', 0) == 1 else "safe",
                            "source": "codexglue"
                        })
            log(f"    Added {len([s for s in all_samples if s['source'] == 'codexglue'])} CodeXGLUE samples")
        except Exception as e:
            log(f"    Error: {e}")

    # Big-Vul CSV
    bigvul_csv = DATA_DIR / "bigvul" / "all_c_cpp_release2.0.csv"
    if bigvul_csv.exists():
        log("  Processing Big-Vul...")
        import csv
        import sys
        csv.field_size_limit(sys.maxsize)
        count = 0
        try:
            with open(bigvul_csv, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if count >= 10000:  # Limit for speed
                        break
                    # Vulnerable code
                    vul_code = row.get('func_before', '')
                    if vul_code and len(vul_code) > 50:
                        all_samples.append({
                            "code": vul_code[:3000],
                            "label": "vulnerable",
                            "cwe": row.get('cwe_id', ''),
                            "source": "bigvul"
                        })
                        count += 1
                    # Fixed (safe) code
                    safe_code = row.get('func_after', '')
                    if safe_code and len(safe_code) > 50:
                        all_samples.append({
                            "code": safe_code[:3000],
                            "label": "safe",
                            "source": "bigvul_fixed"
                        })
            log(f"    Added {count} Big-Vul samples")
        except Exception as e:
            log(f"    Error: {e}")

    # Add our curated samples
    for ds_name in ["juliet", "sard", "python_vulns", "cwe_samples"]:
        ds_file = DATA_DIR / ds_name / "samples.json"
        if ds_file.exists():
            with open(ds_file) as f:
                samples = json.load(f)
                for s in samples:
                    all_samples.append({
                        "code": s.get("code", ""),
                        "label": s.get("label", ""),
                        "cwe": s.get("cwe", ""),
                        "lang": s.get("lang", ""),
                        "source": ds_name
                    })
            log(f"    Added {len(samples)} {ds_name} samples")

    # Save combined
    output_file = DATA_DIR / "combined_training" / "samples.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_samples, f, indent=2)

    # Statistics
    vuln = sum(1 for s in all_samples if s['label'] == 'vulnerable')
    safe = sum(1 for s in all_samples if s['label'] == 'safe')
    log(f"\n  Total: {len(all_samples)} samples ({vuln} vulnerable, {safe} safe)")
    log(f"  Saved to: {output_file}")

    return all_samples


def main():
    log("="*60)
    log("PREPARING COMBINED TRAINING DATA")
    log("="*60)

    # Try to download additional datasets
    datasets_to_try = [
        # ("devign_hf", "benjis/devign"),  # May not exist
        # ("reveal", "claudios/reveal"),   # May not exist
    ]

    for name, hf_path in datasets_to_try:
        download_dataset(name, hf_path)

    # Convert all to training samples
    samples = convert_to_samples()

    # Show by source
    log("\n" + "="*60)
    log("SAMPLES BY SOURCE")
    log("="*60)
    sources = {}
    for s in samples:
        src = s.get('source', 'unknown')
        sources[src] = sources.get(src, 0) + 1
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        log(f"  {src}: {count}")


if __name__ == "__main__":
    main()
