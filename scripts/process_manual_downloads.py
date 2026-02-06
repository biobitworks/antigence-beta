#!/usr/bin/env python3
"""
Process manually downloaded datasets (PrimeVul, DiverseVul).
Run after placing files in ~/.antigence/data/
"""

import json
import csv
import sys
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.home() / ".antigence" / "data"

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def process_primevul():
    """Process PrimeVul JSONL files."""
    primevul_dir = DATA_DIR / "primevul"
    samples = []

    # Look for JSONL files
    jsonl_files = list(primevul_dir.glob("*.jsonl"))
    if not jsonl_files:
        # Check subdirectories
        jsonl_files = list(primevul_dir.glob("**/*.jsonl"))

    if not jsonl_files:
        log("  No PrimeVul JSONL files found")
        return []

    log(f"  Found {len(jsonl_files)} JSONL files")

    for jsonl_file in jsonl_files:
        log(f"    Processing {jsonl_file.name}...")
        try:
            with open(jsonl_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        func = item.get('func', '') or item.get('code', '') or item.get('function', '')
                        target = item.get('target', item.get('label', item.get('vulnerable', 0)))

                        if func and len(func) > 50:
                            samples.append({
                                "code": func[:4000],
                                "label": "vulnerable" if str(target) == "1" else "safe",
                                "cwe": item.get('cwe', item.get('cwe_id', '')),
                                "source": "primevul"
                            })
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            log(f"      Error: {e}")

    log(f"  Loaded {len(samples)} PrimeVul samples")
    return samples


def process_diversevul():
    """Process DiverseVul JSON/CSV file."""
    diversevul_dir = DATA_DIR / "diversevul"
    samples = []

    # Look for JSON or CSV files
    json_files = list(diversevul_dir.glob("*.json"))
    csv_files = list(diversevul_dir.glob("*.csv"))

    # Process JSON
    for json_file in json_files:
        if json_file.name == "samples.json":  # Skip our output
            continue
        log(f"    Processing {json_file.name}...")
        try:
            with open(json_file, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)

            # Handle different formats
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get('data', data.get('samples', [data]))
            else:
                items = []

            for item in items:
                func = item.get('func', '') or item.get('code', '') or item.get('vulnerable_code', '')
                target = item.get('target', item.get('label', item.get('vulnerable', 1)))

                if func and len(func) > 50:
                    samples.append({
                        "code": func[:4000],
                        "label": "vulnerable" if str(target) == "1" else "safe",
                        "cwe": item.get('cwe', item.get('cwe_id', '')),
                        "source": "diversevul"
                    })
        except Exception as e:
            log(f"      Error: {e}")

    # Process CSV
    csv.field_size_limit(sys.maxsize)
    for csv_file in csv_files:
        log(f"    Processing {csv_file.name}...")
        try:
            with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    func = row.get('func', '') or row.get('code', '') or row.get('vulnerable_code', '')
                    target = row.get('target', row.get('label', row.get('vulnerable', '1')))

                    if func and len(func) > 50:
                        samples.append({
                            "code": func[:4000],
                            "label": "vulnerable" if str(target) == "1" else "safe",
                            "cwe": row.get('cwe', row.get('cwe_id', '')),
                            "source": "diversevul"
                        })
        except Exception as e:
            log(f"      Error: {e}")

    log(f"  Loaded {len(samples)} DiverseVul samples")
    return samples


def main():
    log("="*60)
    log("PROCESSING MANUAL DOWNLOADS")
    log("="*60)

    all_samples = []

    # Check what's available
    log("\nChecking for downloaded files...")

    primevul_dir = DATA_DIR / "primevul"
    diversevul_dir = DATA_DIR / "diversevul"

    log(f"  PrimeVul dir: {primevul_dir}")
    if primevul_dir.exists():
        files = list(primevul_dir.glob("*"))
        log(f"    Files: {[f.name for f in files[:10]]}")
    else:
        log("    Not found")

    log(f"  DiverseVul dir: {diversevul_dir}")
    if diversevul_dir.exists():
        files = list(diversevul_dir.glob("*"))
        log(f"    Files: {[f.name for f in files[:10]]}")
    else:
        log("    Not found")

    # Process PrimeVul
    log("\nProcessing PrimeVul...")
    primevul_samples = process_primevul()
    all_samples.extend(primevul_samples)

    # Process DiverseVul
    log("\nProcessing DiverseVul...")
    diversevul_samples = process_diversevul()
    all_samples.extend(diversevul_samples)

    if not all_samples:
        log("\nNo samples found. Make sure files are in the correct directories:")
        log(f"  PrimeVul: {primevul_dir}")
        log(f"  DiverseVul: {diversevul_dir}")
        return

    # Save combined samples
    output_file = DATA_DIR / "research_downloads" / "samples.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(all_samples, f)

    # Statistics
    vuln = sum(1 for s in all_samples if s['label'] == 'vulnerable')
    safe = sum(1 for s in all_samples if s['label'] == 'safe')

    log("\n" + "="*60)
    log("DOWNLOAD PROCESSING COMPLETE")
    log("="*60)
    log(f"Total samples: {len(all_samples)}")
    log(f"  Vulnerable: {vuln}")
    log(f"  Safe: {safe}")
    log(f"  PrimeVul: {len(primevul_samples)}")
    log(f"  DiverseVul: {len(diversevul_samples)}")
    log(f"\nSaved to: {output_file}")

    # CWE distribution
    cwes = {}
    for s in all_samples:
        cwe = s.get('cwe', 'unknown') or 'unknown'
        cwes[cwe] = cwes.get(cwe, 0) + 1

    log("\nTop CWEs:")
    for cwe, count in sorted(cwes.items(), key=lambda x: -x[1])[:15]:
        log(f"  {cwe}: {count}")


if __name__ == "__main__":
    main()
