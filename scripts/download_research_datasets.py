#!/usr/bin/env python3
"""
Download research datasets and models for B Cell training.

Sources:
- PrimeVul (ICSE 2025): High-quality vulnerability labels
- Devign: FFMpeg+Qemu manual labels
- All-CVE-Records: 300K CVE records from HuggingFace
- VulBERTa: Pre-trained vulnerability detection model
"""

import json
import os
import subprocess
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.home() / ".antigence" / "data"
MODELS_DIR = Path.home() / ".antigence" / "models"

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def download_primevul():
    """Download PrimeVul dataset from GitHub."""
    log("Downloading PrimeVul (ICSE 2025)...")

    primevul_dir = DATA_DIR / "primevul"
    if primevul_dir.exists() and (primevul_dir / "primevul_train.csv").exists():
        log("  Already exists, skipping...")
        return True

    primevul_dir.mkdir(parents=True, exist_ok=True)

    # Clone the repo
    try:
        result = subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/DLVulDet/PrimeVul.git", str(primevul_dir / "repo")],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            log("  ✓ Cloned PrimeVul repo")
            return True
        else:
            log(f"  Error: {result.stderr}")
            return False
    except Exception as e:
        log(f"  Error: {e}")
        return False


def download_devign():
    """Download Devign dataset."""
    log("Downloading Devign dataset...")

    devign_dir = DATA_DIR / "devign"
    if devign_dir.exists() and list(devign_dir.glob("*.json")):
        log("  Already exists, skipping...")
        return True

    devign_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try HuggingFace datasets
        from datasets import load_dataset
        ds = load_dataset("benjis/devign", trust_remote_code=True)
        ds.save_to_disk(str(devign_dir))
        log("  ✓ Downloaded Devign from HuggingFace")
        return True
    except Exception as e:
        log(f"  HuggingFace failed: {e}")

        # Try GitHub
        try:
            result = subprocess.run(
                ["git", "clone", "--depth=1", "https://github.com/VulDetProject/ReVeal.git", str(devign_dir / "reveal_repo")],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                log("  ✓ Cloned ReVeal repo (contains Devign)")
                return True
        except Exception as e2:
            log(f"  Error: {e2}")

    return False


def download_cve_records():
    """Download All-CVE-Records from HuggingFace."""
    log("Downloading All-CVE-Records (300K CVE records)...")

    cve_dir = DATA_DIR / "cve_records"
    if cve_dir.exists() and list(cve_dir.glob("*")):
        log("  Already exists, skipping...")
        return True

    cve_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        ds = load_dataset("AlicanKiraz0/All-CVE-Records-Training-Dataset")
        ds.save_to_disk(str(cve_dir))
        log("  ✓ Downloaded CVE records")
        return True
    except Exception as e:
        log(f"  Error: {e}")
        return False


def download_vulberta():
    """Download VulBERTa model."""
    log("Downloading VulBERTa model...")

    vulberta_dir = MODELS_DIR / "vulberta"
    if vulberta_dir.exists():
        log("  Already exists, skipping...")
        return True

    vulberta_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="claudios/VulBERTa-MLP-Devign",
            local_dir=str(vulberta_dir),
            local_dir_use_symlinks=False
        )
        log("  ✓ Downloaded VulBERTa")
        return True
    except Exception as e:
        log(f"  HuggingFace download failed: {e}")

        # Try cloning the repo for the model
        try:
            result = subprocess.run(
                ["git", "clone", "--depth=1", "https://github.com/ICL-ml4csec/VulBERTa.git", str(vulberta_dir / "repo")],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                log("  ✓ Cloned VulBERTa repo")
                return True
        except Exception as e2:
            log(f"  Error: {e2}")

    return False


def download_draper():
    """Download Draper VDISC dataset."""
    log("Downloading Draper VDISC dataset...")

    draper_dir = DATA_DIR / "draper"
    if draper_dir.exists():
        log("  Already exists, skipping...")
        return True

    draper_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        ds = load_dataset("squareresearch/vdisc-vuln")
        ds.save_to_disk(str(draper_dir))
        log("  ✓ Downloaded Draper VDISC")
        return True
    except Exception as e:
        log(f"  Error: {e}")
        return False


def create_training_samples_from_downloads():
    """Convert downloaded datasets to training samples."""
    log("\nConverting datasets to training samples...")

    all_samples = []

    # Process PrimeVul if available
    primevul_dir = DATA_DIR / "primevul" / "repo" / "data"
    if primevul_dir.exists():
        log("  Processing PrimeVul...")
        import csv
        for csv_file in primevul_dir.glob("*.csv"):
            try:
                with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        func = row.get('func', '') or row.get('code', '')
                        label = row.get('target', row.get('label', ''))
                        if func and len(func) > 50:
                            all_samples.append({
                                "code": func[:3000],
                                "label": "vulnerable" if str(label) == "1" else "safe",
                                "source": "primevul",
                                "cwe": row.get('cwe', '')
                            })
            except Exception as e:
                log(f"    Error reading {csv_file}: {e}")
        log(f"    Loaded {len([s for s in all_samples if s['source'] == 'primevul'])} PrimeVul samples")

    # Process Devign if available
    devign_dir = DATA_DIR / "devign"
    if devign_dir.exists():
        log("  Processing Devign...")
        try:
            from datasets import load_from_disk
            ds = load_from_disk(str(devign_dir))
            count = 0
            for split in ds:
                for item in ds[split]:
                    func = item.get('func', '')
                    target = item.get('target', 0)
                    if func and len(func) > 50:
                        all_samples.append({
                            "code": func[:3000],
                            "label": "vulnerable" if target == 1 else "safe",
                            "source": "devign"
                        })
                        count += 1
            log(f"    Loaded {count} Devign samples")
        except Exception as e:
            log(f"    Error: {e}")

    # Process Draper if available
    draper_dir = DATA_DIR / "draper"
    if draper_dir.exists():
        log("  Processing Draper VDISC...")
        try:
            from datasets import load_from_disk
            ds = load_from_disk(str(draper_dir))
            count = 0
            for split in ds:
                for item in ds[split]:
                    func = item.get('func', '') or item.get('code', '')
                    label = item.get('label', 0)
                    if func and len(func) > 50:
                        all_samples.append({
                            "code": func[:3000],
                            "label": "vulnerable" if label == 1 else "safe",
                            "source": "draper"
                        })
                        count += 1
            log(f"    Loaded {count} Draper samples")
        except Exception as e:
            log(f"    Error: {e}")

    # Save combined samples
    if all_samples:
        output_file = DATA_DIR / "research_combined" / "samples.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(all_samples, f, indent=2)
        log(f"\n  Saved {len(all_samples)} combined samples to {output_file}")

    return all_samples


def show_status():
    """Show download status."""
    log("\n" + "="*50)
    log("DOWNLOAD STATUS")
    log("="*50)

    datasets = {
        "primevul": DATA_DIR / "primevul",
        "devign": DATA_DIR / "devign",
        "draper": DATA_DIR / "draper",
        "cve_records": DATA_DIR / "cve_records",
        "research_combined": DATA_DIR / "research_combined",
    }

    models = {
        "vulberta": MODELS_DIR / "vulberta",
    }

    log("\nDatasets:")
    for name, path in datasets.items():
        exists = path.exists() and any(path.iterdir()) if path.exists() else False
        status = "✓" if exists else "○"
        log(f"  {status} {name}: {path}")

    log("\nModels:")
    for name, path in models.items():
        exists = path.exists() and any(path.iterdir()) if path.exists() else False
        status = "✓" if exists else "○"
        log(f"  {status} {name}: {path}")


def main():
    log("="*60)
    log("DOWNLOADING RESEARCH DATASETS & MODELS")
    log("="*60)

    # Download datasets
    download_primevul()
    download_devign()
    download_draper()
    # download_cve_records()  # Large, optional

    # Download models
    download_vulberta()

    # Convert to training samples
    create_training_samples_from_downloads()

    # Show status
    show_status()


if __name__ == "__main__":
    main()
