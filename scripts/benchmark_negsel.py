#!/usr/bin/env python3
"""
Benchmark: NumPy vs PyTorch Negative Selection
"""

import time

import numpy as np
import torch

from immunos_mcp.algorithms.negsel import NegativeSelectionClassifier, NegSelConfig
from immunos_mcp.algorithms.negsel_torch import NegativeSelectionTorch


def run_benchmark():
    # Setup parameters
    dim = 20
    num_self = 1000
    target_detectors = 1000
    r_self = 0.8

    config = NegSelConfig(num_detectors=target_detectors, r_self=r_self)
    self_samples = np.random.uniform(0, 1, (num_self, dim))

    print(f"--- Benchmark: {target_detectors} Detectors in {dim}-Dim Space ---")
    print(f"Self Samples: {num_self}")

    # 1. NumPy Benchmark
    print("\n[NumPy] Training...")
    start_cpu = time.time()
    cpu_clf = NegativeSelectionClassifier(config=config)
    cpu_clf.fit(self_samples)
    end_cpu = time.time()
    print(f"✅ NumPy Time: {end_cpu - start_cpu:.4f}s")
    print(f"   Detectors found: {len(cpu_clf.valid_detectors)}")

    # 2. PyTorch Benchmark
    print(
        f"\n[PyTorch] Training on {torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')}..."
    )
    start_torch = time.time()
    torch_clf = NegativeSelectionTorch(config=config)
    torch_clf.fit(self_samples)
    end_torch = time.time()
    print(f"✅ PyTorch Time: {end_torch - start_torch:.4f}s")
    print(f"   Detectors found: {len(torch_clf.to_detectors())}")

    speedup = (end_cpu - start_cpu) / (end_torch - start_torch)
    print(f"\n🚀 Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    run_benchmark()
