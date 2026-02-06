#!/usr/bin/env python3
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

print("🚀 Starting Publication Report Generation...")
print("=" * 50)

# Run Benchmarks
os.system("PYTHONPATH=src python3 scripts/reproduce_negsl_ais.py")

print("\n" + "=" * 50)
print("✅ Publication Report Generated!")
print("Location: manuscript/NegSl-AIS_PREPRINT_SUMMARY.md")
print("Metrics Included: Accuracy, MCC, Cohen's Kappa, Confusion Matrices")
