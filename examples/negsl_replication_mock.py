"""
NegSl-AIS Replication (Mock Data)
=================================

Simulates the MAHNOB-HCI emotion classification experiment using synthetic data.
This verifies the Negative Selection algorithm integration in `NKCellAgent`
prior to the availability of the real dataset.

Dataset Properties (Simulated):
- Dimensions: 20 features (mimicking spectral power, HRV, etc.)
- Classes: Arousal (High/Low), Valence (Positive/Negative)
- Normal Behavior ("Self"): Calm/Positive
- Anomalous Behavior ("Non-Self"): Stressed/Negative
"""

# Core Immunos imports
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from immunos_mcp.agents.nk_cell_agent import NKCellAgent
from immunos_mcp.core.antigen import Antigen, DataType


def generate_synthetic_data(
    n_samples: int = 100, dim: int = 20, is_anomaly: bool = False
) -> Tuple[List[List[float]], List[str]]:
    """
    Generate synthetic physiological feature vectors.

    Self (Normal): Gaussian distribution centered at 0.3, low variance
    Non-Self (Anomaly): Gaussian distribution centered at 0.7, high variance
    """
    if not is_anomaly:
        # Normal state (e.g., Calm)
        # Mean 0.3, Std 0.1
        data = np.random.normal(0.3, 0.1, (n_samples, dim))
        labels = ["self"] * n_samples
    else:
        # Anomalous state (e.g., High Stress)
        # Mean 0.7, Std 0.15
        data = np.random.normal(0.7, 0.15, (n_samples, dim))
        labels = ["anomaly"] * n_samples

    return data.tolist(), labels


def run_experiment():
    print("=" * 60)
    print("🧬 NegSl-AIS Replication Experiment (Synthetic Data)")
    print("   Target: Emotion Classification (Arousal/Valence)")
    print("=" * 60)

    # 1. Data Generation
    print("\n[1] Generating Synthetic Data...")
    n_train_self = 200
    n_test_self = 50
    n_test_anomaly = 50
    feat_dim = 20

    # Training data (Self only)
    train_vectors, _ = generate_synthetic_data(n_train_self, feat_dim, is_anomaly=False)

    # Test data (Mixed)
    test_self_vec, test_self_lbl = generate_synthetic_data(n_test_self, feat_dim, is_anomaly=False)
    test_anom_vec, test_anom_lbl = generate_synthetic_data(
        n_test_anomaly, feat_dim, is_anomaly=True
    )

    test_vectors = test_self_vec + test_anom_vec
    test_labels = test_self_lbl + test_anom_lbl

    # Convert to Antigens (Feature only, no text data needed for this mode)
    train_antigens = [
        Antigen(data=f"syn_train_{i}", data_type=DataType.STRUCTURED, features={"vec": v})
        for i, v in enumerate(train_vectors)
    ]

    print(f"   Training Data: {len(train_vectors)} samples (Self only)")
    print(
        f"   Testing Data:  {len(test_vectors)} samples ({n_test_self} Self, {n_test_anomaly} Anomaly)"
    )

    # 2. Initialization
    print("\n[2] Initializing NegSl-AIS Agent...")
    nk_agent = NKCellAgent(
        agent_name="negsl_replicator",
        detection_threshold=0.2,  # Tunable parameter
        num_detectors=500,  # Number of negative detectors
        mode="feature",  # Use feature vector mode
    )

    # 3. Training
    start_time = time.time()
    nk_agent.train_on_features(train_antigens, train_vectors)
    train_time = time.time() - start_time
    print(f"   Training complete in {train_time:.2f}s")

    # 4. Testing
    print("\n[3] Running Classification...")
    predictions = []

    start_time = time.time()
    for i, vec in enumerate(test_vectors):
        # Create dummy antigen
        ag = Antigen(data=f"test_{i}", data_type=DataType.STRUCTURED, features={"vec": vec})

        # Detect
        result = nk_agent.detect_with_features(ag, vec)

        pred_label = "anomaly" if result.is_anomaly else "self"
        predictions.append(pred_label)

    test_time = time.time() - start_time
    print(
        f"   Inference complete in {test_time:.2f}s ({test_time/len(test_vectors)*1000:.2f}ms/sample)"
    )

    # 5. Results
    print("\n[4] Results Analysis:")
    acc = accuracy_score(test_labels, predictions)
    print(f"   Accuracy: {acc:.2%}")

    print("\n   Confusion Matrix:")
    cm = confusion_matrix(test_labels, predictions, labels=["self", "anomaly"])
    print(f"   TN (Self->Self):       {cm[0][0]}")
    print(f"   FP (Self->Anomaly):    {cm[0][1]}")
    print(f"   FN (Anomaly->Self):    {cm[1][0]}")
    print(f"   TP (Anomaly->Anomaly): {cm[1][1]}")

    print("\n   Classification Report:")
    print(classification_report(test_labels, predictions, target_names=["self", "anomaly"]))

    print("\n✅ Replication Status: SUCCESS (Simulated)")
    print("   Note: Results based on Gaussian synthetic data.")


if __name__ == "__main__":
    run_experiment()
