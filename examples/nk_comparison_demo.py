"""
NK Cell Comparison Demo

Compares original NK Cell agent vs Enhanced NK Cell (NegSl-AIS methodology)
on email spam detection task.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from immunos_mcp.core.antigen import Antigen
from immunos_mcp.agents.nk_cell_agent import NKCellAgent
from immunos_mcp.agents.nk_cell_enhanced import EnhancedNKCellAgent


def print_separator(title=""):
    """Print separator"""
    if title:
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)
    else:
        print("-" * 70)


def generate_simple_embeddings(texts):
    """Generate simple embeddings (char-based for demo)"""
    embeddings = []
    for text in texts:
        # Simple bag-of-chars embedding
        vec = np.zeros(256)
        for char in text.lower():
            vec[ord(char) % 256] += 1
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        embeddings.append(vec)
    return embeddings


def main():
    print_separator("NK Cell Comparison: Original vs Enhanced (NegSl-AIS)")
    print("\nComparing anomaly detection performance on email spam dataset")

    # === 1. PREPARE DATA ===
    print_separator("1. Preparing Data")

    # Normal emails (self) - expanded dataset
    normal_emails = [
        "Meeting scheduled for tomorrow at 10am",
        "Please review the attached document",
        "Thanks for your help with the project",
        "Looking forward to our discussion",
        "Can you send me the updated report",
        "Let's schedule a call for next week",
        "The presentation looks great",
        "Could you provide feedback on this",
        "I've attached the files you requested",
        "Thanks for the quick response"
    ]

    # Spam emails (for reference, NOT used in training)
    spam_emails = [
        "CONGRATULATIONS! You've won $1,000,000!!!",
        "Click here NOW for FREE prizes",
        "Hot singles in your area want to meet you",
        "Urgent: Your account has been compromised, click here",
        "AMAZING OFFER - Act now before it's too late!!!",
        "You have been selected for a special reward"
    ]

    # Test emails - mix of normal and spam
    test_cases = [
        ("Let's have lunch tomorrow", False, "normal meeting request"),
        ("WIN BIG MONEY FAST!!!", True, "obvious spam"),
        ("Please send the quarterly report", False, "normal business request"),
        ("FREE iPhone! Click now before it's too late!!!", True, "obvious spam"),
        ("Thanks for the update on the project", False, "normal thank you"),
        ("URGENT: Verify your account NOW!!!", True, "phishing attempt"),
        ("Could we reschedule our meeting", False, "normal scheduling"),
        ("Claim your prize immediately!!!", True, "prize scam"),
        ("The document has been updated", False, "normal notification"),
        ("Hot deal! Limited time only!!!", True, "spam advertisement")
    ]

    print(f"Normal emails (self): {len(normal_emails)}")
    print(f"Test cases: {len(test_cases)} ({sum(1 for _, is_spam, _ in test_cases if is_spam)} spam, "
          f"{sum(1 for _, is_spam, _ in test_cases if not is_spam)} normal)")

    # Generate embeddings
    normal_antigens = [Antigen.from_text(email) for email in normal_emails]
    normal_embeddings = generate_simple_embeddings(normal_emails)

    # === 2. TRAIN ORIGINAL NK CELL ===
    print_separator("2. Training Original NK Cell")

    start_time = time.time()
    nk_original = NKCellAgent(
        agent_name="nk_original",
        detection_threshold=0.5,
        num_detectors=100
    )
    nk_original.train_on_self(normal_antigens, embeddings=normal_embeddings)
    train_time_original = time.time() - start_time

    stats_original = nk_original.get_statistics()
    print(f"\nOriginal NK Cell:")
    print(f"  Self patterns: {stats_original['num_self_patterns']}")
    print(f"  Detectors: {stats_original['num_detectors']} (total)")
    print(f"  Threshold: {stats_original['detection_threshold']:.4f} (fixed)")
    print(f"  Training time: {train_time_original:.3f}s")

    # === 3. TRAIN ENHANCED NK CELL ===
    print_separator("3. Training Enhanced NK Cell (NegSl-AIS)")

    start_time = time.time()
    nk_enhanced = EnhancedNKCellAgent(
        agent_name="nk_enhanced",
        threshold_method="min_distance",
        detectors_per_class=20
    )
    nk_enhanced.train_on_self(normal_antigens, embeddings=normal_embeddings)
    train_time_enhanced = time.time() - start_time

    stats_enhanced = nk_enhanced.get_statistics()
    print(f"\nEnhanced NK Cell:")
    print(f"  Self patterns: {stats_enhanced['training_stats']['total_self_patterns']}")
    print(f"  Detectors: {stats_enhanced['training_stats']['total_detectors']} (per-class)")
    print(f"  Threshold: {stats_enhanced['global_threshold']:.4f} (adaptive)")
    print(f"  Threshold method: {stats_enhanced['threshold_method']}")
    print(f"  Training time: {train_time_enhanced:.3f}s")

    if stats_enhanced['detector_sets']:
        print(f"\n  Detector sets:")
        for class_label, ds_stats in stats_enhanced['detector_sets'].items():
            print(f"    {class_label}: {ds_stats['num_detectors']} detectors, "
                  f"threshold={ds_stats['threshold']:.4f}, "
                  f"avg_quality={ds_stats['avg_quality']:.4f}")

    # === 4. COMPARE ON TEST CASES ===
    print_separator("4. Testing Both Agents")

    results_original = []
    results_enhanced = []

    print("\n{:<45} {:<20} {:<20}".format(
        "Test Case", "Original", "Enhanced"
    ))
    print("-" * 85)

    for email, expected_spam, description in test_cases:
        antigen = Antigen.from_text(email)
        embedding = generate_simple_embeddings([email])[0]

        # Test original
        result_orig = nk_original.detect_novelty(antigen, antigen_embedding=embedding)
        results_original.append((expected_spam, result_orig))

        # Test enhanced
        result_enh = nk_enhanced.detect_novelty(antigen, antigen_embedding=embedding)
        results_enhanced.append((expected_spam, result_enh))

        # Format output
        orig_status = "🚨 ANOMALY" if result_orig.is_anomaly else "✓ NORMAL"
        enh_status = "🚨 ANOMALY" if result_enh.is_anomaly else "✓ NORMAL"

        orig_correct = "✓" if result_orig.is_anomaly == expected_spam else "✗"
        enh_correct = "✓" if result_enh.is_anomaly == expected_spam else "✗"

        # Truncate email for display
        display_email = email[:40] + "..." if len(email) > 40 else email

        print(f"{display_email:<45} "
              f"{orig_status} {orig_correct} ({result_orig.anomaly_score:.2f})  "
              f"{enh_status} {enh_correct} ({result_enh.anomaly_score:.2f})")

    # === 5. CALCULATE METRICS ===
    print_separator("5. Performance Metrics")

    def calculate_metrics(results):
        """Calculate accuracy, precision, recall, F1"""
        tp = sum(1 for expected, result in results if expected and result.is_anomaly)
        tn = sum(1 for expected, result in results if not expected and not result.is_anomaly)
        fp = sum(1 for expected, result in results if not expected and result.is_anomaly)
        fn = sum(1 for expected, result in results if expected and not result.is_anomaly)

        accuracy = (tp + tn) / len(results) if results else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        avg_confidence = np.mean([result.confidence for _, result in results])
        avg_execution_time = np.mean([result.metadata.get('execution_time', 0) for _, result in results])

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'avg_confidence': avg_confidence,
            'avg_execution_time': avg_execution_time
        }

    metrics_original = calculate_metrics(results_original)
    metrics_enhanced = calculate_metrics(results_enhanced)

    print("\n{:<20} {:<15} {:<15}".format("Metric", "Original", "Enhanced"))
    print("-" * 50)
    print(f"{'Accuracy':<20} {metrics_original['accuracy']:.1%}            {metrics_enhanced['accuracy']:.1%}")
    print(f"{'Precision':<20} {metrics_original['precision']:.1%}            {metrics_enhanced['precision']:.1%}")
    print(f"{'Recall':<20} {metrics_original['recall']:.1%}            {metrics_enhanced['recall']:.1%}")
    print(f"{'F1 Score':<20} {metrics_original['f1']:.1%}            {metrics_enhanced['f1']:.1%}")
    print()
    print(f"{'True Positives':<20} {metrics_original['tp']:<15} {metrics_enhanced['tp']}")
    print(f"{'True Negatives':<20} {metrics_original['tn']:<15} {metrics_enhanced['tn']}")
    print(f"{'False Positives':<20} {metrics_original['fp']:<15} {metrics_enhanced['fp']}")
    print(f"{'False Negatives':<20} {metrics_original['fn']:<15} {metrics_enhanced['fn']}")
    print()
    print(f"{'Avg Confidence':<20} {metrics_original['avg_confidence']:.3f}           {metrics_enhanced['avg_confidence']:.3f}")
    print(f"{'Avg Exec Time':<20} {metrics_original['avg_execution_time']*1000:.2f}ms          {metrics_enhanced['avg_execution_time']*1000:.2f}ms")

    # === 6. SUMMARY ===
    print_separator("Summary")

    improvement_accuracy = (metrics_enhanced['accuracy'] - metrics_original['accuracy']) * 100
    improvement_f1 = (metrics_enhanced['f1'] - metrics_original['f1']) * 100

    print("\n✨ Key Improvements with NegSl-AIS Methodology:")
    print(f"  • Accuracy: {improvement_accuracy:+.1f} percentage points")
    print(f"  • F1 Score: {improvement_f1:+.1f} percentage points")
    print(f"  • Adaptive Threshold: {stats_enhanced['global_threshold']:.4f} vs fixed {stats_original['detection_threshold']:.4f}")
    print(f"  • Detector Efficiency: {stats_enhanced['training_stats']['total_detectors']} focused detectors vs {stats_original['num_detectors']} random")

    print("\n🔬 NegSl-AIS Features Demonstrated:")
    print("  1. Adaptive threshold calculation (min_distance method)")
    print("  2. Per-class detector generation (one-vs-rest strategy)")
    print("  3. Enhanced validation rule: d(detector, nearest_self) > τ")
    print("  4. Quality metrics for detector effectiveness")

    print("\n💡 Biological Inspiration:")
    print("  • NK Cells: Detect abnormal cells without prior exposure")
    print("  • Thymic Selection: T-cells undergo negative selection to eliminate self-reactive cells")
    print("  • Adaptive Immunity: System learns and adapts to specific threats")

    print_separator()
    print("\n✓ Comparison demo complete!")
    print("\nNext steps:")
    print("  - Test on larger, real-world datasets")
    print("  - Integrate with LLM embeddings for semantic understanding")
    print("  - Add multi-modal support (text + metadata)")
    print("  - Implement temporal windowing for time-series data")


if __name__ == "__main__":
    main()
