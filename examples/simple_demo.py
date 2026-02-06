"""
Simple IMMUNOS-MCP Demo

Demonstrates B Cell (pattern matching) and NK Cell (anomaly detection)
agents working together for text classification.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from immunos_mcp.core.antigen import Antigen
from immunos_mcp.agents.bcell_agent import BCellAgent
from immunos_mcp.agents.nk_cell_agent import NKCellAgent


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
    print_separator("IMMUNOS-MCP: Simple Demo")
    print("\nDemonstrating B Cell (pattern matching) and NK Cell (anomaly detection)")
    print("for text classification without LLM embeddings (using simple char-based)")

    # === 1. PREPARE DATA ===
    print_separator("1. Preparing Data")

    # Normal emails (self)
    normal_emails = [
        "Meeting scheduled for tomorrow at 10am",
        "Please review the attached document",
        "Thanks for your help with the project",
        "Looking forward to our discussion",
        "Can you send me the updated report"
    ]

    # Spam emails (non-self)
    spam_emails = [
        "CONGRATULATIONS! You've won $1,000,000!!!",
        "Click here NOW for FREE prizes",
        "Hot singles in your area want to meet you",
        "Urgent: Your account has been compromised, click here"
    ]

    # Test emails
    test_emails = [
        "Let's have lunch tomorrow",  # Should be: normal
        "WIN BIG MONEY FAST!!!",  # Should be: spam/anomaly
        "Please send the quarterly report",  # Should be: normal
        "FREE iPhone! Click now before it's too late!!!",  # Should be: spam/anomaly
    ]

    print(f"Normal emails (self): {len(normal_emails)}")
    print(f"Spam emails: {len(spam_emails)}")
    print(f"Test emails: {len(test_emails)}")

    # === 2. TRAIN B CELL AGENT ===
    print_separator("2. Training B Cell Agent")

    # Create training antigens
    train_antigens = []
    for email in normal_emails:
        train_antigens.append(Antigen.from_text(email, class_label="normal"))
    for email in spam_emails:
        train_antigens.append(Antigen.from_text(email, class_label="spam"))

    # Generate embeddings
    train_texts = [a.data for a in train_antigens]
    train_embeddings = generate_simple_embeddings(train_texts)

    # Train B cell
    bcell = BCellAgent(agent_name="bcell_email_classifier", affinity_method="hybrid")
    bcell.train(train_antigens, embeddings=train_embeddings)

    stats = bcell.get_statistics()
    print(f"B Cell trained: {stats['num_patterns']} patterns in {stats['num_clones']} clones")
    print(f"Classes: {stats['classes']}")

    # === 3. TRAIN NK CELL AGENT ===
    print_separator("3. Training NK Cell Agent (Negative Selection)")

    # Train on normal emails only (self patterns)
    normal_antigens = [Antigen.from_text(email) for email in normal_emails]
    normal_embeddings = generate_simple_embeddings(normal_emails)

    nk_cell = NKCellAgent(agent_name="nk_cell_anomaly_detector",
                          detection_threshold=0.5,
                          num_detectors=20)
    nk_cell.train_on_self(normal_antigens, embeddings=normal_embeddings)

    nk_stats = nk_cell.get_statistics()
    print(f"NK Cell trained: {nk_stats['num_self_patterns']} self patterns")
    print(f"Generated {nk_stats['num_detectors']} anomaly detectors")

    # === 4. TEST BOTH AGENTS ===
    print_separator("4. Testing on New Emails")

    test_antigens = [Antigen.from_text(email) for email in test_emails]
    test_embeddings = generate_simple_embeddings(test_emails)

    print("\nExpected classifications:")
    print("  1. Normal (meeting/lunch)")
    print("  2. Spam/Anomaly (WIN BIG MONEY)")
    print("  3. Normal (quarterly report)")
    print("  4. Spam/Anomaly (FREE iPhone)")

    print_separator()

    for i, (antigen, embedding) in enumerate(zip(test_antigens, test_embeddings), 1):
        print(f"\n📧 Email {i}: \"{antigen.data}\"")

        # B Cell classification
        bcell_result = bcell.recognize(antigen, antigen_embedding=embedding, strategy="sha")
        print(f"\n  🔬 B Cell (Pattern Matching):")
        print(f"     Class: {bcell_result.predicted_class}")
        print(f"     Confidence: {bcell_result.confidence:.3f}")
        print(f"     Avidity scores: {{{', '.join(f'{k}: {v:.3f}' for k, v in bcell_result.avidity_scores.items())}}}")

        # NK Cell anomaly detection
        nk_result = nk_cell.detect_novelty(antigen, antigen_embedding=embedding)
        print(f"\n  🛡️  NK Cell (Anomaly Detection):")
        print(f"     Is Anomaly: {nk_result.is_anomaly}")
        print(f"     Anomaly Score: {nk_result.anomaly_score:.3f}")
        print(f"     Confidence: {nk_result.confidence:.3f}")
        print(f"     Explanation: {nk_result.explanation}")

        # Combined decision
        print(f"\n  ✅ Combined Decision:")
        if bcell_result.predicted_class == "spam" or nk_result.is_anomaly:
            print(f"     🚨 SPAM/SUSPICIOUS")
            if bcell_result.predicted_class == "spam" and nk_result.is_anomaly:
                print(f"     (Both agents agree: spam + anomaly)")
            elif bcell_result.predicted_class == "spam":
                print(f"     (B Cell detected spam pattern)")
            else:
                print(f"     (NK Cell detected anomaly)")
        else:
            print(f"     ✓ NORMAL")
            print(f"     (Matches normal patterns)")

    # === 5. SUMMARY ===
    print_separator("Summary")

    print("\n🎯 Key Capabilities Demonstrated:")
    print("\n1. **B Cell Agent (Pattern Matching)**:")
    print("   - Learns patterns from labeled examples (normal vs spam)")
    print("   - Calculates affinity to learned patterns")
    print("   - Uses SHA strategy for classification")
    print("   - Hybrid: traditional affinity + embedding similarity")

    print("\n2. **NK Cell Agent (Anomaly Detection)**:")
    print("   - Trains ONLY on normal (self) data")
    print("   - Generates negative selection detectors")
    print("   - Detects non-self patterns without explicit spam examples")
    print("   - Zero-shot anomaly detection capability")

    print("\n3. **Combined Decision**:")
    print("   - B Cell provides supervised classification")
    print("   - NK Cell provides unsupervised anomaly detection")
    print("   - Together: robust spam/threat detection")

    print("\n💡 Biological Inspiration:")
    print("   - B Cells: Produce antibodies for known patterns")
    print("   - NK Cells: Detect abnormal cells without prior exposure")
    print("   - Immune System: Multi-layered defense (adaptive + innate)")

    print_separator()
    print("\n✓ Demo complete!")
    print("\nNext steps:")
    print("  - Add Dendritic Cell for feature extraction")
    print("  - Add Memory Cell for caching results")
    print("  - Integrate real LLM embeddings (OpenAI, Anthropic)")
    print("  - Create MCP server for tool exposure")


if __name__ == "__main__":
    main()
