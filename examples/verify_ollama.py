"""
Ollama Connection Verification for IMMUNOS-MCP

Tests:
1. Ollama service is running
2. Required models are installed
3. Each model responds correctly
4. Performance check (response time)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess
import time
import json
from typing import Dict, List, Tuple

# Check if ollama package is available, install if not
try:
    import ollama
except ImportError:
    print("📦 Installing ollama Python package...")
    subprocess.run([sys.executable, "-m", "pip", "install", "ollama"], check=True)
    import ollama


# Required models for 16GB configuration
REQUIRED_MODELS = {
    "qwen2.5-coder:7b": {
        "purpose": "B Cell + Dendritic (code analysis)",
        "size": "4.7GB"
    },
    "deepseek-r1:14b": {
        "purpose": "NK Cell + T Cell + QML (reasoning)",
        "size": "8.0GB"
    },
    "qwen2.5:1.5b": {
        "purpose": "Macrophage (fast triage)",
        "size": "1.0GB"
    }
}


def check_ollama_service() -> bool:
    """Check if Ollama service is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-x", "ollama"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error checking Ollama service: {e}")
        return False


def check_ollama_command() -> bool:
    """Check if ollama command is available."""
    try:
        result = subprocess.run(
            ["which", "ollama"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error checking ollama command: {e}")
        return False


def get_installed_models() -> List[str]:
    """Get list of installed models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse output (skip header line)
        lines = result.stdout.strip().split('\n')[1:]
        models = []
        for line in lines:
            if line.strip():
                # Extract model name (first column)
                model_name = line.split()[0]
                models.append(model_name)
        return models
    except Exception as e:
        print(f"Error getting installed models: {e}")
        return []


def test_model(model_name: str) -> Tuple[bool, float, str]:
    """
    Test a model with a simple prompt.

    Returns:
        (success, response_time, error_message)
    """
    test_prompt = "Say 'OK' if you can read this."

    try:
        start_time = time.time()
        response = ollama.generate(
            model=model_name,
            prompt=test_prompt,
            options={"num_predict": 10}  # Limit tokens for quick test
        )
        elapsed = time.time() - start_time

        if response and 'response' in response:
            return True, elapsed, ""
        else:
            return False, elapsed, "No response received"

    except Exception as e:
        return False, 0.0, str(e)


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70 + "\n")


def print_success(text: str):
    """Print success message."""
    print(f"✓ {text}")


def print_error(text: str):
    """Print error message."""
    print(f"✗ {text}")


def print_warning(text: str):
    """Print warning message."""
    print(f"⚠️  {text}")


def main():
    print_header("IMMUNOS-MCP: Ollama Verification")

    all_passed = True

    # Test 1: Ollama command available
    print("🔍 Test 1: Checking Ollama installation...")
    if check_ollama_command():
        print_success("Ollama command is available")

        # Get version
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            version = result.stdout.strip()
            print(f"   Version: {version}")
        except:
            pass
    else:
        print_error("Ollama command not found")
        print("   Install with: ./examples/setup_ollama.sh")
        all_passed = False
        return 1

    # Test 2: Ollama service running
    print("\n🔍 Test 2: Checking Ollama service...")
    if check_ollama_service():
        print_success("Ollama service is running")
    else:
        print_error("Ollama service is not running")
        print("   Start with: ollama serve")
        all_passed = False
        # Continue anyway - service might start automatically

    # Test 3: List installed models
    print("\n🔍 Test 3: Checking installed models...")
    installed = get_installed_models()

    if installed:
        print_success(f"Found {len(installed)} installed model(s):")
        for model in installed:
            print(f"   • {model}")
    else:
        print_error("No models installed")
        print("   Install with: ./examples/setup_ollama.sh")
        all_passed = False
        return 1

    # Test 4: Check required models
    print("\n🔍 Test 4: Verifying required models for 16GB configuration...")
    print(f"\nRequired models:")
    for model_name, info in REQUIRED_MODELS.items():
        print(f"   • {model_name} ({info['size']}) - {info['purpose']}")
    print()

    missing_models = []
    for model_name in REQUIRED_MODELS.keys():
        # Check if any installed model matches (exact or with tags)
        found = any(model_name in inst for inst in installed)

        if found:
            print_success(f"{model_name} is installed")
        else:
            print_error(f"{model_name} is NOT installed")
            missing_models.append(model_name)
            all_passed = False

    if missing_models:
        print(f"\n⚠️  Missing {len(missing_models)} model(s)")
        print("   Install with: ./examples/setup_ollama.sh")
        print("\n   Or manually:")
        for model in missing_models:
            print(f"     ollama pull {model}")
        return 1

    # Test 5: Test each model with a prompt
    print("\n🔍 Test 5: Testing model responses...")
    print("   (This may take 30-60 seconds)\n")

    model_results = {}
    for model_name in REQUIRED_MODELS.keys():
        print(f"   Testing {model_name}...", end=" ", flush=True)

        success, response_time, error = test_model(model_name)
        model_results[model_name] = (success, response_time, error)

        if success:
            print(f"✓ ({response_time:.2f}s)")
        else:
            print(f"✗ Failed")
            print(f"      Error: {error}")
            all_passed = False

    # Summary
    print_header("Verification Summary")

    if all_passed:
        print("🎉 All tests passed!")
        print("\n✓ Ollama is ready for IMMUNOS-MCP")
        print("✓ All required models are installed and working")

        # Performance summary
        print("\n📊 Performance Summary:")
        for model_name, (success, response_time, _) in model_results.items():
            if success:
                speed = "Fast" if response_time < 2 else "Moderate" if response_time < 5 else "Slow"
                print(f"   • {model_name}: {response_time:.2f}s ({speed})")

        print("\n🚀 Next step: Run the demo")
        print("   python examples/llm_agents_demo.py")

        return 0
    else:
        print("⚠️  Some tests failed")
        print("\nPlease fix the issues above before running the demo.")
        print("\nTroubleshooting:")
        print("  1. Run setup: ./examples/setup_ollama.sh")
        print("  2. Start service: ollama serve")
        print("  3. Check logs: tail -f ~/.ollama/logs/server.log")
        print("\nFor offline deployment, see: docs/Offline-Deployment-Architecture.md")

        return 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nVerification cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
