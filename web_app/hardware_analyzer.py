"""
Hardware Analysis for Antigence Platform
Detects system capabilities and recommends optimal Ollama models
"""

import os
import platform
import subprocess
import psutil
import json
from pathlib import Path


def get_cpu_info():
    """Get CPU information"""
    try:
        if platform.system() == "Darwin":  # macOS
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True
            )
            cpu_name = result.stdout.strip()

            # Check for Apple Silicon
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True
            )
            is_apple_silicon = "Apple" in result.stdout

            cpu_count = psutil.cpu_count(logical=False)
            cpu_threads = psutil.cpu_count(logical=True)

            return {
                "name": cpu_name,
                "cores": cpu_count,
                "threads": cpu_threads,
                "is_apple_silicon": is_apple_silicon,
                "architecture": "ARM64" if is_apple_silicon else "x86_64"
            }
        else:
            return {
                "name": platform.processor(),
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
                "is_apple_silicon": False,
                "architecture": platform.machine()
            }
    except Exception as e:
        return {"error": str(e)}


def get_memory_info():
    """Get RAM information"""
    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / (1024**3), 2),
        "available_gb": round(mem.available / (1024**3), 2),
        "percent_used": mem.percent
    }


def get_gpu_info():
    """Get GPU information (if available)"""
    gpu_info = {
        "available": False,
        "name": None,
        "vram_gb": None
    }

    try:
        if platform.system() == "Darwin":
            # Check for Metal support (Apple Silicon)
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if "Chipset Model" in result.stdout or "Apple" in result.stdout:
                gpu_info["available"] = True
                gpu_info["name"] = "Apple Silicon GPU (Metal)"
                # Apple Silicon has unified memory
                mem = psutil.virtual_memory()
                gpu_info["vram_gb"] = "Unified Memory"
        else:
            # Try nvidia-smi for NVIDIA GPUs
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_info["available"] = True
                parts = result.stdout.strip().split(",")
                gpu_info["name"] = parts[0].strip()
                gpu_info["vram_gb"] = parts[1].strip()
    except Exception:
        pass

    return gpu_info


def recommend_ollama_model(cpu_info, memory_info, gpu_info):
    """
    Recommend optimal Ollama model based on hardware

    Model sizes and requirements:
    - 1B-3B params: 2-4GB RAM, fast on CPU
    - 7B params: 8GB RAM minimum, good balance
    - 13B params: 16GB RAM, better quality
    - 34B+ params: 32GB+ RAM, highest quality
    """
    ram_gb = memory_info["total_gb"]
    has_gpu = gpu_info["available"]
    is_apple_silicon = cpu_info.get("is_apple_silicon", False)

    recommendations = []

    # Apple Silicon optimized (Metal acceleration)
    if is_apple_silicon:
        if ram_gb >= 16:
            recommendations.append({
                "model": "qwen2.5-coder:7b",
                "size": "7B",
                "ram_required": "8GB",
                "speed": "Fast (Metal)",
                "quality": "High",
                "use_case": "Code analysis, reasoning",
                "priority": 1
            })
            recommendations.append({
                "model": "llama3.2:3b",
                "size": "3B",
                "ram_required": "4GB",
                "speed": "Very Fast (Metal)",
                "quality": "Good",
                "use_case": "Quick validation",
                "priority": 2
            })
        elif ram_gb >= 8:
            recommendations.append({
                "model": "qwen2.5:3b",
                "size": "3B",
                "ram_required": "4GB",
                "speed": "Very Fast (Metal)",
                "quality": "Good",
                "use_case": "General purpose",
                "priority": 1
            })
            recommendations.append({
                "model": "llama3.2:1b",
                "size": "1B",
                "ram_required": "2GB",
                "speed": "Extremely Fast (Metal)",
                "quality": "Moderate",
                "use_case": "Ultra-fast screening",
                "priority": 2
            })

    # Standard CPU/GPU recommendations
    else:
        if ram_gb >= 32 and has_gpu:
            recommendations.append({
                "model": "llama3.1:13b",
                "size": "13B",
                "ram_required": "16GB",
                "speed": "Medium (GPU)",
                "quality": "Very High",
                "use_case": "Maximum accuracy",
                "priority": 1
            })
        elif ram_gb >= 16:
            recommendations.append({
                "model": "qwen2.5-coder:7b",
                "size": "7B",
                "ram_required": "8GB",
                "speed": "Medium",
                "quality": "High",
                "use_case": "Balanced performance",
                "priority": 1
            })
        elif ram_gb >= 8:
            recommendations.append({
                "model": "qwen2.5:3b",
                "size": "3B",
                "ram_required": "4GB",
                "speed": "Fast",
                "quality": "Good",
                "use_case": "Standard validation",
                "priority": 1
            })

    # Always include lightweight option
    recommendations.append({
        "model": "tinyllama:1.1b",
        "size": "1.1B",
        "ram_required": "2GB",
        "speed": "Extremely Fast",
        "quality": "Basic",
        "use_case": "Emergency fallback",
        "priority": 3
    })

    # Sort by priority
    recommendations.sort(key=lambda x: x["priority"])

    return recommendations


def check_ollama_installed():
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return {
            "installed": True,
            "running": result.returncode == 0,
            "models": parse_ollama_models(result.stdout) if result.returncode == 0 else []
        }
    except FileNotFoundError:
        return {
            "installed": False,
            "running": False,
            "models": []
        }
    except Exception as e:
        return {
            "installed": True,
            "running": False,
            "error": str(e),
            "models": []
        }


def parse_ollama_models(output):
    """Parse ollama list output"""
    models = []
    for line in output.strip().split("\n")[1:]:  # Skip header
        if line.strip():
            parts = line.split()
            if parts:
                models.append(parts[0])
    return models


def analyze_system():
    """Complete system analysis"""
    cpu = get_cpu_info()
    memory = get_memory_info()
    gpu = get_gpu_info()
    ollama = check_ollama_installed()
    recommendations = recommend_ollama_model(cpu, memory, gpu)

    return {
        "system": {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version()
        },
        "cpu": cpu,
        "memory": memory,
        "gpu": gpu,
        "ollama": ollama,
        "recommendations": recommendations,
        "optimal_model": recommendations[0] if recommendations else None
    }


def main():
    """CLI interface for hardware analysis"""
    print("=" * 60)
    print("Antigence Hardware Analysis")
    print("=" * 60)
    print()

    analysis = analyze_system()

    # System Info
    print(f"OS: {analysis['system']['os']} {analysis['system']['os_version']}")
    print()

    # CPU
    cpu = analysis['cpu']
    print(f"CPU: {cpu.get('name', 'Unknown')}")
    print(f"  Cores: {cpu.get('cores', 'Unknown')} physical, {cpu.get('threads', 'Unknown')} threads")
    print(f"  Architecture: {cpu.get('architecture', 'Unknown')}")
    if cpu.get('is_apple_silicon'):
        print(f"  ✅ Apple Silicon detected (Metal acceleration available)")
    print()

    # Memory
    mem = analysis['memory']
    print(f"Memory: {mem['total_gb']} GB total, {mem['available_gb']} GB available ({mem['percent_used']}% used)")
    print()

    # GPU
    gpu = analysis['gpu']
    if gpu['available']:
        print(f"GPU: {gpu['name']}")
        print(f"  VRAM: {gpu['vram_gb']}")
    else:
        print("GPU: None detected")
    print()

    # Ollama
    ollama = analysis['ollama']
    if ollama['installed']:
        if ollama['running']:
            print(f"✅ Ollama: Installed and running")
            if ollama['models']:
                print(f"  Installed models: {', '.join(ollama['models'])}")
            else:
                print(f"  No models installed")
        else:
            print(f"⚠️  Ollama: Installed but not running")
    else:
        print(f"❌ Ollama: Not installed")
        print(f"  Install: https://ollama.ai/download")
    print()

    # Recommendations
    print("Recommended Models:")
    print("-" * 60)
    for i, rec in enumerate(analysis['recommendations'][:3], 1):
        priority_badge = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"{priority_badge} {rec['model']}")
        print(f"   Size: {rec['size']} | RAM: {rec['ram_required']} | Speed: {rec['speed']}")
        print(f"   Quality: {rec['quality']} | Use: {rec['use_case']}")

        # Show install command if not installed
        if ollama['installed'] and ollama['running']:
            if rec['model'] not in ollama.get('models', []):
                print(f"   Install: ollama pull {rec['model']}")
        print()

    # Save to file
    output_path = Path(".immunos/system_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(analysis, indent=2, fp=f)
    print(f"Analysis saved to: {output_path}")

    return analysis


if __name__ == "__main__":
    main()
