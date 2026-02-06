# Antigence: Getting Started Guide

## Overview

**Antigence** is a bio-inspired multi-agent security analysis platform using Artificial Immune System (AIS) principles. AI agents play biological immune cell roles to detect vulnerabilities, anomalies, and threats.

**Version**: v0.2.0-beta (January 2026)

## Data Availability (Public-Only)

Antigence **does not** ship proprietary or user-derived data in the public repo.
Any local training artifacts, user data, or model checkpoints are **excluded**.

Publicly available datasets can be downloaded using the provided scripts:

```bash
# Core public datasets (optional)
python scripts/download_datasets.py

# Additional public datasets for training/evaluation (optional)
python scripts/download_extra_datasets.py
python scripts/download_research_datasets.py
python scripts/download_sard_juliet.py
```

If you do not need training data, you can skip these steps.

---

## Quick Start (5 Minutes)

### 1. Install Antigence

```bash
# Option A: pip (recommended)
pip install antigence

# Option B: From source
git clone https://github.com/biobitworks/antigence.git
cd antigence
pip install -e .

# Option C: Homebrew (macOS)
brew install biobitworks/tap/antigence
```

### 2. Install Ollama (for local embeddings)

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# Pull embedding model
ollama pull nomic-embed-text
```

### 3. Run Your First Analysis

```bash
# Add to PATH (one time)
export PATH="$HOME/projects/bin:$PATH"

# Analyze code for vulnerabilities
antigence analyze 'eval(user_input)'

# Output:
# {
#   "risk_level": "HIGH",
#   "classification": "vulnerable",
#   "anomaly": true,
#   "confidence": 0.94
# }
```

---

## Installation Options

### Minimal (Core Only)

```bash
pip install antigence
```

Includes: B Cell, NK Cell, Dendritic, Memory, Orchestrator agents

### With LLM Support

```bash
pip install antigence[llm]
```

Adds: Anthropic, OpenAI integrations

### With Vector Database

```bash
pip install antigence[vector]
```

Adds: ChromaDB, FAISS for memory storage

### Full Installation

```bash
pip install antigence[full]
```

Includes all optional dependencies

### Development

```bash
git clone https://github.com/biobitworks/antigence.git
cd antigence
pip install -e ".[dev]"
```

---

## CLI Commands

```bash
# Show all commands
antigence help

# B Cell: Pattern matching (classification)
antigence scan 'code_snippet'
antigence scan -f suspicious.py

# NK Cell: Anomaly detection (zero-shot)
antigence detect 'code_snippet'
antigence detect -f untrusted.js

# Full multi-agent analysis
antigence analyze 'code_snippet'
antigence analyze -f file.py

# Dendritic: Feature extraction
antigence inspect 'text or code'

# Memory: Context lookup
antigence recall 'query'
```

---

## Python API

### Basic Usage

```python
from immunos_mcp.core import Antigen
from immunos_mcp.agents import BCellAgent, NKCellAgent
from immunos_mcp.orchestrator import ImmunosOrchestrator

# Create an antigen (input to analyze)
antigen = Antigen.from_code("eval(user_input)", language="python")

# Option 1: Use individual agents
bcell = BCellAgent()
result = bcell.recognize(antigen)
print(f"Classification: {result.predicted_class}")

# Option 2: Use orchestrator (recommended)
orchestrator = ImmunosOrchestrator()
result = orchestrator.analyze(antigen)
print(f"Risk: {result.risk_level}, Confidence: {result.confidence}")
```

### Training Your Own Models

```python
from immunos_mcp.agents import BCellAgent, NKCellAgent
from immunos_mcp.core import Antigen

# Prepare training data
safe_code = [
    Antigen.from_code("print('hello')", class_label="safe"),
    Antigen.from_code("x = 1 + 2", class_label="safe"),
]
unsafe_code = [
    Antigen.from_code("eval(input())", class_label="vulnerable"),
    Antigen.from_code("os.system(cmd)", class_label="vulnerable"),
]

# Train B Cell (needs both classes)
bcell = BCellAgent()
bcell.train(safe_code + unsafe_code)

# Train NK Cell (only "self" - safe examples)
nk = NKCellAgent()
nk.train_on_self(safe_code)

# Test
test = Antigen.from_code("exec(user_data)")
bcell_result = bcell.recognize(test)
nk_result = nk.detect_novelty(test)
```

---

## The 5 Antibody Agents

| Agent | Role | CLI | Training |
|-------|------|-----|----------|
| **B Cell** | Pattern classification | `scan` | Labeled examples |
| **NK Cell** | Anomaly detection | `detect` | "Self" only (no anomalies) |
| **Dendritic** | Feature extraction | `inspect` | None (rule-based) |
| **Memory** | Context storage | `recall` | Auto-populated |
| **Orchestrator** | Coordination | `analyze` | None |

See [docs/kb/antibodies.md](docs/kb/antibodies.md) for detailed documentation.

---

## Package Tiers

| Feature | Individual (Free) | Professional ($29/mo) | Enterprise |
|---------|-------------------|----------------------|------------|
| All 5 agents | Yes | Yes | Yes |
| CLI & Python API | Yes | Yes | Yes |
| Pre-trained security models | - | Yes | Yes |
| Web dashboard | Basic | Full | Custom |
| Hugging Face integration | Manual | Auto | Auto + Private |
| Support | Community | Email | SLA |

See [docs/kb/packages.md](docs/kb/packages.md) for full comparison.

---

## Configuration

### Environment Variables

```bash
# Ollama settings
export OLLAMA_BASE_URL="http://localhost:11434"
export ANTIGENCE_EMBED_MODEL="nomic-embed-text"

# For code analysis
export ANTIGENCE_CODE_EMBED_MODEL="nomic-embed-code"

# User data directory
export ANTIGENCE_USER_DATA="~/.antigence"
```

### Configuration File

Create `~/.antigence/config/settings.json`:

```json
{
  "embedding_model": "nomic-embed-text",
  "ollama_url": "http://localhost:11434",
  "default_threshold": 0.8,
  "hardware_tier": 1
}
```

---

## Hardware Requirements

| Tier | Target | RAM | GPU | Models |
|------|--------|-----|-----|--------|
| 0 | IoT/Airgapped | 4 GB | - | qwen2.5:1.5b |
| 1 | Laptop | 8 GB | - | qwen2.5:7b |
| 2 | Workstation | 16 GB | 8+ GB | deepseek-r1:14b |
| 3 | Cloud | 32+ GB | 24+ GB | claude-opus-4.5 |

---

## Project Structure

```
antigence/
├── src/immunos_mcp/
│   ├── core/           # Antigen, Affinity, Protocols
│   ├── agents/         # B Cell, NK Cell, Dendritic, Memory
│   ├── orchestrator/   # Multi-agent coordination
│   ├── embeddings/     # Ollama, HuggingFace embedders
│   ├── training/       # Code, emotion, research trainers
│   └── algorithms/     # NegSl-AIS, opt-ainet
├── web_app/            # Flask dashboard
├── examples/           # Demo scripts
├── docs/
│   └── kb/             # Knowledge base
│       ├── antibodies.md
│       ├── packages.md
│       └── huggingface-integration.md
├── bin/antigence       # CLI tool
└── tests/
```

---

## Examples

### Code Security Scanner

```bash
# Analyze a file
antigence analyze -f suspicious.py

# Batch analysis
for f in src/*.py; do antigence analyze -f "$f"; done
```

### Integration with CI/CD

```yaml
# .github/workflows/security.yml
- name: Antigence Security Scan
  run: |
    pip install antigence
    antigence analyze -f src/ --output json > security-report.json
```

### Web Dashboard

```bash
# Start dashboard
python -m immunos_mcp.web_app.app --port 5001

# Open in browser
open http://localhost:5001
```

---

## Troubleshooting

### Ollama Connection Error

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve
```

### Missing Embedding Model

```bash
# Pull the model
ollama pull nomic-embed-text

# Verify
ollama list
```

### Python Version Issues

ChromaDB requires Python 3.10-3.12. Check your version:

```bash
python --version

# If needed, use pyenv
pyenv install 3.11.0
pyenv local 3.11.0
```

---

## Documentation

- **[Knowledge Base](docs/kb/README.md)** - Full technical documentation
- **[Antibodies Guide](docs/kb/antibodies.md)** - The 5 immune cell agents
- **[Package Tiers](docs/kb/packages.md)** - Individual vs Organization
- **[Hugging Face Integration](docs/kb/huggingface-integration.md)** - Models and datasets
- **[TRAITS Framework](docs/kb/traits.md)** - Core principles

---

## Support

- **Community**: [GitHub Issues](https://github.com/biobitworks/antigence/issues)
- **Documentation**: [docs/kb/](docs/kb/)
- **Professional**: support@biobitworks.com

---

## License

Apache 2.0 (Code) | CC-BY-4.0 (Documentation)

---

**Last Updated**: 2026-01-12
**Version**: Antigence v0.2.0-alpha
