# Antigence Knowledge Base

## Overview

Technical knowledge base for the Antigence Platform - a bio-inspired multi-agent security analysis system.

## Index

### Core Concepts
- [Antibodies & Cell Agents](antibodies.md) - The 5 immune cell types and their roles
- [Antigen Receptor Logic](antigen-receptor-logic.md) - How inputs are processed and routed
- [Domain Packs](domain-pack-spec.md) - Portable domain-specific configurations

### Architecture
- [System Architecture](architecture.md) - Installation and deployment structure
- [Model Architecture](models.md) - LLM and embedding model mappings

### Integration
- [Hugging Face Integration](huggingface-integration.md) - Models and datasets from HF
- [Ollama Setup](ollama-setup.md) - Local model configuration

### Deployment
- [Package Tiers](packages.md) - Individual vs Organization offerings
- [Installation Guide](installation.md) - Getting started
- [Packaging Standards](packaging.md) - PyPI, Homebrew, Docker specs

### Reference
- [TRAITS Framework](traits.md) - Core principles
- [Evaluation Plan](evaluation.md) - Testing and validation
- [Lexicon](../lexicon.md) - Terminology definitions

---

## Quick Reference

### The 5 Antibodies (Cell Agents)

| Cell | Role | CLI Command |
|------|------|-------------|
| **B Cell** | Pattern matching & classification | `antigence scan` |
| **NK Cell** | Anomaly detection (negative selection) | `antigence detect` |
| **Dendritic** | Feature extraction & signal processing | `antigence inspect` |
| **Memory** | Adaptive memory with decay | `antigence recall` |
| **Orchestrator** | Multi-agent coordination | `antigence analyze` |

### Hardware Tiers

| Tier | Target | Models | Memory |
|------|--------|--------|--------|
| 0 | Airgapped/IoT | qwen2.5:1.5b | 4 GB |
| 1 | Laptop | qwen2.5:7b | 8 GB |
| 2 | Workstation | deepseek-r1:14b | 16 GB |
| 3 | Cloud | claude-opus-4.5 | 32+ GB |

---

**Last Updated**: 2026-01-12
**Version**: Antigence v0.2.0-alpha
