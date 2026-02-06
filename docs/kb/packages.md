# Antigence Package Tiers

## Overview

Antigence is available in three deployment tiers designed for different use cases: **Individual** (open source), **Professional** (small teams), and **Enterprise** (organizations).

---

## Tier Comparison

| Feature | Individual (OSS) | Professional | Enterprise |
|---------|------------------|--------------|------------|
| **Price** | Free | $29/mo | Custom |
| **License** | Apache 2.0 | Commercial | Commercial |
| **Support** | Community | Email | SLA + Dedicated |

### Core Agents

| Agent | Individual | Professional | Enterprise |
|-------|------------|--------------|------------|
| B Cell (Pattern Matching) | Yes | Yes | Yes |
| NK Cell (Anomaly Detection) | Yes | Yes | Yes |
| Dendritic (Feature Extraction) | Yes | Yes | Yes |
| Memory (T Cell) | Yes | Yes | Yes |
| Orchestrator | Yes | Yes | Yes |
| Sentinel (File Monitor) | - | Yes | Yes |

### Models & Training

| Feature | Individual | Professional | Enterprise |
|---------|------------|--------------|------------|
| Pre-trained SciFact models | Yes | Yes | Yes |
| Pre-trained code security models | - | Yes | Yes |
| Hugging Face model auto-download | - | Yes | Yes |
| Custom model training scripts | - | Yes | Yes |
| Fine-tuning on private data | - | - | Yes |
| Model versioning & rollback | - | - | Yes |

### Interfaces

| Interface | Individual | Professional | Enterprise |
|-----------|------------|--------------|------------|
| CLI (`antigence` command) | Yes | Yes | Yes |
| Python API | Yes | Yes | Yes |
| Web Dashboard | Basic | Full | Full + Custom |
| REST API | - | Yes | Yes |
| MCP Server | - | Yes | Yes |

### Deployment

| Option | Individual | Professional | Enterprise |
|--------|------------|--------------|------------|
| Local (Ollama) | Yes | Yes | Yes |
| Docker | Yes | Yes | Yes |
| Kubernetes | - | - | Yes |
| Air-gapped | - | - | Yes |
| Multi-tenant | - | - | Yes |

### Data & Integrations

| Feature | Individual | Professional | Enterprise |
|---------|------------|--------------|------------|
| Local embeddings (nomic-embed-text) | Yes | Yes | Yes |
| Hugging Face datasets | Manual | Auto | Auto + Private |
| CI/CD integration | - | GitHub Actions | Any |
| SIEM integration | - | - | Yes |
| SSO/SAML | - | - | Yes |

---

## Individual (Open Source)

**Best for**: Researchers, students, hobbyists, and open source contributors.

### What's Included

- **All 5 core agents**: B Cell, NK Cell, Dendritic, Memory, Orchestrator
- **CLI tool**: Full `antigence` command suite
- **Python API**: Import and use in your projects
- **Local-first**: Ollama embeddings, no external APIs required
- **Pre-trained models**: SciFact claim verification
- **Basic web dashboard**: View analysis results
- **Documentation**: Full access to KB and guides

### Installation

```bash
# Via pip
pip install antigence

# Via Homebrew (macOS)
brew install biobitworks/tap/antigence

# From source
git clone https://github.com/biobitworks/antigence.git
cd antigence && pip install -e .
```

### Limitations

- No pre-trained code security models (must train your own)
- No Sentinel file monitoring
- Community support only (GitHub Issues)
- No commercial use without attribution

---

## Professional

**Best for**: Small teams, startups, security consultants, and DevSecOps.

### What's Included

Everything in Individual, plus:

- **Pre-trained security models**: CodeBERT, GraphCodeBERT vulnerability detection
- **Sentinel file monitor**: Real-time file integrity monitoring
- **Full web dashboard**: Advanced visualizations and ticket management
- **REST API**: Integrate with your tools
- **MCP Server**: Claude Code integration
- **Hugging Face auto-download**: Automatic model fetching
- **Training scripts**: Train on your own datasets
- **Email support**: 48-hour response time

### Recommended Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| GPU | - | 8+ GB VRAM |
| Storage | 50 GB | 100 GB SSD |

### Pre-trained Models Included

| Model | Task | Source |
|-------|------|--------|
| `antigence-bcell-scifact-v1` | Claim verification | SciFact |
| `antigence-bcell-codesec-v1` | Vulnerability detection | DiverseVul |
| `antigence-nk-codesec-v1` | Code anomaly detection | Safe code corpus |

### Pricing

- **Monthly**: $29/user/month
- **Annual**: $290/user/year (2 months free)
- **Team (5+)**: Contact for volume discount

---

## Enterprise

**Best for**: Large organizations, government, healthcare, and finance.

### What's Included

Everything in Professional, plus:

- **Custom model fine-tuning**: Train on your proprietary data
- **Model versioning**: Rollback and A/B testing
- **Multi-tenant deployment**: Isolate teams/projects
- **Air-gapped deployment**: No internet required
- **Kubernetes operator**: Scalable deployment
- **CI/CD integration**: Jenkins, GitLab CI, Azure DevOps
- **SIEM integration**: Splunk, Elastic, QRadar
- **SSO/SAML**: Okta, Azure AD, OneLogin
- **Audit logging**: Full compliance trail
- **SLA support**: 4-hour response, dedicated CSM
- **On-premise option**: Deploy in your data center

### Compliance & Certifications

| Standard | Status |
|----------|--------|
| SOC 2 Type II | Planned Q2 2026 |
| HIPAA | Available on request |
| FedRAMP | Roadmap |
| ISO 27001 | Roadmap |

### Deployment Options

```
┌─────────────────────────────────────────────────────────────┐
│                    ENTERPRISE DEPLOYMENT                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐          │
│  │  Team A   │    │  Team B   │    │  Team C   │  Tenants  │
│  └─────┬─────┘    └─────┬─────┘    └─────┬─────┘          │
│        │                │                │                 │
│        └────────────────┼────────────────┘                 │
│                         │                                  │
│                         ▼                                  │
│              ┌─────────────────────┐                       │
│              │   API Gateway       │                       │
│              │   (Auth + Rate Limit)│                      │
│              └──────────┬──────────┘                       │
│                         │                                  │
│        ┌────────────────┼────────────────┐                 │
│        │                │                │                 │
│        ▼                ▼                ▼                 │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐           │
│  │ Antigence │   │ Antigence │   │ Antigence │  Replicas  │
│  │  Pod 1    │   │  Pod 2    │   │  Pod 3    │           │
│  └───────────┘   └───────────┘   └───────────┘           │
│                         │                                  │
│                         ▼                                  │
│              ┌─────────────────────┐                       │
│              │  Shared Model Store │                       │
│              │  (S3/MinIO/NFS)     │                       │
│              └─────────────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Pricing

- **Base**: $500/month (includes 10 users)
- **Additional users**: $30/user/month
- **Air-gapped/On-premise**: Custom pricing
- **Government**: GSA Schedule available

---

## Feature Matrix (Detailed)

### Agents & Algorithms

| Feature | Individual | Professional | Enterprise |
|---------|------------|--------------|------------|
| B Cell (Immunos-81 + embeddings) | Yes | Yes | Yes |
| NK Cell (NegSl-AIS) | Yes | Yes | Yes |
| Enhanced NK Cell (adaptive threshold) | Yes | Yes | Yes |
| Dendritic (20 features) | Yes | Yes | Yes |
| Memory (priority decay) | Yes | Yes | Yes |
| Orchestrator (multi-agent) | Yes | Yes | Yes |
| Sentinel (file integrity) | - | Yes | Yes |
| T Killer (output validation) | Roadmap | Roadmap | Roadmap |
| T Regulatory (calibration) | Roadmap | Roadmap | Roadmap |

### Embedding Models

| Model | Individual | Professional | Enterprise |
|-------|------------|--------------|------------|
| nomic-embed-text | Yes | Yes | Yes |
| nomic-embed-text-v2-moe | - | Yes | Yes |
| nomic-embed-code | - | Yes | Yes |
| mxbai-embed-large | - | Yes | Yes |
| Custom embeddings | - | - | Yes |

### Domain Packs

| Domain | Individual | Professional | Enterprise |
|--------|------------|--------------|------------|
| Research (SciFact) | Yes | Yes | Yes |
| Code Security | - | Yes | Yes |
| Hallucination | - | Yes | Yes |
| Network (NSL-KDD) | - | - | Yes |
| Custom domains | - | - | Yes |

---

## Getting Started

### Individual

```bash
pip install antigence
antigence --help
```

### Professional

```bash
pip install antigence[pro]
antigence activate --license YOUR_LICENSE_KEY
```

### Enterprise

Contact sales@biobitworks.com for deployment consultation.

---

## FAQ

**Q: Can I use Individual for commercial projects?**
A: Yes, with Apache 2.0 attribution. Pre-trained security models require Professional license.

**Q: Can I train my own models with Individual?**
A: Yes, using the training scripts in `training/`. Professional includes pre-trained models.

**Q: What's the difference between Professional REST API and Python API?**
A: Python API is local; REST API runs as a service for team access.

**Q: Is my data sent to the cloud?**
A: No. All processing is local by default. Enterprise cloud option available.

**Q: Can I downgrade from Professional to Individual?**
A: Yes. You'll lose access to pre-trained security models and Sentinel.

---

**Last Updated**: 2026-01-12
**Version**: Antigence v0.2.0-alpha
