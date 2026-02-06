# Antigence Architecture

## Overview

Antigence is a **hybrid AI security platform** that combines:
- **Classical ML algorithms** (Artificial Immune System / AIS)
- **LLM orchestration** (Ollama) for high-level reasoning

The system uses an **immune system metaphor** where different "cell types" perform specialized security analysis tasks.

---

## Core Concept: Antigents (Not LLMs)

The "antigents" (B Cell, NK Cell, Dendritic, etc.) are **NOT separate LLM models**. They are:

| Agent | Type | Algorithm | LLM Required? |
|-------|------|-----------|---------------|
| **B Cell** | ML Classifier | Affinity-based pattern matching | No |
| **NK Cell** | Anomaly Detector | Negative Selection (NegSl-AIS) | No |
| **Dendritic** | Feature Extractor | Rule-based regex + heuristics | No |
| **Thymus** | Orchestrator | LLM-based reasoning | **Yes** (Ollama) |

---

## Risk Determination Pipeline

```
INPUT (code/claim/text)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  CLASSICAL ML AGENTS (the "antibodies")                 │
│                                                         │
│  [Dendritic] → Feature extraction (20 numeric signals)  │
│       │                                                 │
│       ├──► [B Cell] → Pattern match against trained     │
│       │              clones → verdict + confidence      │
│       │                                                 │
│       └──► [NK Cell] → Negative selection → anomaly     │
│                       score (is this "non-self"?)       │
└─────────────────────────────────────────────────────────┘
         │
         │  B Cell: "SUPPORTS" @ 0.72 confidence
         │  NK Cell: anomaly=True, score=0.85
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  LLM ORCHESTRATOR (Thymus) - OPTIONAL                   │
│                                                         │
│  Weighs signals:                                        │
│  - B Cell says SUPPORTS but NK Cell flagged anomaly     │
│  - → Downgrade confidence, output UNCERTAIN             │
│                                                         │
│  Or resolves conflicts:                                 │
│  - B Cell low confidence + NK Cell no anomaly           │
│  - → LLM reasoning to produce final verdict             │
└─────────────────────────────────────────────────────────┘
         │
         ▼
    FINAL RISK OUTPUT
    - verdict: SUPPORTS | CONTRADICTS | UNCERTAIN | DANGER
    - confidence: 0.0 - 1.0
    - explanation: "B Cell matched pattern X, NK flagged anomaly"
```

---

## Agent Details

### B Cell Agent (`bcell_agent.py`)

**Purpose**: Pattern-based classification (like antibodies recognizing antigens)

**Algorithm**: Affinity-based pattern matching (inspired by Immunos-81)

**How it works**:
1. **Training**: Learns patterns from labeled data, groups them into "clones" by class
2. **Recognition**: Calculates "affinity" (similarity) between new input and learned patterns
3. **Output**: Predicted class + confidence score

**Affinity Calculation** (hybrid mode):
- 70% embedding similarity (cosine distance)
- 30% traditional (character overlap, numeric distance)

**The "Antibodies"**: When trained, B Cell creates pattern clones stored in `.pkl` files:
```python
bcell.train(antigens=[
    Antigen(data="safe code", class_label="safe"),
    Antigen(data="SQL injection", class_label="vulnerable"),
])
bcell.save_state("scifact-bcell-2026-01-05.pkl")
```

---

### NK Cell Agent (`nk_cell_agent.py`)

**Purpose**: Anomaly detection (detecting "non-self" / unknown threats)

**Algorithm**: Negative Selection Algorithm (NegSl-AIS)

**How it works**:
1. **Training**: Learn what "self" (normal) looks like
2. **Detector Generation**: Create detectors that DON'T match self patterns
3. **Detection**: If input matches a detector → it's anomalous (non-self)

**Key Equation** (NegSl-AIS):
```
detector is valid if: distance(detector, nearest_self) > threshold
```

**Modes**:
- `embedding`: Uses pre-computed embeddings for semantic similarity
- `feature`: Uses Dendritic-extracted numeric features

**Output**: `AnomalyResult` with:
- `is_anomaly`: Boolean
- `anomaly_score`: 0.0 - 1.0

---

### Enhanced NK Cell (`nk_cell_enhanced.py`)

**Improvements over basic NK Cell**:
- Per-class detector generation (one-vs-rest)
- Adaptive threshold calculation
- Quality metrics for detector effectiveness

---

### Dendritic Agent (`dendritic_agent.py`)

**Purpose**: Feature extraction (like dendritic cells processing antigens)

**Algorithm**: Rule-based regex + hand-crafted heuristics

**Extracts 20 numeric features**:

| Category | Features |
|----------|----------|
| Text Structure (5) | length, tokens, sentences, avg word length, avg sentence length |
| Claim Characteristics (6) | citations, numbers, hedging words, certainty words, questions, exclamations |
| Semantic Signals (5) | exaggeration, specificity, subjectivity, sentiment, controversy |
| Danger Signals (4) | PAMP score, danger count, contradictions, credibility |

**Output**:
- Feature dictionary (named features)
- Numeric vector (20 floats) for downstream ML
- Signal classification: DANGER / SAFE / NEUTRAL

---

### Thymus Orchestrator (`ollama_integration.py`)

**Purpose**: Coordinate agent outputs into final verdict using LLM reasoning

**Algorithm**: Prompt-based LLM reasoning via Ollama

**How it works**:
1. Receives outputs from B Cell and NK Cell
2. Constructs prompt with agent signals
3. Queries Ollama for final verdict
4. Adjusts confidence based on signal agreement/conflict

**Input**:
```python
orchestrator.orchestrate_validation(
    claim="Aspirin reduces heart attack risk",
    bcell_verdict="SUPPORTS",
    bcell_confidence=0.72,
    nk_anomaly=True,
    evidence_sentences=["Study X showed..."]
)
```

**Output**:
```python
{
    "final_verdict": "UNCERTAIN",
    "confidence_adjustment": -0.15,
    "llm_reasoning": "B Cell supports but NK flagged anomaly...",
    "llm_model": "llama3.2:3b"
}
```

**Fallback**: If Ollama unavailable, uses deterministic logic:
- NK anomaly + B Cell support → downgrade to UNCERTAIN
- High B Cell confidence + no anomaly → trust B Cell

---

## LLM Role Matrix

The LLM Role Matrix allows assigning **different Ollama models to different roles**.

### Default Roles

| Role | Label | Purpose |
|------|-------|---------|
| `orchestrator` | Thymus | Coordinates antigents; final verdicts |
| `bcell` | B Cell | Evidence extraction, citation checks |
| `nk` | NK Cell | Flag anomalies, risk summaries |
| `tcell_security` | T Cell | Secure-code review |
| `dendritic_summarizer` | Dendritic | Summarize artifacts |

### Configuration

**Option 1: Environment Variable**
```bash
export ANTIGENCE_LLM_ROLE_MODELS='{"orchestrator":"llama3.2:3b","tcell_security":"codellama:7b"}'
```

**Option 2: Config File** (`~/.antigence/llm/roles.json`)
```json
{
  "orchestrator": "llama3.2:3b",
  "tcell_security": "codellama:7b",
  "dendritic_summarizer": "mistral:7b"
}
```

### Current Status

Currently, only the **orchestrator role** is actively used (Thymus). Other roles are placeholders for future LLM-augmented agents. The core B Cell/NK Cell agents use classical ML, not LLMs.

---

## File Locations

```
/Users/byron/projects/active_projects/antigence/
├── src/immunos_mcp/
│   ├── agents/
│   │   ├── bcell_agent.py          # Pattern matcher
│   │   ├── nk_cell_agent.py        # Anomaly detector (basic)
│   │   ├── nk_cell_enhanced.py     # Anomaly detector (advanced)
│   │   ├── dendritic_agent.py      # Feature extractor
│   │   └── hallucination_dendritic.py
│   ├── core/
│   │   ├── affinity.py             # Affinity calculations
│   │   ├── antigen.py              # Data structure
│   │   └── protocols.py            # Shared types
│   ├── algorithms/
│   │   ├── negsel.py               # NegSl-AIS (Equation 20)
│   │   ├── negsel_torch.py         # GPU-accelerated
│   │   └── opt_ainet.py            # Optimization AINet
│   └── embeddings/
│       └── simple_text_embedder.py
├── web_app/
│   ├── app.py                      # Flask application
│   ├── ollama_integration.py       # Thymus orchestrator
│   ├── llm_roles.py                # LLM Role Matrix
│   └── templates/sentinel.html     # Ops dashboard
└── .immunos/models/
    ├── scifact-bcell-*.pkl         # Trained B Cell patterns
    └── scifact-nk-*.pkl            # Trained NK Cell detectors
```

---

## Training & Embeddings

### B Cell Training

```python
from immunos_mcp.agents.bcell_agent import BCellAgent
from immunos_mcp.core.antigen import Antigen

bcell = BCellAgent(affinity_method="hybrid")

# Training data with labels
antigens = [
    Antigen.from_text("safe pattern", class_label="safe"),
    Antigen.from_text("dangerous pattern", class_label="vulnerable"),
]

# Pre-computed embeddings (from Sentence-BERT, etc.)
embeddings = compute_embeddings(antigens)

# Train and save
bcell.train(antigens, embeddings=embeddings)
bcell.save_state("bcell-model.pkl")
```

### NK Cell Training

```python
from immunos_mcp.agents.nk_cell_agent import NKCellAgent

nk = NKCellAgent(mode="embedding")

# Train on "self" (normal) data only
normal_antigens = [...]
normal_embeddings = compute_embeddings(normal_antigens)

nk.train_on_self(normal_antigens, embeddings=normal_embeddings)
nk.save_state("nk-model.pkl")
```

### Embedding Sources

Agents don't generate embeddings—they USE pre-computed ones:
- Sentence-BERT (`all-MiniLM-L6-v2`)
- OpenAI embeddings
- Custom embedders

---

## Summary Table

| Component | ML Type | Trained? | Uses LLM? | Output |
|-----------|---------|----------|-----------|--------|
| **Dendritic** | Rule-based | No | No | 20 features + signal type |
| **B Cell** | Affinity classifier | Yes (patterns/clones) | No | Class + confidence |
| **NK Cell** | Neg. selection | Yes (detectors) | No | Anomaly bool + score |
| **Thymus** | LLM orchestrator | No | **Yes** | Final verdict + reasoning |

---

## Key Takeaways

1. **Hybrid Architecture**: Classical AIS algorithms + optional LLM coordination
2. **Antibodies = Trained Patterns**: B Cell clones and NK Cell detectors are the "immune memory"
3. **LLM is Optional**: System works without Ollama (uses deterministic fallback)
4. **Role Matrix**: Future-proofing for per-role LLM customization
5. **Interpretable**: Each agent produces structured, explainable output

---

## Antibody → Antigent Trigger Implementation

The unified LLM caller is in `antigent_llm.py`:

```python
from antigent_llm import get_antigent_llm

antigent = get_antigent_llm()

# B Cell LLM (when confidence < 0.7)
response = antigent.bcell_analyze(claim, antibody_result)

# NK Cell LLM (when anomaly detected)
response = antigent.nk_risk_summary(input_data, antibody_result)

# T Cell Security LLM (always for code)
response = antigent.tcell_security_review(code, quick_scan_result)

# Dendritic Summarizer (for long text)
response = antigent.dendritic_summarize(text, features)

# Thymus Orchestrator (final coordination)
response = antigent.orchestrate(all_signals)
```

### Trigger Conditions

| Role | Trigger Condition | Endpoint |
|------|-------------------|----------|
| `bcell` | `confidence < 0.7` in deep/orchestrated mode | `/api/validate_publications` |
| `nk` | `is_anomaly == True` in deep/orchestrated mode | `/api/validate_publications` |
| `tcell_security` | Always in `pipeline=deep` | `/api/scan` |
| `dendritic_summarizer` | `len(text) > 500` | (future) |
| `orchestrator` | Always in `mode=orchestrated` | `/api/validate_publications` |

### Events Logging

All antigent calls are logged to `~/.antigence/antigent_events.jsonl`:

```json
{
  "ts": "2026-01-11T10:52:00",
  "type": "antigent_call",
  "role": "tcell_security",
  "model": "codellama:7b",
  "success": true,
  "tokens": 256,
  "latency_ms": 1234.5,
  "prompt_preview": "Review this code...",
  "context_keys": ["bcell_vulnerable", "nkcell_anomaly"]
}
```

View antigent events in the Sentinel Ops dashboard at `/sentinel`.

---

## Related Documentation

- [README.md](../README.md) - Quickstart and setup
- [IMMUNE_SYSTEM_MAP.md](./IMMUNE_SYSTEM_MAP.md) - Full immune metaphor
- [Sentinel Ops](../templates/sentinel.html) - Ops dashboard

---

*Last Updated: 2026-01-11*
