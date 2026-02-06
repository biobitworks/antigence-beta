# Antibodies & Cell Agents

## Overview

Antigence implements **Artificial Immune System (AIS)** principles using 5 core "antibody" agents, each mapping to a biological immune cell type. This document explains what each agent does, how they work together, and when to use them.

---

## The 5 Core Antibodies

### 1. B Cell Agent (Pattern Matcher)

**Biological Inspiration**: B cells produce antibodies that bind to specific antigens (pathogens).

**Role**: Supervised classification against learned patterns.

| Property | Value |
|----------|-------|
| **File** | `agents/bcell_agent.py` |
| **CLI** | `antigence scan 'code'` |
| **Training** | Labeled examples (safe/unsafe, spam/ham, support/contradict) |
| **Output** | Classification + confidence + avidity scores |

**How It Works**:
1. Learns patterns from labeled training data
2. Groups patterns into **clones** (families of similar patterns)
3. Calculates **affinity** (similarity) using hybrid method:
   - 30% traditional (Immunos-81 character matching)
   - 70% embedding (semantic similarity via LLM)
4. Computes **avidity**: `sum(affinities) * log(1 + clone_size)`
5. Classifies via **RHA** (Relative Highest Avidity) with 5% threshold

**Use Cases**:
- Vulnerability classification (safe vs vulnerable code)
- Claim verification (support/contradict/NEI)
- Spam/sentiment detection
- Multi-class text classification

**Example**:
```bash
antigence scan 'eval(user_input)'
# Output: {"classification": "vulnerable", "confidence": 0.92, "cwe": "CWE-95"}
```

---

### 2. NK Cell Agent (Anomaly Detector)

**Biological Inspiration**: Natural Killer cells detect abnormal cells without prior sensitization.

**Role**: Zero-shot anomaly detection via negative selection.

| Property | Value |
|----------|-------|
| **File** | `agents/nk_cell_agent.py`, `agents/nk_cell_enhanced.py` |
| **CLI** | `antigence detect 'code'` |
| **Training** | Only "self" (normal/safe) patterns - NO anomaly examples needed |
| **Output** | is_anomaly + anomaly_score + confidence |

**How It Works** (Negative Selection Algorithm - NegSl-AIS):
1. Train ONLY on "self" patterns (known-good examples)
2. Generate random **detector** vectors
3. Keep detectors that DON'T match self (negative selection)
4. Detection: if ANY detector matches input → anomaly

**Key Equation** (from Umair et al. 2025):
```
Detector valid if: distance(detector, nearest_self) > r_self
```

**Optimal Parameters** (from NegSl-AIS paper):
| Preset | Detectors | r_self | Use Case |
|--------|-----------|--------|----------|
| GENERAL | 20 | 0.85 | Default |
| LLM_HALLUCINATION | 50 | 0.15 | Hallucination detection |
| CODE_SECURITY | 25 | 0.90 | Vulnerability detection |

**Use Cases**:
- Intrusion detection (train on normal traffic)
- Code security (train on safe code, detect malicious)
- Novel threat detection (no prior exposure needed)
- Content moderation (train on acceptable content)

**Example**:
```bash
antigence detect 'os.system(user_input)'
# Output: {"anomaly": true, "score": 0.87, "severity": "HIGH"}
```

---

### 3. Dendritic Cell Agent (Feature Extractor)

**Biological Inspiration**: Dendritic cells capture, process, and present antigens to other immune cells.

**Role**: Feature extraction and signal classification.

| Property | Value |
|----------|-------|
| **File** | `agents/dendritic_agent.py` |
| **CLI** | `antigence inspect 'code'` |
| **Training** | Rule-based (no training required) |
| **Output** | 20 features + signal classification |

**Features Extracted** (20 total):

| Category | Features | Count |
|----------|----------|-------|
| Text Structure | word_count, avg_sentence_length, complexity | 5 |
| Claim Characteristics | hedging, certainty, exaggeration | 6 |
| Semantic Signals | citations, negation, domain_terms | 5 |
| Danger Signals | medical_claims, guarantees, extreme_language | 4 |

**Signal Types** (from Danger Theory):
| Signal | Meaning | Example |
|--------|---------|---------|
| **PAMP** | Known threat pattern | SQL injection signature |
| **DANGER** | Suspicious context | Absolute certainty + no citations |
| **SAFE** | Verified benign | Proper hedging + citations |
| **INFLAMMATORY** | Escalation needed | Multiple danger signals |

**Example**:
```bash
antigence inspect 'This GUARANTEED method will 100% cure your illness!'
# Output: {"danger_score": 0.95, "signals": {"DANGER": 0.8, "PAMP": 0.3}}
```

---

### 4. Memory Agent (T Cell Memory)

**Biological Inspiration**: T cells remember past infections for rapid future response.

**Role**: Adaptive memory with priority-based retention.

| Property | Value |
|----------|-------|
| **File** | `agents/memory_agent.py` |
| **CLI** | `antigence recall 'query'` |
| **Storage** | JSON + optional vector DB (ChromaDB/FAISS) |
| **Output** | Similar past cases + confidence |

**Memory Entry Structure**:
```python
@dataclass
class MemoryEntry:
    key: str
    value: Any
    priority: str       # critical, high, medium, low
    embedding: List[float]
    access_count: int
    relevance_score: float
```

**Adaptive Decay Formula**:
```
score = base_score * exp(-decay_rate * age_days / priority_multiplier)
```

| Priority | Multiplier | Half-life |
|----------|------------|-----------|
| critical | 10x | ~100 days |
| high | 5x | ~50 days |
| medium | 2x | ~20 days |
| low | 1x | ~10 days |

**Use Cases**:
- Few-shot learning (retrieve similar examples)
- Context persistence across sessions
- Pattern stabilization (reduce false positives)

---

### 5. Orchestrator (T Helper / Thymus)

**Biological Inspiration**: T Helper cells coordinate the immune response; Thymus trains T cells.

**Role**: Multi-agent coordination and final verdict.

| Property | Value |
|----------|-------|
| **File** | `orchestrator/orchestrator.py` |
| **CLI** | `antigence analyze 'code'` |
| **Agents Used** | All (B Cell, NK Cell, Dendritic, Memory) |
| **Output** | Aggregated verdict + risk level |

**Coordination Flow**:
```
Input → Dendritic (features) → B Cell (classify) + NK Cell (anomaly)
                             ↓
                    Memory (context)
                             ↓
                    Orchestrator (aggregate)
                             ↓
                    Final Verdict
```

**Confidence Calibration**:
```python
base_conf = (bcell_confidence + nk_confidence) / 2
adjusted = base_conf * dendritic_signal_weight * domain_factor
```

**Risk Levels**:
| Level | Criteria |
|-------|----------|
| **HIGH** | anomaly=True AND confidence > 0.8 |
| **MEDIUM** | anomaly=True OR confidence 0.5-0.8 |
| **LOW** | anomaly=False AND confidence > 0.8 |
| **UNCERTAIN** | confidence < 0.5 |

---

## Additional Components

### Sentinel (File System Immunity)

**Role**: Monitor file changes against baseline hashes.

| Property | Value |
|----------|-------|
| **Location** | `~/.antigence/sentinel/` |
| **Baseline** | `baseline.json` (known-good file hashes) |
| **Events** | `events.jsonl` (change log) |

**Detection Types**:
- `modified`: File content changed
- `new`: File not in baseline
- `deleted`: File removed
- `unexpected`: Sensitive file changed

---

## Pipeline Diagram

```
                    INPUT ANTIGEN (code/text/claim)
                              │
                              ▼
                    ┌─────────────────┐
                    │  DENDRITIC CELL │
                    │  (preprocessing)│
                    └────────┬────────┘
                             │ features + signals
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
   ┌───────────┐       ┌───────────┐       ┌───────────┐
   │  B CELL   │       │  NK CELL  │       │  MEMORY   │
   │ (pattern) │       │ (anomaly) │       │ (context) │
   └─────┬─────┘       └─────┬─────┘       └─────┬─────┘
         │                   │                   │
         │ classification    │ is_anomaly        │ similar_cases
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  ORCHESTRATOR   │
                    │  (T Helper)     │
                    └────────┬────────┘
                             │
                             ▼
                    FINAL VERDICT
                    {
                      risk_level: "HIGH",
                      classification: "vulnerable",
                      confidence: 0.92,
                      anomaly: true
                    }
```

---

## When to Use Each Agent

| Scenario | Agent | Why |
|----------|-------|-----|
| "Is this code vulnerable?" | B Cell (`scan`) | Need classification |
| "Is this unusual?" | NK Cell (`detect`) | Zero-shot anomaly |
| "What features does this have?" | Dendritic (`inspect`) | Feature analysis |
| "Have we seen this before?" | Memory (`recall`) | Context lookup |
| "Full security analysis" | Orchestrator (`analyze`) | All agents |

---

## Training Requirements

| Agent | Training Data | Examples Needed |
|-------|---------------|-----------------|
| B Cell | Labeled examples | 100+ per class |
| NK Cell | "Self" only | 50+ normal examples |
| Dendritic | None | Rule-based |
| Memory | Auto-populated | N/A |
| Orchestrator | None | Coordination logic |

---

## References

1. **Hunt & Cooke (2000)**: Immunos-81 pattern recognition - *JAMIA 7(1):28-41*
2. **de Castro & Von Zuben (2002)**: Negative Selection Algorithm
3. **Greensmith et al. (2005)**: Dendritic Cell Algorithm
4. **Muhammad Umair et al. (2025)**: NegSl-AIS - *Results in Engineering 27:106601*
5. **Matzinger (2002)**: Danger Theory

---

**Last Updated**: 2026-01-12
**Version**: Antigence v0.2.0-alpha
