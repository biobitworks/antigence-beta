# Antigence Immune System Architecture Map

## Core Metaphor

The Antigence platform maps AI security analysis to the biological immune system:

```
BIOLOGICAL IMMUNE SYSTEM          ANTIGENCE PLATFORM
─────────────────────────────────────────────────────────────
Antigen (foreign substance)   →   Input (code, claims, text)
Antibody (recognition protein)→   Trained patterns (clones, detectors)
Antigen Receptor (binding)    →   Matching algorithm (affinity, NegSel)
Immune Cell (processor)       →   Agent (B Cell, NK Cell, T Cell, etc.)
Thymus (T cell education)     →   LLM Orchestrator (coordinates all)
Cytokines (cell signals)      →   Agent outputs (verdict, confidence, anomaly)
Immune Response               →   Final risk assessment
```

---

## Complete Role Map

### 1. B CELL (Adaptive Immunity - Pattern Recognition)

| Immune Concept | Antigence Implementation |
|----------------|--------------------------|
| **Immune Cell** | `BCellAgent` class |
| **Antigen** | Input text/code/claim (`Antigen.from_text()`) |
| **Antibody** | Trained pattern clones (stored in `.pkl`) |
| **Antigen Receptor** | Affinity calculation (70% embedding + 30% traditional) |
| **Recognition** | `bcell.recognize(antigen)` → `RecognitionResult` |
| **Cytokine Output** | `{verdict, confidence, avidity_scores}` |
| **LLM Role** | `bcell` - Evidence extraction, citation checks |
| **LLM Trigger** | When confidence < threshold OR for detailed analysis |

```
INPUT ANTIGEN
     │
     ▼
┌─────────────────────────────────────────┐
│  B CELL AGENT                           │
│  ┌─────────────┐    ┌────────────────┐  │
│  │  Antibodies │───▶│ Antigen Receptor│ │
│  │  (clones)   │    │ (affinity calc) │ │
│  └─────────────┘    └────────────────┘  │
│         │                    │          │
│         ▼                    ▼          │
│  ┌─────────────────────────────────┐    │
│  │ ML Recognition (pattern match)  │    │
│  └─────────────────────────────────┘    │
│                   │                     │
│                   ▼                     │
│  ┌─────────────────────────────────┐    │
│  │ LLM Enhancement (bcell role)    │◀───┼── TRIGGERS LLM
│  │ - Explain verdict               │    │   when needed
│  │ - Extract evidence sentences    │    │
│  │ - Citation verification         │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
     │
     ▼
CYTOKINE: {verdict, confidence, evidence, explanation}
```

---

### 2. NK CELL (Innate Immunity - Anomaly Detection)

| Immune Concept | Antigence Implementation |
|----------------|--------------------------|
| **Immune Cell** | `NKCellAgent` / `EnhancedNKCellAgent` |
| **Antigen** | Input to check for "non-self" |
| **Antibody** | Negative selection detectors |
| **Antigen Receptor** | Distance threshold (NegSl-AIS Equation 20) |
| **Recognition** | `nk.detect_novelty(antigen)` → `AnomalyResult` |
| **Cytokine Output** | `{is_anomaly, anomaly_score, confidence}` |
| **LLM Role** | `nk` - Risk summaries, anomaly explanations |
| **LLM Trigger** | When anomaly detected (is_anomaly=True) |

```
INPUT ANTIGEN
     │
     ▼
┌─────────────────────────────────────────┐
│  NK CELL AGENT                          │
│  ┌─────────────┐    ┌────────────────┐  │
│  │  Detectors  │───▶│ Distance Calc   │ │
│  │  (anti-self)│    │ (NegSel thresh) │ │
│  └─────────────┘    └────────────────┘  │
│         │                    │          │
│         ▼                    ▼          │
│  ┌─────────────────────────────────┐    │
│  │ ML Detection (negative select)  │    │
│  └─────────────────────────────────┘    │
│                   │                     │
│          ┌───────┴───────┐              │
│          ▼               ▼              │
│     is_anomaly?     is_anomaly?         │
│        NO              YES              │
│          │               │              │
│          ▼               ▼              │
│       [skip]    ┌─────────────────┐     │
│                 │ LLM Risk Summary│◀────┼── TRIGGERS LLM
│                 │ (nk role)       │     │   on anomaly
│                 │ - Why flagged?  │     │
│                 │ - Risk level    │     │
│                 │ - Next steps    │     │
│                 └─────────────────┘     │
└─────────────────────────────────────────┘
     │
     ▼
CYTOKINE: {is_anomaly, score, risk_summary, explanation}
```

---

### 3. T CELL - SECURITY (Adaptive Immunity - Code Review)

| Immune Concept | Antigence Implementation |
|----------------|--------------------------|
| **Immune Cell** | `TCellSecurityAgent` (NEW) |
| **Antigen** | Source code to review |
| **Antibody** | Security patterns (CWE, OWASP) |
| **Antigen Receptor** | Pattern matching + LLM analysis |
| **Recognition** | `tcell.review_code(code)` |
| **Cytokine Output** | `{vulnerabilities, severity, remediation}` |
| **LLM Role** | `tcell_security` - Secure-code review |
| **LLM Trigger** | Always (LLM-first for code review) |

```
INPUT CODE
     │
     ▼
┌─────────────────────────────────────────┐
│  T CELL SECURITY AGENT                  │
│  ┌─────────────┐    ┌────────────────┐  │
│  │  CWE/OWASP  │───▶│ Pattern Match   │ │
│  │  Patterns   │    │ (regex/AST)     │ │
│  └─────────────┘    └────────────────┘  │
│         │                    │          │
│         ▼                    ▼          │
│  ┌─────────────────────────────────┐    │
│  │ Quick Scan (heuristic patterns) │    │
│  └─────────────────────────────────┘    │
│                   │                     │
│                   ▼                     │
│  ┌─────────────────────────────────┐    │
│  │ LLM Deep Review (tcell_security)│◀───┼── ALWAYS TRIGGERS
│  │ - Vulnerability analysis        │    │   (LLM-first)
│  │ - CWE classification            │    │
│  │ - Remediation suggestions       │    │
│  │ - Severity scoring              │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
     │
     ▼
CYTOKINE: {vulnerabilities[], severity, cwe_ids[], remediation}
```

---

### 4. DENDRITIC CELL (Feature Extraction & Presentation)

| Immune Concept | Antigence Implementation |
|----------------|--------------------------|
| **Immune Cell** | `DendriticAgent` |
| **Antigen** | Raw input to process |
| **Antibody** | Feature extraction rules |
| **Antigen Receptor** | Regex patterns, word lists |
| **Recognition** | `dendritic.extract_features(antigen)` |
| **Cytokine Output** | `{features[20], signal_type, danger_score}` |
| **LLM Role** | `dendritic_summarizer` - Summarization |
| **LLM Trigger** | For long artifacts needing summary |

```
INPUT ANTIGEN
     │
     ▼
┌─────────────────────────────────────────┐
│  DENDRITIC CELL AGENT                   │
│  ┌─────────────┐    ┌────────────────┐  │
│  │  Feature    │───▶│ Regex/Word     │ │
│  │  Rules      │    │ Extraction     │ │
│  └─────────────┘    └────────────────┘  │
│         │                    │          │
│         ▼                    ▼          │
│  ┌─────────────────────────────────┐    │
│  │ Extract 20 Numeric Features     │    │
│  │ - Text structure (5)            │    │
│  │ - Claim characteristics (6)     │    │
│  │ - Semantic signals (5)          │    │
│  │ - Danger signals (4)            │    │
│  └─────────────────────────────────┘    │
│                   │                     │
│          ┌───────┴───────┐              │
│          ▼               ▼              │
│     len < 500       len >= 500          │
│          │               │              │
│          ▼               ▼              │
│       [skip]    ┌─────────────────┐     │
│                 │ LLM Summarizer  │◀────┼── TRIGGERS LLM
│                 │ (dendritic role)│     │   for long text
│                 │ - Key points    │     │
│                 │ - Bullet summary│     │
│                 └─────────────────┘     │
└─────────────────────────────────────────┘
     │
     ▼
CYTOKINE: {features[], signal_type, summary, danger_score}
```

---

### 5. THYMUS (Central Orchestration)

| Immune Concept | Antigence Implementation |
|----------------|--------------------------|
| **Organ** | `OllamaOrchestrator` |
| **Function** | T cell education, self/non-self discrimination |
| **Input** | All cytokines from other cells |
| **Output** | Final coordinated immune response |
| **LLM Role** | `orchestrator` - Final verdicts |
| **LLM Trigger** | Always (coordinates all signals) |

```
         CYTOKINES FROM ALL CELLS
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
 B Cell          NK Cell        T Cell
 verdict         anomaly        vulns
    │               │               │
    └───────────────┼───────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  THYMUS ORCHESTRATOR                    │
│  ┌─────────────────────────────────┐    │
│  │ Collect All Cytokines           │    │
│  │ - B Cell: verdict + confidence  │    │
│  │ - NK Cell: anomaly + score      │    │
│  │ - T Cell: vulnerabilities       │    │
│  │ - Dendritic: features + summary │    │
│  └─────────────────────────────────┘    │
│                   │                     │
│                   ▼                     │
│  ┌─────────────────────────────────┐    │
│  │ LLM Coordination (orchestrator) │◀───┼── ALWAYS TRIGGERS
│  │ - Weigh conflicting signals     │    │
│  │ - Adjust confidence             │    │
│  │ - Produce final verdict         │    │
│  │ - Generate explanation          │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
                    │
                    ▼
          FINAL IMMUNE RESPONSE
    {verdict, confidence, explanation, risk_level}
```

---

## Complete Pipeline

```
                         INPUT ANTIGEN
                              │
                              ▼
                    ┌─────────────────┐
                    │  DENDRITIC CELL │
                    │  (preprocessing)│
                    └────────┬────────┘
                             │ features + summary
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
   ┌───────────┐       ┌───────────┐       ┌───────────┐
   │  B CELL   │       │  NK CELL  │       │  T CELL   │
   │ (pattern) │       │ (anomaly) │       │(security) │
   └─────┬─────┘       └─────┬─────┘       └─────┬─────┘
         │                   │                   │
         │ verdict           │ anomaly           │ vulns
         │ + LLM explain     │ + LLM risk        │ + LLM review
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     THYMUS      │
                    │  (orchestrator) │
                    │   LLM coords    │
                    └────────┬────────┘
                             │
                             ▼
                    FINAL RISK VERDICT
                    {
                      verdict: "DANGER|SAFE|UNCERTAIN",
                      confidence: 0.92,
                      explanation: "...",
                      vulnerabilities: [...],
                      anomaly_detected: true/false
                    }
```

---

## LLM Role Trigger Summary

| Role | LLM Model | Trigger Condition | Purpose |
|------|-----------|-------------------|---------|
| `bcell` | qwen2.5-coder:7b | Low confidence OR detailed mode | Explain verdict, extract evidence |
| `nk` | qwen2.5-coder:7b | Anomaly detected | Risk summary, explain why flagged |
| `tcell_security` | codellama:7b | Always (code input) | Deep security review |
| `dendritic_summarizer` | qwen2.5:1.5b | Long text (>500 chars) | Summarize to bullets |
| `orchestrator` | qwen2.5-coder:7b | Always (final step) | Coordinate all signals |

---

## Antibody → Antigent Trigger Map

```python
# Antibody (trained pattern) triggers Antigent (LLM) when needed

class BCellAgent:
    def recognize(self, antigen) -> RecognitionResult:
        # 1. ML pattern matching (antibody)
        result = self._affinity_match(antigen)

        # 2. Trigger LLM (antigent) if needed
        if result.confidence < 0.7 or self.detailed_mode:
            result.explanation = self._call_llm("bcell", antigen, result)

        return result

class NKCellAgent:
    def detect_novelty(self, antigen) -> AnomalyResult:
        # 1. ML negative selection (antibody)
        result = self._negsel_detect(antigen)

        # 2. Trigger LLM (antigent) if anomaly
        if result.is_anomaly:
            result.risk_summary = self._call_llm("nk", antigen, result)

        return result

class TCellSecurityAgent:
    def review_code(self, code) -> SecurityResult:
        # 1. Quick heuristic scan (antibody)
        quick_result = self._pattern_scan(code)

        # 2. Always trigger LLM (antigent) for deep review
        result = self._call_llm("tcell_security", code, quick_result)

        return result

class DendriticAgent:
    def process(self, antigen) -> FeatureResult:
        # 1. Extract features (antibody)
        features = self._extract_features(antigen)

        # 2. Trigger LLM (antigent) for long text
        if len(antigen.data) > 500:
            features.summary = self._call_llm("dendritic_summarizer", antigen)

        return features
```

---

---

## Sentinel Antibodies (File System Immune Response)

The Sentinel is a **separate immune subsystem** focused on file system monitoring. It has its own antibodies:

### Sentinel Architecture

```
FILE SYSTEM (Environment)
         │
         ▼
┌─────────────────────────────────────────┐
│  SENTINEL WATCHER (Innate Immunity)     │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ ANTIBODIES (Detection Patterns)  │   │
│  │ - File hash baseline (self)      │   │
│  │ - Sensitive file patterns        │   │
│  │ - Change classification rules    │   │
│  └─────────────────────────────────┘    │
│                   │                     │
│                   ▼                     │
│  ┌─────────────────────────────────┐    │
│  │ DETECTION                        │   │
│  │ - Compare current vs baseline    │   │
│  │ - Classify: expected/unexpected  │   │
│  │ - Flag anomalies                 │   │
│  └─────────────────────────────────┘    │
│                   │                     │
│                   ▼                     │
│  CYTOKINE: Event logged to JSONL        │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  SENTINEL DASHBOARD (Thymus)            │
│  - Reviews flagged events               │
│  - Creates tickets for investigation    │
│  - Coordinates response                 │
└─────────────────────────────────────────┘
```

### Sentinel Antibody Types

| Antibody | What It Detects | Classification |
|----------|-----------------|----------------|
| **Baseline hashes** | File content changes | `modified` |
| **Sensitive patterns** | Config files, scripts, secrets | `unexpected` |
| **New file detection** | Files not in baseline | `new` |
| **Deletion detection** | Missing files | `deleted` |

### Sentinel vs Analysis Antibodies

| Aspect | Sentinel Antibodies | Analysis Antibodies |
|--------|--------------------|--------------------|
| **Target** | File system changes | Content/code analysis |
| **Self** | Baseline file hashes | Trained patterns |
| **Non-self** | Unexpected changes | Anomalies/threats |
| **Response** | Log event + ticket | Classification + LLM |
| **Speed** | Fast (hash comparison) | Variable (ML + LLM) |

### Sentinel Files

```
~/.antigence/sentinel/
├── antigence_sentinel_watch.py  # Watcher (antibody logic)
├── baseline.json                # "Self" definition (known-good hashes)
├── events.jsonl                 # Cytokine signals (change events)
└── logs/
    ├── antigence_sentinel.out.log
    └── antigence_sentinel.err.log
```

### Running the Sentinel

```bash
# Single scan (creates baseline on first run)
python3 ~/.antigence/sentinel/antigence_sentinel_watch.py

# Watch specific directories
python3 ~/.antigence/sentinel/antigence_sentinel_watch.py --dirs ~/projects,~/.config

# Daemon mode (continuous monitoring)
python3 ~/.antigence/sentinel/antigence_sentinel_watch.py --daemon --interval 300

# Via launchd (macOS)
launchctl load ~/Library/LaunchAgents/com.antigence.sentinel.watch.plist
```

---

*Last Updated: 2026-01-11*
