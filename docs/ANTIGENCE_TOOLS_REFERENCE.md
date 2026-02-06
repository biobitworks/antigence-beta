# Antigence Tools Reference

**Updated**: 2026-01-05 - New tool names!

## 🎯 Available Tools

| Tool | Purpose | When to Use | Example |
|------|---------|-------------|---------|
| **antigence_scan** | Quick security scan | Find known vulnerabilities | "Scan this code with antigence" |
| **antigence_detect** | Threat detection | Find anomalies/unknown threats | "Can antigence detect any issues?" |
| **antigence_analyze** | Comprehensive review | Deep security analysis | "Analyze this with antigence" |
| **antigence_inspect** | Code inspection | Understand code structure | "Inspect this code with antigence" |
| **antigence_recall** | Check history | See if analyzed before | "Has antigence seen this code?" |

---

## 📖 Detailed Tool Descriptions

### 1. antigence_scan
**Quick security vulnerability scan**

**What it does:**
- Pattern matching for known security issues
- Returns classification (safe/vulnerable)
- Provides confidence scores
- Lists detected patterns

**Best for:**
- SQL injection detection
- XSS vulnerabilities
- Command injection
- Known security anti-patterns

**Usage:**
```
Can you scan this code with antigence?

def login(username, password):
    query = f"SELECT * FROM users WHERE name='{username}'"
    cursor.execute(query)
```

**Output:**
- Classification: vulnerable/safe
- Confidence: 0-100%
- Matched patterns: List of security issues found

---

### 2. antigence_detect
**Zero-shot anomaly and threat detection**

**What it does:**
- Detects unusual/suspicious patterns
- Works without prior training on threats
- Identifies novel attack vectors
- Negative selection algorithm

**Best for:**
- Novel threats
- Obfuscated code
- Unusual behavior patterns
- Code that "feels wrong"

**Usage:**
```
Can antigence detect any threats in this?

import os
cmd = input("Enter command: ")
os.system(cmd)
```

**Output:**
- Anomaly: yes/no
- Severity: high/medium/low
- Confidence: 0-100%

---

### 3. antigence_analyze
**Comprehensive multi-agent security analysis**

**What it does:**
- Runs all agents (pattern matching + anomaly detection + features)
- Multi-agent consensus
- Detailed security assessment
- Feature extraction

**Best for:**
- Thorough security review
- High-stakes code
- Production code review
- Complete assessment

**Usage:**
```
Can you do a full antigence analysis of this function?

def process_payment(amount, card):
    eval(f"charge({amount}, {card})")
```

**Output:**
- Overall risk level
- All agent results
- Features + signals
- Execution details
- Comprehensive recommendation

---

### 4. antigence_inspect
**Code structure and feature inspection**

**What it does:**
- Extracts structural features
- Analyzes complexity
- Identifies code characteristics
- Signal classification (danger/safe/threat)

**Best for:**
- Understanding code complexity
- Code quality assessment
- Architecture analysis
- Feature-based analysis

**Usage:**
```
Can you inspect this code with antigence?

class UserManager:
    def __init__(self):
        self.users = []
        self.cache = {}
```

**Output:**
- Structural features
- Complexity metrics
- Signal classifications
- Code characteristics

---

### 5. antigence_recall
**Query analysis history**

**What it does:**
- Checks if code was analyzed before
- Returns previous results
- Compares current vs previous
- Can clear memory cache

**Best for:**
- Checking analysis history
- Comparing results over time
- Memory management
- Avoiding duplicate analysis

**Usage:**
```
Has antigence analyzed this code before?

def hash_password(pwd):
    return hashlib.md5(pwd.encode()).hexdigest()
```

**Output:**
- Memory hit: yes/no
- Previous analysis (if exists)
- Current analysis
- Comparison

---

## 🚀 Access Methods

### Claude Code / Claude Desktop (MCP)
```
# Natural language - tools are auto-invoked
Can you scan this code with antigence?
Can antigence detect any issues here?
Do an antigence analysis of this function
```

### Ollama Agents (HTTP API)
```bash
# Direct API calls
curl -X POST http://127.0.0.1:5555/antigence/scan \
  -H "Content-Type: application/json" \
  -d '{"code": "eval(user_input)"}'

curl -X POST http://127.0.0.1:5555/antigence/detect \
  -H "Content-Type: application/json" \
  -d '{"code": "import os; os.system(cmd)"}'

curl -X POST http://127.0.0.1:5555/antigence/analyze \
  -H "Content-Type: application/json" \
  -d '{"code": "def process(data): exec(data)"}'
```

---

## 💡 Usage Tips

### When to use each tool:

**Quick check → antigence_scan**
- "Is this SQL injection vulnerable?"
- "Any security issues?"

**Suspicious code → antigence_detect**
- "This code looks weird, check it"
- "Detect any threats here"

**Important code → antigence_analyze**
- "Full security review please"
- "Comprehensive analysis needed"

**Understand code → antigence_inspect**
- "What features does this have?"
- "How complex is this?"

**Check history → antigence_recall**
- "Have we seen this before?"
- "Previous analysis results?"

---

## 🎨 Natural Language Examples

### Good prompts:
- ✅ "Scan this with antigence"
- ✅ "Can antigence detect issues?"
- ✅ "Analyze this using antigence"
- ✅ "Inspect this code with antigence"
- ✅ "Has antigence seen this?"

### Also works:
- ✅ "Use antigence to scan..."
- ✅ "Check this with antigence"
- ✅ "Run antigence analysis"
- ✅ "Antigence, analyze this"

---

## 🔧 Tool Selection Guide

```
┌─────────────────────────────────────────────────────┐
│  Need quick vulnerability check?                    │
│  → antigence_scan                                   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Code looks suspicious/unusual?                     │
│  → antigence_detect                                 │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Need complete security review?                     │
│  → antigence_analyze                                │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Want to understand code structure?                 │
│  → antigence_inspect                                │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Check if analyzed before?                          │
│  → antigence_recall                                 │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Comparison: Old vs New Names

| Old Name | New Name | Why Changed |
|----------|----------|-------------|
| immune_scan | **antigence_scan** | Brand alignment, clearer |
| immune_anomaly | **antigence_detect** | More action-oriented |
| immune_full | **antigence_analyze** | Better describes purpose |
| immune_features | **antigence_inspect** | More intuitive verb |
| immune_memory | **antigence_recall** | Clearer memory action |

---

**Antigence Tools Reference**
*Updated: 2026-01-05*
*5 tools for comprehensive security analysis*
