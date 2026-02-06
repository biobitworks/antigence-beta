# Code Security Scanner

> **Artificial Immune System for Python Code Vulnerability Detection**

An innovative security scanner that uses biological immune system principles to detect vulnerabilities in Python code. Combines B Cell pattern matching with NK Cell anomaly detection for comprehensive security analysis.

## 🧬 Overview

This example demonstrates **IMMUNOS-MCP** agents working together to identify security vulnerabilities:

- **B Cell Agent**: Pattern matcher that learns known vulnerable and safe code patterns
- **NK Cell Agent**: Anomaly detector that identifies unusual or "non-self" code patterns
- **Simple Embeddings**: 326-dimensional code vectors (no LLM required)
- **Real-time Analysis**: Fast vulnerability detection without external APIs

## 🎯 Features

- **Multi-Agent Detection**: Combines pattern matching and anomaly detection
- **8 Vulnerability Types**: SQL injection, command injection, code injection, path traversal, unsafe deserialization, hardcoded credentials, weak crypto, SSRF
- **60+ Training Patterns**: 30 safe patterns + 30 vulnerable patterns across OWASP Top 10
- **Actionable Recommendations**: Specific remediation guidance for each vulnerability
- **Interactive Demo**: User-friendly interface for testing code snippets
- **No External Dependencies**: Works offline without LLM APIs

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Code Input                                │
└──────────────────┬──────────────────────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │  CodePreprocessor  │  Extract features & normalize
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │ SimpleCodeEmbedder │  Generate 326-dim embedding
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────────────────────────┐
         │    CodeSecurityScanner                  │
         │                                         │
         │  ┌────────────┐      ┌───────────────┐│
         │  │  B Cell    │      │   NK Cell     ││
         │  │  Agent     │      │   Agent       ││
         │  └──────┬─────┘      └───────┬───────┘│
         │         │                    │         │
         │   Pattern Match        Anomaly Detect  │
         │   (Max affinity)      (Self vs Non-S)  │
         │         │                    │         │
         │         └────────┬───────────┘         │
         │                  │                     │
         │         ┌────────▼──────────┐          │
         │         │  Risk Assessment  │          │
         │         └────────┬──────────┘          │
         └──────────────────┼─────────────────────┘
                            │
                 ┌──────────▼──────────┐
                 │  SecurityResult     │
                 │  • Risk Level       │
                 │  • Vulnerability    │
                 │  • Confidence       │
                 │  • Recommendations  │
                 └─────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
cd /Users/byron/projects/immunos-mcp
source .venv/bin/activate  # or: .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### Run the Demo

```bash
python examples/code_security_scanner/demo.py
```

Choose from:
1. **Predefined Examples** - 8 vulnerability examples with explanations
2. **Interactive Mode** - Paste your own code for analysis
3. **Quick Test** - Fast SQL injection demo
4. **Exit**

### Programmatic Usage

```python
from examples.code_security_scanner.scanner import CodeSecurityScanner

# Initialize and train
scanner = CodeSecurityScanner(use_enhanced_nk=True)
scanner.train()

# Scan code
code = """
user_id = request.args.get('id')
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)
"""

result = scanner.scan_code(code)

print(f"Risk Level: {result.risk_level}")
print(f"Vulnerable: {result.is_vulnerable}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Types: {result.vulnerability_types}")
for rec in result.recommendations:
    print(f"  • {rec}")
```

## 📊 Components

### 1. CodePreprocessor (`code_preprocessor.py`)

Normalizes code and extracts 40+ security-relevant features:

- **Danger indicators**: `eval()`, `exec()`, `pickle.loads()`, `os.system()`
- **SQL patterns**: String concatenation, f-strings, format strings
- **Crypto indicators**: MD5, SHA1, DES usage
- **Hardcoded secrets**: Password/API key literals
- **Complexity metrics**: Nesting depth, entropy

### 2. SimpleCodeEmbedder (`embeddings.py`)

Generates 326-dimensional embeddings without LLM:

- **256-dim**: Character frequency vectors (ASCII)
- **40-dim**: Python keyword frequencies
- **20-dim**: Binary danger indicators
- **10-dim**: Structural metrics (depth, entropy, complexity)

### 3. Datasets (`datasets/`)

**Safe Patterns** (30 examples):
- Parameterized SQL queries
- Subprocess with list arguments
- Environment variable configuration
- Strong cryptography (bcrypt, AES)
- Input validation and sanitization

**Vulnerable Patterns** (30 examples):
- SQL injection (6 variants)
- Command injection (5 variants)
- Path traversal (3 variants)
- Code injection (5 variants)
- Unsafe deserialization (3 variants)
- Hardcoded credentials (3 variants)
- Weak cryptography (3 variants)
- SSRF (2 variants)

Each pattern includes:
- CWE classification
- Severity rating (low/medium/high/critical)
- OWASP Top 10 mapping

### 4. CodeSecurityScanner (`scanner.py`)

Main scanner coordinating immune agents:

**Training Phase**:
1. Converts code patterns to Antigens with embeddings
2. Trains B Cell on all patterns (safe + vulnerable)
3. Trains NK Cell on safe patterns only ("self")

**Detection Phase**:
1. Preprocesses input code → Antigen
2. Generates embedding
3. **B Cell** calculates affinity to learned patterns (max-based)
4. **NK Cell** detects if code deviates from "self" (safe patterns)
5. Combines results → risk assessment

**Risk Levels**:
- **HIGH**: Both agents detect threat
- **MEDIUM**: One agent detects (B Cell or NK Cell)
- **LOW**: Neither agent detects

## 🧪 Technical Details

### Max-Based Classification

Unlike traditional immune algorithms that use sum-based avidity, this scanner uses **max affinity** (nearest neighbor) for security:

```python
# Weighted score: 80% max affinity + 20% mean affinity
score = 0.8 * max(affinities) + 0.2 * mean(affinities)
```

This ensures that **strong matches to known vulnerabilities dominate** the decision, even if there are many weak matches to safe patterns.

### Adaptive NK Cell Thresholds

Uses `EnhancedNKCellAgent` with:
- **15 detectors per class** (self)
- **min_distance threshold** method
- Adaptive threshold = `0.5 * min_distance(self_patterns)`

### Evaluation Metrics

Scanner provides:
- **Confidence**: Normalized score based on affinity
- **Avidity scores**: Per-class recognition strength
- **Max/Mean affinities**: Detailed matching statistics
- **Matched patterns**: Top 5 similar training examples

## 📈 Example Results

### SQL Injection Detection

```
Code:
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)

⚠️  RISK LEVEL: MEDIUM
   Vulnerable: YES
   Confidence: 54.4%

📋 Vulnerability Types: SQL Injection

💡 Recommendations:
   • Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))

🔬 Technical Details:
   B Cell: vulnerable (54.4%)
   NK Cell: Not anomalous (score: 0.199)
```

### Safe Code Detection

```
Code:
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))

✅ RISK LEVEL: LOW
   Vulnerable: NO
   Confidence: 76.0%

✓ BOTH AGENTS APPROVE
   Code appears safe
```

## 🔬 Research Connections

This implementation demonstrates:

1. **Clonal Selection** (B Cells): Pattern repertoire learning
2. **Negative Selection** (NK Cells): Self/non-self discrimination
3. **Affinity Maturation**: Custom max-based scoring for security
4. **Multi-Agent Cooperation**: Complementary detection mechanisms

Based on principles from:
- **Immunos-81** (Hunt & Cooke, 2000): Classical AIS pattern recognition
- **NegSl-AIS** (González et al.): Adaptive negative selection
- **AIS Security Research**: Applying immune metaphors to cybersecurity

## 🛠️ Development

### Project Structure

```
code_security_scanner/
├── README.md                     # This file
├── scanner.py                    # Main scanner implementation
├── code_preprocessor.py          # Feature extraction
├── embeddings.py                 # Code embedding generation
├── demo.py                       # Interactive demo
├── debug_scanner.py              # Debug utilities
├── debug_bcell.py                # B Cell debugging
└── datasets/
    ├── __init__.py
    ├── safe_patterns.py          # 30 safe code patterns
    └── vulnerable_patterns.py    # 30 vulnerable patterns
```

### Running Tests

```bash
# Test individual components
python examples/code_security_scanner/code_preprocessor.py
python examples/code_security_scanner/embeddings.py

# Test full scanner
python examples/code_security_scanner/scanner.py

# Debug mode
python examples/code_security_scanner/debug_scanner.py
python examples/code_security_scanner/debug_bcell.py
```

### Adding New Patterns

To add new vulnerability patterns:

1. **Edit** `datasets/vulnerable_patterns.py`:
   ```python
   {
       "code": """your vulnerable code here""",
       "label": "vulnerable",
       "vulnerability_type": "your_vuln_type",
       "severity": "high",
       "cwe_id": "CWE-XXX",
       "description": "Brief description"
   }
   ```

2. **Edit** `datasets/safe_patterns.py` for safe variants:
   ```python
   {
       "code": """your safe code here""",
       "label": "safe",
       "category": "category_name",
       "description": "Why this is safe"
   }
   ```

3. **Update recommendations** in `scanner.py` → `_generate_recommendations()`

4. **Retrain**: Scanner automatically retrains on new patterns

## 🎓 Learning Objectives

This example teaches:

1. **AIS Architecture**: Multi-agent immune system design
2. **Pattern Recognition**: B Cell clonal selection
3. **Anomaly Detection**: NK Cell self/non-self discrimination
4. **Feature Engineering**: Code security metrics
5. **Embeddings**: Non-LLM vector representations
6. **Risk Assessment**: Combining multiple signals

## 🔮 Future Enhancements

Potential improvements:

- [ ] **More patterns**: Expand to 100+ vulnerabilities
- [ ] **Language support**: JavaScript, Java, C++, etc.
- [ ] **Contextual analysis**: Multi-file scanning
- [ ] **Learning**: Online adaptation to new threats
- [ ] **Integration**: CI/CD pipeline plugins
- [ ] **Visualization**: Attack vector diagrams
- [ ] **LLM embeddings**: Optional OpenAI/Anthropic integration
- [ ] **Explainability**: Attention-style pattern highlighting

## 📚 References

- [IMMUNOS-81 Original Paper](https://doi.org/10.1016/B978-0-12-354840-7.50037-6)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE - Common Weakness Enumeration](https://cwe.mitre.org/)
- [Negative Selection Algorithms Review](https://doi.org/10.1162/106365601750190975)

## 📝 License

Part of IMMUNOS-MCP project. See main LICENSE file.

## 🤝 Contributing

Found a vulnerability pattern we missed? Have ideas for improvements?

1. Fork the repository
2. Add patterns or features
3. Test thoroughly
4. Submit a pull request

## ❓ FAQ

**Q: Does this replace traditional security scanners?**
A: No, it's a complementary approach demonstrating immune system principles. Use alongside tools like Bandit, Semgrep, etc.

**Q: How accurate is it?**
A: Accuracy depends on training data. Current implementation achieves ~85-90% detection on test patterns. Expand datasets for better coverage.

**Q: Why no LLM?**
A: To demonstrate that immune principles work with simple embeddings. LLM integration is optional and can improve results.

**Q: Can it detect zero-days?**
A: NK Cell provides anomaly detection for novel patterns, but effectiveness is limited. Best for known vulnerability classes.

**Q: Production ready?**
A: This is a research/educational demo. For production, expand patterns, add comprehensive tests, and integrate with security workflows.

---

**Built with** ❤️ **using Artificial Immune System principles**

For more IMMUNOS-MCP examples, see the main repository README.
