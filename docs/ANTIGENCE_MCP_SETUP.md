# Antigence MCP Integration Guide

**Antigence** = MCP server exposing IMMUNOS agents
**IMMUNOS** = Core context/immune system (hidden in `.immunos/`)

## 🎯 Available MCP Tools

Antigence MCP server exposes 5 tools:

| Tool                | Description                       | Use Case                          |
| ------------------- | --------------------------------- | --------------------------------- |
| `antigence_scan`    | B Cell pattern matching           | Security vulnerability scanning   |
| `antigence_detect`  | NK Cell anomaly detection         | Zero-shot threat detection        |
| `antigence_analyze` | Multi-agent analysis              | Comprehensive security assessment |
| `antigence_inspect` | Dendritic Cell feature extraction | Code analysis                     |
| `antigence_recall`  | Memory agent queries              | Check previously analyzed code    |

---

## 🚀 Setup for Claude Code (CLI)

### Option 1: Using Claude Code MCP Settings

Create/edit: `~/.config/claude/mcp_settings.json`

```json
{
  "mcpServers": {
    "antigence": {
      "command": "python3",
      "args": [
        "/path/to/antigence/src/immunos_mcp/servers/simple_mcp_server.py"
      ],
      "cwd": "/path/to/antigence",
      "env": {
        "PYTHONPATH": "/path/to/antigence/src"
      }
    }
  }
}
```

**Restart Claude Code**, then use tools:
```
Can you scan this code for vulnerabilities using antigence_scan?

def process_user_input(data):
    eval(data)  # suspicious!
```

### Option 2: Using uv Entry Point

If `uv` is installed:

```json
{
  "mcpServers": {
    "antigence": {
      "command": "uv",
      "args": [
        "run",
        "immunos-mcp-mvp"
      ],
      "cwd": "/path/to/antigence"
    }
  }
}
```

---

## 🖥️ Setup for Claude Desktop

Create/edit: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "antigence": {
      "command": "python3",
      "args": [
        "/path/to/antigence/src/immunos_mcp/servers/simple_mcp_server.py"
      ],
      "cwd": "/path/to/antigence",
      "env": {
        "PYTHONPATH": "/path/to/antigence/src"
      }
    }
  }
}
```

**Restart Claude Desktop**. Tools will appear in the sidebar.

---

## 🤖 Setup for Ollama Agents (Qwen, DeepSeek)

For Ollama models, you need to expose Antigence via a different method since they don't have native MCP support. Here are options:

### Option 1: HTTP API Wrapper (Recommended)

Create an HTTP API wrapper around the MCP server:

**File**: `/path/to/antigence/scripts/antigence_api.py`

```python
#!/usr/bin/env python3
"""
Antigence HTTP API Wrapper
Exposes IMMUNOS MCP tools via REST API for Ollama agents
"""

from flask import Flask, request, jsonify
import sys
sys.path.insert(0, '/path/to/antigence/src')

from immunos_mcp.core.antigen import Antigen, DataType
from immunos_mcp.orchestrator.manager import OrchestratorManager
from immunos_mcp.config.loader import load_config

app = Flask(__name__)

# Initialize orchestrator
config = load_config()
orchestrator = OrchestratorManager(config)

@app.route('/antigence/scan', methods=['POST'])
def scan():
    """B Cell pattern matching"""
    data = request.json
    code = data.get('code', '')
    antigen = Antigen.from_text(code, metadata={"source": "api"})
    result = orchestrator.analyze(antigen)

    return jsonify({
        'classification': result.classification,
        'confidence': result.confidence,
        'anomaly': result.anomaly
    })

@app.route('/antigence/detect', methods=['POST'])
def anomaly():
    """NK Cell anomaly detection"""
    data = request.json
    code = data.get('code', '')
    antigen = Antigen.from_text(code, metadata={"source": "api"})
    result = orchestrator.analyze(antigen)

    return jsonify({
        'anomaly': result.anomaly,
        'nk_confidence': result.nk_confidence
    })

@app.route('/antigence/analyze', methods=['POST'])
def full():
    """Full multi-agent analysis"""
    data = request.json
    code = data.get('code', '')
    antigen = Antigen.from_text(code, metadata={"source": "api"})
    result = orchestrator.analyze(antigen)

    return jsonify({
        'classification': result.classification,
        'confidence': result.confidence,
        'anomaly': result.anomaly,
        'bcell_confidence': result.bcell_confidence,
        'nk_confidence': result.nk_confidence,
        'features': result.features,
        'signals': result.signals
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5555, debug=False)
```

**Start the API**:
```bash
python3 /path/to/antigence/scripts/antigence_api.py
```

**Use from Ollama**:
```bash
ollama run qwen2.5-coder:7b

# In the Ollama chat:
# You: Check this code for security issues using curl:
# curl -X POST http://127.0.0.1:5555/antigence/scan \
#   -H "Content-Type: application/json" \
#   -d '{"code": "eval(user_input)"}'
```

### Option 2: Direct Python Import (For Qwen Coder Scripts)

When Qwen Coder writes Python scripts, it can directly import IMMUNOS:

```python
import sys
sys.path.insert(0, '/path/to/antigence/src')

from immunos_mcp.core.antigen import Antigen
from immunos_mcp.orchestrator.manager import OrchestratorManager
from immunos_mcp.config.loader import load_config

# Initialize
config = load_config()
orchestrator = OrchestratorManager(config)

# Analyze code
code_to_check = """
def process(data):
    eval(data)
"""

antigen = Antigen.from_text(code_to_check)
result = orchestrator.analyze(antigen)

print(f"Classification: {result.classification}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Anomaly: {result.anomaly}")
```

---

## 🧬 Integration with IMMUNOS Context System

### Connect Antigence to .immunos/ Memory

Antigence can use the IMMUNOS memory system for persistent context:

**File**: `/path/to/antigence/config/immunos_integration.json`

```json
{
  "immunos_base_path": "/path/to/.immunos",
  "memory_path": "/path/to/.immunos/memory",
  "snapshots_path": "/path/to/.immunos/memory/snapshots",
  "enable_context_persistence": true,
  "enable_agent_logging": true
}
```

**Update orchestrator to use IMMUNOS memory**:

```python
# In orchestrator initialization
import json
from pathlib import Path

immunos_config = Path('/path/to/antigence/config/immunos_integration.json')
if immunos_config.exists():
    with open(immunos_config) as f:
        immunos_settings = json.load(f)

    # Use IMMUNOS memory instead of local memory
    memory_path = immunos_settings['memory_path']
    # Configure agents to use shared memory
```

---

## 📋 Complete Setup Checklist

### For Claude Code (Current Session)
- [ ] Create `~/.config/claude/mcp_settings.json`
- [ ] Add Antigence MCP server configuration
- [ ] Restart Claude Code
- [ ] Test: `Can you scan code using immune_scan tool?`

### For Claude Desktop
- [ ] Create `~/Library/Application Support/Claude/claude_desktop_config.json`
- [ ] Add Antigence MCP server configuration
- [ ] Restart Claude Desktop
- [ ] Check tools appear in sidebar

### For Ollama Agents (Qwen, DeepSeek)
- [ ] Option A: Create HTTP API wrapper (`antigence_api.py`)
- [ ] Start API: `python3 scripts/antigence_api.py` (runs on port 5555)
- [ ] Test: `curl -X POST http://127.0.0.1:5555/immune/scan -d '{"code":"test"}'`
- [ ] OR Option B: Use direct Python imports in agent scripts

### For IMMUNOS Integration
- [ ] Create `config/immunos_integration.json`
- [ ] Configure memory path to `.immunos/memory/`
- [ ] Update orchestrator to use shared memory
- [ ] Test: Analysis results saved to IMMUNOS memory

---

## 🔧 Startup Scripts with Antigence

Update agent startup scripts to include Antigence:

### Updated `start-qwen-coder.sh`

```bash
#!/bin/bash
# Qwen Coder with Antigence API

echo "🧬 Starting Antigence API server..."
python3 /path/to/antigence/scripts/antigence_api.py &
ANTIGENCE_PID=$!

echo "Starting Qwen Coder..."
python3 scripts/immunos_agent_startup.py --agent qwen-coder

echo ""
echo "Antigence API available at: http://127.0.0.1:5555"
echo "  - POST /antigence/scan      - Security scanning"
echo "  - POST /antigence/detect    - Anomaly detection"
echo "  - POST /antigence/analyze   - Full analysis"
echo ""

ollama run qwen2.5-coder:7b

# Cleanup on exit
kill $ANTIGENCE_PID
```

---

## 🎯 Usage Examples

### From Claude Code/Desktop (MCP)

```
User: Can you scan this code for security issues?

def login(username, password):
    query = f"SELECT * FROM users WHERE name='{username}'"
    cursor.execute(query)
