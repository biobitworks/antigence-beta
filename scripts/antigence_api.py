#!/usr/bin/env python3
"""
Antigence HTTP API Wrapper
Exposes IMMUNOS MCP tools via REST API for Ollama agents

Usage:
    python3 scripts/antigence_api.py

    # Then from any client:
    curl -X POST http://127.0.0.1:5555/immune/scan \
         -H "Content-Type: application/json" \
         -d '{"code": "eval(user_input)"}'
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from flask import Flask, request, jsonify
from immunos_mcp.core.antigen import Antigen, DataType
from immunos_mcp.orchestrator.manager import OrchestratorManager
from immunos_mcp.config.loader import load_config

app = Flask(__name__)

# Initialize orchestrator
print("Initializing Antigence IMMUNOS orchestrator...", file=sys.stderr)
config = load_config()
orchestrator = OrchestratorManager(config)
print("✓ Orchestrator ready", file=sys.stderr)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'service': 'antigence-immunos'})

@app.route('/antigence/scan', methods=['POST'])
def scan():
    """Scan code for security vulnerabilities using pattern matching"""
    data = request.json
    code = data.get('code', '')

    if not code:
        return jsonify({'error': 'No code provided'}), 400

    antigen = Antigen.from_text(code, metadata={"source": "api"})
    result = orchestrator.analyze(antigen)

    return jsonify({
        'classification': result.classification,
        'confidence': result.confidence,
        'anomaly': result.anomaly,
        'bcell_confidence': result.bcell_confidence,
        'patterns': list(result.metadata.get('bcell', {}).get('avidity_scores', {}).keys())
    })

@app.route('/antigence/detect', methods=['POST'])
def detect():
    """Detect anomalies and threats using zero-shot detection"""
    data = request.json
    code = data.get('code', '')

    if not code:
        return jsonify({'error': 'No code provided'}), 400

    antigen = Antigen.from_text(code, metadata={"source": "api"})
    result = orchestrator.analyze(antigen)

    return jsonify({
        'anomaly': result.anomaly,
        'nk_confidence': result.nk_confidence,
        'severity': 'high' if result.nk_confidence > 0.8 else 'medium' if result.nk_confidence > 0.5 else 'low'
    })

@app.route('/antigence/analyze', methods=['POST'])
def analyze():
    """Complete comprehensive multi-agent analysis"""
    data = request.json
    code = data.get('code', '')

    if not code:
        return jsonify({'error': 'No code provided'}), 400

    antigen = Antigen.from_text(code, metadata={"source": "api"})
    result = orchestrator.analyze(antigen)

    return jsonify({
        'classification': result.classification,
        'confidence': result.confidence,
        'anomaly': result.anomaly,
        'bcell_confidence': result.bcell_confidence,
        'nk_confidence': result.nk_confidence,
        'features': result.features,
        'signals': result.signals,
        'agents': result.agents,
        'execution_time': result.metadata.get('execution_time', 0),
        'risk_level': _get_risk_level(result)
    })

@app.route('/antigence/inspect', methods=['POST'])
def inspect():
    """Inspect code structure and extract features"""
    data = request.json
    code = data.get('code', '')

    if not code:
        return jsonify({'error': 'No code provided'}), 400

    antigen = Antigen.from_text(code, metadata={"source": "api"})
    result = orchestrator.analyze(antigen)

    return jsonify({
        'features': result.features,
        'signals': result.signals
    })

@app.route('/antigence/recall', methods=['POST'])
def recall():
    """Check if code has been analyzed before (query memory)"""
    data = request.json
    code = data.get('code', '')
    action = data.get('action', 'check')

    if not code and action != 'clear':
        return jsonify({'error': 'No code provided'}), 400

    if action == 'clear':
        # Clear memory
        current_orch = orchestrator.get_current_orchestrator()
        if current_orch.get_mode() == "local":
            current_orch.orchestrator.memory.clear()
            return jsonify({'status': 'cleared'})
        else:
            return jsonify({'error': 'Cannot clear remote memory'}), 400

    antigen = Antigen.from_text(code, metadata={"source": "api"})
    memory_key = antigen.identifier or str(hash(antigen.get_text_content()))

    current_orch = orchestrator.get_current_orchestrator()
    if current_orch.get_mode() == "local":
        previous = current_orch.orchestrator.memory.retrieve(memory_key)
        hit = previous is not None
    else:
        previous = None
        hit = False

    result = orchestrator.analyze(antigen)

    return jsonify({
        'memory_hit': hit,
        'previous_analysis': previous,
        'current_analysis': {
            'classification': result.classification,
            'confidence': result.confidence,
            'anomaly': result.anomaly
        }
    })

def _get_risk_level(result):
    """Determine risk level from analysis result"""
    if result.anomaly and result.confidence > 0.7:
        return 'HIGH'
    elif result.anomaly:
        return 'MEDIUM'
    elif result.confidence > 0.8:
        return 'LOW'
    else:
        return 'UNCERTAIN'

if __name__ == '__main__':
    print("\n" + "="*60, file=sys.stderr)
    print("🧬 Antigence IMMUNOS API Server", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print("\nEndpoints:", file=sys.stderr)
    print("  GET  /health                 - Health check", file=sys.stderr)
    print("  POST /antigence/scan         - Scan for vulnerabilities", file=sys.stderr)
    print("  POST /antigence/detect       - Detect anomalies/threats", file=sys.stderr)
    print("  POST /antigence/analyze      - Comprehensive analysis", file=sys.stderr)
    print("  POST /antigence/inspect      - Inspect code features", file=sys.stderr)
    print("  POST /antigence/recall       - Check analysis history", file=sys.stderr)
    print("\nListening on: http://127.0.0.1:5555", file=sys.stderr)
    print("="*60 + "\n", file=sys.stderr)

    app.run(host='127.0.0.1', port=5555, debug=False)
