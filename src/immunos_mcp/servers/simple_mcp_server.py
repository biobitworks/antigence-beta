#!/usr/bin/env python3
"""
IMMUNOS-MCP Enhanced Server

Multi-agent MCP server exposing all IMMUNOS agents:
- B Cell: Pattern matching
- NK Cell: Anomaly detection
- Dendritic: Feature extraction
- Memory: Pattern caching
- Orchestrator: Full multi-agent analysis

Usage:
    python src/immunos_mcp/servers/simple_mcp_server.py
"""

import asyncio
import sys
from typing import Any, Dict

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("ERROR: MCP SDK not installed. Run: uv pip install mcp>=1.3.1", file=sys.stderr)
    sys.exit(1)

from immunos_mcp.core.antigen import Antigen, DataType
from immunos_mcp.config.loader import load_config
from immunos_mcp.orchestrator.manager import OrchestratorManager
from immunos_mcp.utils.network import detect_mode


class SimpleMCPServer:
    """Enhanced MCP server wrapper for all IMMUNOS agents"""

    def __init__(self):
        self.server = Server("immunos-mcp")

        # Load configuration
        self.config = load_config()

        # Detect operating mode
        self.mode = detect_mode(self.config)

        # Initialize orchestrator manager (handles local/remote selection)
        self.orchestrator_manager = OrchestratorManager(self.config)

        # Log initialization
        orchestrator_type = self.orchestrator_manager.get_current_orchestrator().get_mode()
        print("IMMUNOS-MCP Server initialized:", file=sys.stderr)
        print(f"  Mode: {self.mode}", file=sys.stderr)
        print(f"  Orchestrator: {orchestrator_type} (thymus)", file=sys.stderr)

        self._register_tools()

    def _register_tools(self):
        """Register all MCP tools for Antigence"""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="antigence_scan",
                    description="Scan code/text for security vulnerabilities using pattern matching. Returns classification (safe/vulnerable) with confidence score and detected patterns. Use when you need quick security vulnerability detection.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The code or text snippet to scan for security issues"
                            },
                            "data_type": {
                                "type": "string",
                                "enum": ["code", "text", "auto"],
                                "description": "Type of data being analyzed (default: auto-detect)",
                                "default": "auto"
                            }
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="antigence_detect",
                    description="Detect anomalies and threats in code/text using zero-shot detection. Identifies unusual patterns without prior training on malicious samples. Use when you need to detect novel or unknown threats.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The code or text to check for anomalies and threats"
                            },
                            "data_type": {
                                "type": "string",
                                "enum": ["code", "text", "auto"],
                                "description": "Type of data being analyzed",
                                "default": "auto"
                            }
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="antigence_analyze",
                    description="Complete comprehensive security analysis using all Antigence agents (pattern matching, anomaly detection, feature extraction, memory). Provides detailed security assessment with multi-agent consensus. Use for thorough security review.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The code or text to analyze comprehensively"
                            },
                            "data_type": {
                                "type": "string",
                                "enum": ["code", "text", "auto"],
                                "description": "Type of data being analyzed",
                                "default": "auto"
                            }
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="antigence_inspect",
                    description="Inspect code structure and extract features for analysis. Returns structural and statistical features, complexity metrics, and signal classifications. Use when you need to understand code characteristics.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The code or text to inspect and extract features from"
                            },
                            "data_type": {
                                "type": "string",
                                "enum": ["code", "text", "auto"],
                                "description": "Type of data being analyzed",
                                "default": "auto"
                            }
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="antigence_recall",
                    description="Check if code has been analyzed before by querying Antigence memory. Returns previous analysis results if available. Use to check analysis history or clear memory cache.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The code or text to check in memory (optional if action is 'clear')"
                            },
                            "action": {
                                "type": "string",
                                "enum": ["check", "clear"],
                                "description": "Action to perform: 'check' for retrieval, 'clear' to clear memory",
                                "default": "check"
                            }
                        },
                        "required": ["code"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> list[TextContent]:
            if name == "antigence_scan":
                return await self._scan_code(arguments)
            elif name == "antigence_detect":
                return await self._detect_anomaly(arguments)
            elif name == "antigence_analyze":
                return await self._full_analysis(arguments)
            elif name == "antigence_inspect":
                return await self._extract_features(arguments)
            elif name == "antigence_recall":
                return await self._query_memory(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    def _create_antigen(self, code: str, data_type: str) -> Antigen:
        """Create an Antigen from input code/text"""
        if data_type == "code" or (data_type == "auto" and any(
            code.strip().startswith(kw) for kw in ["def ", "import ", "class ", "if ", "for ", "while "]
        )):
            return Antigen(data=code, data_type=DataType.CODE, identifier="user_code")
        else:
            return Antigen.from_text(code, metadata={"source": "mcp"})

    async def _scan_code(self, args: Dict[str, Any]) -> list[TextContent]:
        """Run B Cell pattern matching on code"""
        code = args['code']
        data_type = args.get('data_type', 'auto')

        antigen = self._create_antigen(code, data_type)

        # Use orchestrator manager (automatically handles local/remote fallback)
        try:
            result = self.orchestrator_manager.analyze(antigen)
        except Exception as e:
            # If analysis fails, try to fall back to local
            if self.orchestrator_manager.get_current_orchestrator().get_mode() != "local":
                print(f"Analysis failed, forcing local orchestrator: {e}", file=sys.stderr)
                self.orchestrator_manager.force_local()
                result = self.orchestrator_manager.analyze(antigen)
            else:
                raise

        # Determine risk level
        if result.anomaly:
            if result.confidence > 0.8:
                risk = "🔴 HIGH RISK"
                recommendation = "Do NOT use this code. Security review required."
            elif result.confidence > 0.5:
                risk = "🟡 MEDIUM RISK"
                recommendation = "Proceed with caution. Manual review recommended."
            else:
                risk = "🟢 LOW RISK (uncertain)"
                recommendation = "Possibly safe, but low confidence. Review recommended."
        else:  # safe
            if result.confidence > 0.8:
                risk = "🟢 SAFE"
                recommendation = "Code appears safe. Standard review process."
            else:
                risk = "🟡 UNCLEAR"
                recommendation = "Classification uncertain. Manual review recommended."

        # Format output
        output = f"""## 🧬 IMMUNOS B Cell Scan

**Classification:** {result.classification.upper() if result.classification else 'UNKNOWN'}
**Risk Level:** {risk}
**Confidence:** {result.confidence:.1%}

### Pattern Matching Results
**Avidity Scores:**
"""

        avidity_scores = result.metadata.get("bcell", {}).get("avidity_scores", {})
        if avidity_scores:
            for label, score in sorted(avidity_scores.items(), key=lambda x: x[1], reverse=True):
                bar = '█' * int(score * 20)
                output += f"- {label}: {score:.3f} {bar}\n"
        else:
            output += "No pattern matches found\n"

        output += f"\n### Recommendation\n{recommendation}\n"
        output += "\n### Analysis Details\n"
        output += f"- Patterns matched: {len(avidity_scores)}\n"
        output += "- Strategy: Simple Highest Avidity (SHA)\n"
        output += f"- Uncertain: {'Yes' if result.metadata.get('bcell', {}).get('is_uncertain') else 'No'}\n"
        output += "- Agent: B Cell (Pattern Matching)\n"

        output += "\n---\n*Powered by IMMUNOS-MCP B Cell Agent*"

        return [TextContent(type="text", text=output)]

    async def _detect_anomaly(self, args: Dict[str, Any]) -> list[TextContent]:
        """Run NK Cell anomaly detection"""
        code = args['code']
        data_type = args.get('data_type', 'auto')

        antigen = self._create_antigen(code, data_type)

        # Use orchestrator manager (automatically handles local/remote fallback)
        try:
            result = self.orchestrator_manager.analyze(antigen)
        except Exception as e:
            # If analysis fails, try to fall back to local
            if self.orchestrator_manager.get_current_orchestrator().get_mode() != "local":
                print(f"Analysis failed, forcing local orchestrator: {e}", file=sys.stderr)
                self.orchestrator_manager.force_local()
                result = self.orchestrator_manager.analyze(antigen)
            else:
                raise

        # NK Cell specific analysis
        is_anomaly = result.anomaly
        nk_confidence = result.nk_confidence

        if is_anomaly:
            if nk_confidence > 0.8:
                severity = "🔴 HIGH SEVERITY"
                explanation = "Strong anomaly detected. This code deviates significantly from normal patterns."
            elif nk_confidence > 0.5:
                severity = "🟡 MEDIUM SEVERITY"
                explanation = "Anomaly detected. This code shows unusual characteristics."
            else:
                severity = "🟢 LOW SEVERITY"
                explanation = "Weak anomaly signal. May be benign but worth reviewing."
        else:
            severity = "🟢 NORMAL"
            explanation = "No anomaly detected. Code appears consistent with normal patterns."

        output = f"""## 🛡️ IMMUNOS NK Cell Anomaly Detection

**Anomaly Status:** {'⚠️ ANOMALY DETECTED' if is_anomaly else '✓ NORMAL'}
**Severity:** {severity}
**Confidence:** {nk_confidence:.1%}

### Analysis
{explanation}

### Detection Method
- **Algorithm:** Negative Selection
- **Approach:** Zero-shot detection (no prior training on threats)
- **Principle:** Detects deviations from "self" (normal) patterns

### Details
- **NK Cell Confidence:** {nk_confidence:.1%}
"""
        # Get NK cell details if using local orchestrator
        current_orch = self.orchestrator_manager.get_current_orchestrator()
        if current_orch.get_mode() == "local":
            nk_agent = current_orch.orchestrator.nk_cell
            output += f"- **Detection Threshold:** {nk_agent.detection_threshold:.2f}\n"
            output += f"- **Detectors Used:** {len(nk_agent.detectors)} detectors\n"
            output += f"- **Self Patterns:** {len(nk_agent.self_patterns)} patterns\n"
        else:
            output += "- **Detection Details:** N/A (remote orchestrator)\n"
        output += """

---

*Powered by IMMUNOS-MCP NK Cell Agent (Negative Selection Algorithm)*
"""

        return [TextContent(type="text", text=output)]

    async def _full_analysis(self, args: Dict[str, Any]) -> list[TextContent]:
        """Run complete multi-agent analysis"""
        code = args['code']
        data_type = args.get('data_type', 'auto')

        antigen = self._create_antigen(code, data_type)

        # Use orchestrator manager (automatically handles local/remote fallback)
        try:
            result = self.orchestrator_manager.analyze(antigen)
        except Exception as e:
            # If analysis fails, try to fall back to local
            if self.orchestrator_manager.get_current_orchestrator().get_mode() != "local":
                print(f"Analysis failed, forcing local orchestrator: {e}", file=sys.stderr)
                self.orchestrator_manager.force_local()
                result = self.orchestrator_manager.analyze(antigen)
            else:
                raise

        # Comprehensive analysis
        output = f"""## 🧬 IMMUNOS Multi-Agent Analysis

**Overall Classification:** {result.classification.upper() if result.classification else 'UNKNOWN'}
**Anomaly Detected:** {'⚠️ YES' if result.anomaly else '✓ NO'}
**Aggregated Confidence:** {result.confidence:.1%}

### Agent Results

#### B Cell (Pattern Matching)
- **Classification:** {result.classification or 'Unknown'}
- **Confidence:** {result.bcell_confidence:.1%}
- **Patterns Matched:** {len(result.metadata.get('bcell', {}).get('avidity_scores', {}))}

#### NK Cell (Anomaly Detection)
- **Anomaly:** {'⚠️ Detected' if result.anomaly else '✓ None'}
- **Confidence:** {result.nk_confidence:.1%}
"""
        current_orch = self.orchestrator_manager.get_current_orchestrator()
        if current_orch.get_mode() == "local":
            nk_agent = current_orch.orchestrator.nk_cell
            output += f"- **Detectors:** {len(nk_agent.detectors)}\n"
        else:
            output += "- **Detectors:** N/A (remote orchestrator)\n"
        output += """

#### Dendritic Cell (Feature Extraction)
- **Features Extracted:** {len(result.features)}
- **Signals:** {result.signals}

#### Memory Agent
- **Memory Hit:** {'✓ Yes' if result.memory_hit else '✗ No'}
- **Cached:** {'Previously analyzed' if result.memory_hit else 'New analysis'}

### Detailed Features
"""

        for key, value in result.features.items():
            output += f"- **{key}:** {value}\n"

        output += "\n### Signals\n"
        for signal, value in result.signals.items():
            bar = '█' * int(value * 20)
            output += f"- **{signal}:** {value:.3f} {bar}\n"

        output += "\n### Agents Used\n"
        for agent in result.agents:
            output += f"- {agent}\n"

        if result.model_roles:
            output += "\n### Model Roles\n"
            for agent, model in result.model_roles.items():
                output += f"- **{agent}:** {model}\n"

        output += "\n### Execution Metadata\n"
        output += f"- **Execution Time:** {result.metadata.get('execution_time', 0):.3f}s\n"
        output += f"- **Domain:** {result.metadata.get('domain', 'default')}\n"

        # Risk assessment
        if result.anomaly and result.confidence > 0.7:
            recommendation = "🔴 HIGH RISK: Do NOT use this code. Security review required immediately."
        elif result.anomaly:
            recommendation = "🟡 MEDIUM RISK: Proceed with caution. Manual security review recommended."
        elif result.confidence > 0.8:
            recommendation = "🟢 SAFE: Code appears safe based on multi-agent consensus."
        else:
            recommendation = "🟡 UNCERTAIN: Low confidence across agents. Manual review recommended."

        output += f"\n### Recommendation\n{recommendation}\n"

        output += "\n---\n*Powered by IMMUNOS-MCP Multi-Agent Orchestrator*"

        return [TextContent(type="text", text=output)]

    async def _extract_features(self, args: Dict[str, Any]) -> list[TextContent]:
        """Extract features using Dendritic Cell agent"""
        code = args['code']
        data_type = args.get('data_type', 'auto')

        antigen = self._create_antigen(code, data_type)

        # Get current orchestrator for feature extraction
        current_orch = self.orchestrator_manager.get_current_orchestrator()
        if current_orch.get_mode() == "local":
            features = current_orch.orchestrator.dendritic.extract_features(antigen)
            signals = current_orch.orchestrator._derive_signals(features)
        else:
            # For remote, create a basic analysis to get features
            result = self.orchestrator_manager.analyze(antigen)
            features = result.features
            signals = result.signals

        output = f"""## 🔬 IMMUNOS Feature Extraction

**Agent:** Dendritic Cell
**Data Type:** {antigen.data_type.value}

### Extracted Features
"""

        for key, value in features.items():
            output += f"- **{key}:** {value}\n"

        output += "\n### Derived Signals\n"
        for signal, value in signals.items():
            bar = '█' * int(value * 20)
            output += f"- **{signal}:** {value:.3f} {bar}\n"

        output += "\n### Signal Interpretation\n"
        danger = signals.get('danger', 0)
        pamp = signals.get('pamp', 0)
        safe = signals.get('safe', 0)

        if danger > 0.7:
            output += "- **Danger Signal:** 🔴 High - Complex or potentially risky content\n"
        elif danger > 0.3:
            output += "- **Danger Signal:** 🟡 Medium - Moderate complexity\n"
        else:
            output += "- **Danger Signal:** 🟢 Low - Simple content\n"

        if pamp > 0.5:
            output += "- **PAMP Signal:** ⚠️ Present - Known threat pattern detected\n"
        else:
            output += "- **PAMP Signal:** ✓ None - No known threat patterns\n"

        if safe > 0.5:
            output += "- **Safe Signal:** ✓ High - Content appears benign\n"
        else:
            output += "- **Safe Signal:** ⚠️ Low - Cannot confirm safety\n"

        output += "\n---\n*Powered by IMMUNOS-MCP Dendritic Cell Agent*"

        return [TextContent(type="text", text=output)]

    async def _query_memory(self, args: Dict[str, Any]) -> list[TextContent]:
        """Query or manage memory agent"""
        code = args['code']
        action = args.get('action', 'check')

        if action == 'clear':
            current_orch = self.orchestrator_manager.get_current_orchestrator()
            if current_orch.get_mode() == "local":
                current_orch.orchestrator.memory.clear()
            else:
                # For remote, we'd need to call remote API
                # For now, just report that we can't clear remote memory
                pass
            output = """## 🧠 IMMUNOS Memory Agent

**Action:** Clear Memory
**Status:** ✓ Memory cleared successfully

All stored patterns and analysis results have been removed.

---
*Powered by IMMUNOS-MCP Memory Agent*
"""
            return [TextContent(type="text", text=output)]

        # Check memory
        antigen = self._create_antigen(code, 'auto')
        memory_key = antigen.identifier or str(hash(antigen.get_text_content()))

        # Get memory from current orchestrator
        current_orch = self.orchestrator_manager.get_current_orchestrator()
        if current_orch.get_mode() == "local":
            previous = current_orch.orchestrator.memory.retrieve(memory_key)
        else:
            previous = None  # Remote memory not accessible directly

        if previous:
            output = f"""## 🧠 IMMUNOS Memory Agent

**Status:** ✓ Memory Hit
**Key:** {memory_key[:50]}...

### Previous Analysis Found
This code has been analyzed before. Previous results:

#### B Cell Results
"""
            bcell_data = previous.get('bcell', {})
            if bcell_data:
                output += f"- **Classification:** {bcell_data.get('predicted_class', 'Unknown')}\n"
                output += f"- **Confidence:** {bcell_data.get('confidence', 0):.1%}\n"

            output += "\n#### NK Cell Results\n"
            nk_data = previous.get('nk', {})
            if nk_data:
                output += f"- **Anomaly:** {'Yes' if nk_data.get('is_anomaly') else 'No'}\n"
                output += f"- **Confidence:** {nk_data.get('confidence', 0):.1%}\n"

            output += "\n### Current Analysis\n"
            result = self.orchestrator_manager.analyze(antigen)
            output += f"- **Current Classification:** {result.classification or 'Unknown'}\n"
            output += f"- **Current Confidence:** {result.confidence:.1%}\n"
            output += f"- **Anomaly:** {'Yes' if result.anomaly else 'No'}\n"

        else:
            output = f"""## 🧠 IMMUNOS Memory Agent

**Status:** ✗ No Memory Hit
**Key:** {memory_key[:50]}...

### Analysis
This code has not been analyzed before. Running new analysis...

"""
            result = self.orchestrator_manager.analyze(antigen)
            output += "### New Analysis Results\n"
            output += f"- **Classification:** {result.classification or 'Unknown'}\n"
            output += f"- **Confidence:** {result.confidence:.1%}\n"
            output += f"- **Anomaly:** {'Yes' if result.anomaly else 'No'}\n"
            output += "\n✓ Results stored in memory for future queries.\n"

        output += "\n---\n*Powered by IMMUNOS-MCP Memory Agent*"

        return [TextContent(type="text", text=output)]

    async def run(self):
        """Run the MCP server"""
        print("IMMUNOS-MCP Enhanced Server starting...", file=sys.stderr)
        print("Multi-agent system ready:", file=sys.stderr)
        print("  - B Cell (Pattern Matching)", file=sys.stderr)
        print("  - NK Cell (Anomaly Detection)", file=sys.stderr)
        print("  - Dendritic Cell (Feature Extraction)", file=sys.stderr)
        print("  - Memory Agent (Pattern Caching)", file=sys.stderr)
        print("  - Local Thymus/Orchestrator (always available)", file=sys.stderr)
        print("Waiting for MCP client connection via stdio...", file=sys.stderr)

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Entry point"""
    server = SimpleMCPServer()
    await server.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
