#!/bin/bash
# Install Antigence as MCP Server
# Updated: 2026-01-11 - Fixed paths for active_projects location

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Installing Antigence MCP Server                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

ANTIGENCE_PATH="/Users/byron/projects/active_projects/antigence"
CONFIG_SOURCE="/Users/byron/projects/antigence_mcp_config.json"

# 1. Install for Claude Code
echo "1. Installing for Claude Code..."
CLAUDE_CONFIG_DIR="$HOME/.config/claude"
CLAUDE_MCP_CONFIG="$CLAUDE_CONFIG_DIR/mcp_settings.json"

mkdir -p "$CLAUDE_CONFIG_DIR"

if [ -f "$CLAUDE_MCP_CONFIG" ]; then
    echo "   ⚠️  Existing mcp_settings.json found"
    echo "   Backing up to mcp_settings.json.backup"
    cp "$CLAUDE_MCP_CONFIG" "$CLAUDE_MCP_CONFIG.backup"
fi

cp "$CONFIG_SOURCE" "$CLAUDE_MCP_CONFIG"
echo "   ✓ Installed to $CLAUDE_MCP_CONFIG"

# 2. Install for Claude Desktop
echo ""
echo "2. Installing for Claude Desktop..."
DESKTOP_CONFIG_DIR="$HOME/Library/Application Support/Claude"
DESKTOP_MCP_CONFIG="$DESKTOP_CONFIG_DIR/claude_desktop_config.json"

mkdir -p "$DESKTOP_CONFIG_DIR"

if [ -f "$DESKTOP_MCP_CONFIG" ]; then
    echo "   ⚠️  Existing claude_desktop_config.json found"
    echo "   Backing up to claude_desktop_config.json.backup"
    cp "$DESKTOP_MCP_CONFIG" "$DESKTOP_MCP_CONFIG.backup"
fi

cp "$CONFIG_SOURCE" "$DESKTOP_MCP_CONFIG"
echo "   ✓ Installed to $DESKTOP_MCP_CONFIG"

# 3. Test MCP server
echo ""
echo "3. Testing MCP server..."
cd "$ANTIGENCE_PATH"

# Check dependencies
if ! python3 -c "import mcp" 2>/dev/null; then
    echo "   ⚠️  MCP SDK not found, installing..."
    pip3 install "mcp>=1.3.1" --quiet
fi

echo "   ✓ Dependencies OK"

# 4. Summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Installation Complete!                                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Antigence MCP Server installed for:"
echo "  ✓ Claude Code     ($CLAUDE_MCP_CONFIG)"
echo "  ✓ Claude Desktop  ($DESKTOP_MCP_CONFIG)"
echo ""
echo "Next steps:"
echo "  1. Restart Claude Code or Claude Desktop"
echo "  2. Test with: 'Can you scan code using antigence_scan tool?'"
echo ""
echo "Available MCP tools:"
echo "  - antigence_scan      - Security vulnerability scanning"
echo "  - antigence_detect    - Anomaly detection (zero-shot)"
echo "  - antigence_analyze   - Complete multi-agent analysis"
echo "  - antigence_inspect   - Feature extraction"
echo "  - antigence_recall    - Memory queries"
echo ""
