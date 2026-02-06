#!/bin/bash
# Ollama Setup Script for IMMUNOS-MCP (16GB Laptop Configuration)
#
# This script:
# 1. Installs Ollama for macOS
# 2. Downloads recommended models for 16GB RAM (~14GB total)
# 3. Verifies installation
#
# For offline/air-gapped deployment, see docs/Offline-Deployment-Architecture.md

set -e  # Exit on error

echo "========================================================================"
echo "IMMUNOS-MCP: Ollama Setup for 16GB Configuration"
echo "========================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check system requirements
echo "🔍 Checking system requirements..."
TOTAL_MEM_GB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
echo "   Total RAM: ${TOTAL_MEM_GB}GB"

if [ "$TOTAL_MEM_GB" -lt 16 ]; then
    echo -e "${YELLOW}⚠️  Warning: System has ${TOTAL_MEM_GB}GB RAM. 16GB recommended.${NC}"
    echo "   Consider using smaller models (see docs/Model-Selection-By-Agent-Role.md)"
    read -p "   Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if Ollama is already installed
echo ""
echo "📦 Checking Ollama installation..."

if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama is already installed${NC}"
    ollama --version
else
    echo "   Ollama not found. Installing..."

    # Check if we're on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo ""
        echo "⬇️  Installing Ollama for macOS..."

        # Check if Homebrew is available
        if command -v brew &> /dev/null; then
            echo "   Using Homebrew..."
            brew install ollama
        else
            echo "   Downloading Ollama.app..."
            echo ""
            echo "   Please follow these steps:"
            echo "   1. Download Ollama from: https://ollama.com/download/mac"
            echo "   2. Open the downloaded Ollama.app"
            echo "   3. Run this script again"
            echo ""
            echo "   Or install Homebrew first:"
            echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            echo "   Then run: brew install ollama"
            exit 1
        fi
    else
        # Linux installation
        echo ""
        echo "⬇️  Downloading Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
    fi

    if command -v ollama &> /dev/null; then
        echo -e "${GREEN}✓ Ollama installed successfully${NC}"
    else
        echo -e "${RED}✗ Failed to install Ollama${NC}"
        echo ""
        echo "Manual installation:"
        echo "  1. Visit: https://ollama.com/download"
        echo "  2. Download for your platform"
        echo "  3. Run this script again"
        exit 1
    fi
fi

# Start Ollama service if not running
echo ""
echo "🚀 Starting Ollama service..."
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve > /dev/null 2>&1 &
    sleep 3
    echo -e "${GREEN}✓ Ollama service started${NC}"
else
    echo -e "${GREEN}✓ Ollama service already running${NC}"
fi

# Define models for 16GB configuration
# Total: ~14GB (leaves 2GB for system + application)
echo ""
echo "========================================================================"
echo "Model Download Plan (16GB Configuration)"
echo "========================================================================"
echo ""
echo "Model                    Size    Purpose"
echo "------------------------------------------------------------------------"
echo "qwen2.5-coder:7b        4.7GB   B Cell + Dendritic (code analysis)"
echo "deepseek-r1:14b         8.0GB   NK Cell + T Cell + QML (reasoning)"
echo "qwen2.5:1.5b            1.0GB   Macrophage (fast triage)"
echo "------------------------------------------------------------------------"
echo "TOTAL                   ~14GB"
echo ""

# Ask for confirmation
read -p "Download these models? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Skipping model download. Run this script again when ready."
    exit 0
fi

# Download models
echo ""
echo "⬇️  Downloading models (this may take 10-30 minutes)..."
echo ""

# Model 1: qwen2.5-coder:7b
echo "📥 Downloading qwen2.5-coder:7b (4.7GB)..."
echo "   Purpose: Code pattern matching and feature extraction"
ollama pull qwen2.5-coder:7b
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ qwen2.5-coder:7b downloaded${NC}"
else
    echo -e "${RED}✗ Failed to download qwen2.5-coder:7b${NC}"
fi
echo ""

# Model 2: deepseek-r1:14b
echo "📥 Downloading deepseek-r1:14b (8GB)..."
echo "   Purpose: Deep reasoning for anomaly detection and coordination"
ollama pull deepseek-r1:14b
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ deepseek-r1:14b downloaded${NC}"
else
    echo -e "${RED}✗ Failed to download deepseek-r1:14b${NC}"
fi
echo ""

# Model 3: qwen2.5:1.5b (fast triage)
echo "📥 Downloading qwen2.5:1.5b (1.0GB)..."
echo "   Purpose: Fast initial triage"
ollama pull qwen2.5:1.5b
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ qwen2.5:1.5b downloaded${NC}"
else
    echo -e "${RED}✗ Failed to download qwen2.5:1.5b${NC}"
fi
echo ""

# Verify installation
echo "========================================================================"
echo "Verifying Installation"
echo "========================================================================"
echo ""

echo "📋 Installed models:"
ollama list

echo ""
echo "========================================================================"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Verify installation: python examples/verify_ollama.py"
echo "  2. Run demo: python examples/llm_agents_demo.py"
echo ""
echo "For offline deployment, see: docs/Offline-Deployment-Architecture.md"
echo ""
echo "💡 Tip: Models are cached in ~/.ollama/models/"
echo "   You can copy this directory for offline deployment."
echo ""
