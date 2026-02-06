#!/bin/bash
# IMMUNOS-MCP Local Demo Deployment Script
# One-command setup for portfolio demo

set -e  # Exit on error

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧬 IMMUNOS-MCP Portfolio Demo Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. Check Python version
echo "📋 Step 1: Checking Python version..."
python3 --version

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

echo "✅ Python 3 found"
echo ""

# 2. Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found."
    echo "   Please run this script from the web_app/ directory:"
    echo "   cd /path/to/immunos-mcp/web_app && ./deploy_local.sh"
    exit 1
fi

# 3. Install dependencies
echo "📦 Step 2: Installing dependencies..."
echo "   This may take a few minutes..."

pip3 install -r requirements.txt --quiet

echo "✅ Dependencies installed"
echo ""

# 4. Initialize database
echo "🗄️  Step 3: Initializing database..."

if [ -f "data/immunos.db" ]; then
    echo "⚠️  Database already exists at data/immunos.db"
    read -p "   Do you want to reset it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f data/immunos.db
        python3 database.py init
        python3 database.py seed
        echo "✅ Database reset and seeded"
    else
        echo "ℹ️  Keeping existing database"
    fi
else
    python3 database.py init
    python3 database.py seed
    echo "✅ Database initialized and seeded"
fi

echo ""

# 5. Check database stats
echo "📊 Step 4: Database statistics..."
python3 database.py stats
echo ""

# 6. Optional: Test Ollama connection
echo "🔍 Step 5: Checking for Ollama (optional)..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama found"
    if ollama list | tail -n +2 | grep -q .; then
        echo "✅ Ollama models available"
    else
        echo "⚠️  No Ollama models found. Example: 'ollama pull llama3.2:3b'"
    fi
else
    echo "ℹ️  Ollama not installed (optional for advanced features)"
    echo "   Install from: https://ollama.ai"
fi

echo ""

# 7. Final message
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🚀 Starting IMMUNOS-MCP demo server..."
echo ""
PORT="${PORT:-5001}"
HOST="${HOST:-127.0.0.1}"
echo "📱 Access the demo at: http://${HOST}:${PORT}"
echo "🛑 Press Ctrl+C to stop the server"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 8. Launch Flask server
python3 app.py
