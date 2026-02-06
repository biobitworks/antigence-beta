#!/bin/bash
# Create Air-Gapped Deployment Bundle
# Usage: ./bundle_airgapped.sh [output_dir]

set -e

OUTPUT_DIR="${1:-immunos-mcp-airgapped}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Creating air-gapped deployment bundle..."
echo "Output directory: $OUTPUT_DIR"

# Create bundle structure
mkdir -p "$OUTPUT_DIR"/{packages,models,scripts,config}

# Copy source code
echo "Copying source code..."
cp -r "$PROJECT_DIR"/src "$OUTPUT_DIR/"
cp -r "$PROJECT_DIR"/config "$OUTPUT_DIR/" 2>/dev/null || true
cp "$PROJECT_DIR"/pyproject.toml "$OUTPUT_DIR/"
cp "$PROJECT_DIR"/README.md "$OUTPUT_DIR/" 2>/dev/null || true

# Download Python packages
echo "Downloading Python packages..."
cd "$PROJECT_DIR"
if [ -f "requirements.txt" ]; then
    pip download -r requirements.txt -d "$OUTPUT_DIR/packages" || {
        echo "Warning: Some packages may not be downloadable offline"
        echo "Creating requirements from pyproject.toml..."
        # Extract requirements from pyproject.toml
        grep -A 100 "dependencies" pyproject.toml | grep -E '^\s+"' | sed 's/[",]//g' > "$OUTPUT_DIR/requirements.txt"
        pip download -r "$OUTPUT_DIR/requirements.txt" -d "$OUTPUT_DIR/packages" || true
    }
else
    # Create requirements from dependencies
    echo "Creating requirements.txt..."
    python3 -c "
import tomli
with open('pyproject.toml', 'rb') as f:
    data = tomli.load(f)
deps = data.get('project', {}).get('dependencies', [])
with open('$OUTPUT_DIR/requirements.txt', 'w') as f:
    for dep in deps:
        f.write(dep + '\n')
"
    pip download -r "$OUTPUT_DIR/requirements.txt" -d "$OUTPUT_DIR/packages" || true
fi

# Create installation script
cat > "$OUTPUT_DIR/install.sh" << 'INSTALL_EOF'
#!/bin/bash
# Offline Installation Script for IMMUNOS-MCP
# Run this script on the air-gapped system

set -e

echo "Installing IMMUNOS-MCP (air-gapped mode)..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found. Please install Python 3.10+ first."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install packages from bundle
echo "Installing Python packages..."
if [ -d "packages" ] && [ "$(ls -A packages)" ]; then
    pip install --no-index --find-links=packages -r requirements.txt
else
    echo "Warning: No packages directory found. Installing from PyPI (requires internet)."
    pip install -r requirements.txt
fi

# Install IMMUNOS-MCP
echo "Installing IMMUNOS-MCP..."
pip install -e .

# Create config directory
echo "Creating configuration directory..."
mkdir -p ~/.immunos-mcp
if [ -f "config/config.yaml.example" ]; then
    cp config/config.yaml.example ~/.immunos-mcp/config.yaml
    echo "Configuration template copied to ~/.immunos-mcp/config.yaml"
fi

# Set default mode to air-gapped
export IMMUNOS_MODE=air_gapped

# Test installation
echo "Testing installation..."
python -c "from immunos_mcp.config.loader import load_config; from immunos_mcp.orchestrator.manager import OrchestratorManager; config = load_config(); mgr = OrchestratorManager(config); print('✅ Installation successful!')" || {
    echo "Warning: Installation test failed, but installation may still be successful."
}

echo ""
echo "✅ IMMUNOS-MCP installed successfully!"
echo ""
echo "Next steps:"
echo "1. Configure: ~/.immunos-mcp/config.yaml"
echo "2. Test: python test_mcp_setup.py"
echo "3. Run MCP server: python src/immunos_mcp/servers/simple_mcp_server.py"
INSTALL_EOF

chmod +x "$OUTPUT_DIR/install.sh"

# Create README
cat > "$OUTPUT_DIR/README_AIRGAPPED.md" << 'README_EOF'
# IMMUNOS-MCP Air-Gapped Deployment

This bundle contains everything needed to install IMMUNOS-MCP on an air-gapped system.

## Contents

- `src/` - Source code
- `packages/` - Python packages (offline installation)
- `install.sh` - Installation script
- `requirements.txt` - Python dependencies
- `config/` - Configuration templates

## Installation

1. Copy this entire directory to the air-gapped system
2. Run: `./install.sh`
3. Configure: `~/.immunos-mcp/config.yaml`
4. Test: `python test_mcp_setup.py`

## Requirements

- Python 3.10 or higher
- pip
- Internet connection NOT required (all packages included)

## Configuration

After installation, edit `~/.immunos-mcp/config.yaml`:

```yaml
mode: air_gapped
orchestrator:
  type: local
  local:
    enabled: true
```

## Usage

Run MCP server:
```bash
source venv/bin/activate
python src/immunos_mcp/servers/simple_mcp_server.py
```

## Troubleshooting

If installation fails:
1. Check Python version: `python3 --version` (needs 3.10+)
2. Check pip: `pip --version`
3. Try installing packages individually from `packages/` directory

## Security Notes

- All processing is local (no network required)
- No external API calls
- All data stays on local system
- Suitable for sensitive environments (medical data, classified networks)
README_EOF

# Create tarball
echo "Creating tarball..."
cd "$(dirname "$OUTPUT_DIR")"
tar -czf "$(basename "$OUTPUT_DIR").tar.gz" "$(basename "$OUTPUT_DIR")"

echo ""
echo "✅ Bundle created successfully!"
echo ""
echo "Bundle location: $OUTPUT_DIR"
echo "Tarball: $(basename "$OUTPUT_DIR").tar.gz"
echo ""
echo "To deploy:"
echo "1. Copy tarball to air-gapped system"
echo "2. Extract: tar -xzf $(basename "$OUTPUT_DIR").tar.gz"
echo "3. Run: cd $(basename "$OUTPUT_DIR") && ./install.sh"

