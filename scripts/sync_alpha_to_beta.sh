#!/usr/bin/env bash
# sync_alpha_to_beta.sh — Sync alpha (private) to beta (public) with sanitization
#
# Usage: ./scripts/sync_alpha_to_beta.sh [--dry-run]
#
# This script:
# 1. Copies source files from alpha to beta
# 2. Excludes private/local-only content
# 3. Scans for hardcoded paths and secrets
# 4. Reports any issues before you commit

set -euo pipefail

ALPHA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BETA_DIR="${ALPHA_DIR}/../antigence-beta"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] No files will be modified."
fi

# Verify directories exist
if [[ ! -d "$ALPHA_DIR/.git" ]]; then
    echo "ERROR: Alpha directory not found at $ALPHA_DIR" >&2
    exit 1
fi
if [[ ! -d "$BETA_DIR/.git" ]]; then
    echo "ERROR: Beta directory not found at $BETA_DIR" >&2
    exit 1
fi

echo "=== Alpha → Beta Sync ==="
echo "  Alpha: $ALPHA_DIR"
echo "  Beta:  $BETA_DIR"
echo ""

# ── Exclusions ──────────────────────────────────────────────────────────────
RSYNC_EXCLUDES=(
    # Git and environments
    --exclude='.git/'
    --exclude='.venv/'
    --exclude='venv/'
    --exclude='.env'
    --exclude='.env.*'
    --exclude='!.env.example'

    # Local system data
    --exclude='.immunos/'
    --exclude='.DS_Store'
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='.pytest_cache/'
    --exclude='*.egg-info/'

    # Shadow files
    --exclude='*.py.md'
    --exclude='*.json.md'
    --exclude='*.sh.md'
    --exclude='*.csv.md'
    --exclude='*.html.md'
    --exclude='*.css.md'
    --exclude='*.toml.md'
    --exclude='*.cff.md'
    --exclude='*.bib.md'
    --exclude='*.db.md'
    --exclude='*.log.md'
    --exclude='*.png.md'
    --exclude='*.txt.md'
    --exclude='*.yaml.md'
    --exclude='*.yml.md'
    --exclude='.DS_Store.md'
    --exclude='.gitignore.md'
    --exclude='LICENSE.md'
    --exclude='VERSION.md'

    # Runtime artifacts
    --exclude='*.db'
    --exclude='*.sqlite3'
    --exclude='*.log'
    --exclude='*.pkl'

    # Local config (use .example instead)
    --exclude='config/antigence_mcp_config.json'
    --exclude='config/models.json'

    # Internal cases
    --exclude='cases/'

    # Data directories with runtime content
    --exclude='data/immunos_data/'
    --exclude='data/dictionaries/'
    --exclude='runs/'
    --exclude='blog/'
    --exclude='context.json'
)

# ── Step 1: Sync files ──────────────────────────────────────────────────────
echo "Step 1: Syncing files..."
if $DRY_RUN; then
    rsync -avn --delete "${RSYNC_EXCLUDES[@]}" "$ALPHA_DIR/" "$BETA_DIR/" 2>&1 | tail -20
    echo "  ... (truncated, use without --dry-run to execute)"
else
    rsync -av --delete "${RSYNC_EXCLUDES[@]}" "$ALPHA_DIR/" "$BETA_DIR/" 2>&1 | tail -5
fi
echo ""

# ── Step 2: Audit for leaked paths ─────────────────────────────────────────
echo "Step 2: Scanning beta for hardcoded paths..."
LEAKS=0

# Check for /Users/byron (only in git-tracked files)
if git -C "$BETA_DIR" grep -l "/Users/byron" 2>/dev/null; then
    echo "  WARNING: Found /Users/byron references in the above files!"
    LEAKS=$((LEAKS + 1))
else
    echo "  OK: No /Users/byron paths found."
fi

# Check for potential secrets (basic patterns)
if grep -rE "(sk-[a-zA-Z0-9]{20,}|AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{36})" "$BETA_DIR" --include="*.py" --include="*.json" --include="*.env" -l 2>/dev/null; then
    echo "  WARNING: Potential API keys found in the above files!"
    LEAKS=$((LEAKS + 1))
else
    echo "  OK: No API key patterns detected."
fi

# Check for .env files (not .example)
if find "$BETA_DIR" -name ".env" -not -name "*.example" -not -path "*/.git/*" | grep -q .; then
    echo "  WARNING: Found .env files (not .example)!"
    LEAKS=$((LEAKS + 1))
else
    echo "  OK: No .env files."
fi

# Check for database files
if find "$BETA_DIR" -name "*.db" -o -name "*.sqlite3" | grep -q .; then
    echo "  WARNING: Found database files!"
    LEAKS=$((LEAKS + 1))
else
    echo "  OK: No database files."
fi

# Check for .immunos directories
if find "$BETA_DIR" -name ".immunos" -type d | grep -q .; then
    echo "  WARNING: Found .immunos directories!"
    LEAKS=$((LEAKS + 1))
else
    echo "  OK: No .immunos directories."
fi

echo ""

# ── Step 3: Summary ─────────────────────────────────────────────────────────
if [[ $LEAKS -gt 0 ]]; then
    echo "=== SYNC COMPLETE WITH $LEAKS WARNING(S) ==="
    echo "Review warnings above before committing to beta."
    exit 1
else
    echo "=== SYNC COMPLETE — ALL CLEAN ==="
    echo ""
    echo "Next steps:"
    echo "  cd $BETA_DIR"
    echo "  git status"
    echo "  git add -A"
    echo "  git commit -m 'Sync from alpha: <description>'"
    echo "  git push origin main"
fi
