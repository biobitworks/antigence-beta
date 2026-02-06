# Antigence Multi-Agent Startup Guide

**Antigence**: The next-generation AI agent coordination system
**IMMUNOS**: The underlying operating system / context management layer

## 🏗️ Architecture Overview

```
/Users/byron/projects/              ← Working directory (all projects)
├── .immunos/                       ← Hidden OS layer (context database)
│   ├── memory/                     ← Shared memory across all agents
│   ├── snapshots/                  ← Context checkpoints
│   ├── recovery/                   ← Fast recovery files
│   ├── journal/                    ← Daily activity logs
│   ├── model-contexts/             ← Agent-specific contexts
│   └── db/                         ← SQLite databases
│
├── antigence-alpha/                ← Antigence web app (MCP server?)
├── scripts/                        ← IMMUNOS system scripts
├── CLAUDE.md                       ← User context (main)
└── ANTIGENCE_STARTUP.md            ← This file (visible)
```

## 📍 What is .immunos?

**`.immunos`** is the **hidden operating system layer** that provides:

### Core Directories (sorted by importance)

| Directory | Purpose | Size | Why Hidden? |
|-----------|---------|------|-------------|
| `memory/` | Shared context database for all agents | 1.7M | Implementation detail - agents read via scripts |
| `agents/` | Agent sessions and hallucination logs | 15M | Debug/audit logs - not for daily review |
| `logs/` | System operation logs | 13M | Technical logs - reviewed on demand |
| `runs/` | Execution history | 3.5M | Historical data - accessed programmatically |
| `db/` | SQLite databases for dashboard | 256K | Binary data - accessed via web UI |
| `journal/` | Daily journals (2025-12-26.md, etc.) | 84K | **Should be visible?** Daily review files |
| `recovery/` | Fast context recovery files | 16K | Read at startup - don't need to see |
| `model-contexts/` | Agent startup contexts | 20K | Read at startup - don't need to see |
| `config/` | System configuration | 16K | Rarely changed - set and forget |

### Files That Should Be Visible

**Journals** (`journal/*.md`) - You might want these in `daily/` instead?
**Quick Start** (moved to parent directory below)
**Recovery Context** (accessed via startup script)

## 🎯 Antigence vs IMMUNOS

### IMMUNOS (Operating System Layer)
- **Location**: `.immunos/` (hidden)
- **Purpose**: Context persistence, memory management, agent coordination
- **Analogy**: Like Linux kernel - runs in background
- **Components**:
  - T Cell (memory)
  - NK Cell (scanner)
  - B Cell (verifier)
  - Dendritic Cell (reporter)
  - Snapshot/recovery system

### Antigence (Application Layer)
- **Location**: `antigence-alpha/` (visible project)
- **Purpose**: MCP server, web UI, agent orchestration
- **Analogy**: Like user applications - what you interact with
- **Future**: Could be MCP server working with online/local models

**Relationship**: Antigence runs *on top of* IMMUNOS, using its context management

## 🚀 Agent Startup Scripts

### Quick Start (All Agents)

```bash
# Claude Sonnet 4.5 (via Claude Code)
./start-claude.sh

# Qwen Coder 7B (via Ollama)
./start-qwen-coder.sh

# DeepSeek R1 14B (via Ollama)
./start-deepseek.sh

# Qwen Quick 1.5B (via Ollama)
./start-qwen-quick.sh
```

Each script:
1. Runs context recovery from `.immunos/`
2. Shows agent-specific context
3. Shows latest work summary
4. Launches the model (if Ollama)

### Universal Startup (Python)

```bash
# Single command for any agent
python3 scripts/immunos_agent_startup.py --agent <agent-name>

# Options: claude-sonnet, qwen-coder, deepseek-r1, qwen-quick
```

## 🔄 Typical Workflow

### 1. Start Agent Session
```bash
./start-claude.sh
# or
./start-qwen-coder.sh
```

**What happens:**
- Reads latest snapshot from `.immunos/memory/snapshots/`
- Displays recovery context
- Shows agent role and capabilities
- Ready to work

### 2. Do Work
All agents share the same hidden context:
- Changes saved to `.immunos/memory/`
- Snapshots created in `.immunos/memory/snapshots/`
- Conversations logged in `.immunos/memory/conversations/`

### 3. End Session
```bash
# Create snapshot (preserves context for next agent)
python3 scripts/immunos_snapshot.py create \
  --trigger manual \
  --summary "What you accomplished this session"
```

### 4. Switch Agents
```bash
# Different agent picks up where you left off
./start-deepseek.sh
```

## 📋 What's in .immunos/ and Why?

### Directory Structure Explained

```
.immunos/
├── memory/                         ← Shared context database
│   ├── conversations/              ← All conversations (JSON)
│   ├── decisions/                  ← Key decisions
│   ├── snapshots/                  ← Context checkpoints
│   │   ├── latest.json             ← Symlink to latest
│   │   └── snap_YYYY-MM-DD_*.json  ← Timestamped snapshots
│   └── index.json                  ← Memory index
│
├── recovery/                       ← Fast context recovery
│   ├── CONTEXT_RECOVERY.md         ← Human-readable recovery file
│   └── quick_start.sh              ← Auto-generated startup
│
├── model-contexts/                 ← Agent-specific contexts
│   ├── claude-sonnet-context.md    ← Claude's role & capabilities
│   ├── qwen-coder-context.md       ← Qwen's role & capabilities
│   ├── deepseek-r1-context.md      ← DeepSeek's role & capabilities
│   └── qwen-quick-context.md       ← Quick agent's context
│
├── journal/                        ← Daily activity logs
│   ├── 2026-01-05.md               ← Today's journal
│   └── YYYY-MM-DD.md               ← Historical journals
│
├── agents/                         ← Agent session logs
│   ├── sessions/                   ← Session transcripts
│   └── hallucination_*.json        ← Hallucination detection logs
│
├── logs/                           ← System operation logs
│   └── changes/                    ← File change tracking
│
├── runs/                           ← Execution history
│
├── db/                             ← SQLite databases
│   └── dashboard.db                ← Web dashboard data
│
└── config/                         ← System configuration
    └── *.json                      ← Config files
```

### Why Hidden (.immunos)?

**Pros of hiding:**
1. Keeps `/Users/byron/projects/` clean for actual projects
2. System files don't clutter file browser
3. Conventional (like `.git/`, `.vscode/`)
4. Prevents accidental edits to critical files

**Cons of hiding:**
1. You can't see daily journals easily
2. Can't browse snapshots in Finder
3. Less transparent

**Alternative**: Move journals to `daily/`, keep system files hidden

## 🛠️ Key Commands (From Parent Directory)

```bash
# View latest context
cat .immunos/recovery/CONTEXT_RECOVERY.md

# View today's journal (hidden)
cat .immunos/journal/$(date +%Y-%m-%d).md

# Create snapshot
python3 scripts/immunos_snapshot.py create --trigger manual --summary "Work summary"

# Check memory status
python3 scripts/immunos_memory.py stats

# Check todos
python3 scripts/immunos_todo.py list --overdue
```

## 🎮 Shell Aliases (Optional)

Add to `~/.zshrc`:

```bash
source ~/projects/.immunos/immunos-aliases.sh
```

Then use:
```bash
immunos-claude      # Start Claude
immunos-coder       # Start Qwen Coder
immunos-context     # Show recovery context
immunos-journal     # Today's journal
immunos-save "msg"  # Quick snapshot
```

## 🔮 Future: Antigence as MCP Server

**Vision:**
- **IMMUNOS**: Hidden OS layer (context management, memory, persistence)
- **Antigence**: MCP server exposing IMMUNOS capabilities
- **Agents**: Work via MCP protocol, all share IMMUNOS context
- **Models**: Online (Claude, GPT) or airgapped local frontier models

**Benefits:**
- Any model can use IMMUNOS context
- Seamless handoffs between models
- Centralized memory and decision tracking
- Works with airgapped servers for sensitive work

## 📊 Current State

**Working Directory**: `/Users/byron/projects/`

**Active Projects**:
- `antigence-alpha/` - Web app (future MCP server?)
- `prion-clock/` - Research hypothesis
- `papers/` - Literature archive
- `bioviztech/` - Portfolio work
- `daily/` - Daily notes (visible)

**IMMUNOS Status**: Operational
- ✅ Memory system
- ✅ Snapshot/recovery
- ✅ Multi-agent contexts
- ✅ Todo tracking
- ⏳ NK Cell scanner
- ⏳ Citation verifier

## 🆘 Troubleshooting

**Can't find journals?**
```bash
ls -la .immunos/journal/
cat .immunos/journal/$(date +%Y-%m-%d).md
```

**Want journals visible?**
```bash
# Option: Symlink to daily/
ln -s /Users/byron/projects/.immunos/journal ~/projects/journals-immunos
```

**Recovery file outdated?**
```bash
python3 scripts/immunos_recover.py
cat .immunos/recovery/CONTEXT_RECOVERY.md
```

**Need to see snapshot contents?**
```bash
cat .immunos/memory/snapshots/latest.json | jq
```

---

**Antigence Multi-Agent System**
*Built on IMMUNOS Context Management*
*Working Directory: `/Users/byron/projects/`*
