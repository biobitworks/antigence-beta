# Antigence Web App

Flask-based ops dashboard for Antigence Platform with Sentinel monitoring, ticketing, and multi-domain AI analysis.

## Quickstart (macOS)

```bash
# 1. Install dependencies
cd web_app
pip install -r requirements.txt

# 2. Set up Sentinel directories (optional but recommended)
mkdir -p ~/.antigence/sentinel/logs
touch ~/.antigence/sentinel/events.jsonl

# 3. Start the web app
python3 app.py
# -> http://localhost:5001

# 4. (Optional) Set up launchd for scheduled Sentinel runs
# Copy the plist to ~/Library/LaunchAgents/
# Then load with: launchctl load ~/Library/LaunchAgents/com.antigence.sentinel.watch.plist

# 5. (Optional) Start Ollama for LLM orchestration
brew install ollama
ollama serve &
ollama pull llama3.2:3b
```

## Quickstart (Linux)

```bash
# 1. Install dependencies
cd web_app
pip install -r requirements.txt

# 2. Set up Sentinel directories
mkdir -p ~/.antigence/sentinel/logs
touch ~/.antigence/sentinel/events.jsonl

# 3. Start the web app
python3 app.py
# -> http://localhost:5001

# 4. (Optional) Schedule Sentinel with cron
# Add to crontab (crontab -e):
# */5 * * * * python3 ~/.antigence/sentinel/antigence_sentinel_watch.py

# 5. (Optional) Install Ollama for LLM orchestration
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2:3b
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTIGENCE_HOME` | Base directory for Antigence data | `~/.antigence` |
| `ANTIGENCE_SENTINEL_EVENTS_PATH` | Path to Sentinel JSONL events file | `~/.antigence/sentinel/events.jsonl` |
| `ANTIGENCE_SENTINEL_WATCHER` | Path to watcher script | `~/.antigence/sentinel/antigence_sentinel_watch.py` |
| `ANTIGENCE_SENTINEL_OUT_LOG` | Stdout log path | `~/.antigence/sentinel/logs/antigence_sentinel.out.log` |
| `ANTIGENCE_SENTINEL_ERR_LOG` | Stderr log path | `~/.antigence/sentinel/logs/antigence_sentinel.err.log` |
| `ANTIGENCE_OPS_TOKEN` | Token for ops endpoints (required for remote access) | (empty = localhost only) |
| `ANTIGENCE_SENSITIVE_REPO_PREFIXES` | Comma-separated repo prefixes for ticketing | (empty = no repo ticketing) |
| `ANTIGENCE_LLM_ROLE_MODELS` | JSON dict mapping LLM role → Ollama model | (empty = use `~/.antigence/llm/roles.json` or fallback to `OLLAMA_MODEL`) |
| `ANTIGENCE_ANTIGENT_EVENTS_PATH` | JSONL log for antigent LLM calls | `~/.antigence/antigent_events.jsonl` |
| `OLLAMA_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | LLM model for orchestration | `llama3.2:3b` |
| `FLASK_DEBUG` | Enable debug mode | `0` |
| `HOST` | Bind address | `127.0.0.1` |
| `PORT` | Bind port | `5001` |

## Multi-Model Antigents (Orchestrator + Specialists)

Antigence supports a “big orchestrator + smaller role models” pattern via **role → model mapping**.

### Roles

- `orchestrator` (Thymus): coordinates all immune signals
- `bcell`: evidence extraction / claim review
- `nk`: anomaly / risk summaries
- `tcell_security`: code security review
- `dendritic_summarizer`: summarization of long artifacts

### Configure role models (env)

```bash
export ANTIGENCE_LLM_ROLE_MODELS='{
  "orchestrator": "qwen2.5:14b",
  "bcell": "llama3.2:3b",
  "nk": "llama3.2:3b",
  "tcell_security": "qwen2.5-coder:7b",
  "dendritic_summarizer": "llama3.2:3b"
}'
```

### Configure role models (file)

Create `~/.antigence/llm/roles.json` with the same JSON mapping.

## Sentinel Ops Features

The `/sentinel` page provides:

- **Status Dashboard**: Sentinel running status, Ollama availability, platform info
- **Ticket Management**: View, filter, and close security tickets
- **Detection Events**: Track immunOS detection results with filtering by domain/result
- **Sentinel Events**: File system monitoring with unexpected write alerts
- **Controls**: Start/Stop (macOS), Run Once, Refresh

### Filters

- **Tickets**: Filter by HIGH or MED severity
- **Detections**: Filter by domain or result type (danger, non_self)
- **Events**: Show only events with unexpected writes

## Ops: Sentinel + Ticketing

- Sentinel events file (default): `~/.antigence/sentinel/events.jsonl`
  - Override with env var: `ANTIGENCE_SENTINEL_EVENTS_PATH`
- Watcher script path (for "Run Once"): `~/.antigence/sentinel/antigence_sentinel_watch.py`
  - Override with env var: `ANTIGENCE_SENTINEL_WATCHER`
- Start/stop controls:
  - macOS only (launchd): user LaunchAgent `com.antigence.sentinel.watch`
  - non-macOS: controls are disabled; use cron or systemd instead
- Log paths shown in UI are configurable:
  - `ANTIGENCE_SENTINEL_OUT_LOG`, `ANTIGENCE_SENTINEL_ERR_LOG`
- Optional repo-change ticketing:
  - By default, tickets are created for `fs.unexpected` only.
  - To enable "sensitive repo change" tickets, set `ANTIGENCE_SENSITIVE_REPO_PREFIXES`.

## Security Notes

- By default, ops endpoints only accept requests from localhost
- If binding beyond localhost, set `ANTIGENCE_OPS_TOKEN` and include it as `X-Antigence-Token` header
- Secret key is auto-generated in production mode (non-DEBUG)

## Verification Checklist

After setup, verify your installation:

- [ ] Web app starts without errors: `python3 app.py`
- [ ] `/sentinel` page loads at http://localhost:5001/sentinel
- [ ] Sentinel status shows platform correctly (darwin/linux)
- [ ] Events file status shows "found" or setup warning if missing
- [ ] Watcher script status shows "found" or setup warning if missing
- [ ] Ollama card shows installation and running status
- [ ] Refresh button works (reloads without full page navigation)
- [ ] Ticket filters work (HIGH/MED toggles)
- [ ] Detection filters work (domain dropdown, danger/non_self toggles)
- [ ] Event filter works (Unexpected Only toggle)
- [ ] (macOS) Start/Stop buttons are enabled if plist exists
- [ ] (non-macOS) Start/Stop shows "macOS only" hint with cron example

### API Verification

```bash
# Check sentinel status
curl http://localhost:5001/api/sentinel/status

# Check LLM role→model status (installed vs missing)
curl http://localhost:5001/api/models/status

# List tickets
curl http://localhost:5001/api/tickets

# List detection events
curl "http://localhost:5001/api/events?type=detection"

# Run sentinel once (localhost only)
curl -X POST http://localhost:5001/api/sentinel/run_once
```

## Directory Structure

```
web_app/
├── app.py              # Main Flask application
├── database.py         # SQLAlchemy models
├── requirements.txt    # Python dependencies
├── deploy_local.sh     # Setup script
├── data/
│   └── immunos.db      # SQLite database (auto-created)
├── static/
│   ├── css/style.css
│   └── img/
└── templates/
    ├── base.html
    ├── sentinel.html   # Ops dashboard
    ├── index.html
    └── ...
```

## Enabling OPS Token (Remote Access)

If you need to access the app from another machine:

```bash
# Set token in environment
export ANTIGENCE_OPS_TOKEN="your-secure-token-here"

# Bind to all interfaces
export HOST="0.0.0.0"

# Start app
python3 app.py

# From remote, include token in requests:
curl -H "X-Antigence-Token: your-secure-token-here" \
     http://your-server:5001/api/sentinel/start -X POST
```
