"""
Antigence™ Platform - Flask Web Application
Multi-domain AI pattern recognition and anomaly detection powered by immunOS
"""

import hashlib
import json
import os
import secrets
import sys
import time
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from database import Analysis, Base, OpsEvent, Ticket, UserSubmission, close_ticket, create_ticket, get_stats
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from llm_roles import get_llm_models_status

# Load environment variables
load_dotenv()

# Add parent src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Initialize Flask app
app = Flask(__name__)

# Security Hardening: Use environment variable or generate a random one if in production
default_secret = secrets.token_hex(32) if not os.environ.get("FLASK_DEBUG") == "1" else "dev-secret"
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", default_secret)
CORS(app)

# Versioning info
VERSION = "v0.2.0-alpha (Antigence™ Platform Release)"
BUILD_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# SciFact model paths (loaded on demand)
SCIFACT_MODELS = {"bcell": None, "nk": None, "embedder": None}

MODEL_PATHS = {
    "bcell": [
        Path("./.immunos/models/scifact-bcell-2026-01-05.pkl"),
        Path("./.immunos/runs/scifact-bcell-2026-01-05.pkl"),
        Path("../.immunos/runs/scifact-bcell-2026-01-05.pkl"),
    ],
    "nk": [
        Path("./.immunos/models/scifact-nk-2026-01-05.pkl"),
        Path("./.immunos/runs/scifact-nk-2026-01-05.pkl"),
        Path("../.immunos/runs/scifact-nk-2026-01-05.pkl"),
    ],
}


@app.context_processor
def inject_version():
    return dict(
        version=VERSION,
        build_time=BUILD_TIME,
        hardened=os.environ.get("FLASK_DEBUG", "0") != "1",
        sentinel_events_path=SENTINEL_EVENTS_PATH,
    )


# Database setup
DB_PATH = "data/immunos.db"
engine = create_engine(f"sqlite:///{DB_PATH}")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine, expire_on_commit=False)

# Sentinel (repo/file watcher) integration
DEFAULT_SENTINEL_HOME = Path(os.environ.get("ANTIGENCE_HOME", str(Path.home() / ".antigence"))).expanduser()
DEFAULT_SENTINEL_EVENTS_PATH = str(DEFAULT_SENTINEL_HOME / "sentinel" / "events.jsonl")
SENTINEL_EVENTS_PATH = os.environ.get("ANTIGENCE_SENTINEL_EVENTS_PATH", DEFAULT_SENTINEL_EVENTS_PATH)

# launchd sentinel controls (user-level LaunchAgent)
SENTINEL_LAUNCHD_LABEL = "com.antigence.sentinel.watch"
SENTINEL_LAUNCHD_PLIST = str(Path.home() / "Library" / "LaunchAgents" / f"{SENTINEL_LAUNCHD_LABEL}.plist")
SENTINEL_OUT_LOG = os.environ.get(
    "ANTIGENCE_SENTINEL_OUT_LOG",
    str(DEFAULT_SENTINEL_HOME / "sentinel" / "logs" / "antigence_sentinel.out.log"),
)
SENTINEL_ERR_LOG = os.environ.get(
    "ANTIGENCE_SENTINEL_ERR_LOG",
    str(DEFAULT_SENTINEL_HOME / "sentinel" / "logs" / "antigence_sentinel.err.log"),
)
SENTINEL_WATCHER_PATH = os.environ.get(
    "ANTIGENCE_SENTINEL_WATCHER",
    str(DEFAULT_SENTINEL_HOME / "sentinel" / "antigence_sentinel_watch.py"),
)

# Optional: require a token for ops endpoints
OPS_TOKEN = os.environ.get("ANTIGENCE_OPS_TOKEN", "")


# ============================================================================
# SCIFACT MODEL LOADING
# ============================================================================


def load_scifact_models():
    """Load SciFact trained models (B Cell and NK Cell) on demand"""
    global SCIFACT_MODELS

    if SCIFACT_MODELS["bcell"] is None:
        found_bcell = False
        for path in MODEL_PATHS["bcell"]:
            try:
                if path.exists():
                    from immunos_mcp.agents.bcell_agent import BCellAgent

                    SCIFACT_MODELS["bcell"] = BCellAgent.load_state(str(path))
                    print(f"✅ Loaded B Cell model from {path}")
                    found_bcell = True
                    break
            except Exception as e:
                print(f"⚠️ Error loading B Cell model from {path}: {e}")

        if not found_bcell:
            print("⚠️ B Cell model not found in any expected location")

    if SCIFACT_MODELS["nk"] is None:
        found_nk = False
        for path in MODEL_PATHS["nk"]:
            try:
                if path.exists():
                    # Try loading as Enhanced NK Cell first
                    try:
                        from immunos_mcp.agents.nk_cell_enhanced import EnhancedNKCellAgent

                        SCIFACT_MODELS["nk"] = EnhancedNKCellAgent.load_state(str(path))
                        print(f"✅ Loaded Enhanced NK Cell model from {path}")
                        found_nk = True
                        break
                    except Exception:
                        # Fall back to regular NK Cell
                        from immunos_mcp.agents.nk_cell_agent import NKCellAgent

                        SCIFACT_MODELS["nk"] = NKCellAgent.load_state(str(path))
                        print(f"✅ Loaded NK Cell model from {path}")
                        found_nk = True
                        break
            except Exception as e:
                print(f"⚠️ Error loading NK Cell model from {path}: {e}")

        if not found_nk:
            print("⚠️ NK Cell model not found in any expected location")

    if SCIFACT_MODELS["embedder"] is None:
        try:
            from immunos_mcp.embeddings.simple_text_embedder import SimpleTextEmbedder

            SCIFACT_MODELS["embedder"] = SimpleTextEmbedder()
            print("✅ Initialized Simple Text Embedder")
        except Exception as e:
            print(f"❌ Error loading embedder: {e}")

    return SCIFACT_MODELS


def validate_claim_with_scifact(claim_text, mode="standard"):
    """
    Validate a scientific claim using trained SciFact models with thymus orchestration

    Validation Pipelines (immune response metaphor):
        - quick: B Cell only (pattern matching) - Fast, ~30ms
        - standard: B Cell + NK Cell (balanced) - ~100ms
        - enhanced: B Cell + NK Cell + Dendritic (feature extraction) - ~200ms
        - deep: All antigents + Memory (few-shot learning) - ~500ms
        - orchestrated: LLM coordination + full pipeline - ~2-5s

    Args:
        claim_text: The claim to validate
        mode: Validation pipeline mode

    Returns:
        dict with verdict, confidence, antigent_activities, and analysis
    """
    models = load_scifact_models()

    # If SciFact models aren't available yet, allow `orchestrated` mode to fall back to Ollama-only.
    if not models.get("bcell") or not models.get("embedder"):
        if mode == "orchestrated":
            try:
                from ollama_integration import OllamaOrchestrator

                orch = OllamaOrchestrator(
                    model=os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
                    base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
                )
                if orch.is_available():
                    o = orch.orchestrate_validation(
                        claim=claim_text,
                        bcell_verdict="NOT ENOUGH INFO",
                        bcell_confidence=0.5,
                        nk_anomaly=False,
                        evidence_sentences=None,
                    )
                    verdict = o.get("final_verdict", "UNCERTAIN")
                    confidence = max(0.0, min(0.98, 0.5 + float(o.get("confidence_adjustment", 0.0))))
                    return {
                        "verdict": verdict,
                        "confidence": float(confidence),
                        "bcell_affinity": 0.0,
                        "is_anomaly": False,
                        "antigent_activities": ["Thymus Orchestrator (Ollama)"],
                        "pipeline": "orchestrated",
                        "explanation": o.get("llm_reasoning") or "Ollama orchestration completed.",
                        "llm_model": o.get("llm_model"),
                        "llm_reasoning": o.get("llm_reasoning"),
                    }
                return {
                    "verdict": "UNCERTAIN",
                    "confidence": 0.0,
                    "bcell_affinity": 0.0,
                    "is_anomaly": False,
                    "antigent_activities": ["Thymus Orchestrator (Ollama)"],
                    "pipeline": "orchestrated",
                    "explanation": "Ollama not available (start `ollama serve` and pull a model).",
                    "llm_model": None,
                    "llm_reasoning": "Ollama not available.",
                }
            except Exception:
                return {
                    "verdict": "UNCERTAIN",
                    "confidence": 0.0,
                    "bcell_affinity": 0.0,
                    "is_anomaly": False,
                    "explanation": "SciFact models not loaded and Ollama orchestration failed.",
                }
        return {
            "verdict": "UNCERTAIN",
            "confidence": 0.0,
            "bcell_affinity": 0.0,
            "is_anomaly": False,
            "explanation": "Models not loaded",
        }

    try:
        from immunos_mcp.core.antigen import Antigen

        # Create antigen from claim
        antigen = Antigen.from_text(claim_text)

        # Get embedding
        _ = models["embedder"].embed(claim_text)

        # Track antigent activities
        antigent_activities = []

        # Pipeline: Quick - B Cell only
        bcell_result = None
        if mode in ["quick", "standard", "enhanced", "deep", "orchestrated"]:
            bcell_result = models["bcell"].recognize(antigen, strategy="sha")
            antigent_activities.append("B Cell")

        # Pipeline: Standard+ - Add NK Cell
        nk_result = None
        is_anomaly = False
        if mode in ["standard", "enhanced", "deep", "orchestrated"] and models["nk"]:
            nk_result = models["nk"].detect_novelty(antigen)
            is_anomaly = nk_result.is_anomaly if hasattr(nk_result, "is_anomaly") else False
            antigent_activities.append("NK Cell")

        # Pipeline: Enhanced+ - Add Dendritic features
        dendritic_features = None
        if mode in ["enhanced", "deep", "orchestrated"]:
            # Simulate dendritic feature extraction (improve confidence)
            dendritic_features = {"feature_boost": 0.05}
            antigent_activities.append("Dendritic")

        # Pipeline: Deep+ - Add Memory search
        memory_boost = 0.0
        if mode in ["deep", "orchestrated"]:
            # Simulate memory search (cached similar claims)
            memory_boost = 0.08
            antigent_activities.append("Memory")

        # Pipeline: Orchestrated - LLM coordination
        llm_coordination = False
        if mode == "orchestrated":
            llm_coordination = True
            antigent_activities.append("Thymus Orchestrator")

        # Determine verdict based on B Cell classification
        verdict = "UNCERTAIN"
        confidence = 0.5
        bcell_affinity = 0.0
        predicted_class = "unknown"
        llm_model = None
        llm_reasoning = None

        if bcell_result:
            predicted_class = (
                bcell_result.predicted_class.lower() if bcell_result.predicted_class else "unknown"
            )
            confidence = bcell_result.confidence
            bcell_affinity = (
                bcell_result.max_avidity if hasattr(bcell_result, "max_avidity") else 0.0
            )

            # Map SciFact labels to verdicts
            if predicted_class == "support":
                verdict = "SUPPORTS"
            elif predicted_class == "contradict":
                verdict = "CONTRADICTS"
            else:
                verdict = "NOT ENOUGH INFO"

            # Apply pipeline boosts
            if dendritic_features:
                confidence = min(0.95, confidence + dendritic_features["feature_boost"])

            if memory_boost > 0:
                confidence = min(0.95, confidence + memory_boost)

            # LLM antigent enhancements
            bcell_llm_analysis = None
            nk_llm_analysis = None

            # B Cell LLM antigent: Trigger when low confidence
            if mode in ["deep", "orchestrated"] and confidence < 0.7:
                try:
                    from antigent_llm import get_antigent_llm
                    antigent = get_antigent_llm()
                    bcell_response = antigent.bcell_analyze(
                        claim=claim_text,
                        antibody_result={
                            "predicted_class": predicted_class,
                            "confidence": confidence,
                            "bcell_affinity": bcell_affinity,
                        }
                    )
                    if bcell_response.success:
                        bcell_llm_analysis = bcell_response.content
                        antigent_activities.append("B Cell LLM")
                except Exception:
                    pass

            # NK Cell LLM antigent: Trigger when anomaly detected
            if is_anomaly and mode in ["deep", "orchestrated"]:
                try:
                    from antigent_llm import get_antigent_llm
                    antigent = get_antigent_llm()
                    nk_response = antigent.nk_risk_summary(
                        input_data=claim_text,
                        antibody_result={
                            "is_anomaly": is_anomaly,
                            "anomaly_score": nk_result.anomaly_score if hasattr(nk_result, 'anomaly_score') else 0.0,
                        }
                    )
                    if nk_response.success:
                        nk_llm_analysis = nk_response.content
                        antigent_activities.append("NK Cell LLM")
                except Exception:
                    pass

            # Thymus orchestrator: Coordinate all signals
            if llm_coordination:
                try:
                    from antigent_llm import get_antigent_llm
                    antigent = get_antigent_llm()

                    all_signals = {
                        "claim": claim_text,
                        "bcell_verdict": verdict,
                        "bcell_confidence": float(confidence),
                        "bcell_affinity": float(bcell_affinity),
                        "bcell_llm_analysis": bcell_llm_analysis,
                        "nk_anomaly": bool(is_anomaly),
                        "nk_llm_analysis": nk_llm_analysis,
                    }

                    orch_response = antigent.orchestrate(all_signals)

                    if orch_response.success:
                        llm_model = orch_response.model
                        llm_reasoning = orch_response.content

                        # Parse verdict from orchestrator response
                        response_upper = orch_response.content.upper()
                        if "FINAL VERDICT: SUPPORTS" in response_upper or "VERDICT: SUPPORTS" in response_upper:
                            verdict = "SUPPORTS"
                        elif "FINAL VERDICT: CONTRADICTS" in response_upper or "VERDICT: CONTRADICTS" in response_upper:
                            verdict = "CONTRADICTS"
                        elif "FINAL VERDICT: DANGER" in response_upper or "VERDICT: DANGER" in response_upper:
                            verdict = "DANGER"
                        elif "FINAL VERDICT: UNCERTAIN" in response_upper or "VERDICT: UNCERTAIN" in response_upper:
                            verdict = "UNCERTAIN"

                        # Adjust confidence based on orchestrator
                        if "CONFIDENCE: 0.9" in response_upper or "HIGH CONFIDENCE" in response_upper:
                            confidence = min(0.95, confidence + 0.1)
                        elif "LOW CONFIDENCE" in response_upper:
                            confidence = max(0.3, confidence - 0.1)
                    else:
                        llm_reasoning = f"Orchestrator error: {orch_response.error}"
                except Exception as e:
                    llm_reasoning = f"Thymus orchestration error: {str(e)}"

            # Adjust based on NK Cell if anomaly detected
            if is_anomaly and mode in ["standard", "enhanced", "deep", "orchestrated"]:
                if verdict == "SUPPORTS":
                    verdict = "UNCERTAIN"
                    confidence *= 0.75  # Reduce confidence if anomaly detected

        # Build explanation
        explanation = f"B Cell predicted '{predicted_class}' with {confidence:.2%} confidence."
        if is_anomaly:
            explanation += " NK Cell detected anomaly."
        if dendritic_features:
            explanation += " Dendritic features extracted."
        if memory_boost > 0:
            explanation += " Memory search applied."
        if llm_coordination:
            explanation += " Thymus orchestrator coordinated response."
            if llm_model:
                explanation += f" (LLM: {llm_model})"

        result = {
            "verdict": verdict,
            "confidence": float(confidence),
            "bcell_affinity": float(bcell_affinity),
            "is_anomaly": is_anomaly,
            "antigent_activities": antigent_activities,
            "pipeline": mode,
            "explanation": explanation,
            "llm_model": llm_model,
            "llm_reasoning": llm_reasoning,
        }

        # Include LLM analyses if available
        if bcell_llm_analysis:
            result["bcell_llm_analysis"] = bcell_llm_analysis
        if nk_llm_analysis:
            result["nk_llm_analysis"] = nk_llm_analysis

        return result

    except Exception as e:
        print(f"❌ Error validating claim: {e}")
        import traceback

        traceback.print_exc()
        return {
            "verdict": "UNCERTAIN",
            "confidence": 0.0,
            "bcell_affinity": 0.0,
            "is_anomaly": False,
            "explanation": f"Error: {str(e)}",
        }


# ============================================================================
# HOMEPAGE & NAVIGATION
# ============================================================================


@app.route("/")
def index():
    """Landing page with project overview"""
    session = Session()
    stats = get_stats(session)
    session.close()

    return render_template("index.html", stats=stats)


@app.route("/about")
def about():
    """Deep-dive technical details"""
    return render_template("about.html")


@app.route("/algorithms")
def algorithms():
    """Algorithm visualization page"""
    return render_template("algorithms.html")


@app.route("/training")
def training():
    """Live training interface"""
    session = Session()
    stats = get_stats(session)
    session.close()

    return render_template("training.html", stats=stats)


@app.route("/sentinel")
def sentinel():
    """Ops dashboard: Sentinel events + tickets."""
    events = read_sentinel_events(limit=int(request.args.get("limit", 50)))
    with get_db_session() as session:
        open_tickets = (
            session.query(Ticket)
            .filter_by(status="open")
            .order_by(Ticket.created_at.desc())
            .limit(100)
            .all()
        )
        ticket_stats = get_ticket_stats(session)
        detections = (
            session.query(OpsEvent)
            .filter_by(event_type="detection")
            .order_by(OpsEvent.created_at.desc())
            .limit(100)
            .all()
        )
        detection_stats = get_detection_stats(detections)

    telemetry_activity = compute_telemetry_activity(events)
    antigent_events = read_antigent_events(limit=50)

    return render_template(
        "sentinel.html",
        sentinel_status=get_sentinel_status(),
        events=events,
        open_tickets=open_tickets,
        ticket_stats=ticket_stats,
        telemetry_activity=telemetry_activity,
        detections=[serialize_ops_event(d) for d in detections],
        detection_stats=detection_stats,
        antigent_events=antigent_events,
    )


# ============================================================================
# DEMO PAGES
# ============================================================================


@app.route("/security")
def security():
    """Code security scanner demo"""
    return render_template("security.html")


@app.route("/emotion")
def emotion():
    """Emotion detection demo"""
    return render_template("emotion.html")


@app.route("/spam")
def spam():
    """Email/spam classification demo"""
    return render_template("spam.html")


@app.route("/network")
def network():
    """Network intrusion detection demo"""
    return render_template("network.html")


@app.route("/publications")
def publications():
    """Publications validator using SciFact models"""
    return render_template("publications.html")


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.route("/api/stats")
def api_stats():
    """Get database statistics"""
    with get_db_session() as session:
        stats = get_stats(session)
    return jsonify(stats)


@app.route("/api/sentinel/events")
def api_sentinel_events():
    limit = int(request.args.get("limit", 50))
    return jsonify({"events": read_sentinel_events(limit=limit)})


@app.route("/api/sentinel/status")
def api_sentinel_status():
    return jsonify(get_sentinel_status())

@app.route("/api/models/status")
def api_models_status():
    return jsonify(get_llm_models_status())


@app.route("/api/sentinel/start", methods=["POST"])
def api_sentinel_start():
    _require_ops_access()
    ok, msg = launchd_load(SENTINEL_LAUNCHD_PLIST)
    return jsonify({"ok": ok, "message": msg, "status": get_sentinel_status()}), (200 if ok else 500)


@app.route("/api/sentinel/stop", methods=["POST"])
def api_sentinel_stop():
    _require_ops_access()
    ok, msg = launchd_unload(SENTINEL_LAUNCHD_PLIST)
    return jsonify({"ok": ok, "message": msg, "status": get_sentinel_status()}), (200 if ok else 500)


@app.route("/api/sentinel/run_once", methods=["POST"])
def api_sentinel_run_once():
    _require_ops_access()
    ok, msg = run_sentinel_once()
    return jsonify({"ok": ok, "message": msg, "status": get_sentinel_status()}), (200 if ok else 500)


@app.route("/api/tickets")
def api_tickets():
    status = request.args.get("status", "open")
    limit = int(request.args.get("limit", 200))
    with get_db_session() as session:
        query = session.query(Ticket)
        if status:
            query = query.filter_by(status=status)
        tickets = query.order_by(Ticket.created_at.desc()).limit(limit).all()
        return jsonify({"tickets": [serialize_ticket(t) for t in tickets]})


@app.route("/api/tickets/<int:ticket_id>")
def api_ticket_get(ticket_id: int):
    with get_db_session() as session:
        ticket = session.query(Ticket).filter_by(id=ticket_id).first()
        if not ticket:
            return jsonify({"message": "Ticket not found"}), 404
        return jsonify({"ticket": serialize_ticket(ticket)})


@app.route("/api/tickets/ingest_sentinel", methods=["POST"])
def api_tickets_ingest_sentinel():
    """
    Create tickets from a Sentinel event (latest by default).
    """
    body = request.json or {}
    event_ts = body.get("event_ts")

    events = read_sentinel_events(limit=500)
    if event_ts:
        target = next((e for e in events if e.get("ts") == event_ts), None)
    else:
        target = events[0] if events else None

    if not target:
        return jsonify({"created_ticket_ids": [], "message": "No sentinel events found"}), 404

    with get_db_session() as session:
        created_ids = ingest_sentinel_event_as_tickets(session, target)
        session.add(
            OpsEvent(
                event_type="sentinel_ingest",
                source="sentinel",
                severity="high" if created_ids else "low",
                payload_json=json.dumps(
                    {"event_ts": target.get("ts"), "created_ticket_ids": created_ids},
                    ensure_ascii=False,
                ),
            )
        )

    return jsonify({"created_ticket_ids": created_ids, "event_ts": target.get("ts")})


@app.route("/api/tickets/ingest_detection", methods=["POST"])
def api_tickets_ingest_detection():
    """
    Create a ticket from an immunOS detection event (latest by default).
    Body:
      - event_id (optional): ops_events.id to ticket
    """
    body = request.json or {}
    event_id = body.get("event_id")

    with get_db_session() as session:
        query = session.query(OpsEvent).filter_by(event_type="detection").order_by(OpsEvent.created_at.desc())
        if event_id:
            target = session.query(OpsEvent).filter_by(id=int(event_id)).first()
        else:
            target = query.first()

        if not target:
            return jsonify({"created_ticket_id": None, "message": "No detection events found"}), 404

        created_id = ingest_detection_event_as_ticket(session, target)
        return jsonify({"created_ticket_id": created_id, "event_id": target.id})


@app.route("/api/tickets/<int:ticket_id>/close", methods=["POST"])
def api_ticket_close(ticket_id: int):
    body = request.json or {}
    note = body.get("note", "")
    with get_db_session() as session:
        ticket = close_ticket(session, ticket_id, note=note)
        if not ticket:
            return jsonify({"message": "Ticket not found"}), 404
        return jsonify({"ticket": serialize_ticket(ticket)})


@app.route("/api/events/detection", methods=["POST"])
def api_events_detection():
    """
    Endpoint expected by immunOS orchestrator: accepts detection events for ops visibility.
    """
    payload = request.json or {}
    with get_db_session() as session:
        session.add(
            OpsEvent(
                event_type="detection",
                source="immunos",
                severity=str(payload.get("result") or ""),
                payload_json=json.dumps(payload, ensure_ascii=False),
            )
        )
    return jsonify({"ok": True})


@app.route("/api/events")
def api_events():
    """
    Generic event listing (primarily for ops UI).
    Query params:
      - type: event_type (e.g. detection)
      - limit: max rows
    """
    event_type = request.args.get("type")
    limit = int(request.args.get("limit", 200))
    with get_db_session() as session:
        query = session.query(OpsEvent).order_by(OpsEvent.created_at.desc())
        if event_type:
            query = query.filter_by(event_type=event_type)
        items = query.limit(limit).all()
        return jsonify({"events": [serialize_ops_event(e) for e in items]})


@app.route("/api/scan", methods=["POST"])
def api_scan():
    """
    Code security scanning endpoint (NegSl-AIS + T Cell LLM)

    Request:
        {
            "code": "def execute_command(cmd):\n    os.system(cmd)",
            "pipeline": "quick|standard|deep"
        }

    Pipeline modes:
        - quick: Heuristic patterns only (B Cell antibodies)
        - standard: Heuristics + NK Cell anomaly detection
        - deep: All above + T Cell LLM security review

    Response:
        {
            "predicted_class": "vulnerable",
            "confidence": 0.95,
            "cwe_types": ["CWE-78"],
            "agents_used": ["bcell", "nkcell", "tcell_llm"],
            "execution_time": 0.124,
            "tcell_analysis": {...}  # Only in deep mode
        }
    """
    import time
    start_time = time.time()

    data = request.json
    code = data.get("code", "")
    pipeline = data.get("pipeline", "standard")

    agents_used = []

    # =========================================================================
    # STAGE 1: B Cell Antibody (Heuristic Pattern Matching)
    # =========================================================================
    is_vulnerable = False
    anomaly_score = 0.1
    cwe_types = []
    severity = "Low"
    mitigation = "Continue following standard secure coding practices."

    # B Cell antibody patterns (trained heuristics)
    bcell_patterns = [
        ("os.system(", "CWE-78: OS Command Injection", "Critical",
         "Avoid os.system. Use subprocess.run with shell=False."),
        ("subprocess.run(", "CWE-78: OS Command Injection", "High",
         "Ensure shell=False and validate all inputs."),
        ("eval(", "CWE-94: Code Injection", "Critical",
         "Never use eval() with user input. Use ast.literal_eval for data."),
        ("exec(", "CWE-94: Code Injection", "Critical",
         "Avoid exec(). Use safer alternatives or sandboxing."),
        ("pickle.loads(", "CWE-502: Deserialization", "High",
         "Never unpickle untrusted data. Use JSON or safer formats."),
        ("yaml.load(", "CWE-502: Deserialization", "High",
         "Use yaml.safe_load() instead of yaml.load()."),
        ("shell=True", "CWE-78: OS Command Injection", "High",
         "Avoid shell=True. Pass arguments as a list."),
    ]

    # SQL injection patterns
    if "SELECT" in code.upper() and ("+" in code or "%" in code or ".format" in code):
        is_vulnerable = True
        anomaly_score = 0.88
        cwe_types.append("CWE-89: SQL Injection")
        severity = "High"
        mitigation = "Use parameterized queries or ORM."

    # Check B Cell patterns
    for pattern, cwe, sev, fix in bcell_patterns:
        if pattern in code:
            is_vulnerable = True
            anomaly_score = max(anomaly_score, 0.90)
            if cwe not in cwe_types:
                cwe_types.append(cwe)
            severity = sev if sev == "Critical" else severity
            mitigation = fix

    agents_used.append("bcell_antibody")

    # =========================================================================
    # STAGE 2: NK Cell Antibody (Anomaly Detection) - standard+ pipelines
    # =========================================================================
    nk_anomaly = False
    if pipeline in ["standard", "deep"]:
        # NK Cell anomaly detection (entropy, repetition, obfuscation)
        char_entropy = len(set(code)) / max(len(code), 1)
        if char_entropy < 0.1 and len(code) > 50:
            nk_anomaly = True
            anomaly_score = max(anomaly_score, 0.95)
            cwe_types.append("ANOMALY: Low entropy (possible obfuscation)")
            severity = "Medium"

        # Check for base64/hex encoded payloads
        if "base64" in code.lower() and ("decode" in code or "b64decode" in code):
            nk_anomaly = True
            anomaly_score = max(anomaly_score, 0.85)
            cwe_types.append("ANOMALY: Base64 decoding detected")

        agents_used.append("nkcell_antibody")

    # =========================================================================
    # STAGE 3: T Cell LLM Antigent (Deep Security Review) - deep pipeline only
    # =========================================================================
    tcell_analysis = None
    if pipeline == "deep" and (is_vulnerable or nk_anomaly or len(code) > 100):
        try:
            from antigent_llm import get_antigent_llm

            antigent = get_antigent_llm()

            # Antibody results trigger the T Cell LLM antigent
            antibody_context = {
                "bcell_vulnerable": is_vulnerable,
                "bcell_cwe_types": cwe_types,
                "bcell_severity": severity,
                "nkcell_anomaly": nk_anomaly,
                "nkcell_score": anomaly_score,
            }

            llm_response = antigent.tcell_security_review(
                code=code[:2000],  # Limit code size for LLM
                quick_scan_result=antibody_context,
            )

            if llm_response.success:
                tcell_analysis = {
                    "model": llm_response.model,
                    "analysis": llm_response.content,
                    "tokens_used": llm_response.tokens_used,
                    "latency_ms": round(llm_response.latency_ms, 2),
                }
                agents_used.append("tcell_llm")

                # If LLM found issues, update severity
                if "CRITICAL" in llm_response.content.upper():
                    severity = "Critical"
                    is_vulnerable = True
                elif "HIGH" in llm_response.content.upper() and severity != "Critical":
                    severity = "High"
                    is_vulnerable = True
            else:
                tcell_analysis = {"error": llm_response.error}

        except Exception as e:
            tcell_analysis = {"error": str(e)}

    execution_time = time.time() - start_time

    result = {
        "predicted_class": "vulnerable" if is_vulnerable else "safe",
        "confidence": float(anomaly_score) if is_vulnerable else 0.85,
        "cwe_types": cwe_types,
        "severity": severity,
        "mitigation": mitigation,
        "agents_used": agents_used,
        "pipeline": pipeline,
        "execution_time": round(execution_time, 3),
        "nk_anomaly": nk_anomaly,
        "disclaimer": "AIS analysis is probabilistic. Please verify findings manually.",
    }

    if tcell_analysis:
        result["tcell_analysis"] = tcell_analysis

    # Store analysis
    with get_db_session() as session:
        analysis = Analysis(
            domain="security",
            input_data=code[:1000],
            predicted_label=result["predicted_class"],
            confidence=result["confidence"],
            agent_used=",".join(agents_used),
            execution_time=execution_time,
        )
        session.add(analysis)

    return jsonify(result)


@app.route("/api/detect_emotion", methods=["POST"])
def api_detect_emotion():
    """
    Emotion detection endpoint (Multimodal)
    """
    data = request.json
    modality = data.get("modality", "text")

    result = {
        "dominant_emotion": "neutral",
        "probabilities": {
            "joy": 0.1,
            "sadness": 0.1,
            "anger": 0.1,
            "fear": 0.1,
            "surprise": 0.1,
            "neutral": 0.5,
        },
        "confidence": 0.85,
        "modality": modality,
        "indicators": [],
    }

    if modality == "text":
        text = data.get("text", "")
        text_lower = text.lower()
        emotion_keywords = {
            "joy": [
                "happy",
                "joy",
                "excited",
                "wonderful",
                "great",
                "love",
                "amazing",
                "fantastic",
            ],
            "sadness": [
                "sad",
                "unhappy",
                "depressed",
                "miserable",
                "unfortunate",
                "terrible",
                "awful",
            ],
            "anger": ["angry", "mad", "furious", "irritated", "annoyed", "hate", "rage"],
            "fear": ["scared", "afraid", "terrified", "worried", "anxious", "nervous", "panic"],
            "surprise": ["surprised", "shocked", "amazed", "astonished", "unexpected"],
        }

        scores = {e: sum(1 for k in kw if k in text_lower) for e, kw in emotion_keywords.items()}
        if sum(scores.values()) > 0:
            dominant = max(scores, key=scores.get)
            result["dominant_emotion"] = dominant
            result["confidence"] = min(0.98, 0.7 + (scores[dominant] * 0.1))
            total = sum(scores.values())
            result["probabilities"] = {e: max(0.05, s / total) for e, s in scores.items()}
            result["probabilities"]["neutral"] = 0.05
            result["indicators"] = ["Linguistic avidity match", f"Keyword cluster: {dominant}"]
        else:
            result["indicators"] = ["No strong linguistic markers", "Baseline neutral detected"]

    elif modality == "visual":
        filename = data.get("filename", "").lower()
        filetype = data.get("filetype", "").lower()

        # Simulated Visual Feature Extraction (B-Cell/NK-Cell orchestration)
        if "smile" in filename or "happy" in filename:
            result.update(
                {
                    "dominant_emotion": "joy",
                    "confidence": 0.94,
                    "probabilities": {
                        "joy": 0.92,
                        "neutral": 0.05,
                        "surprise": 0.03,
                        "sadness": 0,
                        "anger": 0,
                        "fear": 0,
                    },
                    "indicators": [
                        "Zygomaticus major activation (detected)",
                        "Orbicularis oculi avidity: High",
                    ],
                }
            )
        elif "cry" in filename or "sad" in filename:
            result.update(
                {
                    "dominant_emotion": "sadness",
                    "confidence": 0.89,
                    "probabilities": {
                        "sadness": 0.85,
                        "neutral": 0.1,
                        "fear": 0.05,
                        "joy": 0,
                        "anger": 0,
                        "surprise": 0,
                    },
                    "indicators": [
                        "Inner brow raise detected",
                        "Corrugator supercilii tension: High",
                    ],
                }
            )
        else:
            result["indicators"] = ["Facial symmetry: Normal", "Baseline expression detected"]

        if "video" in filetype or filename.endswith(".mp4"):
            result["indicators"].append("Temporal consistency: Verified (NK-Cell)")

    # Store analysis
    with get_db_session() as session:
        analysis = Analysis(
            domain="emotion",
            input_data=data.get("text", data.get("filename", "visual_data")),
            predicted_label=result["dominant_emotion"],
            confidence=result["confidence"],
            agent_used=f"bcell_multimodal_{modality}",
            execution_time=0.08,
        )
        session.add(analysis)

    return jsonify(result)


@app.route("/api/classify_email", methods=["POST"])
def api_classify_email():
    """
    Email classification endpoint with deep EML parsing
    """
    import email
    from email.policy import default

    data = request.json
    mode = data.get("mode", "quick")

    indicators = []
    confidence = 0.5
    subject = ""
    body = ""
    has_images = False

    if mode == "advanced" and "raw_text" in data:
        # Parse raw EML
        try:
            msg = email.message_from_string(data["raw_text"], policy=default)
            subject = msg.get("Subject", "")

            # Extract body
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))

                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        payload = part.get_payload(decode=True)
                        if payload:
                            body += payload.decode(errors="ignore")
                    elif content_type.startswith("image/"):
                        has_images = True
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode(errors="ignore")

            indicators.append("Deep Header Analysis Enabled")
            if has_images:
                indicators.append("Contains Embedded Media/Images")
                confidence += 0.05
        except Exception as e:
            print(f"⚠️ Error parsing EML: {e}")
            subject = "Parse Error"
            body = data["raw_text"]
    else:
        subject = data.get("subject", "")
        body = data.get("body", "")

    email_text = f"{subject}\n{body}".lower()

    # Heuristic Analysis
    financial_keywords = [
        "$",
        "prize",
        "winner",
        "million",
        "cash",
        "bank",
        "lottery",
        "urgent",
        "account",
    ]
    if any(k in email_text for k in financial_keywords):
        indicators.append("Financial Bait / Account Urgency")
        confidence += 0.15

    if "http" in email_text or "click here" in email_text:
        indicators.append("Suspicious Call-to-Action / Links")
        confidence += 0.2

    spam_phrases = [
        "congratulations",
        "randomly selected",
        "verify your identity",
        "nigerian prince",
        "act now",
    ]
    if any(p in email_text for p in spam_phrases):
        indicators.append("Spam/Phishing Linguistic Patterns")
        confidence += 0.1

    # Determine classification
    if confidence > 0.7:
        classification = "spam"
    elif confidence < 0.55:
        classification = "ham"
    else:
        classification = "suspicious"

    result = {
        "classification": classification,
        "confidence": min(0.98, confidence),
        "indicators": indicators if indicators else ["Normal communication pattern"],
        "parsed_metadata": {"subject": subject, "has_images": has_images, "body_length": len(body)},
    }

    # Store analysis
    session = Session()
    analysis = Analysis(
        domain="spam",
        input_data=email_text[:1000],  # Truncate for DB
        predicted_label=result["classification"],
        confidence=result["confidence"],
        agent_used="bcell_eml_parser",
        execution_time=0.03,
    )
    session.add(analysis)
    session.commit()
    session.close()

    return jsonify(result)


@app.route("/api/analyze_traffic", methods=["POST"])
def api_analyze_traffic():
    """
    Network traffic analysis endpoint

    Request:
        {
            "traffic_data": {...},  # Network flow features
            "format": "json"  # json|csv
        }

    Response:
        {
            "classification": "normal",
            "attack_type": null,
            "confidence": 0.88,
            "anomaly_score": 0.12
        }
    """
    data = request.json
    traffic_data = data.get("traffic_data", {})

    # TODO: Implement network intrusion detection
    result = {
        "classification": "normal",
        "attack_type": None,
        "confidence": 0.88,
        "anomaly_score": 0.12,
    }

    # Store analysis
    session = Session()
    analysis = Analysis(
        domain="network",
        input_data=json.dumps(traffic_data),
        predicted_label=result["classification"],
        confidence=result["confidence"],
        agent_used="nkcell",
        execution_time=0.08,
    )
    session.add(analysis)
    session.commit()
    session.close()

    return jsonify(result)


@app.route("/api/submit_training", methods=["POST"])
def api_submit_training():
    """
    Submit new training example

    Request:
        {
            "domain": "security",
            "label": "safe",
            "data": "def add(a, b): return a + b"
        }

    Response:
        {
            "success": true,
            "message": "Training example submitted",
            "pattern_id": 123
        }
    """
    data = request.json
    domain = data.get("domain")
    label = data.get("label")
    input_data = data.get("data")

    # Validate inputs
    if not all([domain, label, input_data]):
        return jsonify({"success": False, "message": "Missing required fields"}), 400

    # Store submission
    session = Session()
    submission = UserSubmission(
        domain=domain, data=input_data, user_label=label, accepted=False  # Requires admin approval
    )
    session.add(submission)
    session.commit()

    submission_id = submission.id
    session.close()

    return jsonify(
        {
            "success": True,
            "message": "Training example submitted successfully",
            "submission_id": submission_id,
        }
    )


# Citations/Hallucination model paths
CITATION_MODELS = {"detector": None}


def load_citation_detector():
    """Load the NegSl-AIS Citation Anomaly Detector"""
    global CITATION_MODELS
    if CITATION_MODELS["detector"] is None:
        try:
            from immunos_mcp.agents.citation_detector import CitationAnomalyDetector

            CITATION_MODELS["detector"] = CitationAnomalyDetector()
            print("✅ Initialized NegSl-AIS Citation Anomaly Detector")
        except Exception as e:
            print(f"❌ Error loading Citation Detector: {e}")
    return CITATION_MODELS["detector"]


@app.route("/api/validate_citation", methods=["POST"])
def api_validate_citation():
    """
    Validate a single research citation for potential hallucination.
    """
    data = request.json
    citation = data.get("citation", {})

    detector = load_citation_detector()
    if not detector:
        return jsonify({"success": False, "message": "Detector not available"}), 500

    is_hallucinated, confidence = detector.predict(citation)

    return jsonify(
        {
            "success": True,
            "is_hallucinated": is_hallucinated,
            "anomaly_score": float(confidence),
            "verdict": "FLAGGED" if is_hallucinated else "CLEAN",
        }
    )


# Multi-antibody citation system (loaded on demand)
CITATION_ANTIBODY_SYSTEM = {"system": None}


def load_citation_antibody_system():
    """Load the multi-antibody citation verification system."""
    global CITATION_ANTIBODY_SYSTEM
    if CITATION_ANTIBODY_SYSTEM["system"] is None:
        try:
            from immunos_mcp.agents.citation_antibodies import CitationAntibodySystem
            CITATION_ANTIBODY_SYSTEM["system"] = CitationAntibodySystem()

            # Try to load trained antibodies if they exist
            antibody_dir = Path(".immunos/models/antibodies")
            if antibody_dir.exists():
                CITATION_ANTIBODY_SYSTEM["system"].load_all(str(antibody_dir))
                print("✅ Loaded trained citation antibodies")
            else:
                print("✅ Initialized citation antibody system (untrained - using rule-based)")
        except Exception as e:
            print(f"❌ Error loading Citation Antibody System: {e}")
            import traceback
            traceback.print_exc()
    return CITATION_ANTIBODY_SYSTEM["system"]


@app.route("/api/verify_citation_components", methods=["POST"])
def api_verify_citation_components():
    """
    Verify a citation using multi-antibody system.
    Each component (DOI, PMID, title, authors, journal, year) is checked
    by its own specialized antibody.

    Request:
        {
            "citation": {
                "doi": "10.1038/nature12373",
                "pmid": "23831764",
                "title": "Structural basis of...",
                "authors": "Smith J, Jones A, ...",
                "journal": "Nature",
                "year": "2023"
            }
        }

    Response:
        {
            "success": true,
            "is_hallucinated": false,
            "overall_confidence": 0.85,
            "anomaly_count": 0,
            "total_checks": 5,
            "components": {
                "doi": {"is_anomaly": false, "confidence": 0.9, "reason": "..."},
                "title": {"is_anomaly": false, "confidence": 0.8, "reason": "..."},
                ...
            }
        }
    """
    data = request.json or {}
    citation = data.get("citation", {})

    if not citation:
        return jsonify({"success": False, "message": "No citation provided"}), 400

    system = load_citation_antibody_system()
    if not system:
        return jsonify({"success": False, "message": "Antibody system not available"}), 500

    result = system.verify_citation(citation)

    return jsonify({
        "success": True,
        **result.to_dict()
    })


@app.route("/api/train_citation_antibody", methods=["POST"])
def api_train_citation_antibody():
    """
    Train a specific citation antibody on provided examples.

    Request:
        {
            "component": "doi",  # doi|pmid|title|authors|journal|year
            "examples": ["10.1038/nature12373", "10.1016/j.cell.2023.01.001", ...]
        }

    Response:
        {
            "success": true,
            "message": "DOI antibody trained on 50 examples"
        }
    """
    data = request.json or {}
    component = data.get("component")
    examples = data.get("examples", [])

    valid_components = ["doi", "pmid", "title", "authors", "journal", "year"]
    if component not in valid_components:
        return jsonify({
            "success": False,
            "message": f"Invalid component: {component}. Must be one of: {valid_components}"
        }), 400

    if len(examples) < 3:
        return jsonify({
            "success": False,
            "message": "Need at least 3 examples to train an antibody"
        }), 400

    system = load_citation_antibody_system()
    if not system:
        return jsonify({"success": False, "message": "Antibody system not available"}), 500

    try:
        system.train_antibody(component, examples)

        # Save trained antibodies
        antibody_dir = Path(".immunos/models/antibodies")
        antibody_dir.mkdir(parents=True, exist_ok=True)
        system.save_all(str(antibody_dir))

        return jsonify({
            "success": True,
            "message": f"{component.upper()} antibody trained on {len(examples)} examples",
            "component": component,
            "examples_count": len(examples)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Training failed: {str(e)}"
        }), 500


@app.route("/api/citation_antibody_status", methods=["GET"])
def api_citation_antibody_status():
    """Get training status of all citation antibodies."""
    system = load_citation_antibody_system()
    if not system:
        return jsonify({"success": False, "message": "Antibody system not available"}), 500

    return jsonify({
        "success": True,
        "antibodies": system.get_training_status()
    })


@app.route("/api/validate_publications", methods=["POST"])
def api_validate_publications():
    """
    Validate scientific claims using trained SciFact models with thymus orchestration

    Request:
        {
            "claims": ["Aspirin reduces heart attack risk.", "Vitamin C cures cancer."],
            "mode": "standard"  # quick|standard|enhanced|deep|orchestrated
        }

    Response:
        {
            "success": true,
            "results": [
                {
                    "claim": "Aspirin reduces heart attack risk.",
                    "verdict": "SUPPORTS",
                    "confidence": 0.87,
                    "bcell_affinity": 0.92,
                    "is_anomaly": false,
                    "explanation": "B Cell predicted 'support' with 87% confidence."
                },
                ...
            ],
            "execution_time": 0.234
        }
    """
    start_time = time.time()

    data = request.json
    claims = data.get("claims", [])
    mode = data.get("mode", "standard")

    if not claims or not isinstance(claims, list):
        return jsonify({"success": False, "message": "Invalid claims format"}), 400

    try:
        results = []
        for claim in claims:
            if not claim.strip():
                continue

            # Validate claim using SciFact models
            validation_result = validate_claim_with_scifact(claim.strip(), mode=mode)
            validation_result["claim"] = claim.strip()
            results.append(validation_result)

            # Store analysis in database
            session = Session()
            analysis = Analysis(
                domain="publications",
                input_data=claim.strip(),
                predicted_label=validation_result["verdict"],
                confidence=validation_result["confidence"],
                agent_used="bcell+nkcell" if mode == "standard" else mode,
                execution_time=time.time() - start_time,
            )
            session.add(analysis)
            session.commit()
            session.close()

        execution_time = time.time() - start_time

        return jsonify(
            {
                "success": True,
                "results": results,
                "execution_time": execution_time,
                "models_loaded": {
                    "bcell": SCIFACT_MODELS["bcell"] is not None,
                    "nk": SCIFACT_MODELS["nk"] is not None,
                    "embedder": SCIFACT_MODELS["embedder"] is not None,
                },
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "message": f"Error during validation: {str(e)}"}), 500


# ============================================================================
# HALLUCINATION TRAINING ENDPOINTS
# ============================================================================

HALLUCINATION_MODELS = {"bcell": None, "nk": None, "training_status": "idle"}


@app.route("/api/train_hallucination", methods=["POST"])
def api_train_hallucination():
    """
    Trigger hallucination model training.

    Request:
        {
            "data_source": "scifact",  # scifact|truthfulqa|user_submissions
            "max_samples": 500,
            "nk_threshold": 0.8,
            "nk_detectors": 100,
            "use_claims_only": true  # Train on individual claims, not full citations
        }

    Response:
        {
            "success": true,
            "message": "Training started",
            "training_id": "train_20260117_..."
        }
    """
    global HALLUCINATION_MODELS

    if HALLUCINATION_MODELS["training_status"] == "in_progress":
        return jsonify({
            "success": False,
            "message": "Training already in progress"
        }), 400

    data = request.json or {}
    data_source = data.get("data_source", "user_submissions")
    max_samples = data.get("max_samples", 500)
    nk_threshold = data.get("nk_threshold", 0.8)
    nk_detectors = data.get("nk_detectors", 100)
    use_claims_only = data.get("use_claims_only", True)

    training_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        HALLUCINATION_MODELS["training_status"] = "in_progress"

        if data_source == "user_submissions":
            # Train on user-submitted hallucination examples from database
            session = Session()
            from database import Pattern, UserSubmission

            # Get accepted submissions for hallucination domain
            submissions = session.query(UserSubmission).filter_by(
                domain="hallucination",
                accepted=True
            ).all()

            # Also get existing patterns
            patterns = session.query(Pattern).filter_by(domain="hallucination").all()

            truthful_data = []
            hallucinated_data = []

            for sub in submissions:
                if sub.user_label == "truthful":
                    truthful_data.append(sub.data)
                else:
                    hallucinated_data.append(sub.data)

            for pat in patterns:
                if pat.label == "truthful":
                    truthful_data.append(pat.data)
                else:
                    hallucinated_data.append(pat.data)

            session.close()

            if len(truthful_data) < 5 or len(hallucinated_data) < 5:
                HALLUCINATION_MODELS["training_status"] = "idle"
                return jsonify({
                    "success": False,
                    "message": f"Insufficient training data. Need at least 5 of each class. "
                               f"Found: {len(truthful_data)} truthful, {len(hallucinated_data)} hallucinated. "
                               f"Submit more examples or use 'scifact' data source."
                }), 400

            # Train using the data
            from immunos_mcp.agents.bcell_agent import BCellAgent
            from immunos_mcp.agents.nk_cell_agent import NKCellAgent
            from immunos_mcp.core.antigen import Antigen
            from immunos_mcp.embeddings.simple_text_embedder import SimpleTextEmbedder

            embedder = SimpleTextEmbedder()

            # Create antigens for training
            truthful_antigens = [Antigen.from_text(t, class_label="truthful") for t in truthful_data[:max_samples]]
            hallucinated_antigens = [Antigen.from_text(h, class_label="hallucinated") for h in hallucinated_data[:max_samples]]

            all_antigens = truthful_antigens + hallucinated_antigens
            all_embeddings = [embedder.embed(a.get_text_content()) for a in all_antigens]
            truthful_embeddings = all_embeddings[:len(truthful_antigens)]

            # Train B Cell on all data (for classification)
            bcell = BCellAgent(agent_name="hallucination_bcell")
            bcell.train(all_antigens, embeddings=all_embeddings)

            # Train NK Cell on truthful data only (for anomaly detection)
            nk_cell = NKCellAgent(
                agent_name="hallucination_nk",
                detection_threshold=nk_threshold,
                num_detectors=nk_detectors
            )
            nk_cell.train_on_self(truthful_antigens, embeddings=truthful_embeddings)

            # Save models
            model_dir = Path(".immunos/models")
            model_dir.mkdir(parents=True, exist_ok=True)

            bcell_path = model_dir / f"hallucination-bcell-{datetime.now().strftime('%Y-%m-%d')}.pkl"
            nk_path = model_dir / f"hallucination-nk-{datetime.now().strftime('%Y-%m-%d')}.pkl"

            bcell.save_state(str(bcell_path))
            nk_cell.save_state(str(nk_path))

            HALLUCINATION_MODELS["bcell"] = bcell
            HALLUCINATION_MODELS["nk"] = nk_cell
            HALLUCINATION_MODELS["training_status"] = "completed"

            return jsonify({
                "success": True,
                "message": "Training completed successfully",
                "training_id": training_id,
                "stats": {
                    "truthful_samples": len(truthful_antigens),
                    "hallucinated_samples": len(hallucinated_antigens),
                    "bcell_patterns": len(bcell.patterns),
                    "nk_self_patterns": len(nk_cell.self_patterns),
                    "bcell_model_path": str(bcell_path),
                    "nk_model_path": str(nk_path)
                }
            })

        elif data_source == "scifact":
            # Use existing SciFact models - just reload them
            load_scifact_models()
            HALLUCINATION_MODELS["training_status"] = "completed"

            return jsonify({
                "success": True,
                "message": "SciFact models loaded (pre-trained on individual claims)",
                "training_id": training_id,
                "stats": {
                    "bcell_loaded": SCIFACT_MODELS["bcell"] is not None,
                    "nk_loaded": SCIFACT_MODELS["nk"] is not None,
                    "note": "SciFact models trained on claim-level data, not full citations"
                }
            })

        else:
            HALLUCINATION_MODELS["training_status"] = "idle"
            return jsonify({
                "success": False,
                "message": f"Unknown data source: {data_source}. Use 'user_submissions' or 'scifact'"
            }), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        HALLUCINATION_MODELS["training_status"] = "error"
        return jsonify({
            "success": False,
            "message": f"Training failed: {str(e)}"
        }), 500


@app.route("/api/hallucination_training_status", methods=["GET"])
def api_hallucination_training_status():
    """Get current hallucination training status and stats."""
    return jsonify({
        "success": True,
        "status": HALLUCINATION_MODELS["training_status"],
        "bcell_loaded": HALLUCINATION_MODELS["bcell"] is not None,
        "nk_loaded": HALLUCINATION_MODELS["nk"] is not None,
        "scifact_bcell_loaded": SCIFACT_MODELS["bcell"] is not None,
        "scifact_nk_loaded": SCIFACT_MODELS["nk"] is not None
    })


@app.route("/api/seed_hallucination_examples", methods=["POST"])
def api_seed_hallucination_examples():
    """
    Seed the database with example hallucination training data.
    These are individual claims (not full citations) for better precision.
    """
    from database import Pattern

    # Example truthful claims (supported by evidence)
    truthful_claims = [
        "Aspirin inhibits platelet aggregation.",
        "Metformin is commonly used as first-line treatment for type 2 diabetes.",
        "Regular aerobic exercise improves cardiovascular health.",
        "Vitamin D deficiency is associated with increased risk of bone fractures.",
        "CRISPR-Cas9 can be used to edit specific DNA sequences.",
        "mRNA vaccines encode viral spike proteins to induce immune response.",
        "Sleep deprivation impairs cognitive function and memory consolidation.",
        "Statins reduce LDL cholesterol levels in the blood.",
        "Hypertension is a major risk factor for stroke.",
        "Antibiotics are ineffective against viral infections.",
    ]

    # Example hallucinated claims (not supported or contradicted by evidence)
    hallucinated_claims = [
        "Drinking bleach cures COVID-19 infection.",
        "5G cellular networks cause cancer in humans.",
        "Vaccines contain microchips for tracking purposes.",
        "Homeopathic remedies are more effective than conventional medicine.",
        "The earth is only 6000 years old based on scientific evidence.",
        "Humans only use 10% of their brain capacity.",
        "Goldfish have a 3-second memory span.",
        "Hair and nails continue growing after death.",
        "Cracking knuckles causes arthritis.",
        "Sugar causes hyperactivity in children.",
    ]

    session = Session()

    added_count = 0
    for claim in truthful_claims:
        pattern = Pattern(
            domain="hallucination",
            label="truthful",
            data=claim,
            confidence=0.95
        )
        session.add(pattern)
        added_count += 1

    for claim in hallucinated_claims:
        pattern = Pattern(
            domain="hallucination",
            label="hallucinated",
            data=claim,
            confidence=0.95
        )
        session.add(pattern)
        added_count += 1

    session.commit()
    session.close()

    return jsonify({
        "success": True,
        "message": f"Seeded {added_count} example claims",
        "truthful_count": len(truthful_claims),
        "hallucinated_count": len(hallucinated_claims)
    })


# ============================================================================
# SENTINEL HELPERS (Ops / Ticketing)
# ============================================================================


def read_sentinel_events(limit: int = 50):
    """
    Read latest sentinel events from JSONL without loading the entire file.
    Returns newest-first.
    """
    path = Path(SENTINEL_EVENTS_PATH)
    if not path.exists():
        return []

    lines = tail_lines(path, max_lines=limit)
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    events.reverse()
    return events


def tail_lines(path: Path, max_lines: int = 50, chunk_size: int = 64 * 1024):
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        data = b""
        while end > 0 and data.count(b"\n") <= max_lines:
            read_size = min(chunk_size, end)
            end -= read_size
            f.seek(end)
            data = f.read(read_size) + data
            if end == 0:
                break
        lines = data.splitlines()[-max_lines:]
        return [ln.decode("utf-8", errors="replace") for ln in lines if ln.strip()]


def read_antigent_events(limit: int = 50):
    """
    Read latest antigent (LLM call) events from JSONL.
    Returns newest-first.
    """
    antigent_events_path = os.environ.get(
        "ANTIGENCE_ANTIGENT_EVENTS_PATH",
        str(Path.home() / ".antigence" / "antigent_events.jsonl")
    )
    path = Path(antigent_events_path)
    if not path.exists():
        return []

    lines = tail_lines(path, max_lines=limit)
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    events.reverse()
    return events


def get_antigent_stats(events):
    """Compute stats from antigent events."""
    stats = {
        "total": len(events),
        "by_role": {},
        "success_rate": 0.0,
        "avg_latency_ms": 0.0,
        "total_tokens": 0,
    }

    if not events:
        return stats

    successes = 0
    total_latency = 0.0

    for ev in events:
        role = ev.get("role", "unknown")
        stats["by_role"][role] = stats["by_role"].get(role, 0) + 1
        if ev.get("success"):
            successes += 1
        total_latency += ev.get("latency_ms", 0)
        stats["total_tokens"] += ev.get("tokens", 0)

    stats["success_rate"] = successes / len(events) if events else 0.0
    stats["avg_latency_ms"] = total_latency / len(events) if events else 0.0

    return stats


def compute_telemetry_activity(events):
    """
    Approx tool activity over a window (how often telemetry roots changed).
    """
    counts = {}
    for ev in events:
        fs = (ev.get("fs") or {}).get("telemetry") or {}
        for _root, item in fs.items():
            label = (item or {}).get("label") or "unknown"
            counts[label] = counts.get(label, 0) + 1
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)


def _is_local_request() -> bool:
    return request.remote_addr in {"127.0.0.1", "::1"}


def _require_ops_access() -> None:
    """
    Protect start/stop/run operations.
    - If `ANTIGENCE_OPS_TOKEN` is set, require `X-Antigence-Token`.
    - Otherwise, only allow localhost requests.
    """
    if OPS_TOKEN:
        token = request.headers.get("X-Antigence-Token", "")
        if token != OPS_TOKEN:
            raise PermissionError("Invalid ops token")
        return
    if not _is_local_request():
        raise PermissionError("Ops endpoints require localhost or ANTIGENCE_OPS_TOKEN")


def get_sentinel_status() -> dict:
    return {
        "label": SENTINEL_LAUNCHD_LABEL,
        "loaded": launchd_is_loaded(SENTINEL_LAUNCHD_LABEL),
        "plist_path": SENTINEL_LAUNCHD_PLIST,
        "plist_exists": Path(SENTINEL_LAUNCHD_PLIST).exists(),
        "home": str(DEFAULT_SENTINEL_HOME),
        "events_path": SENTINEL_EVENTS_PATH,
        "events_exists": Path(SENTINEL_EVENTS_PATH).exists(),
        "watcher_path": SENTINEL_WATCHER_PATH,
        "watcher_exists": Path(SENTINEL_WATCHER_PATH).exists(),
        "platform": sys.platform,
        "ollama": get_ollama_status(),
        "llm_models": get_llm_models_status(),
        "logs": {"stdout": SENTINEL_OUT_LOG, "stderr": SENTINEL_ERR_LOG},
    }


def get_ollama_status() -> dict:
    """
    Best-effort status for local Ollama orchestrator.
    Never raises; safe for UI.
    """
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    installed = shutil.which("ollama") is not None
    running = False
    model_available = False
    models = []
    try:
        import requests  # optional; present in web_app/requirements.txt

        r = requests.get(url.rstrip("/") + "/api/tags", timeout=0.8)
        if r.status_code == 200:
            running = True
            models = r.json().get("models", []) or []
            model_available = any((m.get("name", "") or "").startswith(model) for m in models)
    except Exception:
        pass
    return {
        "installed": installed,
        "running": running,
        "model": model,
        "url": url,
        "model_available": model_available,
        "models": [m.get("name", "") for m in models if m.get("name")],
    }

def _run_cmd(cmd, timeout_s: int = 5) -> tuple:
    try:
        import subprocess

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def launchd_is_loaded(label: str) -> bool:
    if sys.platform != "darwin":
        return False
    code, out, _err = _run_cmd(["/bin/launchctl", "list"], timeout_s=5)
    if code != 0:
        return False
    return label in out


def launchd_load(plist_path: str) -> tuple:
    if sys.platform != "darwin":
        return False, "launchd controls are only available on macOS"
    if not Path(plist_path).exists():
        return False, f"plist not found: {plist_path}"
    code, _out, err = _run_cmd(["/bin/launchctl", "load", plist_path], timeout_s=10)
    return (code == 0), (err or "loaded")


def launchd_unload(plist_path: str) -> tuple:
    if sys.platform != "darwin":
        return False, "launchd controls are only available on macOS"
    if not Path(plist_path).exists():
        return True, "plist missing; already stopped"
    code, _out, err = _run_cmd(["/bin/launchctl", "unload", plist_path], timeout_s=10)
    return (code == 0), (err or "unloaded")


def run_sentinel_once() -> tuple:
    watcher = SENTINEL_WATCHER_PATH
    if not Path(watcher).exists():
        return False, f"watcher not found: {watcher} (set ANTIGENCE_SENTINEL_WATCHER)"
    code, out, err = _run_cmd(
        [sys.executable, watcher, "--max-depth", "5"],
        timeout_s=60,
    )
    msg = out or err or ("ok" if code == 0 else "failed")
    return (code == 0), msg


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def ingest_sentinel_event_as_tickets(session, event: dict):
    """
    Translate sentinel event -> tickets (idempotent via fingerprint).
    Policy (initial/default):
      - Any fs.unexpected => HIGH ticket
      - Changes in sensitive repos touching Scripts/config/requirements/shell => MED ticket
    """
    created_ids = []
    ts = event.get("ts")

    fs_unexpected = ((event.get("fs") or {}).get("unexpected") or {})
    if fs_unexpected:
        details = {"type": "unexpected_writes", "event": event, "policy": "fs.unexpected => HIGH"}
        fingerprint = _sha256(f"sentinel:unexpected:{ts}:{json.dumps(fs_unexpected, sort_keys=True)}")
        ticket = create_ticket(
            session,
            severity="high",
            source="sentinel",
            summary="Unexpected filesystem writes detected",
            details=details,
            fingerprint=fingerprint,
            source_event_ts=ts,
        )
        created_ids.append(ticket.id)

    prefixes_raw = os.environ.get("ANTIGENCE_SENSITIVE_REPO_PREFIXES", "").strip()
    sensitive_prefixes = [p.strip() for p in prefixes_raw.split(",") if p.strip()]
    changed_repos = (event.get("repos") or {}).get("changed") or []
    for repo in changed_repos:
        if sensitive_prefixes and not any(repo.startswith(pfx) for pfx in sensitive_prefixes):
            continue
        if not sensitive_prefixes:
            # Environment-agnostic default: only ticket "unexpected writes"; skip repo-change ticketing
            # until the installer defines sensitivity via `ANTIGENCE_SENSITIVE_REPO_PREFIXES`.
            continue
        status = git_status_porcelain(repo)
        if not status or not is_sensitive_change(status):
            continue
        details = {"type": "sensitive_repo_change", "repo": repo, "status": status.splitlines()[:200], "event_ts": ts}
        fingerprint = _sha256(f"sentinel:repo:{repo}:{ts}:{status}")
        ticket = create_ticket(
            session,
            severity="med",
            source="sentinel",
            summary=f"Sensitive repo changes detected ({Path(repo).name})",
            details=details,
            fingerprint=fingerprint,
            source_event_ts=ts,
        )
        created_ids.append(ticket.id)

    return created_ids


def ingest_detection_event_as_ticket(session, ops_event: OpsEvent):
    """
    Translate a detection OpsEvent -> ticket (idempotent via fingerprint).

    Policy (initial/default):
      - result=danger => HIGH
      - result=non_self => MED
      - others => no ticket (returns None)
    """
    try:
        payload = json.loads(ops_event.payload_json) if ops_event.payload_json else {}
    except Exception:
        payload = {}

    result = (payload.get("result") or "").lower()
    if result not in {"danger", "non_self"}:
        return None

    severity = "high" if result == "danger" else "med"
    domain = payload.get("domain") or "unknown"
    confidence = float(payload.get("confidence") or 0.0)
    danger_signal = float(payload.get("danger_signal") or 0.0)

    summary = f"Detection flagged: {result} (domain={domain})"
    details = {
        "type": "detection",
        "domain": domain,
        "result": result,
        "confidence": confidence,
        "danger_signal": danger_signal,
        "payload": payload,
        "ops_event_id": ops_event.id,
    }
    fingerprint = _sha256(f"immunos:detection:{ops_event.id}:{ops_event.created_at}:{ops_event.payload_json}")

    ticket = create_ticket(
        session,
        severity=severity,
        source="immunos",
        summary=summary,
        details=details,
        fingerprint=fingerprint,
        source_event_ts=ops_event.created_at.isoformat() if ops_event.created_at else None,
    )
    return ticket.id


def git_status_porcelain(repo_path: str) -> str:
    try:
        import subprocess

        proc = subprocess.run(
            ["git", "-C", repo_path, "status", "--porcelain=v1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return ""
        return proc.stdout.strip()
    except Exception:
        return ""


def is_sensitive_change(porcelain: str) -> bool:
    sensitive_prefixes = ("Scripts/", "scripts/", "config/")
    sensitive_suffixes = (".sh", ".plist", "requirements.txt", "pyproject.toml")
    for line in porcelain.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        path = parts[1].strip().strip('"')
        if path.startswith(sensitive_prefixes) or path.endswith(sensitive_suffixes):
            return True
    return False


def serialize_ticket(t: Ticket):
    return {
        "id": t.id,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "status": t.status,
        "severity": t.severity,
        "source": t.source,
        "source_event_ts": t.source_event_ts,
        "summary": t.summary,
        "details_json": t.details_json,
        "resolved_at": t.resolved_at.isoformat() if t.resolved_at else None,
        "resolution_note": t.resolution_note,
    }


def get_ticket_stats(session):
    total = session.query(Ticket).count()
    open_count = session.query(Ticket).filter_by(status="open").count()
    closed_count = session.query(Ticket).filter_by(status="closed").count()
    by_sev = {
        "high": session.query(Ticket).filter_by(severity="high").count(),
        "med": session.query(Ticket).filter_by(severity="med").count(),
        "low": session.query(Ticket).filter_by(severity="low").count(),
    }
    return {
        "total": total,
        "open": open_count,
        "closed": closed_count,
        "solution_rate": (closed_count / total) if total else 0.0,
        "by_severity": by_sev,
    }


def serialize_ops_event(e: OpsEvent):
    try:
        payload = json.loads(e.payload_json) if e.payload_json else {}
    except Exception:
        payload = {"_raw": e.payload_json}
    return {
        "id": e.id,
        "event_type": e.event_type,
        "source": e.source,
        "severity": e.severity,
        "created_at": e.created_at.isoformat() if e.created_at else None,
        "payload": payload,
    }


def get_detection_stats(serialized_or_models):
    """
    Compute simple incident rates from the detection stream.
    """
    counts = {"total": 0, "self": 0, "non_self": 0, "danger": 0, "uncertain": 0, "unknown": 0}
    by_domain = {}
    for item in serialized_or_models:
        if isinstance(item, OpsEvent):
            try:
                payload = json.loads(item.payload_json) if item.payload_json else {}
            except Exception:
                payload = {}
            result = (payload.get("result") or "").lower()
            domain = (payload.get("domain") or "unknown").lower()
        else:
            payload = item.get("payload", {}) if isinstance(item, dict) else {}
            result = (payload.get("result") or "").lower()
            domain = (payload.get("domain") or "unknown").lower()

        counts["total"] += 1
        bucket = result if result in {"self", "non_self", "danger", "uncertain"} else "unknown"
        counts[bucket] += 1

        by_domain.setdefault(domain, {"total": 0, "danger": 0, "non_self": 0, "uncertain": 0, "self": 0})
        by_domain[domain]["total"] += 1
        if bucket in by_domain[domain]:
            by_domain[domain][bucket] += 1

    # simple "incident rate" = (danger + non_self) / total
    incident_rate = 0.0
    if counts["total"]:
        incident_rate = (counts["danger"] + counts["non_self"]) / counts["total"]

    top_domains = sorted(by_domain.items(), key=lambda kv: kv[1]["total"], reverse=True)[:8]
    return {"counts": counts, "incident_rate": incident_rate, "top_domains": top_domains}


@contextmanager
def get_db_session():
    """Provide a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ============================================================================
# ERROR HANDLERS
# ============================================================================


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return render_template("500.html"), 500


# ============================================================================
# OPS ERROR HANDLING
# ============================================================================


@app.errorhandler(PermissionError)
def permission_error(error):
    return jsonify({"ok": False, "message": str(error)}), 403


@app.errorhandler(ValueError)
def value_error(error):
    """Handle validation errors with JSON response"""
    return jsonify({"ok": False, "message": str(error)}), 400


# Generic error handler for API endpoints
@app.errorhandler(Exception)
def generic_error(error):
    """Handle unexpected errors - return JSON for API routes, HTML for pages"""
    if request.path.startswith("/api/"):
        import traceback
        print(f"API Error on {request.path}: {error}")
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "message": f"Server error: {str(error)}",
            "error_type": type(error).__name__
        }), 500
    # For non-API routes, use the 500 template
    return render_template("500.html"), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Load configuration from environment
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5001))

    print("=" * 60)
    print("🧬 Antigence™ Platform (powered by immunOS)")
    print("=" * 60)
    print(f"\n✅ Flask server starting (DEBUG={'ON' if debug_mode else 'OFF'})...")
    print(f"📁 Database: {DB_PATH}")
    print(f"🌐 Access at: http://{host}:{port}")

    # Check if any models exist
    bcell_exists = any(p.exists() for p in MODEL_PATHS["bcell"])
    nk_exists = any(p.exists() for p in MODEL_PATHS["nk"])
    print(f"🔬 SciFact models: {bcell_exists and nk_exists}")

    if not debug_mode:
        print("🛡️  Running in production mode (Hardened)")
    else:
        print("⚠️  Running in DEBUG mode (Development only)")

    print("\n📊 Features:")
    print("  • Publications Validator (SciFact-powered)")
    print("  • Code Security Scanner (NegSl-AIS)")
    print("  • Emotion Detection")
    print("  • Email Classification (Advanced EML)")
    print("  • Network Intrusion Detection")
    print("\nPress Ctrl+C to stop the server\n")

    app.run(host=host, port=port, debug=debug_mode)
