import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class LLMRole:
    role: str
    label: str
    purpose: str


DEFAULT_ROLES: List[LLMRole] = [
    LLMRole(
        role="orchestrator",
        label="Thymus (Orchestrator)",
        purpose="Coordinates other antigents; produces final verdicts/summaries.",
    ),
    LLMRole(
        role="bcell",
        label="B Cell (Evidence/Claims)",
        purpose="Evidence extraction, citation checks, claim comparison, writeups.",
    ),
    LLMRole(
        role="nk",
        label="NK Cell (Anomaly/Sentinel)",
        purpose="Flag anomalies/out-of-scope behavior; produce risk summaries.",
    ),
    LLMRole(
        role="tcell_security",
        label="T Cell (Security Review)",
        purpose="Secure-code review and remediation suggestions.",
    ),
    LLMRole(
        role="dendritic_summarizer",
        label="Dendritic (Summarizer)",
        purpose="Summarize long artifacts into presentation-ready bullets.",
    ),
]


def _parse_json_mapping(raw: str) -> Dict[str, str]:
    try:
        obj = json.loads(raw)
    except Exception:
        return {}
    if isinstance(obj, dict):
        out: Dict[str, str] = {}
        for k, v in obj.items():
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                out[k.strip()] = v.strip()
        return out
    return {}


def load_role_model_mapping(home: Optional[Path] = None) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Resolve role→model mapping.
    Precedence:
      1) `ANTIGENCE_LLM_ROLE_MODELS` (JSON dict)
      2) `~/.antigence/llm/roles.json` (or `ANTIGENCE_HOME`)

    Returns:
      (mapping, meta)
    """
    meta: Dict[str, Any] = {"source": None, "path": None, "errors": []}

    raw = os.environ.get("ANTIGENCE_LLM_ROLE_MODELS", "").strip()
    if raw:
        mapping = _parse_json_mapping(raw)
        if mapping:
            meta["source"] = "env"
            return mapping, meta
        meta["errors"].append("ANTIGENCE_LLM_ROLE_MODELS present but invalid JSON mapping")

    home_dir = home or Path(os.environ.get("ANTIGENCE_HOME", str(Path.home() / ".antigence"))).expanduser()
    roles_path = home_dir / "llm" / "roles.json"
    meta["path"] = str(roles_path)
    if roles_path.exists():
        try:
            mapping = _parse_json_mapping(roles_path.read_text(encoding="utf-8"))
            if mapping:
                meta["source"] = "file"
                return mapping, meta
            meta["errors"].append(f"{roles_path} exists but does not contain a valid JSON mapping")
        except Exception as e:
            meta["errors"].append(f"Failed reading {roles_path}: {e}")

    meta["source"] = "default"
    return {}, meta


def _list_ollama_models(url: str) -> List[str]:
    try:
        import requests

        r = requests.get(url.rstrip("/") + "/api/tags", timeout=0.8)
        if r.status_code != 200:
            return []
        models = r.json().get("models", []) or []
        return [m.get("name", "") for m in models if m.get("name")]
    except Exception:
        return []


def get_llm_models_status() -> Dict[str, Any]:
    """
    Return a role-aware LLM status block for the UI/API.
    Designed to be environment-agnostic and never raise.
    """
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    default_model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

    mapping, meta = load_role_model_mapping()
    installed_models = _list_ollama_models(url)
    installed_set = set(installed_models)

    roles: List[Dict[str, Any]] = []
    missing: List[Dict[str, str]] = []
    for r in DEFAULT_ROLES:
        model = mapping.get(r.role, default_model)
        available = any(m.startswith(model) for m in installed_set) if installed_set else False
        roles.append(
            {
                "role": r.role,
                "label": r.label,
                "purpose": r.purpose,
                "model": model,
                "available": available,
                "source": "mapping" if r.role in mapping else "default",
            }
        )
        if not available:
            missing.append({"role": r.role, "model": model})

    return {
        "ollama_url": url,
        "default_model": default_model,
        "roles": roles,
        "installed_models": installed_models,
        "missing": missing,
        "config": {
            "env": "ANTIGENCE_LLM_ROLE_MODELS",
            "file": meta.get("path") or str(Path.home() / ".antigence" / "llm" / "roles.json"),
            "source": meta.get("source"),
            "errors": meta.get("errors", []),
        },
    }

