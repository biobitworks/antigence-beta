"""
Configuration loader for IMMUNOS-MCP.

Supports three-tier configuration:
1. Defaults (built-in)
2. User config (~/.immunos-mcp/config.yaml)
3. Environment variables (override)
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from ..config.validator import validate_config


# Default configuration
DEFAULT_CONFIG = {
    "mode": "air_gapped",  # Default to safest mode
    "orchestrator": {
        "type": "local",
        "local": {
            "enabled": True,
            "path": None,
            "models_path": "~/.immunos-mcp/models",
            "state_path": "~/.immunos-mcp/state",
        },
        "remote": {
            "enabled": False,
            "endpoint": "http://localhost:8000",
            "api_key": None,
            "timeout": 30,
            "retry_count": 3,
            "health_check_interval": 60,
        },
    },
    "agents": {
        "bcell": {"enabled": True, "model": None, "state_path": None},
        "nk_cell": {"enabled": True, "model": None, "state_path": None},
        "dendritic": {"enabled": True},
        "memory": {"enabled": True, "path": "~/.immunos-mcp/memory"},
    },
    "llm": {
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "api_key": None,
        "models": {
            "bcell": "qwen2.5-coder:7b",
            "nk_cell": "deepseek-r1:14b",
            "t_cell": "deepseek-r1:14b",
            "dendritic": "qwen2.5:1.5b",
            "memory": "qwen2.5:1.5b",
        },
    },
    "network": {
        "auto_detect": True,
        "test_endpoints": [
            "https://www.google.com",
            "https://api.openai.com",
        ],
        "test_timeout": 5,
    },
    "logging": {
        "level": "INFO",
        "file": "~/.immunos-mcp/logs/immunos-mcp.log",
        "console": True,
    },
}


def expand_path(path: str) -> Path:
    """Expand ~ and environment variables in path."""
    return Path(os.path.expanduser(os.path.expandvars(path)))


def load_user_config() -> Dict[str, Any]:
    """Load user configuration from ~/.immunos-mcp/config.yaml"""
    config_path = expand_path("~/.immunos-mcp/config.yaml")
    
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}
        return user_config
    except Exception as e:
        print(f"Warning: Could not load user config: {e}", file=os.sys.stderr)
        return {}


def load_env_overrides() -> Dict[str, Any]:
    """Load configuration overrides from environment variables."""
    overrides = {}
    
    # Mode
    if os.getenv("IMMUNOS_MODE"):
        overrides["mode"] = os.getenv("IMMUNOS_MODE")
    
    # Orchestrator remote endpoint
    if os.getenv("IMMUNOS_ORCHESTRATOR_ENDPOINT"):
        if "orchestrator" not in overrides:
            overrides["orchestrator"] = {}
        if "remote" not in overrides["orchestrator"]:
            overrides["orchestrator"]["remote"] = {}
        overrides["orchestrator"]["remote"]["endpoint"] = os.getenv("IMMUNOS_ORCHESTRATOR_ENDPOINT")
    
    # Orchestrator API key
    if os.getenv("IMMUNOS_ORCHESTRATOR_API_KEY"):
        if "orchestrator" not in overrides:
            overrides["orchestrator"] = {}
        if "remote" not in overrides["orchestrator"]:
            overrides["orchestrator"]["remote"] = {}
        overrides["orchestrator"]["remote"]["api_key"] = os.getenv("IMMUNOS_ORCHESTRATOR_API_KEY")
    
    # LLM provider
    if os.getenv("IMMUNOS_LLM_PROVIDER"):
        if "llm" not in overrides:
            overrides["llm"] = {}
        overrides["llm"]["provider"] = os.getenv("IMMUNOS_LLM_PROVIDER")
    
    # LLM base URL
    if os.getenv("IMMUNOS_LLM_BASE_URL"):
        if "llm" not in overrides:
            overrides["llm"] = {}
        overrides["llm"]["base_url"] = os.getenv("IMMUNOS_LLM_BASE_URL")
    
    # Logging level
    if os.getenv("IMMUNOS_LOG_LEVEL"):
        if "logging" not in overrides:
            overrides["logging"] = {}
        overrides["logging"]["level"] = os.getenv("IMMUNOS_LOG_LEVEL")
    
    return overrides


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def load_config() -> Dict[str, Any]:
    """
    Load configuration with three-tier priority:
    1. Defaults
    2. User config (~/.immunos-mcp/config.yaml)
    3. Environment variables
    """
    # Start with defaults
    config = DEFAULT_CONFIG.copy()
    
    # Merge user config
    user_config = load_user_config()
    config = deep_merge(config, user_config)
    
    # Apply environment overrides
    env_overrides = load_env_overrides()
    config = deep_merge(config, env_overrides)
    
    # Expand paths
    if "orchestrator" in config and "local" in config["orchestrator"]:
        local = config["orchestrator"]["local"]
        if "models_path" in local:
            local["models_path"] = str(expand_path(local["models_path"]))
        if "state_path" in local:
            local["state_path"] = str(expand_path(local["state_path"]))
    
    if "agents" in config and "memory" in config["agents"]:
        if "path" in config["agents"]["memory"]:
            config["agents"]["memory"]["path"] = str(expand_path(config["agents"]["memory"]["path"]))
    
    if "logging" in config and "file" in config["logging"]:
        config["logging"]["file"] = str(expand_path(config["logging"]["file"]))
    
    # Validate configuration
    validate_config(config)
    
    return config


def get_config() -> Dict[str, Any]:
    """Get cached configuration (singleton pattern)."""
    if not hasattr(get_config, "_cache"):
        get_config._cache = load_config()
    return get_config._cache


def reload_config() -> Dict[str, Any]:
    """Reload configuration (clear cache and reload)."""
    if hasattr(get_config, "_cache"):
        delattr(get_config, "_cache")
    return get_config()


