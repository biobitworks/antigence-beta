"""
Configuration validator for IMMUNOS-MCP.
"""

from typing import Dict, Any


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration structure and values."""
    # Validate mode
    if "mode" in config:
        if config["mode"] not in ["air_gapped", "online"]:
            raise ValueError(f"Invalid mode: {config['mode']}. Must be 'air_gapped' or 'online'")

    # Validate orchestrator
    if "orchestrator" in config:
        orch = config["orchestrator"]
        if "type" in orch:
            if orch["type"] not in ["local", "remote"]:
                raise ValueError(f"Invalid orchestrator type: {orch['type']}. Must be 'local' or 'remote'")

    # Validate logging level
    if "logging" in config and "level" in config["logging"]:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config["logging"]["level"] not in valid_levels:
            raise ValueError(f"Invalid log level: {config['logging']['level']}. Must be one of {valid_levels}")

    # Validate LLM provider
    if "llm" in config and "provider" in config["llm"]:
        valid_providers = ["ollama", "openai", "anthropic", None]
        if config["llm"]["provider"] not in valid_providers:
            raise ValueError(f"Invalid LLM provider: {config['llm']['provider']}. Must be one of {valid_providers}")


