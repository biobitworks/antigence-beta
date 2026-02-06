"""
Network detection utilities for mode detection.
"""

import os
import time
from typing import Dict, Any, List
import requests


def can_reach_internet(config: Dict[str, Any]) -> bool:
    """
    Test connectivity to test endpoints.
    
    Returns True if at least one endpoint is reachable.
    """
    endpoints = config.get("network", {}).get("test_endpoints", [])
    timeout = config.get("network", {}).get("test_timeout", 5)
    
    if not endpoints:
        return False
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=timeout)
            if response.status_code == 200:
                return True
        except Exception:
            continue
    
    return False


def detect_mode(config: Dict[str, Any]) -> str:
    """
    Detect operating mode based on configuration and network status.
    
    Priority:
    1. Explicit config mode
    2. Environment variable
    3. Auto-detect (if enabled)
    4. Default to air_gapped (safest)
    """
    # 1. Check explicit configuration
    if config.get("mode"):
        return config["mode"]
    
    # 2. Check environment variable
    env_mode = os.getenv("IMMUNOS_MODE")
    if env_mode in ["air_gapped", "online"]:
        return env_mode
    
    # 3. Auto-detect if enabled
    if config.get("network", {}).get("auto_detect", False):
        if can_reach_internet(config):
            return "online"
        else:
            return "air_gapped"
    
    # 4. Default to air_gapped (safest)
    return "air_gapped"


def check_rate_limit(headers: Dict[str, str]) -> bool:
    """
    Check if rate limit headers indicate we're hitting limits.
    
    Common headers:
    - X-RateLimit-Remaining: 0
    - Retry-After: <seconds>
    - 429 status code
    """
    # Check remaining rate limit
    remaining = headers.get("X-RateLimit-Remaining")
    if remaining is not None:
        try:
            if int(remaining) == 0:
                return True
        except ValueError:
            pass
    
    # Check retry-after header
    retry_after = headers.get("Retry-After")
    if retry_after is not None:
        return True
    
    return False


def should_use_local_orchestrator(config: Dict[str, Any], last_error: Exception = None) -> bool:
    """
    Determine if we should use local orchestrator.
    
    Reasons to use local:
    1. Mode is air_gapped
    2. Remote orchestrator not enabled
    3. Network unreachable
    4. Rate limit hit
    5. Remote orchestrator unavailable
    """
    mode = detect_mode(config)
    
    # Air-gapped mode always uses local
    if mode == "air_gapped":
        return True
    
    # Check if remote orchestrator is enabled
    remote_enabled = config.get("orchestrator", {}).get("remote", {}).get("enabled", False)
    if not remote_enabled:
        return True
    
    # Check for rate limit errors
    if last_error:
        error_str = str(last_error).lower()
        if any(term in error_str for term in ["429", "rate limit", "quota", "limit exceeded"]):
            return True
    
    # If online mode and remote enabled, try remote first
    # (fallback logic handled by orchestrator manager)
    return False


