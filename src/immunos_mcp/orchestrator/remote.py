"""
Remote orchestrator implementation.

This orchestrator connects to a remote IMMUNOS orchestrator service
via HTTP. Used when online mode is enabled and remote is available.
"""

import time
from typing import Dict, Any, Optional
import httpx
from ..core.antigen import Antigen
from ..orchestrator.orchestrator import OrchestratorResult
from ..orchestrator.interface import OrchestratorInterface


class RemoteOrchestrator(OrchestratorInterface):
    """
    Remote HTTP-based orchestrator.
    
    This orchestrator:
    - Connects to remote service via HTTP
    - Requires network connectivity
    - Can handle rate limits and retries
    - Falls back to local if unavailable
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize remote orchestrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.mode = "remote"
        
        remote_config = config.get("orchestrator", {}).get("remote", {})
        self.endpoint = remote_config.get("endpoint", "http://localhost:8000")
        self.api_key = remote_config.get("api_key")
        self.timeout = remote_config.get("timeout", 30)
        self.retry_count = remote_config.get("retry_count", 3)
        
        # Create HTTP client
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.client = httpx.Client(
            base_url=self.endpoint,
            headers=headers,
            timeout=self.timeout
        )
        
        self.last_health_check = 0
        self.health_check_interval = remote_config.get("health_check_interval", 60)
        self._available = None
    
    def analyze(self, antigen: Antigen) -> OrchestratorResult:
        """
        Run multi-agent analysis using remote orchestrator.
        
        Args:
            antigen: The antigen to analyze
            
        Returns:
            OrchestratorResult with analysis results
            
        Raises:
            httpx.HTTPError: If request fails
        """
        # Convert antigen to dict for JSON serialization
        payload = {
            "data": antigen.data,
            "data_type": antigen.data_type.value,
            "identifier": antigen.identifier,
            "metadata": antigen.metadata or {},
        }
        
        # Retry logic
        last_error = None
        for attempt in range(self.retry_count):
            try:
                response = self.client.post(
                    "/analyze",
                    json=payload
                )
                response.raise_for_status()
                
                # Parse response
                result_dict = response.json()
                
                # Convert back to OrchestratorResult
                # (This is a simplified conversion - adjust based on actual API)
                return OrchestratorResult(
                    classification=result_dict.get("classification"),
                    confidence=result_dict.get("confidence", 0.0),
                    anomaly=result_dict.get("anomaly", False),
                    bcell_confidence=result_dict.get("bcell_confidence", 0.0),
                    nk_confidence=result_dict.get("nk_confidence", 0.0),
                    features=result_dict.get("features", {}),
                    memory_hit=result_dict.get("memory_hit", False),
                    signals=result_dict.get("signals", {}),
                    agents=result_dict.get("agents", []),
                    model_roles=result_dict.get("model_roles", {}),
                    metadata=result_dict.get("metadata", {}),
                )
                
            except httpx.HTTPStatusError as e:
                # Check for rate limit (429)
                if e.response.status_code == 429:
                    # Rate limit hit - should fall back to local
                    raise RuntimeError("Rate limit exceeded") from e
                last_error = e
                if attempt < self.retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < self.retry_count - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
    
    def is_available(self) -> bool:
        """
        Check if remote orchestrator is available.
        
        Uses cached health check to avoid excessive requests.
        
        Returns:
            True if remote orchestrator is reachable
        """
        now = time.time()
        
        # Use cached result if recent
        if self._available is not None and (now - self.last_health_check) < self.health_check_interval:
            return self._available
        
        # Perform health check
        try:
            response = self.client.get("/health", timeout=5)
            self._available = response.status_code == 200
        except Exception:
            self._available = False
        
        self.last_health_check = now
        return self._available
    
    def get_mode(self) -> str:
        """Get orchestrator mode."""
        return self.mode
    
    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "client"):
            try:
                self.client.close()
            except Exception:
                pass


