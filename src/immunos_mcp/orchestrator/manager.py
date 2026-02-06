"""
Orchestrator manager for IMMUNOS-MCP.

Manages orchestrator selection and fallback logic.
Automatically selects local (thymus) when:
- Offline/air-gapped
- Rate limits hit
- Remote unavailable
"""

from typing import Optional, Dict, Any
from ..core.antigen import Antigen
from ..orchestrator.orchestrator import OrchestratorResult
from ..orchestrator.interface import OrchestratorInterface
from ..orchestrator.local import LocalOrchestrator
from ..orchestrator.remote import RemoteOrchestrator
from ..utils.network import detect_mode, should_use_local_orchestrator


class OrchestratorManager:
    """
    Manages orchestrator selection and fallback.
    
    Automatically selects the best orchestrator based on:
    - Configuration mode (air_gapped vs online)
    - Network availability
    - Rate limit status
    - Remote orchestrator availability
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize orchestrator manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.mode = detect_mode(config)
        self.orchestrator: Optional[OrchestratorInterface] = None
        self.last_error: Optional[Exception] = None
        
        # Initialize orchestrator
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize orchestrator based on mode and availability."""
        # Always create local orchestrator (fallback)
        self.local_orchestrator = LocalOrchestrator(self.config)
        
        # Determine which orchestrator to use
        use_local = should_use_local_orchestrator(self.config, self.last_error)
        
        if use_local:
            self.orchestrator = self.local_orchestrator
        else:
            # Try remote orchestrator
            try:
                remote = RemoteOrchestrator(self.config)
                if remote.is_available():
                    self.orchestrator = remote
                else:
                    # Remote not available, use local
                    self.orchestrator = self.local_orchestrator
            except Exception as e:
                # Failed to create remote, use local
                self.orchestrator = self.local_orchestrator
                self.last_error = e
    
    def analyze(self, antigen: Antigen) -> OrchestratorResult:
        """
        Run analysis using current orchestrator.
        
        Automatically falls back to local if remote fails.
        
        Args:
            antigen: The antigen to analyze
            
        Returns:
            OrchestratorResult with analysis results
        """
        if not self.orchestrator:
            self._initialize()
        
        # Try current orchestrator
        try:
            result = self.orchestrator.analyze(antigen)
            self.last_error = None  # Clear error on success
            return result
        except Exception as e:
            self.last_error = e
            
            # If using remote and it failed, fall back to local
            if self.orchestrator.get_mode() == "remote":
                print(f"Remote orchestrator failed: {e}. Falling back to local.", file=__import__("sys").stderr)
                self.orchestrator = self.local_orchestrator
                
                # Retry with local
                return self.orchestrator.analyze(antigen)
            else:
                # Already using local, re-raise
                raise
    
    def is_available(self) -> bool:
        """Check if orchestrator is available."""
        if not self.orchestrator:
            self._initialize()
        return self.orchestrator.is_available()
    
    def get_current_orchestrator(self) -> OrchestratorInterface:
        """Get the currently active orchestrator."""
        if not self.orchestrator:
            self._initialize()
        return self.orchestrator
    
    def get_mode(self) -> str:
        """Get current operating mode."""
        return self.mode
    
    def force_local(self) -> None:
        """Force use of local orchestrator (e.g., after rate limit)."""
        self.orchestrator = self.local_orchestrator
    
    def retry_remote(self) -> None:
        """Try to switch back to remote orchestrator if available."""
        if self.mode == "online":
            try:
                remote = RemoteOrchestrator(self.config)
                if remote.is_available():
                    self.orchestrator = remote
                    self.last_error = None
            except Exception:
                pass  # Keep using local


