"""
Orchestrator interface for IMMUNOS-MCP.

Defines the abstract interface that all orchestrators must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional
from ..core.antigen import Antigen
from ..orchestrator.orchestrator import OrchestratorResult


class OrchestratorInterface(ABC):
    """
    Abstract interface for IMMUNOS orchestrators.
    
    This allows pluggable orchestrator implementations:
    - LocalOrchestrator: In-process, always available
    - RemoteOrchestrator: HTTP-based, optional
    """
    
    @abstractmethod
    def analyze(self, antigen: Antigen) -> OrchestratorResult:
        """
        Run multi-agent analysis on an antigen.
        
        Args:
            antigen: The antigen to analyze
            
        Returns:
            OrchestratorResult with analysis results
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if orchestrator is available and ready.
        
        Returns:
            True if orchestrator can process requests
        """
        pass
    
    @abstractmethod
    def get_mode(self) -> str:
        """
        Get the mode this orchestrator operates in.
        
        Returns:
            "local" or "remote"
        """
        pass


