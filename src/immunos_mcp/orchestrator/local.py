"""
Local orchestrator implementation (Thymus).

This is the built-in, always-available orchestrator that runs
in-process. It's the "thymus" that coordinates local agents.
"""

from typing import Optional, Dict, Any
from ..core.antigen import Antigen
from ..orchestrator.orchestrator import ImmunosOrchestrator, OrchestratorResult
from ..orchestrator.interface import OrchestratorInterface


class LocalOrchestrator(OrchestratorInterface):
    """
    Local in-process orchestrator (Thymus).
    
    This orchestrator:
    - Always available (no network required)
    - Runs all agents locally
    - Works in air-gapped mode
    - No external dependencies
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize local orchestrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.mode = "local"
        
        # Get agent state paths from config
        bcell_state = None
        nk_state = None
        
        if "agents" in config:
            if "bcell" in config["agents"] and config["agents"]["bcell"].get("state_path"):
                bcell_state = config["agents"]["bcell"]["state_path"]
            if "nk_cell" in config["agents"] and config["agents"]["nk_cell"].get("state_path"):
                nk_state = config["agents"]["nk_cell"]["state_path"]
        
        # Initialize the orchestrator
        self.orchestrator = ImmunosOrchestrator(
            bcell_state=bcell_state,
            nk_state=nk_state
        )
    
    def analyze(self, antigen: Antigen) -> OrchestratorResult:
        """
        Run multi-agent analysis using local orchestrator.
        
        Args:
            antigen: The antigen to analyze
            
        Returns:
            OrchestratorResult with analysis results
        """
        return self.orchestrator.analyze(antigen)
    
    def is_available(self) -> bool:
        """
        Local orchestrator is always available.
        
        Returns:
            True (always available)
        """
        return True
    
    def get_mode(self) -> str:
        """Get orchestrator mode."""
        return self.mode


