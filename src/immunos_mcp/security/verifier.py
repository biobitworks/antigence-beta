"""
Secure Verifier for IMMUNOS

Verifies agent state file signatures before loading.
Ensures integrity and detects tampering.
"""

import json
import hashlib
import base64
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .signer import SignatureManifest, HAS_CRYPTOGRAPHY

if HAS_CRYPTOGRAPHY:
    from cryptography.hazmat.primitives import serialization
    from cryptography.exceptions import InvalidSignature


class VerificationError(Exception):
    """Raised when verification fails."""
    pass


@dataclass
class VerificationResult:
    """Result of signature verification."""
    valid: bool
    state_hash_match: bool
    signature_valid: bool
    agent_type: str
    domain: str
    signed_at: str
    signed_by: str
    error: Optional[str] = None


class SecureVerifier:
    """
    Verifies agent state file signatures.

    Usage:
        verifier = SecureVerifier(public_key_path="/path/to/key.pub")
        result = verifier.verify_file("/path/to/bcell_state.json")
        if not result.valid:
            raise VerificationError("Tampered state file!")
    """

    def __init__(
        self,
        public_key_path: Optional[str] = None,
        public_key_bytes: Optional[bytes] = None,
        key_id: str = "default",
        require_signature: bool = True
    ):
        """
        Initialize verifier with public key.

        Args:
            public_key_path: Path to PEM-encoded Ed25519 public key
            public_key_bytes: Raw public key bytes (alternative to path)
            key_id: Expected key identifier (for HMAC fallback)
            require_signature: If True, reject unsigned files
        """
        self.key_id = key_id
        self.require_signature = require_signature
        self._public_key = None

        if HAS_CRYPTOGRAPHY:
            if public_key_path:
                self._load_key_from_file(public_key_path)
            elif public_key_bytes:
                self._load_key_from_bytes(public_key_bytes)

    def _load_key_from_file(self, path: str) -> None:
        """Load Ed25519 public key from PEM file."""
        try:
            key_data = Path(path).read_bytes()
            self._public_key = serialization.load_pem_public_key(key_data)
        except Exception as e:
            raise VerificationError(f"Failed to load public key: {e}")

    def _load_key_from_bytes(self, key_bytes: bytes) -> None:
        """Load Ed25519 public key from raw bytes."""
        try:
            self._public_key = serialization.load_pem_public_key(key_bytes)
        except Exception as e:
            raise VerificationError(f"Failed to load public key: {e}")

    def compute_hash(self, data: bytes) -> str:
        """Compute SHA-256 hash of data."""
        return "sha256:" + hashlib.sha256(data).hexdigest()

    def verify_signature(self, data: bytes, signature: str) -> bool:
        """
        Verify signature on data.

        Args:
            data: Original data bytes
            signature: Base64-encoded signature

        Returns:
            True if valid, False otherwise
        """
        if signature.startswith("hmac:"):
            # HMAC fallback verification
            import hmac
            expected = hmac.new(
                self.key_id.encode(),
                data,
                hashlib.sha256
            ).digest()
            actual = base64.b64decode(signature[5:])
            return hmac.compare_digest(expected, actual)

        elif HAS_CRYPTOGRAPHY and self._public_key:
            # Ed25519 verification
            try:
                sig_bytes = base64.b64decode(signature)
                self._public_key.verify(sig_bytes, data)
                return True
            except InvalidSignature:
                return False
            except Exception:
                return False

        return False

    def verify_file(self, file_path: str) -> VerificationResult:
        """
        Verify an agent state file.

        Checks:
        1. Signature file exists
        2. State hash matches
        3. Signature is valid

        Args:
            file_path: Path to the state file

        Returns:
            VerificationResult with verification details
        """
        path = Path(file_path)
        sig_path = Path(str(file_path) + ".sig")

        # Check state file exists
        if not path.exists():
            return VerificationResult(
                valid=False,
                state_hash_match=False,
                signature_valid=False,
                agent_type="",
                domain="",
                signed_at="",
                signed_by="",
                error=f"State file not found: {file_path}"
            )

        # Check signature file exists
        if not sig_path.exists():
            if self.require_signature:
                return VerificationResult(
                    valid=False,
                    state_hash_match=False,
                    signature_valid=False,
                    agent_type="",
                    domain="",
                    signed_at="",
                    signed_by="",
                    error="Signature file not found - unsigned state"
                )
            else:
                return VerificationResult(
                    valid=True,
                    state_hash_match=True,
                    signature_valid=False,
                    agent_type="unsigned",
                    domain="unsigned",
                    signed_at="",
                    signed_by="",
                    error="No signature (not required)"
                )

        # Read files
        content = path.read_bytes()
        sig_data = json.loads(sig_path.read_text())
        manifest = SignatureManifest.from_dict(sig_data)

        # Verify hash
        computed_hash = self.compute_hash(content)
        hash_match = (computed_hash == manifest.state_hash)

        if not hash_match:
            return VerificationResult(
                valid=False,
                state_hash_match=False,
                signature_valid=False,
                agent_type=manifest.agent_type,
                domain=manifest.domain,
                signed_at=manifest.signed_at,
                signed_by=manifest.signed_by,
                error="Hash mismatch - file may be tampered"
            )

        # Verify signature
        payload = json.dumps({
            "state_hash": manifest.state_hash,
            "agent_type": manifest.agent_type,
            "domain": manifest.domain,
            "chain_previous": manifest.chain_previous,
            "signed_at": manifest.signed_at
        }, sort_keys=True).encode()

        sig_valid = self.verify_signature(payload, manifest.signature)

        if not sig_valid:
            return VerificationResult(
                valid=False,
                state_hash_match=True,
                signature_valid=False,
                agent_type=manifest.agent_type,
                domain=manifest.domain,
                signed_at=manifest.signed_at,
                signed_by=manifest.signed_by,
                error="Invalid signature - possible tampering"
            )

        return VerificationResult(
            valid=True,
            state_hash_match=True,
            signature_valid=True,
            agent_type=manifest.agent_type,
            domain=manifest.domain,
            signed_at=manifest.signed_at,
            signed_by=manifest.signed_by
        )

    def verify_state(self, state_data: Dict[str, Any]) -> VerificationResult:
        """
        Verify state data with embedded signature.

        Args:
            state_data: State dict with _signature field

        Returns:
            VerificationResult
        """
        if "_signature" not in state_data:
            if self.require_signature:
                return VerificationResult(
                    valid=False,
                    state_hash_match=False,
                    signature_valid=False,
                    agent_type="",
                    domain="",
                    signed_at="",
                    signed_by="",
                    error="No embedded signature"
                )
            return VerificationResult(
                valid=True,
                state_hash_match=True,
                signature_valid=False,
                agent_type="unsigned",
                domain="unsigned",
                signed_at="",
                signed_by=""
            )

        sig_info = state_data["_signature"]

        # Extract state without signature
        state_only = {k: v for k, v in state_data.items() if k != "_signature"}
        content = json.dumps(state_only, sort_keys=True).encode()

        # Verify hash
        computed_hash = "sha256:" + hashlib.sha256(content).hexdigest()
        hash_match = (computed_hash == sig_info.get("state_hash"))

        if not hash_match:
            return VerificationResult(
                valid=False,
                state_hash_match=False,
                signature_valid=False,
                agent_type=sig_info.get("agent_type", ""),
                domain=sig_info.get("domain", ""),
                signed_at=sig_info.get("signed_at", ""),
                signed_by=sig_info.get("signed_by", ""),
                error="Hash mismatch"
            )

        # Verify signature
        payload = json.dumps({
            "state_hash": sig_info["state_hash"],
            "agent_type": sig_info.get("agent_type", "unknown"),
            "domain": sig_info.get("domain", "unknown"),
            "signed_at": sig_info["signed_at"]
        }, sort_keys=True).encode()

        sig_valid = self.verify_signature(payload, sig_info["signature"])

        return VerificationResult(
            valid=sig_valid and hash_match,
            state_hash_match=hash_match,
            signature_valid=sig_valid,
            agent_type=sig_info.get("agent_type", ""),
            domain=sig_info.get("domain", ""),
            signed_at=sig_info.get("signed_at", ""),
            signed_by=sig_info.get("signed_by", ""),
            error=None if sig_valid else "Invalid signature"
        )


class SecureAgentLoader:
    """
    Secure loader that verifies signatures before loading agent states.

    Usage:
        loader = SecureAgentLoader(public_key_path="/path/to/key.pub")
        state = loader.load("/path/to/bcell_state.json")
        # Raises VerificationError if tampered
    """

    def __init__(
        self,
        public_key_path: Optional[str] = None,
        require_signature: bool = True
    ):
        self.verifier = SecureVerifier(
            public_key_path=public_key_path,
            require_signature=require_signature
        )

    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load and verify agent state.

        Raises VerificationError if verification fails.
        """
        result = self.verifier.verify_file(file_path)

        if not result.valid:
            raise VerificationError(
                f"Verification failed: {result.error}"
            )

        return json.loads(Path(file_path).read_text())
