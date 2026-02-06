"""
Secure Signer for IMMUNOS

Signs agent state files to ensure integrity and authenticity.
Uses Ed25519 for cryptographic signatures.
"""

import json
import hashlib
import base64
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


class SignatureError(Exception):
    """Raised when signing fails."""
    pass


@dataclass
class SignatureManifest:
    """Manifest containing signature metadata."""
    version: str = "1.0"
    agent_type: str = ""
    domain: str = ""
    state_hash: str = ""
    signature: str = ""
    signed_at: str = ""
    signed_by: str = ""
    chain_previous: str = ""
    algorithm: str = "ed25519"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "SignatureManifest":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SecureSigner:
    """
    Signs agent state files with Ed25519 or SHA-256 HMAC.

    Usage:
        signer = SecureSigner(private_key_path="/path/to/key")
        signer.sign_file("/path/to/bcell_state.json")
        # Creates /path/to/bcell_state.json.sig
    """

    def __init__(
        self,
        private_key_path: Optional[str] = None,
        private_key_bytes: Optional[bytes] = None,
        key_id: str = "default"
    ):
        """
        Initialize signer with private key.

        Args:
            private_key_path: Path to PEM-encoded Ed25519 private key
            private_key_bytes: Raw private key bytes (alternative to path)
            key_id: Identifier for the signing key
        """
        self.key_id = key_id
        self._private_key = None

        if HAS_CRYPTOGRAPHY:
            if private_key_path:
                self._load_key_from_file(private_key_path)
            elif private_key_bytes:
                self._load_key_from_bytes(private_key_bytes)
        # If no cryptography library, will use HMAC fallback

    def _load_key_from_file(self, path: str) -> None:
        """Load Ed25519 private key from PEM file."""
        try:
            key_data = Path(path).read_bytes()
            self._private_key = serialization.load_pem_private_key(key_data, password=None)
        except Exception as e:
            raise SignatureError(f"Failed to load private key: {e}")

    def _load_key_from_bytes(self, key_bytes: bytes) -> None:
        """Load Ed25519 private key from raw bytes."""
        try:
            self._private_key = serialization.load_pem_private_key(key_bytes, password=None)
        except Exception as e:
            raise SignatureError(f"Failed to load private key: {e}")

    def compute_hash(self, data: bytes) -> str:
        """Compute SHA-256 hash of data."""
        return "sha256:" + hashlib.sha256(data).hexdigest()

    def sign_data(self, data: bytes) -> str:
        """
        Sign data bytes.

        Returns base64-encoded signature.
        """
        if HAS_CRYPTOGRAPHY and self._private_key:
            signature = self._private_key.sign(data)
            return base64.b64encode(signature).decode('utf-8')
        else:
            # Fallback: HMAC-SHA256 with key_id as secret
            import hmac
            sig = hmac.new(
                self.key_id.encode(),
                data,
                hashlib.sha256
            ).digest()
            return "hmac:" + base64.b64encode(sig).decode('utf-8')

    def sign_file(
        self,
        file_path: str,
        agent_type: str = "unknown",
        domain: str = "unknown",
        chain_previous: str = ""
    ) -> SignatureManifest:
        """
        Sign an agent state file.

        Creates a .sig file alongside the original.

        Args:
            file_path: Path to the state file
            agent_type: Type of agent (bcell, nk, dendritic, memory)
            domain: Training domain (research, hallucination, network)
            chain_previous: Hash of previous state in chain

        Returns:
            SignatureManifest with signature details
        """
        path = Path(file_path)
        if not path.exists():
            raise SignatureError(f"File not found: {file_path}")

        # Read file content
        content = path.read_bytes()

        # Compute hash
        state_hash = self.compute_hash(content)

        # Create signing payload (hash + metadata)
        payload = json.dumps({
            "state_hash": state_hash,
            "agent_type": agent_type,
            "domain": domain,
            "chain_previous": chain_previous,
            "signed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }, sort_keys=True).encode()

        # Sign
        signature = self.sign_data(payload)

        # Create manifest
        manifest = SignatureManifest(
            version="1.0",
            agent_type=agent_type,
            domain=domain,
            state_hash=state_hash,
            signature=signature,
            signed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            signed_by=self.key_id,
            chain_previous=chain_previous,
            algorithm="ed25519" if (HAS_CRYPTOGRAPHY and self._private_key) else "hmac-sha256"
        )

        # Write signature file
        sig_path = Path(str(file_path) + ".sig")
        sig_path.write_text(json.dumps(manifest.to_dict(), indent=2))

        return manifest

    def sign_state(
        self,
        state_data: Dict[str, Any],
        agent_type: str = "unknown",
        domain: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Sign state data in memory.

        Returns state with embedded signature.
        """
        # Serialize state
        content = json.dumps(state_data, sort_keys=True).encode()

        # Compute hash
        state_hash = self.compute_hash(content)

        # Sign
        payload = json.dumps({
            "state_hash": state_hash,
            "agent_type": agent_type,
            "domain": domain,
            "signed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }, sort_keys=True).encode()

        signature = self.sign_data(payload)

        # Return state with signature
        return {
            **state_data,
            "_signature": {
                "version": "1.0",
                "state_hash": state_hash,
                "signature": signature,
                "signed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "signed_by": self.key_id,
                "algorithm": "ed25519" if (HAS_CRYPTOGRAPHY and self._private_key) else "hmac-sha256"
            }
        }


def generate_keypair(output_dir: str = ".") -> tuple:
    """
    Generate Ed25519 keypair for signing.

    Returns (private_key_path, public_key_path)
    """
    if not HAS_CRYPTOGRAPHY:
        raise SignatureError("cryptography library required for key generation")

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    # Generate keypair
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Save private key
    private_path = output / "immunos.key"
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    private_path.write_bytes(private_pem)
    private_path.chmod(0o600)  # Restrict permissions

    # Save public key
    public_path = output / "immunos.pub"
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    public_path.write_bytes(public_pem)

    return str(private_path), str(public_path)
