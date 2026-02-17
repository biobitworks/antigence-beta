"""
Key Store for IMMUNOS

Manages cryptographic keys for signing and verification.
Supports file-based and (future) hardware key storage.
"""

import json
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, asdict
import time

from .signer import HAS_CRYPTOGRAPHY, generate_keypair


@dataclass
class KeyInfo:
    """Metadata about a stored key."""
    key_id: str
    key_type: str  # "ed25519", "hmac"
    created_at: str
    public_key_path: Optional[str] = None
    private_key_path: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "KeyInfo":
        return cls(**data)


class KeyStore:
    """
    Manages cryptographic keys for IMMUNOS signing.

    Features:
    - Generate new keypairs
    - List available keys
    - Get key paths for signer/verifier
    - (Future) Hardware key integration
    """

    def __init__(self, store_path: Optional[str] = None):
        """
        Initialize keystore.

        Args:
            store_path: Directory for key storage (default: ~/.immunos/keys/)
        """
        if store_path:
            self.store_path = Path(store_path)
        else:
            self.store_path = Path.home() / ".immunos" / "keys"

        self.store_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_path / "keystore.json"
        self._index: Dict[str, KeyInfo] = self._load_index()

    def _load_index(self) -> Dict[str, KeyInfo]:
        """Load key index from disk."""
        if not self.index_path.exists():
            return {}
        try:
            data = json.loads(self.index_path.read_text())
            return {
                key_id: KeyInfo.from_dict(info)
                for key_id, info in data.items()
            }
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_index(self) -> None:
        """Save key index to disk."""
        data = {
            key_id: info.to_dict()
            for key_id, info in self._index.items()
        }
        self.index_path.write_text(json.dumps(data, indent=2))

    def generate_key(
        self,
        key_id: str = "default",
        description: str = ""
    ) -> KeyInfo:
        """
        Generate a new Ed25519 keypair.

        Args:
            key_id: Unique identifier for this key
            description: Human-readable description

        Returns:
            KeyInfo with paths to generated keys
        """
        if not HAS_CRYPTOGRAPHY:
            # Fallback to HMAC-based "key" (just stores key_id)
            info = KeyInfo(
                key_id=key_id,
                key_type="hmac",
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                description=description or "HMAC key (cryptography library not available)"
            )
            self._index[key_id] = info
            self._save_index()
            return info

        # Generate Ed25519 keypair
        key_dir = self.store_path / key_id
        key_dir.mkdir(parents=True, exist_ok=True)

        private_path, public_path = generate_keypair(str(key_dir))

        info = KeyInfo(
            key_id=key_id,
            key_type="ed25519",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            private_key_path=private_path,
            public_key_path=public_path,
            description=description or f"Ed25519 key generated {time.strftime('%Y-%m-%d')}"
        )

        self._index[key_id] = info
        self._save_index()

        return info

    def get_key(self, key_id: str = "default") -> Optional[KeyInfo]:
        """Get key info by ID."""
        return self._index.get(key_id)

    def list_keys(self) -> Dict[str, KeyInfo]:
        """List all stored keys."""
        return self._index.copy()

    def get_private_key_path(self, key_id: str = "default") -> Optional[str]:
        """Get path to private key for signing."""
        info = self._index.get(key_id)
        if info and info.private_key_path:
            return info.private_key_path
        return None

    def get_public_key_path(self, key_id: str = "default") -> Optional[str]:
        """Get path to public key for verification."""
        info = self._index.get(key_id)
        if info and info.public_key_path:
            return info.public_key_path
        return None

    def delete_key(self, key_id: str) -> bool:
        """
        Delete a key and its files.

        Returns True if deleted, False if not found.
        """
        if key_id not in self._index:
            return False

        info = self._index[key_id]

        # Delete key files
        if info.private_key_path:
            try:
                Path(info.private_key_path).unlink(missing_ok=True)
            except OSError:
                pass

        if info.public_key_path:
            try:
                Path(info.public_key_path).unlink(missing_ok=True)
            except OSError:
                pass

        # Remove from index
        del self._index[key_id]
        self._save_index()

        return True

    def export_public_key(self, key_id: str = "default") -> Optional[bytes]:
        """Export public key bytes for distribution."""
        path = self.get_public_key_path(key_id)
        if path and Path(path).exists():
            return Path(path).read_bytes()
        return None

    def import_public_key(
        self,
        key_id: str,
        public_key_bytes: bytes,
        description: str = ""
    ) -> KeyInfo:
        """
        Import an external public key for verification.

        Args:
            key_id: ID to assign to this key
            public_key_bytes: PEM-encoded public key
            description: Description of key source

        Returns:
            KeyInfo for imported key
        """
        key_dir = self.store_path / key_id
        key_dir.mkdir(parents=True, exist_ok=True)

        public_path = key_dir / "immunos.pub"
        public_path.write_bytes(public_key_bytes)

        info = KeyInfo(
            key_id=key_id,
            key_type="ed25519",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            public_key_path=str(public_path),
            description=description or "Imported public key"
        )

        self._index[key_id] = info
        self._save_index()

        return info
