"""
IMMUNOS Security Module

Cryptographic protection for the "self" definition (trained agent states).
Implements signing, verification, and audit trails.
"""

from .signer import SecureSigner, SignatureError
from .verifier import SecureVerifier, VerificationError
from .keystore import KeyStore

__all__ = [
    "SecureSigner",
    "SecureVerifier",
    "KeyStore",
    "SignatureError",
    "VerificationError",
]
