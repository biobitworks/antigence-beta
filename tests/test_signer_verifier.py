"""
Falsification tests for SecureSigner and SecureVerifier.

Core claims:
- Sign/verify roundtrip works (HMAC fallback path)
- Tampered data is rejected
- Wrong key is rejected
- File signing creates .sig and verification detects changes
"""

import json
import tempfile
from pathlib import Path

import pytest

from immunos_mcp.security.signer import SecureSigner, SignatureManifest
from immunos_mcp.security.verifier import SecureVerifier, SecureAgentLoader, VerificationError


class TestHMACSignVerify:
    """Test HMAC-SHA256 fallback path (no cryptography library needed)."""

    def test_sign_verify_roundtrip(self):
        """Signed data should verify with the same key."""
        signer = SecureSigner(key_id="test-key-1")
        verifier = SecureVerifier(key_id="test-key-1")

        data = b"hello world this is a test payload"
        signature = signer.sign_data(data)
        assert signature.startswith("hmac:")
        assert verifier.verify_signature(data, signature) is True

    def test_tampered_data_rejected(self):
        """Modified data should fail verification."""
        signer = SecureSigner(key_id="test-key-1")
        verifier = SecureVerifier(key_id="test-key-1")

        original = b"original data"
        signature = signer.sign_data(original)

        tampered = b"tampered data"
        assert verifier.verify_signature(tampered, signature) is False

    def test_wrong_key_rejected(self):
        """Signature from key A should not verify with key B."""
        signer = SecureSigner(key_id="key-A")
        verifier = SecureVerifier(key_id="key-B")

        data = b"test data"
        signature = signer.sign_data(data)
        assert verifier.verify_signature(data, signature) is False

    def test_empty_data(self):
        """Empty data should still sign and verify."""
        signer = SecureSigner(key_id="test")
        verifier = SecureVerifier(key_id="test")

        signature = signer.sign_data(b"")
        assert verifier.verify_signature(b"", signature) is True


class TestHashComputation:
    """Test SHA-256 hashing."""

    def test_hash_deterministic(self):
        signer = SecureSigner(key_id="x")
        h1 = signer.compute_hash(b"test data")
        h2 = signer.compute_hash(b"test data")
        assert h1 == h2

    def test_hash_changes_with_data(self):
        signer = SecureSigner(key_id="x")
        h1 = signer.compute_hash(b"data1")
        h2 = signer.compute_hash(b"data2")
        assert h1 != h2

    def test_hash_format(self):
        signer = SecureSigner(key_id="x")
        h = signer.compute_hash(b"test")
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64  # "sha256:" + 64 hex chars


class TestFileSignVerify:
    """Test file signing and verification."""

    def test_sign_file_creates_sig(self):
        signer = SecureSigner(key_id="file-test")
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "agent_state.json"
            state_path.write_text(json.dumps({"patterns": [1, 2, 3]}))

            manifest = signer.sign_file(str(state_path), agent_type="bcell", domain="test")
            sig_path = Path(str(state_path) + ".sig")
            assert sig_path.exists()
            assert manifest.agent_type == "bcell"
            assert manifest.state_hash.startswith("sha256:")

    def test_verify_file_succeeds(self):
        signer = SecureSigner(key_id="verify-test")
        verifier = SecureVerifier(key_id="verify-test")

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "agent_state.json"
            state_path.write_text(json.dumps({"data": "test"}))
            signer.sign_file(str(state_path), agent_type="nk", domain="security")

            result = verifier.verify_file(str(state_path))
            assert result.valid is True
            assert result.state_hash_match is True
            assert result.signature_valid is True

    def test_verify_tampered_file_fails(self):
        signer = SecureSigner(key_id="tamper-test")
        verifier = SecureVerifier(key_id="tamper-test")

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "agent_state.json"
            state_path.write_text(json.dumps({"data": "original"}))
            signer.sign_file(str(state_path))

            # Tamper with the file
            state_path.write_text(json.dumps({"data": "TAMPERED"}))

            result = verifier.verify_file(str(state_path))
            assert result.valid is False
            assert result.state_hash_match is False

    def test_missing_sig_file_fails(self):
        verifier = SecureVerifier(key_id="x", require_signature=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "unsigned.json"
            state_path.write_text("{}")
            result = verifier.verify_file(str(state_path))
            assert result.valid is False

    def test_missing_sig_file_ok_when_not_required(self):
        verifier = SecureVerifier(key_id="x", require_signature=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "unsigned.json"
            state_path.write_text("{}")
            result = verifier.verify_file(str(state_path))
            assert result.valid is True


class TestStateSignVerify:
    """Test in-memory state signing."""

    def test_sign_state_embeds_signature(self):
        signer = SecureSigner(key_id="state-test")
        state = {"patterns": [1, 2, 3], "agent_name": "test"}
        signed = signer.sign_state(state, agent_type="bcell")
        assert "_signature" in signed
        assert signed["_signature"]["state_hash"].startswith("sha256:")

    def test_verify_signed_state(self):
        signer = SecureSigner(key_id="roundtrip")
        verifier = SecureVerifier(key_id="roundtrip")
        state = {"data": "test", "value": 42}
        signed = signer.sign_state(state)
        result = verifier.verify_state(signed)
        assert result.valid is True

    def test_verify_tampered_state(self):
        signer = SecureSigner(key_id="tamper")
        verifier = SecureVerifier(key_id="tamper")
        state = {"data": "original"}
        signed = signer.sign_state(state)
        signed["data"] = "TAMPERED"  # Modify after signing
        result = verifier.verify_state(signed)
        assert result.valid is False


class TestSecureAgentLoader:
    """Test the high-level loader."""

    def test_load_verified_file(self):
        signer = SecureSigner(key_id="loader-test")
        loader = SecureAgentLoader(require_signature=True)
        loader.verifier = SecureVerifier(key_id="loader-test")

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state_path.write_text(json.dumps({"agent": "bcell"}))
            signer.sign_file(str(state_path))

            data = loader.load(str(state_path))
            assert data["agent"] == "bcell"

    def test_load_tampered_raises(self):
        signer = SecureSigner(key_id="loader-tamper")
        loader = SecureAgentLoader(require_signature=True)
        loader.verifier = SecureVerifier(key_id="loader-tamper")

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state_path.write_text(json.dumps({"agent": "ok"}))
            signer.sign_file(str(state_path))
            state_path.write_text(json.dumps({"agent": "HACKED"}))

            with pytest.raises(VerificationError):
                loader.load(str(state_path))


class TestSignatureManifest:
    """Test manifest serialization."""

    def test_roundtrip(self):
        m = SignatureManifest(
            agent_type="bcell", domain="test",
            state_hash="sha256:abc", signature="hmac:xyz"
        )
        d = m.to_dict()
        m2 = SignatureManifest.from_dict(d)
        assert m2.agent_type == "bcell"
        assert m2.state_hash == "sha256:abc"
