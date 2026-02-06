"""
Ollama-backed embedder for offline semantic embeddings.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Dict, List

import numpy as np


class OllamaEmbedder:
    """
    Embed text using a local Ollama server.
    """

    def __init__(self,
                 model: str = "nomic-embed-text",
                 base_url: str | None = None,
                 timeout_s: float = 30.0):
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = int(os.environ.get("OLLAMA_EMBED_MAX_RETRIES", "3"))
        self.backoff_s = float(os.environ.get("OLLAMA_EMBED_BACKOFF_S", "0.2"))
        self.backoff_multiplier = float(os.environ.get("OLLAMA_EMBED_BACKOFF_MULTIPLIER", "2.0"))
        self.max_chars = int(os.environ.get("OLLAMA_EMBED_MAX_CHARS", "4000"))
        self._cache: Dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]

        if self.max_chars > 0 and len(text) > self.max_chars:
            vector = self._embed_chunked(text)
            self._cache[text] = vector
            return vector

        vector = self._embed_with_retry(text)
        self._cache[text] = vector
        return vector

    def _embed_with_retry(self, text: str) -> np.ndarray:
        backoff = self.backoff_s
        attempt = 0
        while True:
            try:
                return self._embed_once(text)
            except RuntimeError as exc:
                if attempt >= self.max_retries or not self._is_retryable(exc):
                    raise
                time.sleep(backoff)
                backoff *= self.backoff_multiplier
                attempt += 1

    def _is_retryable(self, exc: Exception) -> bool:
        cause = exc.__cause__
        if isinstance(cause, urllib.error.HTTPError):
            return 500 <= cause.code < 600
        if isinstance(cause, urllib.error.URLError):
            return True
        return False

    def _embed_once(self, text: str) -> np.ndarray:
        payload = json.dumps({
            "model": self.model,
            "prompt": text
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Ollama embedding request failed: HTTP {exc.code} {exc.reason}") from exc
        except Exception as exc:
            raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc

        embedding = data.get("embedding")
        if not embedding:
            raise RuntimeError("Ollama response missing embedding data")

        return np.array(embedding, dtype=np.float32)

    def _split_text(self, text: str) -> List[str]:
        if self.max_chars <= 0 or len(text) <= self.max_chars:
            return [text]

        sentences = []
        current = ""
        for chunk in text.split(". "):
            if not chunk:
                continue
            sentence = chunk if chunk.endswith(".") else f"{chunk}."
            if len(sentence) > self.max_chars:
                # Hard split oversized sentence.
                for i in range(0, len(sentence), self.max_chars):
                    sentences.append(sentence[i:i + self.max_chars])
                continue
            if len(current) + len(sentence) + 1 > self.max_chars:
                if current:
                    sentences.append(current.strip())
                current = sentence + " "
            else:
                current += sentence + " "

        if current.strip():
            sentences.append(current.strip())
        return sentences

    def _embed_chunked(self, text: str) -> np.ndarray:
        chunks = self._split_text(text)
        vectors = [self._embed_with_retry(chunk) for chunk in chunks]
        if len(vectors) == 1:
            return vectors[0]
        return np.mean(vectors, axis=0)
