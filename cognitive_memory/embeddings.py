"""Embedding backends for hybrid (dense + lexical) retrieval.

The default backend is Ollama's `nomic-embed-text` model (768-dim), served
locally at http://localhost:11434/api/embeddings. No external vector DB and
no heavy Python ML stack (no torch/numpy) are required — the plugin only
does an HTTP POST and Ollama runs the inference.

If the backend is unavailable (Ollama down, model not pulled, disabled by
config), retrieval degrades gracefully to lexical-only via `NoOpBackend`.
"""

from __future__ import annotations

import abc
import json
import logging
import struct
import urllib.error
import urllib.request
from typing import List, Optional

logger = logging.getLogger("cognitive-memory.embeddings")

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_URL = "http://localhost:11434/api/embeddings"
# nomic-embed-text produces 768-dim vectors.
DEFAULT_DIM = 768


def pack_vector(vec: List[float]) -> bytes:
    """Pack a list[float] into a little-endian Float32 BLOB (no numpy)."""
    return struct.pack("<%df" % len(vec), *vec)


def unpack_vector(blob: bytes) -> List[float]:
    """Inverse of pack_vector."""
    return list(struct.unpack("<%df" % (len(blob) // 4), blob))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity, clamped to [-1, 1]. Returns 0.0 on degenerate input."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


class EmbeddingBackend(abc.ABC):
    """Abstract embedding provider."""

    model: str = ""
    dim: int = 0

    @property
    @abc.abstractmethod
    def available(self) -> bool:
        """True if the backend can serve embeddings right now."""

    @abc.abstractmethod
    def embed(self, text: str) -> Optional[List[float]]:
        """Return the embedding for one text, or None if unavailable."""


class NoOpBackend(EmbeddingBackend):
    """Fallback backend: embeddings disabled/unavailable -> lexical-only."""

    model = "noop"
    dim = 0

    @property
    def available(self) -> bool:
        return False

    def embed(self, text: str) -> Optional[List[float]]:
        return None


class OllamaEmbeddingBackend(EmbeddingBackend):
    """Embed via Ollama's /api/embeddings endpoint (local, no external API)."""

    def __init__(self, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL,
                 timeout: float = 30.0):
        self.model = model
        self._url = url
        self._timeout = timeout
        self.dim = DEFAULT_DIM
        self._checked = False
        self._reachable = False

    @property
    def available(self) -> bool:
        if not self._checked:
            self._reachable = self._ping()
            self._checked = True
        return self._reachable

    def _ping(self) -> bool:
        # Cheap reachability check: HEAD-ish via a tiny request to the tags API.
        try:
            req = urllib.request.Request(self._url.replace("/api/embeddings", "/api/tags"),
                                         method="GET")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return resp.status == 200
        except Exception as e:  # noqa: BLE001 - any failure => not available
            logger.debug("cognitive-memory: Ollama unreachable (%s)", e)
            return False

    def embed(self, text: str) -> Optional[List[float]]:
        if not text or not text.strip():
            return None
        if not self.available:
            return None
        payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            vec = body.get("embedding")
            if not vec:
                logger.warning("cognitive-memory: Ollama returned empty embedding")
                return None
            return [float(x) for x in vec]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning(
                    "cognitive-memory: embedding model '%s' not pulled in Ollama. "
                    "Run: ollama pull %s", self.model, self.model)
            else:
                logger.warning("cognitive-memory: Ollama embed HTTP %s", e.code)
            return None
        except Exception as e:  # noqa: BLE001 - degrade, don't crash retrieval
            logger.warning("cognitive-memory: Ollama embed failed (%s)", e)
            return None


def get_embedding_backend(
    enabled: bool = True,
    model: str = DEFAULT_MODEL,
    url: str = DEFAULT_URL,
) -> EmbeddingBackend:
    """Factory: return a real backend if enabled + reachable, else NoOp."""
    if not enabled:
        return NoOpBackend()
    backend = OllamaEmbeddingBackend(model=model, url=url)
    if backend.available:
        return backend
    logger.info(
        "cognitive-memory: embedding backend unavailable (Ollama down or model "
        "'%s' not pulled) — falling back to lexical-only retrieval", model)
    return NoOpBackend()
