"""Small shared helpers: device resolution, content hashing, timing."""
from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def resolve_device(spec: str = "auto") -> str:
    """Map ``auto|cpu|cuda|mps`` onto something torch can actually use.

    Every extractor takes ``--device``; the default resolves to CUDA when a GPU appears, so the
    same command works unchanged on this laptop and on a GPU box.
    """
    if spec != "auto":
        return spec
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def content_hash(obj: Any) -> str:
    """Stable short hash of any JSON-serialisable object, for cache keys."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def file_hash(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()[:12]


@contextmanager
def timed(label: str) -> Iterator[dict]:
    """Wall-clock timer. The assignment asks for runtimes, so we measure rather than estimate."""
    rec: dict = {"label": label}
    t0 = time.perf_counter()
    print(f"[{label}] start", flush=True)
    try:
        yield rec
    finally:
        rec["seconds"] = time.perf_counter() - t0
        print(f"[{label}] done in {rec['seconds']:.1f}s", flush=True)
