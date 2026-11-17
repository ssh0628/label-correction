"""Atomic JSON and NumPy writes shared by training and offline metrics."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    from torch import Tensor
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _commit(temporary: Path, destination: Path, overwrite: bool) -> None:
    if overwrite:
        os.replace(temporary, destination)
    else:
        os.link(temporary, destination)
        temporary.unlink()


def read_json(path: Path | str, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _temporary_file(path: Path | str, overwrite: bool):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        _commit(temporary, path, overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path | str, payload, *, overwrite: bool = False) -> Path:
    with _temporary_file(path, overwrite) as temporary:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2,
                      default=_json_default, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    return Path(path)


def atomic_save_npz(
    path: Path | str, *, compressed: bool = True, overwrite: bool = False, **arrays,
) -> None:
    with _temporary_file(path, overwrite) as temporary:
        with temporary.open("wb") as stream:
            save = np.savez_compressed if compressed else np.savez
            save(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())


def atomic_torch_save(payload, path: Path | str) -> None:
    import torch
    with _temporary_file(path, overwrite=True) as temporary:
        torch.save(payload, temporary)


class JsonlWriter:
    """Write one JSON record per line; a missing path disables only recording."""

    def __init__(self, path: Path | str | None, *, append: bool = False, flush: bool = True):
        self._handle = None
        self._flush = flush
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a" if append else "w", encoding="utf-8")

    def write(self, values) -> None:
        if self._handle is not None:
            self._handle.write(json.dumps(values, default=_json_default, ensure_ascii=False, allow_nan=False) + "\n")
            if self._flush:
                self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
