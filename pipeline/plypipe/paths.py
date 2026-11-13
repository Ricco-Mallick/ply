"""Filesystem probes that must not raise.

A client that runs the Modal scripts locally does not mount the container's paths, and `/root` is typically
unreadable to a normal user. `pathlib` raises `PermissionError` from `is_dir()` in that case instead of returning
False, which took down every local `modal run pipeline/modal/ply_eval.py` before it started.
"""
from __future__ import annotations

import pathlib


def import_root(path: pathlib.Path, marker: str = "plypipe") -> bool:
    """Whether `path` can be added to `sys.path`: it holds a directory named `marker`, and saying so does not raise."""
    try:
        return (path / marker).is_dir()
    except OSError:  # EACCES on a parent the user cannot traverse, ELOOP, ENAMETOOLONG
        return False
