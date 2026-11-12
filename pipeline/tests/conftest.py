import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="session")
def trainer():
    """gsplat's simple_trainer from the pinned checkout, imported on a CPU. Run the tests that take it with
    PLY_GSPLAT_DIR=<gsplat checkout at the pinned SHA>; without it (or without torch and pycolmap) they are skipped."""
    pytest.importorskip("torch")
    pytest.importorskip("pycolmap")
    root = os.environ.get("PLY_GSPLAT_DIR")
    ex = Path(root) / "examples" if root else None
    if not ex or not (ex / "simple_trainer.py").exists():
        pytest.skip("set PLY_GSPLAT_DIR to a gsplat checkout at the pinned SHA")
    import importlib.util
    if importlib.util.find_spec("fused_ssim") is None:
        sys.modules["fused_ssim"] = types.SimpleNamespace(fused_ssim=lambda *a, **k: None)
    sys.path.insert(0, str(ex))
    cwd = os.getcwd()
    os.chdir(ex)
    try:
        import simple_trainer as st
    finally:
        os.chdir(cwd)
    yield st
