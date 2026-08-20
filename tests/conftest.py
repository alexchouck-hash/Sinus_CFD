"""Pytest bootstrap for the Sinus_CFD regression suite (task A3).

Puts the repo root and ``src/`` on ``sys.path`` so tests can ``import
sinus_cfd.*`` and load thin CLI scripts by path, and exposes a fixture that
imports ``scripts/trim_nasopharynx_outlet.py`` as a module (its module-level
code is import-safe: no argparse, main() only runs under ``__main__``).

Run:  py -3.12 -m pytest tests/
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"

for _p in (str(REPO), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_script(name: str) -> ModuleType:
    """Import a scripts/<name>.py file as a standalone module (no side effects)."""
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_sinuscfd_script_{name}", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


import pytest  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def trim_module() -> ModuleType:
    """The trim_nasopharynx_outlet.py script imported as a module (Fix 1 + Fix 3)."""
    return _load_script("trim_nasopharynx_outlet")
