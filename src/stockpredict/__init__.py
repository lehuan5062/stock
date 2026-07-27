"""Vietnamese T+2 swing-trade stock predictor."""
from __future__ import annotations
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

# Must run before anything imports vnstock (all our vnstock imports are lazy, so
# importing any submodule lands here first). See _vnai_guard for why.
from ._vnai_guard import install as _install_vnai_guard

_install_vnai_guard()

__version__ = "0.1.0"
