"""
A Python wrapper to the LibXC compiled object use ctypes.

Extracted from libxc (commit 28c908e86eb74a54060e429ce472964258a16305)
https://gitlab.com/libxc/libxc
License of libxc: Mozilla Public License Version 2.0

Modification to pylibxc
- changed __libxc_path in
"""

from .core import core, get_core_path
from .functional import LibXCFunctional

from . import util
from . import version
