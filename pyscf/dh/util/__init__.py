"""
Utilities for pyscf.dh

File structure
--------------

- 1st Layer: Code without involvement of PySCF in principle

  - ``helper.py``: Simple helper functions.
    Layer functions or classes.

- 2nd Layer: Code with PySCF involvement but minimum involvement of 1st Layer

  - ``helper_pyscf.py``: Simple helper functions that requires PySCF.
  - ``df_addon.py``: Helper additional functions for density fitting
"""

from .helper import (
    calc_batch_size, gen_batch, gen_leggauss_0_1, gen_leggauss_0_inf, sanity_dimension, check_real, parse_incore_flag,
    pad_omega, allocate_array, update_results)
from .df_addon import get_cderi_mo, get_with_df_omega

from .helper_pyscf import (
    parse_frozen_numbers, parse_frozen_list, restricted_biorthogonalize, hermi_sum_last2dim)
from .xccode.xccode import XCInfo, XCList, XCDH
from .xccode.xctype import XCType
from .numint_addon import eval_xc_eff_ssr_generator, eval_xc_eff_ext_param_generator
from .frozen_core import FrozenCore, FrozenRules, FrozenRuleNone
