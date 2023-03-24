iepa_schemes = "IEPA"
""" Flag for IEPA-like schemes.

List including the following schemes is also accepted.

Parameters
----------
"MP2"
    MP2 as basic method.
"IEPA"
    IEPA (independent electron pair approximation).
"sIEPA"
    Screened IEPA.
"DCPT2"
    DCPT2 (degeneracy-corrected second-order perturbation).
"MP2cr"
    MP2/cr I (enhanced second-order treatment of electron pair).
"MP2cr2"
    MP2/cr II (not recommended, restricted only)
"""

omega_list_iepa = [0]
""" Range-separate omega list of MP2.

Zero refers to no range-separate. Long/Short range uses posi/negative values.
"""

integral_scheme_iepa = None
""" Flag for IEPA integral.

By default, it is set to be the same of ``integral_scheme``.

See Also
--------
pyscf.dh.energy.options.integral_scheme
"""

tol_eng_pair_iepa = 1e-10
""" Threshold of convergence of pair-energy in iteration of IEPA or sIEPA. """

max_cycle_pair_iepa = 64
""" Threshold of convergence of pair-energy in iteration of IEPA or sIEPA. """
