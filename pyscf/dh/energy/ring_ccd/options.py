integral_scheme_ring_ccd = None
""" Flag for ring-CCD integral.

By default, it is set to be the same of ``integral_scheme``.

See Also
--------
pyscf.dh.energy.options.integral_scheme
"""

omega_list_ring_ccd = [0]
""" Range-separate omega list of ring-CCD.

Zero refers to no range-separate. Long/Short range uses posi/negative values.
"""

tol_eng_ring_ccd = 1e-8
""" Threshold of ring-CCD energy difference while in DIIS update. """

tol_amp_ring_ccd = 1e-6
""" Threshold of L2 norm of ring-CCD amplitude while in DIIS update. """

max_cycle_ring_ccd = 64
""" Maximum iteration of ring-CCD iteration. """
