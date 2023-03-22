omega_list_mp2 = [0]
""" Range-separate omega list of MP2.

Zero refers to no range-separate. Long/Short range uses posi/negative values.
"""

integral_scheme_mp2 = None
""" Flag for PT2 integral.

By default, it is set to be the same of ``integral_scheme``.

See Also
--------
pyscf.dh.energy.options.integral_scheme
"""

frac_num_mp2 = None
""" Fraction occupation number list for MP2 evaluation.

Should be list of floats, size as ``(nmo, )``.
"""

incore_t_ijab_mp2 = None
""" Flag for MP2 amplitude tensor :math:`t_{ij}^{ab}` stored in memory or disk.

Parameters
----------
True
    Store tensor in memory.
False
    Store tensor in disk.
None
    Do not store tensor in either disk or memory.
"auto"
    Leave program to judge whether tensor locates.
(int)
    If tensor size exceeds this size (in MBytes), then store in disk.
"""
