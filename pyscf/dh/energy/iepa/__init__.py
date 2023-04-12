from . import riepa, uiepa
from .riepa import RIEPAConv, RIEPARI
from .uiepa import UIEPAofDH


__doc__ = r"""
Restricted IEPA (independent electron-pair approximation)
    
Methods included in this pair occupied energy driver are

- IEPA (independent electron pair approximation)
- sIEPA (screened IEPA, using erfc function)
- DCPT2 (degeneracy-corrected second-order perturbation)
- MP2/cr (enhanced second-order treatment of electron pair)
- MP2 (as a basic pair method)

Parameters of these methods are controled by flags.

This function does not make checks, such as SCF convergence.
"""