import numpy as np

from pyscf.dh.dipole.dipolebase import DipoleBase, PolarBase
from pyscf.dh.dipole.dh.rdh import RDHDipole
from pyscf.dh.energy.dh import UDH
from pyscf.dh.response.dh.udh import UDHResp


class UDHDipole(UDHResp, RDHDipole):
    pass


class RDHPolar(UDHResp, PolarBase):
    pass
