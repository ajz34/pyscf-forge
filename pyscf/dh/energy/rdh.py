from pyscf.dh.energy import RDHBase
from pyscf.dh.energy.driver_energy import driver_energy_dh
from pyscf.dh.energy.rdft import (
    kernel_energy_restricted_exactx, kernel_energy_restricted_noxc, kernel_energy_vv10,
    kernel_energy_purexc, get_rho)
from typing import Tuple, List
from pyscf.dh.util import XCList


class RDH(RDHBase):

    inherited: List[Tuple[RDHBase, XCList]]
    """ Advanced correlation methods by inherited instances, accompanied with related exchange-correlation list. """

    def __init__(self, mol_or_mf, xc, params=None, flags=None):
        # make mol_or_mf and xc as non-optional parameters
        self.inherited = []
        super().__init__(mol_or_mf, xc, params=params, flags=flags)

    driver_energy_dh = driver_energy_dh
    kernel = driver_energy_dh

    kernel_energy_exactx = staticmethod(kernel_energy_restricted_exactx)
    kernel_energy_noxc = staticmethod(kernel_energy_restricted_noxc)
    kernel_energy_vv10 = staticmethod(kernel_energy_vv10)
    kernel_energy_purexc = staticmethod(kernel_energy_purexc)
    get_rho = staticmethod(get_rho)

    def to_mp2(self):
        from pyscf.dh import RMP2ofDH, UMP2ofDH
        return RMP2ofDH.from_rdh(self) if self.restricted else UMP2ofDH.from_rdh(self)

    def to_iepa(self):
        from pyscf.dh import RIEPAofDH, UIEPAofDH
        return RIEPAofDH.from_rdh(self) if self.restricted else UIEPAofDH.from_rdh(self)

    def to_ring_ccd(self):
        from pyscf.dh import RRingCCDofDH
        if not self.restricted:
            raise NotImplementedError
        return RRingCCDofDH.from_rdh(self)
