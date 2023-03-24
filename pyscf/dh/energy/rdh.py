from pyscf.dh.energy import RDHBase
from typing import Tuple, List
from pyscf.dh.util import XCList


class RDH(RDHBase):

    inherited: List[Tuple[RDHBase, XCList]]
    """ Advanced correlation methods by inherited instances, accompanied with related exchange-correlation list. """

    def __init__(self, mol_or_mf, xc, params=None, flags=None):
        # make mol_or_mf and xc as non-optional parameters
        self.inherited = []
        super().__init__(mol_or_mf, xc, params=params, flags=flags)

    def kernel(self, **kwargs):
        with self.params.temporary_flags(kwargs):
            results = self.driver_energy_dh()
        self.params.update_results(results)
        return results

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
