from pyscf.dh.dipole.dipolebase import DipoleBase
from pyscf.dh.response.resp_nuc_corr import NucCorrResp, DFTD3Resp
from abc import ABC


class NucCorrDipole(DipoleBase, NucCorrResp, ABC):

    def make_SCR3(self):
        return 0

    def make_pd_rdm1_corr(self):
        return 0


class DFTD3Dipole(NucCorrDipole, DFTD3Resp):
    pass
