from abc import ABC
from pyscf.dh.response.respbase import RespBase
from pyscf.dh.energy.nuc_corr.modeldftd3 import DFTD3Eng


class NucCorrResp(RespBase, ABC):

    @property
    def restricted(self):
        return None

    def make_lag_vo(self):
        return 0

    def make_rdm1_resp(self, ao_repr=False):
        return 0


class DFTD3Resp(NucCorrResp, DFTD3Eng):
    pass
