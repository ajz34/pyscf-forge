""" Base class of response instances. """

from pyscf.dh.energy import EngBase
from pyscf import scf, __config__


CONFIG_max_cycle_cpks = getattr(__config__, "max_cycle_cpks", 20)
CONFIG_tol_cpks = getattr(__config__, "tol_cpks", 1e-9)


class RespBase(EngBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._Ax0_Core = NotImplemented
        self.max_cycle_cpks = CONFIG_max_cycle_cpks
        self.tol_cpks = CONFIG_tol_cpks

    @property
    def Ax0_Core(self):
        """ Fock response of underlying SCF object in MO basis. """
        if self._Ax0_Core is NotImplemented:
            restricted = isinstance(self.scf, scf.rhf.RHF)
            from pyscf.dh.response.hdft import RHDFTResp
            UHDFTResp = NotImplemented
            HDFTResp = RHDFTResp if restricted else UHDFTResp
            mf_scf = HDFTResp(self.scf)
            self._Ax0_Core = mf_scf.get_Ax0_Core
        return self._Ax0_Core

    @Ax0_Core.setter
    def Ax0_Core(self, Ax0_Core):
        self._Ax0_Core = Ax0_Core
