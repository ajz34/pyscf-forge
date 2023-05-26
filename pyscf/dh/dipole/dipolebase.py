""" Base class of dipole electric field. """

from pyscf.dh.response import RespBase
from abc import ABC

α, β = 0, 1
αα, αβ, ββ = 0, 1, 2


class DipoleBase(RespBase, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scf_prop = NotImplemented

    @staticmethod
    def pad_prop(s):
        """ Pad string of property at last of string """
        return s + "_dipole"

    @property
    def nprop(self):
        return self.make_hcore_1_ao().shape[0]

    @property
    def scf_prop(self):
        if self._scf_prop is NotImplemented:
            from pyscf.dh.dipole.hdft.rhdft import RSCFDipole
            from pyscf.dh.dipole.hdft.uhdft import USCFDipole
            SCFDipole = RSCFDipole if self.restricted else USCFDipole
            self._scf_prop = SCFDipole.from_cls(self, self.scf, copy_all=True)
        return self._scf_prop

    @scf_prop.setter
    def scf_prop(self, scf_prop):
        self._scf_prop = scf_prop

    def make_hcore_1_ao(self):
        """ Generate first-order derivative of core hamiltonian in AO-basis. """
        return self.scf_prop.make_hcore_1_ao()

    def make_hcore_1_mo(self):
        """ Generate first-order skeleton-derivative of core hamiltonian in MO-basis. """
        return self.scf_prop.make_hcore_1_mo()

    def make_U_1(self):
        """ Generate first-order derivative of molecular coefficient. """
        return self.scf_prop.make_U_1()
