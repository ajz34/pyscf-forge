from pyscf.dh.energy import RDHBase
from pyscf.dh.energy.driver_energy import driver_energy_dh
from pyscf.dh.energy.iepa.riepa import driver_energy_riepa
from pyscf.dh.energy.rrpa import driver_energy_rring_ccd
from pyscf.dh.energy.rdft import (
    kernel_energy_restricted_exactx, kernel_energy_restricted_noxc, kernel_energy_vv10,
    kernel_energy_purexc, get_rho)


class RDH(RDHBase):

    def __init__(self, mol_or_mf, xc, params=None):
        # make mol_or_mf and xc as non-optional parameters
        super().__init__(mol_or_mf, xc, params)

    driver_energy_iepa = driver_energy_riepa
    driver_energy_ring_ccd = driver_energy_rring_ccd
    driver_energy_dh = driver_energy_dh
    kernel = driver_energy_dh

    kernel_energy_exactx = staticmethod(kernel_energy_restricted_exactx)
    kernel_energy_noxc = staticmethod(kernel_energy_restricted_noxc)
    kernel_energy_vv10 = staticmethod(kernel_energy_vv10)
    kernel_energy_purexc = staticmethod(kernel_energy_purexc)
    get_rho = staticmethod(get_rho)
