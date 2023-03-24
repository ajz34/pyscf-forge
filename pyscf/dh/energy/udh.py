from pyscf.dh.energy import UDHBase
from pyscf.dh.energy.rdh import RDH
from pyscf.dh.energy.udft import kernel_energy_unrestricted_exactx, kernel_energy_unrestricted_noxc


class UDH(UDHBase, RDH):

    kernel_energy_exactx = staticmethod(kernel_energy_unrestricted_exactx)
    kernel_energy_noxc = staticmethod(kernel_energy_unrestricted_noxc)
