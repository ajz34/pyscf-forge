from pyscf.dh.energy import UDHBase
from pyscf.dh.energy.uiepa import driver_energy_uiepa
from pyscf.dh.energy.udft import kernel_energy_unrestricted_exactx, kernel_energy_unrestricted_noxc


class UDH(UDHBase):
    driver_energy_iepa = driver_energy_uiepa

    kernel_energy_exactx = staticmethod(kernel_energy_unrestricted_exactx)
    kernel_energy_noxc = staticmethod(kernel_energy_unrestricted_noxc)
