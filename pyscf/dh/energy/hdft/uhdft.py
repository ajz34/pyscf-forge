from pyscf.dh.energy.hdft.rhdft import RHDFT
from pyscf.dh import util
import numpy as np


def get_energy_unrestricted_exactx(mf, dm, omega=None):
    """ Evaluate exact exchange energy (for either HF and long-range).

    Parameters
    ----------
    mf : dft.rks.RKS
        SCF object that have ``get_k`` member function.
    dm : np.ndarray
        Density matrix.
    omega : float or None
        Parameter of long-range ERI integral :math:`\\mathrm{erfc} (\\omega r_{12}) / r_{12}`.

    See Also
    --------
    pyscf.dh.energy.rdft.kernel_energy_restricted_exactx
    """
    hermi = 1 if np.allclose(dm, dm.swapaxes(-1, -2).conj()) else 0
    if omega == 0:
        vk = mf.get_k(dm=dm, hermi=hermi)
    else:
        vk = mf.get_k(dm=dm, hermi=hermi, omega=omega)
    ex = - 0.5 * np.einsum('sij, sji ->', dm, vk)
    ex = util.check_real(ex)
    # results
    result = dict()
    omega = omega if omega is not None else 0
    result[util.pad_omega("eng_exx_HF", omega)] = ex
    return result


def get_energy_unrestricted_noxc(mf, dm):
    """ Evaluate energy contributions that is not exchange-correlation.

    Note that some contributions (such as vdw) is not considered.

    Parameters
    ----------
    mf : dft.rks.RKS
        SCF object that have ``get_hcore``, ``get_j`` member functions.
    dm : np.ndarray
        Density matrix.

    See Also
    --------
    pyscf.dh.energy.rdft.kernel_energy_restricted_noxc
    """
    hermi = 1 if np.allclose(dm, dm.swapaxes(-1, -2).conj()) else 0
    hcore = mf.get_hcore()
    vj = mf.get_j(dm=dm, hermi=hermi)
    eng_nuc = mf.mol.energy_nuc()
    eng_hcore = np.einsum('sij, ji ->', dm, hcore)
    eng_j = 0.5 * np.einsum('ij, ji ->', dm.sum(axis=0), vj.sum(axis=0))
    eng_hcore = util.check_real(eng_hcore)
    eng_j = util.check_real(eng_j)
    eng_noxc = eng_hcore + eng_nuc + eng_j
    # results
    results = dict()
    results["eng_nuc"] = eng_nuc
    results["eng_hcore"] = eng_hcore
    results["eng_j"] = eng_j
    results["eng_noxc"] = eng_noxc
    return results


class UHDFT(RHDFT):
    """ Unrestricted hybrid (low-rung) DFT wrapper class of convenience. """

    get_energy_exactx = staticmethod(get_energy_unrestricted_exactx)
    get_energy_noxc = staticmethod(get_energy_unrestricted_noxc)


if __name__ == '__main__':
    def main_1():
        from pyscf import gto, scf
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", charge=1, spin=1).build()
        mf_s = scf.UHF(mol)
        mf = UHDFT(mf_s, xc="HF, LYP").run()
        res = mf.make_energy_purexc([", LYP", "B88, ", "HF", "LR_HF(0.5)", "SSR(GGA_X_B88, 0.5), "])
        print(res)

    main_1()

