from pyscf.dh.dipole.dipolebase import DipoleBase, PolarBase
from pyscf.dh.response.hdft.rhdft import RSCFResp, RHDFTResp
from pyscf import lib, dft, gto
import numpy as np

α, β = 0, 1
αα, αβ, ββ = 0, 1, 2


def get_pd_fock_mo(
        fock_mo, hcore_1_mo, mo_occ,
        U_1, Ax0_Core,
        verbose=lib.logger.NOTE):
    """ Full derivative of Fock matrix in MO-basis by perturbation of dipole electric field.

    Note that this function is only used for electric properties, since derivative of ERI contribution is not counted.

    Parameters
    ----------
    fock_mo : np.ndarray
    hcore_1_mo : np.ndarray
    mo_occ : np.ndarray
    U_1 : np.ndarray
    Ax0_Core : callable
    verbose : int

    Returns
    -------
    np.ndarray
    """
    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    nocc = (mo_occ != 0).sum()
    nmo = len(mo_occ)
    so, sa = slice(0, nocc), slice(0, nmo)

    pd_fock_mo = (
        + hcore_1_mo
        + lib.einsum("Amp, mq -> Apq", U_1, fock_mo)
        + lib.einsum("Amq, pm -> Apq", U_1, fock_mo)
        + Ax0_Core(sa, sa, sa, so)(U_1[:, :, so]))

    log.timer("get_pd_fock_mo of dipole", *time0)
    return pd_fock_mo


class RSCFDipole(DipoleBase, RSCFResp):

    def make_hcore_1_ao(self):
        if self.pad_prop("hcore_1_ao") in self.tensors:
            return self.tensors[self.pad_prop("hcore_1_ao")]

        mol = self.mol
        hcore_1_ao = - mol.intor("int1e_r")
        self.tensors[self.pad_prop("hcore_1_ao")] = hcore_1_ao
        return hcore_1_ao

    def make_hcore_1_mo(self):
        if self.pad_prop("hcore_1_mo") in self.tensors:
            return self.tensors[self.pad_prop("hcore_1_mo")]

        hcore_1_ao = self.make_hcore_1_ao()
        if self.restricted:
            hcore_1_mo = self.mo_coeff.T @ hcore_1_ao @ self.mo_coeff
        else:
            hcore_1_mo = [self.mo_coeff[σ].T @ hcore_1_ao @ self.mo_coeff[σ] for σ in (α, β)]

        self.tensors[self.pad_prop("hcore_1_mo")] = hcore_1_mo
        return hcore_1_mo

    def make_U_1(self):
        """ Generate first-order derivative of molecular coefficient. """
        if self.pad_prop("U_1") in self.tensors:
            return self.tensors[self.pad_prop("U_1")]

        nocc, nmo = self.nocc, self.nmo
        so, sv = slice(0, nocc), slice(nocc, nmo)

        hcore_1_mo = self.make_hcore_1_mo()

        U_1_vo = self.solve_cpks(hcore_1_mo[:, sv, so])
        U_1 = np.zeros_like(hcore_1_mo)
        U_1[:, sv, so] = U_1_vo
        U_1[:, so, sv] = - U_1_vo.swapaxes(-1, -2)

        self.tensors[self.pad_prop("U_1")] = U_1
        return U_1

    def make_pd_fock_mo(self):
        if self.pad_prop("pd_fock_mo") in self.tensors:
            return self.tensors[self.pad_prop("pd_fock_mo")]

        mo_coeff = self.mo_coeff
        fock_mo = mo_coeff.T @ self.scf.get_fock() @ mo_coeff
        hcore_1_mo = self.make_hcore_1_mo()
        mo_occ = self.mo_occ
        U_1 = self.make_U_1()
        Ax0_Core = self.Ax0_Core
        verbose = self.verbose
        pd_fock_mo = get_pd_fock_mo(
            fock_mo=fock_mo,
            hcore_1_mo=hcore_1_mo,
            mo_occ=mo_occ,
            U_1=U_1,
            Ax0_Core=Ax0_Core,
            verbose=verbose)

        self.tensors[self.pad_prop("pd_fock_mo")] = pd_fock_mo
        return pd_fock_mo

    def make_SCR3(self):
        return 0

    def make_pd_rdm1_corr(self):
        return 0


class RHDFTDipole(DipoleBase, RHDFTResp):
    def make_pd_fock_mo(self):
        if self.pad_prop("pd_fock_mo") in self.tensors:
            return self.tensors[self.pad_prop("pd_fock_mo")]

        mo_coeff = self.mo_coeff
        fock_mo = mo_coeff.T @ self.hdft.get_fock() @ mo_coeff
        hcore_1_mo = self.make_hcore_1_mo()
        mo_occ = self.mo_occ
        U_1 = self.make_U_1()
        Ax0_Core = self.make_Ax0_Core
        verbose = self.verbose
        pd_fock_mo = get_pd_fock_mo(
            fock_mo=fock_mo,
            hcore_1_mo=hcore_1_mo,
            mo_occ=mo_occ,
            U_1=U_1,
            Ax0_Core=Ax0_Core,
            verbose=verbose)

        self.tensors[self.pad_prop("pd_fock_mo")] = pd_fock_mo
        return pd_fock_mo

    def make_SCR3(self):
        nocc, nmo = self.nocc, self.nmo
        so, sv = slice(0, nocc), slice(nocc, nmo)
        return 4 * self.make_pd_fock_mo()[:, sv, so]

    def make_pd_rdm1_corr(self):
        return np.zeros((self.nprop, self.nmo, self.nmo))


class RSCFPolar(RSCFResp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deriv_dipole = RSCFDipole.from_cls(self, self.scf, copy_all=True)

    def make_polar(self):
        if "polar" in self.tensors:
            return self.tensors["polar"]

        nocc = self.nocc
        so = slice(0, nocc)
        hcore_1_mo = self.deriv_dipole.make_hcore_1_mo()
        U_1 = self.deriv_dipole.make_U_1()
        pol_scf = - 4 * lib.einsum("Api, Bpi -> AB", hcore_1_mo[:, :, so], U_1[:, :, so])

        self.tensors["polar"] = pol_scf
        return pol_scf

    @property
    def de(self):
        return self.make_polar()

    kernel = make_polar


class RHDFTPolar(RHDFTResp, PolarBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deriv_dipole = RHDFTDipole.from_cls(self, self.scf, copy_all=True)

    @property
    def de(self):
        return self.make_polar()
