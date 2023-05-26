from pyscf.dh.dipole.dipolebase import DipoleBase
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


def get_Ax1_contrib_pure_dft(
        ni, mol, grids, xc, dm, dmU, dmR,
        max_memory=2000, verbose=lib.logger.NOTE):
    """ Evaluate contribution to property from Fock second-order response to MO coefficients.

    Parameters
    ----------
    ni : dft.numint.NumInt
    mol : gto.Mole
    grids : dft.Grids
    xc : str
    dm : np.ndarray
    dmU : np.ndarray
    dmR : np.ndarray
    max_memory : float
    verbose : int

    Returns
    -------
    np.ndarray
    """
    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    assert dm.ndim == 2
    assert dmU.ndim == 3
    assert dmR.ndim == 2

    xctype = ni._xc_type(xc)
    if xctype == "HF":
        return 0

    nprop = dmU.shape[0]
    prop = np.zeros((nprop, nprop))

    # todo: check with_lapl if pyscf updates its meta-GGA evaluation scheme
    ao_deriv = 0 if xctype == "LDA" else 1

    idx_grid_start, idx_grid_stop = 0, 0
    for ao, _, weights, _ in ni.block_loop(mol, grids, mol.nao, deriv=ao_deriv, max_memory=max_memory):
        idx_grid_stop = idx_grid_start + len(weights)
        rho = ni.eval_rho(mol, ao, dm, xctype=xctype, with_lapl=False)
        rho_R = ni.eval_rho(mol, ao, dmR, xctype=xctype, with_lapl=False)
        rho_U = np.zeros(tuple([nprop] + list(rho.shape)))
        for idx in range(nprop):
            rho_U[idx] = ni.eval_rho(mol, ao, dmU[idx], xctype=xctype, with_lapl=False)
        _, _, _, kxc = ni.eval_xc_eff(xc, rho, deriv=3, omega=ni.omega)
        if xctype == "LDA":
            kxc = kxc.reshape(-1)
            prop += 2 * lib.einsum("g, Ag, Bg, g, g -> AB", kxc, rho_U, rho_U, rho_R, weights)
        else:
            prop += 2 * lib.einsum("tsrg, Atg, Bsg, rg, g -> AB", kxc, rho_U, rho_U, rho_R, weights)
        idx_grid_start = idx_grid_stop

    log.timer("get_Ax1_contrib_pure_dft of dipole", *time0)
    return prop


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


class RHDFTPolar(RHDFTResp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deriv_dipole = RHDFTDipole.from_cls(self, self.scf, copy_all=True)

    def make_polar(self):
        if "polar" in self.tensors:
            return self.tensors["polar"]

        nocc, nmo = self.nocc, self.nmo
        so, sv, sa = slice(0, nocc), slice(nocc, nmo), slice(0, nmo)
        mo_coeff = self.mo_coeff

        rdm1_corr_resp = self.make_rdm1_resp() - np.diag(self.mo_occ)
        hcore_1_mo = self.deriv_dipole.make_hcore_1_mo()
        U_1 = self.deriv_dipole.make_U_1()
        SCR1 = self.Ax0_Core(sa, sa, sa, sa)(rdm1_corr_resp)
        SCR2 = hcore_1_mo + self.Ax0_Core(sa, sa, sv, so)(U_1[:, sv, so])
        SCR3 = 4 * self.deriv_dipole.make_pd_fock_mo()[:, sv, so]
        pd_fock_mo_scf = self.deriv_dipole.scf_prop.make_pd_fock_mo()

        dmU = mo_coeff @ U_1[:, :, so] @ mo_coeff[:, so].T
        dmU += dmU.swapaxes(-1, -2)
        dmR = mo_coeff @ rdm1_corr_resp @ mo_coeff.T
        dmR += dmR.swapaxes(-1, -2)
        ax1_contrib_pure_dft = get_Ax1_contrib_pure_dft(
            self.scf._numint, self.mol, self.scf.grids, self.scf.xc, self.scf.make_rdm1(), dmU, dmR,
            max_memory=2000, verbose=self.verbose)

        polar_scf = - 4 * lib.einsum("Api, Bpi -> AB", hcore_1_mo[:, :, so], U_1[:, :, so])
        polar_corr = - (
            + lib.einsum("Aai, Bma, mi -> AB", U_1[:, sv, so], U_1[:, :, sv], SCR1[:, so])
            + lib.einsum("Aai, Bmi, ma -> AB", U_1[:, sv, so], U_1[:, :, so], SCR1[:, sv])
            + lib.einsum("Apm, Bmq, pq -> AB", SCR2, U_1, rdm1_corr_resp)
            + lib.einsum("Amq, Bmp, pq -> AB", SCR2, U_1, rdm1_corr_resp)
            + lib.einsum("Bai, Aai -> AB", SCR3, U_1[:, sv, so])
            - lib.einsum("Bki, Aai, ak -> AB", pd_fock_mo_scf[:, so, so], U_1[:, sv, so], rdm1_corr_resp[sv, so])
            + lib.einsum("Bca, Aai, ci -> AB", pd_fock_mo_scf[:, sv, sv], U_1[:, sv, so], rdm1_corr_resp[sv, so])
            + ax1_contrib_pure_dft
        )
        self.tensors["pol_scf"] = polar_scf
        self.tensors["polar_corr"] = polar_corr
        self.tensors["polar"] = polar_scf + polar_corr
        return self.tensors["polar"]

    @property
    def de(self):
        return self.make_polar()

    get_Ax1_contrib_pure_dft = staticmethod(get_Ax1_contrib_pure_dft)
