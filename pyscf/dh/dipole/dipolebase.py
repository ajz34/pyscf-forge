""" Base class of dipole electric field. """

from pyscf.dh.response import RespBase
from pyscf import lib
import numpy as np
from abc import ABC, abstractmethod

α, β = 0, 1
αα, αβ, ββ = 0, 1, 2


def get_Ax1_contrib_pure_dft_restricted(
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
        _, _, _, kxc = ni.eval_xc_eff(xc, rho, deriv=3)
        if xctype == "LDA":
            kxc = kxc.reshape(-1)
            prop += 2 * lib.einsum("g, Ag, Bg, g, g -> AB", kxc, rho_U, rho_U, rho_R, weights)
        else:
            prop += 2 * lib.einsum("tsrg, Atg, Bsg, rg, g -> AB", kxc, rho_U, rho_U, rho_R, weights)
        idx_grid_start = idx_grid_stop

    log.timer("get_Ax1_contrib_pure_dft of dipole", *time0)
    return prop


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
            self._scf_prop = SCFDipole.from_cls(self.scf_resp, self.scf, copy_all=True)
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

    @abstractmethod
    def make_SCR3(self):
        raise NotImplementedError

    @abstractmethod
    def make_pd_rdm1_corr(self):
        raise NotImplementedError


class PolarBase(RespBase, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deriv_dipole = NotImplemented  # type: DipoleBase

    def make_polar_restricted(self):
        if "polar" in self.tensors:
            return self.tensors["polar"]

        # todo: resolve attributes of self not passing into self.deriv_dipole
        self.deriv_dipole.__dict__.update(self.__dict__)

        nocc, nmo = self.nocc, self.nmo
        so, sv, sa = slice(0, nocc), slice(nocc, nmo), slice(0, nmo)
        mo_coeff = self.mo_coeff
        mo_occ = self.mo_occ

        rdm1_corr_resp = self.make_rdm1_resp(ao_repr=False) - np.diag(mo_occ)
        pd_rdm1_corr = self.deriv_dipole.make_pd_rdm1_corr()
        hcore_1_mo = self.deriv_dipole.make_hcore_1_mo()
        U_1 = self.deriv_dipole.make_U_1()
        SCR1 = self.Ax0_Core(sa, sa, sa, sa)(rdm1_corr_resp)
        SCR2 = hcore_1_mo + self.Ax0_Core(sa, sa, sv, so)(U_1[:, sv, so])
        SCR3 = self.deriv_dipole.make_SCR3()
        pd_fock_mo_scf = self.deriv_dipole.scf_prop.make_pd_fock_mo()

        if hasattr(self.scf, "_numint"):
            dmU = mo_coeff @ U_1[:, :, so] @ mo_coeff[:, so].T
            dmU += dmU.swapaxes(-1, -2)
            dmR = mo_coeff @ rdm1_corr_resp @ mo_coeff.T
            dmR += dmR.swapaxes(-1, -2)
            ax1_contrib_pure_dft = self.get_Ax1_contrib_pure_dft(
                self.scf._numint, self.mol, self.scf.grids, self.scf.xc, self.scf.make_rdm1(), dmU, dmR,
                max_memory=2000, verbose=self.verbose)
        else:
            ax1_contrib_pure_dft = 0

        polar_scf = - 4 * lib.einsum("Api, Bpi -> AB", hcore_1_mo[:, :, so], U_1[:, :, so])
        polar_corr = - (
            + lib.einsum("Aai, Bma, mi -> AB", U_1[:, sv, so], U_1[:, :, sv], SCR1[:, so])
            + lib.einsum("Aai, Bmi, ma -> AB", U_1[:, sv, so], U_1[:, :, so], SCR1[:, sv])
            + lib.einsum("Apm, Bmq, pq -> AB", SCR2, U_1, rdm1_corr_resp)
            + lib.einsum("Amq, Bmp, pq -> AB", SCR2, U_1, rdm1_corr_resp)
            + lib.einsum("Apq, Bpq -> AB", SCR2, pd_rdm1_corr)
            + lib.einsum("Bai, Aai -> AB", SCR3, U_1[:, sv, so])
            - lib.einsum("Bki, Aai, ak -> AB", pd_fock_mo_scf[:, so, so], U_1[:, sv, so], rdm1_corr_resp[sv, so])
            + lib.einsum("Bca, Aai, ci -> AB", pd_fock_mo_scf[:, sv, sv], U_1[:, sv, so], rdm1_corr_resp[sv, so])
            + ax1_contrib_pure_dft
        )

        self.tensors["pol_scf"] = polar_scf
        self.tensors["polar_corr"] = polar_corr
        self.tensors["polar"] = polar_scf + polar_corr
        return self.tensors["polar"]

    def get_Ax1_contrib_pure_dft(self, *args, **kwargs):
        if self.restricted:
            return self.get_Ax1_contrib_pure_dft_restricted(*args, **kwargs)
        else:
            raise NotImplementedError

    def make_polar(self):
        if self.restricted:
            return self.make_polar_restricted()
        else:
            raise NotImplementedError

    @property
    def de(self):
        return self.make_polar()

    get_Ax1_contrib_pure_dft_restricted = staticmethod(get_Ax1_contrib_pure_dft_restricted)
