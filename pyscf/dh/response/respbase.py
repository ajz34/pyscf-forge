""" Base class of response instances. """

from pyscf.dh.energy import EngBase
from pyscf import scf, __config__, lib
from pyscf.scf import cphf
import numpy as np
from abc import ABC, abstractmethod

CONFIG_max_cycle_cpks = getattr(__config__, "max_cycle_cpks", 20)
CONFIG_tol_cpks = getattr(__config__, "tol_cpks", 1e-9)


class RespBase(EngBase, ABC):
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
            self._Ax0_Core = mf_scf.make_Ax0_Core
        return self._Ax0_Core

    @Ax0_Core.setter
    def Ax0_Core(self, Ax0_Core):
        self._Ax0_Core = Ax0_Core

    @abstractmethod
    def make_lag_vo(self):
        raise NotImplementedError

    @abstractmethod
    def make_rdm1_resp(self, ao=False):
        raise NotImplementedError

    def make_dipole(self):
        # prepare input
        mol = self.mol
        rdm1_ao = self.make_rdm1_resp(ao=True)
        int1e_r = mol.intor("int1e_r")

        dip_elec = - np.einsum("uv, tuv -> t", rdm1_ao, int1e_r)
        dip_nuc = np.einsum("A, At -> t", mol.atom_charges(), mol.atom_coords())
        dip = dip_elec + dip_nuc
        self.tensors["dipole"] = dip
        return dip


def get_rdm1_resp_vo_restricted(
        lag_vo, mo_energy, mo_occ, Ax0_Core,
        max_cycle=20, tol=1e-9, verbose=lib.logger.NOTE):
    r""" Solution of response 1-RDM vir-occ part by CP-KS equation (restricted).

    .. math::
        A'_{ai, bj} D_{bj} &= L_{ai}

    Parameters
    ----------
    lag_vo : np.ndarray
    mo_energy : np.ndarray
    mo_occ : np.ndarray
    Ax0_Core : callable
    max_cycle : int
    tol : float
    verbose : int

    Returns
    -------
    dict[str, np.ndarray]

    See Also
    --------
    get_rdm1_corr
    """

    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    # dimension definition and check sanity
    nocc = (mo_occ > 0).sum()
    nvir = (mo_occ == 0).sum()
    nmo = nocc + nvir
    assert lag_vo.shape == (nvir, nocc)
    assert len(mo_energy) == nmo
    assert len(mo_occ) == nmo

    # prepare essential matrices and slices
    so, sv = slice(0, nocc), slice(nocc, nmo)

    # cp-ks evaluation
    rdm1_vo = cphf.solve(
        Ax0_Core(sv, so, sv, so), mo_energy, mo_occ, lag_vo,
        max_cycle=max_cycle, tol=tol)[0]

    log.timer("get_rdm1_resp_vo_restricted", *time0)
    return rdm1_vo
