""" Base class of response instances. """

from pyscf.dh.energy import EngBase
from pyscf import scf, __config__, lib
from pyscf.scf import cphf, ucphf
import numpy as np
from abc import ABC, abstractmethod

CONFIG_max_cycle_cpks = getattr(__config__, "max_cycle_cpks", 20)
CONFIG_tol_cpks = getattr(__config__, "tol_cpks", 1e-9)

α, β = 0, 1
αα, αβ, ββ = 0, 1, 2


class RespBase(EngBase, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._scf_resp = NotImplemented
        self._Ax0_Core = NotImplemented
        self.max_cycle_cpks = CONFIG_max_cycle_cpks
        self.tol_cpks = CONFIG_tol_cpks
        self.base = None

    @property
    def scf_resp(self):
        if self._scf_resp is NotImplemented:
            restricted = self.restricted
            from pyscf.dh.response.hdft.rhdft import RSCFResp
            from pyscf.dh.response.hdft.uhdft import USCFResp
            SCFResp = RSCFResp if restricted else USCFResp
            scf_resp = SCFResp(self.scf)
            self._scf_resp = scf_resp
        return self._scf_resp

    @scf_resp.setter
    def scf_resp(self, scf_resp):
        self._scf_resp = scf_resp

    @property
    def Ax0_Core(self):
        """ Fock response of underlying SCF object in MO basis. """
        if self._Ax0_Core is NotImplemented:
            self._Ax0_Core = self.scf_resp.Ax0_Core
        return self._Ax0_Core

    @Ax0_Core.setter
    def Ax0_Core(self, Ax0_Core):
        self._Ax0_Core = Ax0_Core

    @abstractmethod
    def make_lag_vo(self):
        raise NotImplementedError

    @abstractmethod
    def make_rdm1_resp(self, ao_repr=False):
        raise NotImplementedError

    def make_dipole(self):
        # prepare input
        mol = self.mol
        rdm1_ao = self.make_rdm1_resp(ao_repr=True)
        int1e_r = mol.intor("int1e_r")
        restricted = isinstance(self.scf, scf.rhf.RHF)
        rdm1_ao = rdm1_ao if restricted else rdm1_ao.sum(axis=0)
        dip_elec = - np.einsum("uv, tuv -> t", rdm1_ao, int1e_r)
        dip_nuc = np.einsum("A, At -> t", mol.atom_charges(), mol.atom_coords())
        dip = dip_elec + dip_nuc
        self.tensors["dipole"] = dip
        return dip

    def solve_cpks(self, rhs):
        """ Solve CP-KS equation by given RHS (right-hand-side of equation). """
        log = lib.logger.new_logger(verbose=self.verbose)
        time0 = lib.logger.process_clock(), lib.logger.perf_counter()

        # special case handling: RHS is zero
        # zero value by definition
        if isinstance(rhs, int) and rhs == 0:
            return 0
        if isinstance(rhs, tuple) and isinstance(rhs[0], int) and all([r == 0 for r in rhs]):
            return rhs

        # zero by computation
        if self.restricted and np.abs(rhs).max() < self.tol_cpks:
            return np.zeros_like(rhs)
        elif not self.restricted:
            is_zero = True
            for σ in α, β:
                if not np.abs(rhs[σ]).max() < self.tol_cpks:
                    is_zero = False
            if is_zero:
                return [np.zeros_like(rhs[σ]) for σ in (α, β)]

        restricted = isinstance(self.scf, scf.rhf.RHF)
        Ax0_Core = self.Ax0_Core
        so, sv = self.mask_occ, self.mask_vir
        nocc, nvir = self.nocc, self.nvir
        mo_energy = self.mo_energy
        mo_occ = self.mo_occ
        max_cycle = self.max_cycle_cpks
        tol = self.tol_cpks

        if restricted:
            res = cphf.solve(
                Ax0_Core(sv, so, sv, so), mo_energy, mo_occ, rhs,
                max_cycle=max_cycle, tol=tol)[0]
        else:
            # need to reshape X and rhs

            def reshape_inner(X):
                X_shape = X.shape
                X = X.reshape(-1, X.shape[-1])
                nprop = X.shape[0]
                Xα = X[:, :nocc[α]*nvir[α]].reshape(nprop, nvir[α], nocc[α])
                Xβ = X[:, nocc[α]*nvir[α]:].reshape(nprop, nvir[β], nocc[β])
                res = self.Ax0_Core(sv, so, sv, so)((Xα, Xβ))
                flt = np.zeros_like(X)
                for prop, res_pair in enumerate(zip(*res)):
                    flt[prop] = np.concatenate([m.reshape(-1) for m in res_pair])
                flt.shape = X_shape
                return flt

            res = ucphf.solve(
                reshape_inner, mo_energy, mo_occ, rhs,
                max_cycle=max_cycle, tol=tol)[0]

        log.timer("solve_cpks", *time0)
        return res
