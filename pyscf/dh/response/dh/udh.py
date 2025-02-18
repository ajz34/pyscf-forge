""" Doubly-Hybrid Response-Related Utilities. """
import numpy as np

from pyscf.dh.energy.dh import UDH
from pyscf.dh.response.dh.rdh import RDHResp

α, β = 0, 1
αα, αβ, ββ = 0, 1, 2


class UDHResp(UDH, RDHResp):

    @property
    def resp_type(self):
        return "resp"

    def make_lag_vo(self):

        if "lag_vo" in self.tensors:
            return self.tensors["lag_vo"]

        self.update_inherited(self.resp_type)

        nocc, nvir, nmo = self.nocc, self.nvir, self.nmo
        lag_vo = [np.zeros((nvir[σ], nocc[σ])) for σ in (α, β)]
        for key in self.inherited:
            for instance in self.inherited[key][1]:
                lag_vo_component = instance.make_lag_vo()
                if isinstance(lag_vo_component, (int, float)):
                    # some value is actually constant
                    lag_vo_component = (lag_vo_component, lag_vo_component)
                for σ in α, β:
                    lag_vo[σ] += lag_vo_component[σ]

        self.tensors["lag_vo"] = lag_vo
        return lag_vo

    def make_rdm1_resp(self, ao_repr=False):
        # prepare input
        if "rdm1_resp" in self.tensors:
            return self.tensors["rdm1_resp"]

        self.update_inherited(self.resp_type)

        rdm1_resp = np.array([np.diag(self.mo_occ[σ]) for σ in (α, β)])
        for key in self.inherited:
            for instance in self.inherited[key][1]:
                if hasattr(instance, "make_rdm1_corr"):
                    rdm1_resp += instance.make_rdm1_corr()

        lag_vo = self.make_lag_vo()
        rdm1_resp_vo = self.solve_cpks(lag_vo)
        mask_occ, mask_vir = self.mask_occ, self.mask_vir
        for σ in α, β:
            rdm1_resp[np.ix_([σ], mask_vir[σ], mask_occ[σ])] = rdm1_resp_vo[σ]

        self.tensors["rdm1_resp"] = rdm1_resp
        if ao_repr:
            rdm1_resp = np.array([self.mo_coeff[σ] @ rdm1_resp[σ] @ self.mo_coeff[σ].T for σ in (α, β)])
        return rdm1_resp

    to_scf = RDHResp.to_scf
    to_mp2 = RDHResp.to_mp2
    to_iepa = RDHResp.to_iepa
    to_ring_ccd = RDHResp.to_ring_ccd
    update_inherited = RDHResp.update_inherited
