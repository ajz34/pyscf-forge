""" Doubly-Hybrid Response-Related Utilities. """
import numpy as np

from pyscf.dh.energy.dh import RDH
from pyscf.dh.response import RespBase
from pyscf.dh.response.hdft.rhdft import RHDFTResp
from pyscf.dh.response.hdft.uhdft import UHDFTResp


class RDHResp(RDH, RespBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # generate response instance for SCF (to obtain Ax0_Core)
        HDFTResp = RHDFTResp if self.restricted else UHDFTResp
        self.scf_resp = HDFTResp(self.scf)
        self._inherited_updated = False

    @property
    def Ax0_Core(self):
        """ Fock response of underlying SCF object in MO basis. """
        if self._Ax0_Core is NotImplemented:
            self._Ax0_Core = self.scf_resp.Ax0_Core
        return self._Ax0_Core

    @Ax0_Core.setter
    def Ax0_Core(self, Ax0_Core):
        self._Ax0_Core = Ax0_Core

    # in response instance, we first transfer all child instances into response
    def to_scf(self, *args, **kwargs):
        mf_resp = super().to_scf(*args, **kwargs).to_resp(key="resp")
        mf_resp.Ax0_Core = self.scf_resp.Ax0_Core
        return mf_resp

    def to_mp2(self, *args, **kwargs):
        assert len(args) == 0
        mf_resp = super().to_mp2(**kwargs).to_resp(key="resp")
        mf_resp.Ax0_Core = self.scf_resp.Ax0_Core
        return mf_resp

    def to_iepa(self, *args, **kwargs):
        raise NotImplementedError("Not implemented for response functions.")

    def to_ring_ccd(self, *args, **kwargs):
        raise NotImplementedError("Not implemented for response functions.")

    def update_inherited(self):
        """ Update inherited attribute, to transfer all child method instances to response. """

        if self._inherited_updated:
            return

        # if energy not evaluated, then evaluate energy first
        if len(self.inherited) == 0:
            self.kernel()

        # for energy evaluation, instance of low_rung may not be generated.
        if len(self.inherited["low_rung"][1]) == 0:
            HDFTResp = RHDFTResp if self.restricted else UHDFTResp
            self.inherited["low_rung"][1].append(HDFTResp(self.scf, xc=self.inherited["low_rung"][0]))

        # transform instances to response functions
        # note that if Ax0_Core appears, then this object is already response, or have been inherited
        for key in self.inherited:
            for idx in range(len(self.inherited[key][1])):
                instance = self.inherited[key][1][idx]
                if not hasattr(instance, "Ax0_Core"):
                    instance = instance.to_resp(key="resp")
                    self.inherited[key][1][idx] = instance

        self._inherited_updated = True

    def make_lag_vo(self):
        if "lag_vo" in self.tensors:
            return self.tensors["lag_vo"]

        self.update_inherited()

        nocc, nvir, nmo = self.nocc, self.nvir, self.nmo
        lag_vo = np.zeros((nvir, nocc))
        for key in self.inherited:
            for instance in self.inherited[key][1]:
                lag_vo += instance.make_lag_vo()

        self.tensors["lag_vo"] = lag_vo
        return lag_vo

    def make_rdm1_resp(self, ao=False):
        if "rdm1_resp" in self.tensors:
            return self.tensors["rdm1_resp"]

        self.update_inherited()

        rdm1_resp = np.diag(self.mo_occ)
        for key in self.inherited:
            for instance in self.inherited[key][1]:
                if hasattr(instance, "make_rdm1_corr"):
                    rdm1_resp += instance.make_rdm1_corr()

        nocc, nvir, nmo = self.nocc, self.nvir, self.nmo
        so, sv = slice(0, nocc), slice(nocc, nmo)
        lag_vo = self.make_lag_vo()

        rdm1_resp_vo = self.solve_cpks(lag_vo)
        rdm1_resp[sv, so] += rdm1_resp_vo

        self.tensors["rdm1_resp"] = rdm1_resp

        if ao:
            rdm1_resp = self.mo_coeff @ rdm1_resp @ self.mo_coeff.T
        return rdm1_resp


if __name__ == '__main__':
    pass
