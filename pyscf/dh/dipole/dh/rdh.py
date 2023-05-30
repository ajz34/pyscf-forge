import numpy as np

from pyscf.dh.dipole.dipolebase import DipoleBase, PolarBase
from pyscf.dh.energy.dh import RDH
from pyscf.dh.response.dh.rdh import RDHResp


class RDHDipole(RDHResp, DipoleBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def resp_type(self):
        return "dipole"

    def to_scf(self, *args, **kwargs):
        mf_resp = super().to_scf(*args, **kwargs, copy_all=False).to_resp(key="dipole")
        return mf_resp

    def to_mp2(self, *args, **kwargs):
        mf_resp = super().to_mp2(**kwargs, copy_all=False).to_resp(key="dipole")
        return mf_resp

    def make_lag_vo(self):
        return super().make_lag_vo()

    def make_rdm1_resp(self, ao_repr=False):
        return super().make_rdm1_resp(ao_repr=ao_repr)

    def make_SCR3(self):
        if self.pad_prop("SCR3") in self.tensors:
            return self.tensors[self.pad_prop("SCR3")]

        self.update_inherited(self.resp_type)

        nprop, nocc, nvir = self.nprop, self.nocc, self.nvir
        SCR3 = np.zeros((nprop, nvir, nocc))
        for key in self.inherited:
            for instance in self.inherited[key][1]:
                if hasattr(instance, "make_SCR3"):
                    SCR3 += instance.make_SCR3()

        self.tensors[self.pad_prop("SCR3")] = SCR3
        return SCR3

    def make_pd_rdm1_corr(self):
        if self.pad_prop("pd_rdm1_corr") in self.tensors:
            return self.tensors[self.pad_prop("pd_rdm1_corr")]

        self.update_inherited(self.resp_type)

        nprop, nmo = self.nprop, self.nmo
        pd_rdm1_corr = np.zeros((nprop, nmo, nmo))
        for key in self.inherited:
            for instance in self.inherited[key][1]:
                if hasattr(instance, "make_pd_rdm1_corr"):
                    pd_rdm1_corr += instance.make_pd_rdm1_corr()

        self.tensors[self.pad_prop("pd_rdm1_corr")] = pd_rdm1_corr
        return pd_rdm1_corr


class RDHPolar(RDHResp, PolarBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deriv_dipole = RDHDipole.from_cls(self, self.scf, xc=self.xc, copy_all=True)


if __name__ == '__main__':
    pass
