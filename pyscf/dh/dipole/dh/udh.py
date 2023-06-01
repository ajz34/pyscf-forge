import numpy as np

from pyscf.dh.dipole.dipolebase import DipoleBase, PolarBase
from pyscf.dh.dipole.dh.rdh import RDHDipole
from pyscf.dh.energy.dh import UDH
from pyscf.dh.response.dh.udh import UDHResp


α, β = 0, 1


class UDHDipole(UDHResp, RDHDipole):

    @property
    def resp_type(self):
        return "dipole"

    def make_SCR3(self):
        if self.pad_prop("SCR3") in self.tensors:
            return self.tensors[self.pad_prop("SCR3")]

        self.update_inherited(self.resp_type)

        nprop, nocc, nvir = self.nprop, self.nocc, self.nvir
        SCR3 = [np.zeros((nprop, nvir[σ], nocc[σ])) for σ in (α, β)]
        for key in self.inherited:
            for instance in self.inherited[key][1]:
                if hasattr(instance, "make_SCR3"):
                    SCR3_contrib = instance.make_SCR3()
                    for σ in (α, β):
                        SCR3[σ] += SCR3_contrib[σ]

        self.tensors[self.pad_prop("SCR3")] = SCR3
        return SCR3

    def make_pd_rdm1_corr(self):
        if self.pad_prop("pd_rdm1_corr") in self.tensors:
            return self.tensors[self.pad_prop("pd_rdm1_corr")]

        self.update_inherited(self.resp_type)

        nprop, nmo = self.nprop, self.nmo
        pd_rdm1_corr = np.zeros((2, nprop, nmo, nmo))
        for key in self.inherited:
            for instance in self.inherited[key][1]:
                if hasattr(instance, "make_pd_rdm1_corr"):
                    pd_rdm1_corr += instance.make_pd_rdm1_corr()

        self.tensors[self.pad_prop("pd_rdm1_corr")] = pd_rdm1_corr
        return pd_rdm1_corr


class UDHPolar(UDHResp, PolarBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deriv_dipole = UDHDipole.from_cls(self, self.scf, xc=self.xc, copy_all=True)

    def kernel(self, *_args, **_kwargs):
        self.build()
        return self.make_polar()


if __name__ == '__main__':
    def main_1():
        from pyscf import gto, dft, scf
        from pyscf.dh.dipole.dh.rdh import RDHDipole, RDHPolar
        from pyscf.dh.dipole.hdft.uhdft import UHDFTDipole, UHDFTPolar
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1, verbose=0).build()
        mf = UDHPolar(mol, xc="RS-PBE-P86").run()
        print(mf.de)

        # REF = np.array(
        #     [[5.6430117342, -0., -0.9517267271],
        #      [-0., 1.4900634185, 0.],
        #      [-0.9517267305, 0., 5.150744775]])
        # self.assertTrue(np.allclose(mf.de, REF, atol=1e-6, rtol=1e-4))


        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_resp = UDHDipole(mol, xc="RS-PBE-P86").build_scf()
            mf_resp.scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            return mf_resp.make_dipole()

        dip_array = np.zeros((2, 3, 3))
        h = 3e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                dip_array[idx, t] = dipole_with_dipole_field(t, h)
        pol_num = (dip_array[0] - dip_array[1]) / (2 * h)
        print(pol_num)
        print(mf.de - pol_num)

    main_1()

