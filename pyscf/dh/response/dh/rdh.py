""" Doubly-Hybrid Response-Related Utilities. """
import numpy as np

from pyscf.dh.energy.dh import RDH
from pyscf.dh.response import RespBase
from pyscf.dh.response.hdft.rhdft import RHDFTResp
from pyscf.scf import cphf


class RDHResp(RDH, RespBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # generate response instance for SCF (to obtain Ax0_Core)
        HDFTResp = RHDFTResp if self.restricted else NotImplemented
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
        mf_resp = super().to_scf(*args, **kwargs).to_resp()
        mf_resp.Ax0_Core = self.scf_resp.Ax0_Core
        return mf_resp

    def to_mp2(self, *args, **kwargs):
        assert len(args) == 0
        mf_resp = super().to_mp2(**kwargs).to_resp()
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
            HDFTResp = RHDFTResp if self.restricted else NotImplemented
            self.inherited["low_rung"][1].append(HDFTResp(self.scf, xc=self.inherited["low_rung"][0]))

        # transform instances to response functions
        # note that if Ax0_Core appears, then this object is already response, or have been inherited
        for key in self.inherited:
            for idx in range(len(self.inherited[key][1])):
                instance = self.inherited[key][1][idx]
                if not hasattr(instance, "Ax0_Core"):
                    instance = instance.to_resp()
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
        mo_energy = self.mo_energy
        mo_occ = self.mo_occ
        lag_vo = self.make_lag_vo()
        max_cycle = self.max_cycle_cpks
        tol = self.tol_cpks

        Ax0_Core = self.Ax0_Core
        rdm1_resp_vo = cphf.solve(
            Ax0_Core(sv, so, sv, so), mo_energy, mo_occ, lag_vo,
            max_cycle=max_cycle, tol=tol)[0]

        rdm1_resp[sv, so] += rdm1_resp_vo

        self.tensors["rdm1_resp"] = rdm1_resp

        if ao:
            rdm1_resp = self.mo_coeff @ rdm1_resp @ self.mo_coeff.T
        return rdm1_resp

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


if __name__ == '__main__':
    def main_1():
        # test energy is the same for response and general DH
        from pyscf import gto, scf
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf_resp = RDHResp(mol, xc="XYG3").run()
        mf = RDH(mol, xc="XYG3").run()
        print(np.allclose(mf_resp.e_tot, mf.e_tot))

    def main_2():
        # test RSH functional
        from pyscf import gto, scf
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = RDH(mol, xc="RS-PBE-P86").run()
        print(mf.results)
        mf_resp = RDHResp(mol, xc="RS-PBE-P86").run()
        print(mf_resp.results)
        print(np.allclose(mf_resp.e_tot, mf.e_tot))

        from pyscf import lib
        lib.num_threads()

    def main_3():
        # test unbalanced OS contribution
        from pyscf import gto, scf
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = RDH(mol, xc="XYGJ-OS").run()
        mf_resp = RDHResp(mol, xc="XYGJ-OS").run()
        print(np.allclose(mf_resp.e_tot, mf.e_tot))

        from pyscf import lib

    def main_4():
        # try response density and dipole
        from pyscf import gto, scf, dft

        np.set_printoptions(8, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf_resp = RDHResp(mol, xc="XYG3").run()

        def eng_with_dipole_field(t, h):
            mf_scf = dft.RKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_mp = RDH(mf_scf, xc="XYG3").run()
            return mf_mp.e_tot

        eng_array = np.zeros((2, 3))
        h = 1e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                eng_array[idx, t] = eng_with_dipole_field(t, h)
        dip_elec_num = - (eng_array[0] - eng_array[1]) / (2 * h)

        dip_nuc = np.einsum("A, At -> t", mol.atom_charges(), mol.atom_coords())
        dip_anal = mf_resp.make_dipole()
        dip_num = dip_elec_num + dip_nuc
        print(dip_anal)
        print(dip_num)

    main_4()
