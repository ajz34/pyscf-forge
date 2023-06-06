import unittest
from pyscf.dh.response.mp2.ump2ri import UMP2RespRI
from pyscf.dh import UMP2RI
from pyscf import gto, dft, scf
import numpy as np

np.set_printoptions(10, linewidth=150, suppress=True)

α, β = 0, 1
αα, αβ, ββ = 0, 1, 2


class TestUHDFTResp(unittest.TestCase):

    def test_num_dipole_ref_hf(self):
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1).build()

        mf_scf = scf.UHF(mol).density_fit("cc-pVDZ-jkfit").run()
        mf_mp = UMP2RespRI(mf_scf).run()
        dip_anal = mf_mp.make_dipole()
        print(dip_anal)

        REF = np.array([0.7789511165, 0., 1.0060293075])
        self.assertTrue(np.allclose(dip_anal, REF, atol=1e-5, rtol=1e-7))

        """
        # generation of numerical result

        def eng_with_dipole_field(t, h):
            mf_scf = scf.UHF(mol).density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_mp = UMP2RI(mf_scf).run()
            return mf_mp.e_tot

        eng_array = np.zeros((2, 3))
        h = 1e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                eng_array[idx, t] = eng_with_dipole_field(t, h)
        dip_elec_num = - (eng_array[0] - eng_array[1]) / (2 * h)
        dip_nuc = np.einsum("A, At -> t", mol.atom_charges(), mol.atom_coords())
        dip_num = dip_elec_num + dip_nuc

        self.assertTrue(np.allclose(dip_num, dip_anal, atol=1e-5, rtol=1e-7))
        """

    def test_num_dipole_ref_b3lyp(self):
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1).build()

        c_os, c_ss = 1.3, 0.6

        mf_scf = dft.UKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit").run()
        mf_mp = UMP2RespRI(mf_scf).run(c_os=c_os, c_ss=c_ss)
        dip_anal = mf_mp.make_dipole()

        REF = np.array([0.723930342, 0., 0.9349686142])
        self.assertTrue(np.allclose(dip_anal, REF, atol=1e-5, rtol=1e-7))

        """
        # generation of numerical result

        def eng_with_dipole_field(t, h):
            mf_scf = dft.UKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_mp = UMP2RI(mf_scf).run(c_os=c_os, c_ss=c_ss)
            return mf_mp.scf.e_tot + c_os * mf_mp.results["eng_corr_MP2_OS"] + c_ss * mf_mp.results["eng_corr_MP2_SS"]

        eng_array = np.zeros((2, 3))
        h = 1e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                eng_array[idx, t] = eng_with_dipole_field(t, h)
        dip_elec_num = - (eng_array[0] - eng_array[1]) / (2 * h)
        dip_nuc = np.einsum("A, At -> t", mol.atom_charges(), mol.atom_coords())
        dip_num = dip_elec_num + dip_nuc

        print(dip_num)
        print(dip_anal)
        self.assertTrue(np.allclose(dip_num, dip_anal, atol=1e-5, rtol=1e-7))
        """
