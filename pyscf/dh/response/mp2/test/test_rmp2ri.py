import unittest
from pyscf.dh.response.mp2.rmp2ri import RMP2RespRI
from pyscf.dh.energy.mp2.rmp2 import RMP2RI
import numpy as np


class TestRMP2RI(unittest.TestCase):
    def test_num_dipole_ref_hf(self):
        from pyscf import gto, scf
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()

        mf_scf = scf.RHF(mol).density_fit("cc-pVDZ-jkfit").run()
        mf_mp = RMP2RespRI(mf_scf).run()
        dip_anal = mf_mp.make_dipole()

        REF = np.array([6.12636286e-01, 0, 7.91230727e-01])
        self.assertTrue(np.allclose(dip_anal, REF, atol=1e-5, rtol=1e-7))

        """
        # generation of numerical result
        
        def eng_with_dipole_field(t, h):
            mf_scf = scf.RHF(mol).density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_mp = RMP2RI(mf_scf).run()
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
        from pyscf import gto, scf, dft
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()

        c_os, c_ss = 1.3, 0.6

        mf_scf = dft.RKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit").run()
        mf_mp = RMP2RespRI(mf_scf).run(c_os=c_os, c_ss=c_ss)
        dip_anal = mf_mp.make_dipole()

        REF = np.array([5.34275168e-01, 0, 6.90025798e-01])
        self.assertTrue(np.allclose(dip_anal, REF, atol=1e-5, rtol=1e-7))

        """
        # generation of numerical result

        def eng_with_dipole_field(t, h):
            mf_scf = dft.RKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_mp = RMP2RI(mf_scf).run()
            return mf_mp.scf.e_tot + c_os * mf_mp.results["eng_corr_MP2_OS"] + c_ss * mf_mp.results["eng_corr_MP2_SS"]

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


