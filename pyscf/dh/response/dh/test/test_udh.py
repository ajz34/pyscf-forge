import unittest
from pyscf.dh.response.dh.udh import UDHResp
from pyscf.dh.energy.dh import UDH
from pyscf import gto, dft, scf
import numpy as np


class TestRDHResp(unittest.TestCase):
    def test_same_eng_XYG3(self):
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1).build()
        mf_resp = UDHResp(mol, xc="XYG3").run()
        mf = UDH(mol, xc="XYG3").run()
        self.assertAlmostEquals(mf_resp.e_tot, mf.e_tot, 8)

    def test_same_eng_RS_PBE_P86(self):
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1).build()
        mf_resp = UDHResp(mol, xc="RS-PBE-P86").run()
        mf = UDH(mol, xc="RS-PBE-P86").run()
        self.assertAlmostEquals(mf_resp.e_tot, mf.e_tot, 8)

    def test_same_eng_XYGJ_OS(self):
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1).build()
        mf_resp = UDHResp(mol, xc="XYGJ-OS").run()
        mf = UDH(mol, xc="XYGJ-OS").run()
        self.assertAlmostEquals(mf_resp.e_tot, mf.e_tot, 8)

    def test_response_den_XYG3(self):
        np.set_printoptions(10, suppress=True, linewidth=150)
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", spin=2).build()
        mf_resp = UDHResp(mol, xc="XYG3").run()
        dip_anal = mf_resp.make_dipole()

        REF = np.array([-0.0490687712, 0., -0.0633724014])
        self.assertTrue(np.allclose(dip_anal, REF, atol=1e-5, rtol=1e-7))

        """
        # generation of numerical result

        def eng_with_dipole_field(t, h):
            mf_scf = dft.UKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_mp = UDH(mf_scf, xc="XYG3").run()
            return mf_mp.e_tot

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
        self.assertTrue(np.allclose(dip_anal, dip_num, rtol=1e-5, atol=1e-6))
        """

    def test_response_den_RS_PBE_P86(self):
        np.set_printoptions(10, suppress=True, linewidth=150)
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", spin=2).build()
        mf_resp = UDHResp(mol, xc="RS-PBE-P86")
        mf_resp.run()
        dip_anal = mf_resp.make_dipole()
        print(dip_anal)
        REF = np.array([-0.0476602179, -0., -0.0615538847])
        self.assertTrue(np.allclose(dip_anal, REF, atol=1e-5, rtol=1e-7))

        """
        # generation of numerical result

        def eng_with_dipole_field(t, h):
            mf_mp = UDH(mol, xc="RS-PBE-P86")
            mf_mp.scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_mp.scf.conv_tol = 1e-12
            mf_mp.run()
            return mf_mp.e_tot

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
        self.assertTrue(np.allclose(dip_anal, dip_num, rtol=1e-5, atol=1e-7))
        """
