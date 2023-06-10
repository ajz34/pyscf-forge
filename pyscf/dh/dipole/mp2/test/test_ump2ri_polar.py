import unittest
from pyscf.dh.dipole.mp2.ump2ri import UMP2RespRI, UMP2PolarRI
import numpy as np


class TestUMP2PolarRI(unittest.TestCase):
    def test_mp2(self):
        from pyscf import gto, scf
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1, verbose=0).build()
        mf = scf.UHF(mol).density_fit("cc-pVDZ-jkfit").run()
        mf_pol = UMP2PolarRI(mf)
        print(mf_pol.de)
        REF = np.array(
            [[3.6779347281, 0., -0.6313946235],
             [0., 2.7389242733, -0.],
             [-0.6313946235, -0., 3.3513552235]])
        self.assertTrue(np.allclose(mf_pol.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_scf = scf.UHF(mol).density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = UMP2RespRI(mf_scf)
            return mf_resp.make_dipole()

        dip_array = np.zeros((2, 3, 3))
        h = 3e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                dip_array[idx, t] = dipole_with_dipole_field(t, h)
        pol_num = (dip_array[0] - dip_array[1]) / (2 * h)
        print(pol_num)
        print(mf_pol.de - pol_num)

        self.assertTrue(np.allclose(mf_pol.de, pol_num, atol=1e-6, rtol=1e-4))
        """

    def test_mp2_with_scs(self):
        from pyscf import gto, dft, scf
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1, verbose=0).build()
        mf = dft.UKS(mol, xc="HF").density_fit("cc-pVDZ-jkfit").run()
        mf_pol = UMP2PolarRI(mf).run(c_ss=0.6, c_os=1.3)
        print(mf_pol.de)
        REF = np.array(
            [[3.6940469523, -0., -0.634663444],
             [-0., 2.7421887263, -0.],
             [-0.634663444, -0., 3.3657766987]])
        self.assertTrue(np.allclose(mf_pol.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_scf = dft.UKS(mol, xc="HF").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = UMP2RespRI(mf_scf).run(c_ss=0.6, c_os=1.3)
            return mf_resp.make_dipole()

        dip_array = np.zeros((2, 3, 3))
        h = 3e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                dip_array[idx, t] = dipole_with_dipole_field(t, h)
        pol_num = (dip_array[0] - dip_array[1]) / (2 * h)
        print(pol_num)
        print(mf_pol.de - pol_num)

        self.assertTrue(np.allclose(mf_pol.de, pol_num, atol=1e-6, rtol=1e-4))
        """
