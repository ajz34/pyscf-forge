import unittest
from pyscf.dh.dipole.mp2.rmp2ri import RMP2RespRI, RMP2PolarRI
import numpy as np


class TestRMP2PolarRI(unittest.TestCase):
    def test_mp2(self):
        from pyscf import gto, scf
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", verbose=0).build()
        mf = scf.RHF(mol).density_fit("cc-pVDZ-jkfit").run()
        mf_pol = RMP2PolarRI(mf)
        print(mf_pol.de)
        REF = np.array(
            [[5.6190666615, -0., -0.9349582098],
             [-0., 1.5104647786, 0.],
             [-0.9349582098, 0., 5.1354733941]])
        self.assertTrue(np.allclose(mf_pol.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result
        
        def dipole_with_dipole_field(t, h):
            mf_scf = scf.RHF(mol).density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = RMP2RespRI(mf_scf)
            return mf_resp.make_dipole()

        dip_array = np.zeros((2, 3, 3))
        h = 1e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                dip_array[idx, t] = dipole_with_dipole_field(t, h)
        pol_num = (dip_array[0] - dip_array[1]) / (2 * h)
        print(pol_num)

        self.assertTrue(np.allclose(mf_pol.de, pol_num, atol=1e-6, rtol=1e-4))
        """

    def test_mp2_with_scs(self):
        from pyscf import gto, dft, scf
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", verbose=0).build()
        mf = dft.RKS(mol, xc="HF").density_fit("cc-pVDZ-jkfit").run()
        mf_pol = RMP2PolarRI(mf).run(c_ss=0.6, c_os=1.3)
        print(mf_pol.de)
        REF = np.array(
            [[5.6257485328, -0., -0.9315476636],
             [-0., 1.514350452, 0.],
             [-0.9315476636, 0., 5.1439193199]])
        self.assertTrue(np.allclose(mf_pol.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_scf = dft.RKS(mol, xc="HF").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = RMP2RespRI(mf_scf).run(c_ss=0.6, c_os=1.3)
            return mf_resp.make_dipole()

        dip_array = np.zeros((2, 3, 3))
        h = 1e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                dip_array[idx, t] = dipole_with_dipole_field(t, h)
        pol_num = (dip_array[0] - dip_array[1]) / (2 * h)
        print(pol_num)

        self.assertTrue(np.allclose(mf_pol.de, pol_num, atol=1e-6, rtol=1e-4))
        """
