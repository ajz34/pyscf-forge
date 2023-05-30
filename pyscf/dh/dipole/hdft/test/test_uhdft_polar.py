import unittest
from pyscf.dh.dipole.hdft.uhdft import USCFPolar, USCFDipole, USCFResp
import numpy as np


class TestRHDFTPolar(unittest.TestCase):
    def test_b3lyp(self):
        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", charge=1, spin=1, basis="6-31G", verbose=0).build()
        mf = dft.UKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit").run()
        mf_pol = USCFPolar(mf).run()
        print(mf_pol.de)

        REF = np.array(
            [[3.7814963149, -0., -0.6573308352],
             [-0., 2.5774165307, -0.],
             [-0.6573308352, -0., 3.4415021982]])
        self.assertTrue(np.allclose(mf_pol.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_scf = dft.UKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = USCFDipole(mf_scf).run()
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
