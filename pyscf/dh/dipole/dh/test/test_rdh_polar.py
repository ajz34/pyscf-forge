import unittest
from pyscf.dh.dipole.dh.rdh import RDHPolar, RDHDipole, RDHResp
import numpy as np


class TestRDHPolar(unittest.TestCase):
    def test_xyg3(self):
        from pyscf import gto, dft, scf
        np.set_printoptions(10, suppress=True, linewidth=150)
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", verbose=0).build()
        mf_scf = dft.RKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit").run()
        mf = RDHPolar(mf_scf, xc="XYG3").run()

        REF = np.array(
            [[5.6430117342, -0., -0.9517267271],
             [-0., 1.4900634185, 0.],
             [-0.9517267305, 0., 5.150744775]])
        self.assertTrue(np.allclose(mf.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_scf = dft.RKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = RDHDipole(mf_scf, xc="XYG3")
            return mf_resp.make_dipole()

        dip_array = np.zeros((2, 3, 3))
        h = 1e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                dip_array[idx, t] = dipole_with_dipole_field(t, h)
        pol_num = (dip_array[0] - dip_array[1]) / (2 * h)
        print(pol_num)

        self.assertTrue(np.allclose(mf.de, pol_num, atol=1e-6, rtol=1e-4))
        """

    def test_xygjos(self):
        from pyscf import gto, dft, scf
        np.set_printoptions(10, suppress=True, linewidth=150)
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", verbose=0).build()
        mf_scf = dft.RKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit").run()
        mf = RDHPolar(mf_scf, xc="XYGJ-OS").run()

        REF = np.array(
            [[5.6476985545, -0., -0.9436874871],
             [-0., 1.4902875613, 0.],
             [-0.9436874909, 0., 5.1595899522]])
        self.assertTrue(np.allclose(mf.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_scf = dft.RKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = RDHDipole(mf_scf, xc="XYGJ-OS")
            return mf_resp.make_dipole()

        dip_array = np.zeros((2, 3, 3))
        h = 1e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                dip_array[idx, t] = dipole_with_dipole_field(t, h)
        pol_num = (dip_array[0] - dip_array[1]) / (2 * h)
        print(pol_num)

        self.assertTrue(np.allclose(mf.de, pol_num, atol=1e-6, rtol=1e-4))
        """

    def test_rsdh(self):
        from pyscf import gto, dft, scf
        np.set_printoptions(10, suppress=True, linewidth=150)
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", verbose=0).build()
        mf = RDHPolar(mol, xc="RS-PBE-P86").run()

        REF = np.array(
            [[5.4901470804, -0., -0.9391992818],
             [-0., 1.443484457, -0.],
             [-0.9391992817, -0., 5.0043604356]])
        self.assertTrue(np.allclose(mf.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_resp = RDHDipole(mol, xc="RS-PBE-P86").build_scf()
            mf_resp.scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_resp.scf.conv_tol = 1e-12
            return mf_resp.run().make_dipole()

        dip_array = np.zeros((2, 3, 3))
        h = 1e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                dip_array[idx, t] = dipole_with_dipole_field(t, h)
        pol_num = (dip_array[0] - dip_array[1]) / (2 * h)
        print(pol_num)

        self.assertTrue(np.allclose(mf.de, pol_num, atol=1e-6, rtol=1e-4))
        """

    def test_DSD_PBEP86_D3BJ(self):
        from pyscf import gto, dft, scf
        np.set_printoptions(10, suppress=True, linewidth=150)
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", verbose=0).build()
        mf = RDHPolar(mol, xc="DSD-PBEP86-D3BJ").run()

        REF = np.array(
            [[5.673397476, 0., -0.9353920386],
             [0., 1.5088061204, -0.],
             [-0.9353920497, -0., 5.1895879916]])
        self.assertTrue(np.allclose(mf.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_resp = RDHDipole(mol, xc="DSD-PBEP86-D3BJ").build_scf()
            mf_resp.scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_resp.scf.conv_tol = 1e-12
            return mf_resp.run().make_dipole()

        dip_array = np.zeros((2, 3, 3))
        h = 1e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                dip_array[idx, t] = dipole_with_dipole_field(t, h)
        pol_num = (dip_array[0] - dip_array[1]) / (2 * h)
        print(pol_num)

        self.assertTrue(np.allclose(mf.de, pol_num, atol=1e-6, rtol=1e-4))
        """
