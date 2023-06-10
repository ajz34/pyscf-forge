import unittest
from pyscf.dh.dipole.hdft.uhdft import USCFPolar, USCFDipole, UHDFTPolar, UHDFTDipole
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

    def test_b3lyp_cam(self):
        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", charge=1, spin=1, basis="6-31G", verbose=0).build()
        mf = dft.UKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit").run()
        mf_pol = UHDFTPolar(mf, xc="CAM-B3LYP")
        print(mf_pol.de)
        REF = np.array(
            [[3.7441018125, -0., -0.6533705984],
             [-0., 2.5941471595, -0.],
             [-0.6533705984, -0., 3.4061562009]])
        self.assertTrue(np.allclose(mf_pol.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_scf = dft.UKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = UHDFTDipole(mf_scf, xc="CAM-B3LYP")
            return mf_resp.make_dipole()

        dip_array = np.zeros((2, 3, 3))
        h = 1e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                dip_array[idx, t] = dipole_with_dipole_field(t, h)
        pol_num = (dip_array[0] - dip_array[1]) / (2 * h)
        print(pol_num)
        print(mf_pol.de - pol_num)

        self.assertTrue(np.allclose(mf_pol.de, pol_num, atol=1e-6, rtol=1e-4))
        """

    def test_hf_tpss0(self):
        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1, verbose=0).build()
        mf = dft.UKS(mol, xc="HF").density_fit("cc-pVDZ-jkfit").run()
        mf_pol = UHDFTPolar(mf, xc="TPSS0")
        print(mf_pol.de)
        REF = np.array(
            [[3.7161194081, 0., -0.6480115453],
             [0., 2.5642810727, 0.],
             [-0.6480115453, 0., 3.3809478891]])
        self.assertTrue(np.allclose(mf_pol.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_scf = dft.UKS(mol, xc="HF").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = UHDFTDipole(mf_scf, xc="TPSS0")
            return mf_resp.make_dipole()

        dip_array = np.zeros((2, 3, 3))
        h = 1e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                dip_array[idx, t] = dipole_with_dipole_field(t, h)
        pol_num = (dip_array[0] - dip_array[1]) / (2 * h)
        print(pol_num)
        print(mf_pol.de - pol_num)

        self.assertTrue(np.allclose(mf_pol.de, pol_num, atol=1e-6, rtol=1e-4))
        """

    def test_tpss0_hf(self):
        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1, verbose=0).build()
        mf = dft.UKS(mol, xc="TPSS0").density_fit("cc-pVDZ-jkfit").run()
        mf_pol = UHDFTPolar(mf, xc="HF")
        print(mf_pol.de)

        REF = np.array(
            [[3.5470333811, -0., -0.6318031061],
             [-0., 2.6666604229, -0.],
             [-0.6318031061, -0., 3.2202485771]])
        self.assertTrue(np.allclose(mf_pol.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_scf = dft.UKS(mol, xc="TPSS0").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = UHDFTDipole(mf_scf, xc="HF")
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

    def test_svwn_tpss0(self):
        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1, verbose=0).build()
        mf = dft.UKS(mol, xc="SVWN").density_fit("cc-pVDZ-jkfit").run()
        mf_pol = UHDFTPolar(mf, xc="TPSS0")
        print(mf_pol.de)
        REF = np.array(
            [[3.7298806194, -0., -0.6437351888],
             [-0., 2.5585396333, -0.],
             [-0.6437351888, -0., 3.3969155279]])
        self.assertTrue(np.allclose(mf_pol.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_scf = dft.UKS(mol, xc="SVWN").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = UHDFTDipole(mf_scf, xc="TPSS0")
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

    def test_tpss0_svwn(self):
        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1, verbose=0).build()
        mf = dft.UKS(mol, xc="TPSS0").density_fit("cc-pVDZ-jkfit").run()
        mf_pol = UHDFTPolar(mf, xc="SVWN")
        print(mf_pol.de)
        REF = np.array(
            [[3.7595008511, 0., -0.6269043025],
             [0., 2.6534119967, -0.],
             [-0.6269043025, -0., 3.4352487597]])
        self.assertTrue(np.allclose(mf_pol.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_scf = dft.UKS(mol, xc="TPSS0").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = UHDFTDipole(mf_scf, xc="SVWN")
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
