import unittest
from pyscf.dh.dipole.hdft.rhdft import RSCFPolar, RSCFResp, RHDFTPolar, RHDFTResp
import numpy as np


class TestRHDFTPolar(unittest.TestCase):
    def test_b3lyp(self):
        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", verbose=0).build()
        mf = dft.RKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit").run()
        mf_pol = RSCFPolar(mf).run()
        print(mf_pol.de)

        REF = np.array(
            [[5.8439851862, 0., -0.9538125151],
             [0., 1.5152303269, -0.],
             [-0.9538125151, -0., 5.3506412225]])
        self.assertTrue(np.allclose(mf_pol.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result
        
        def dipole_with_dipole_field(t, h):
            mf_scf = dft.RKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = RSCFResp(mf_scf).run()
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

    def test_b3lyp_cam(self):
        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", verbose=0).build()
        mf = dft.RKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit").run()
        mf_pol = RHDFTPolar(mf, xc="CAM-B3LYP")
        print(mf_pol.de)
        REF = np.array(
            [[5.7565090385, -0., -0.9567747422],
             [-0., 1.4921888876, -0.],
             [-0.9567747422, -0., 5.2616312231]])
        self.assertTrue(np.allclose(mf_pol.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_scf = dft.RKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = RHDFTResp(mf_scf, xc="CAM-B3LYP")
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

    def test_hf_tpss0(self):
        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", verbose=0).build()
        mf = dft.RKS(mol, xc="HF").density_fit("cc-pVDZ-jkfit").run()
        mf_pol = RHDFTPolar(mf, xc="TPSS0")
        print(mf_pol.de)
        REF = np.array(
            [[5.6996863302, 0., -0.9570246008],
             [0., 1.4947978961, -0.],
             [-0.9570246008, -0., 5.2046766405]])
        self.assertTrue(np.allclose(mf_pol.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_scf = dft.RKS(mol, xc="HF").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = RHDFTResp(mf_scf, xc="TPSS0")
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

    def test_tpss0_hf(self):
        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", verbose=0).build()
        mf = dft.RKS(mol, xc="TPSS0").density_fit("cc-pVDZ-jkfit").run()
        mf_pol = RHDFTPolar(mf, xc="HF")
        print(mf_pol.de)
        REF = np.array(
            [[5.4235257403, -0., -1.0019995009],
             [-0., 1.4055046383, 0.],
             [-1.0019995009, 0., 4.9052549459]])
        self.assertTrue(np.allclose(mf_pol.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_scf = dft.RKS(mol, xc="TPSS0").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = RHDFTResp(mf_scf, xc="HF")
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

    def test_svwn_tpss0(self):
        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", verbose=0).build()
        mf = dft.RKS(mol, xc="SVWN").density_fit("cc-pVDZ-jkfit").run()
        mf_pol = RHDFTPolar(mf, xc="TPSS0")
        print(mf_pol.de)
        REF = np.array(
            [[5.7215914478, -0., -0.947151482],
             [-0., 1.5089416744, -0.],
             [-0.947151482, -0., 5.2316860189]])
        self.assertTrue(np.allclose(mf_pol.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_scf = dft.RKS(mol, xc="SVWN").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = RHDFTResp(mf_scf, xc="TPSS0")
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

    def test_tpss0_svwn(self):
        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", verbose=0).build()
        mf = dft.RKS(mol, xc="TPSS0").density_fit("cc-pVDZ-jkfit").run()
        mf_pol = RHDFTPolar(mf, xc="SVWN")
        print(mf_pol.de)
        REF = np.array(
            [[5.8318926122, -0., -0.8954774575],
             [-0., 1.5478365106, -0.],
             [-0.8954774575, -0., 5.3687189265]])
        self.assertTrue(np.allclose(mf_pol.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_scf = dft.RKS(mol, xc="TPSS0").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = RHDFTResp(mf_scf, xc="SVWN")
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
