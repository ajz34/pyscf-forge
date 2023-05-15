import unittest
from pyscf.dh.response.hdft.rhdft import RHDFTResp
from pyscf import gto, dft
import numpy as np


class TestRHDFTResp(unittest.TestCase):

    def test_rsh_ax0_cpks(self):
        # test that RSH functional is correct for Ax0_cpks
        np.random.seed(0)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = dft.RKS(mol, xc="CAM-B3LYP").density_fit().run()
        mf_scf = RHDFTResp(mf)
        mf_scf.grids_cpks = mf_scf.scf.grids

        nocc, nvir, nmo = mf_scf.nocc, mf_scf.nvir, mf_scf.nmo
        so, sv = slice(0, nocc), slice(nocc, nmo)
        X = np.random.randn(3, nvir, nocc)
        ax_cpks = mf_scf.make_Ax0_cpks()(X)
        ax_core = mf_scf.make_Ax0_Core_resp(sv, so, sv, so)(X)
        self.assertTrue("eri_cpks_vovo" in mf_scf.tensors)
        self.assertTrue(np.allclose(ax_cpks, ax_core))

    def test_meta_ax0_cpks(self):
        # test that meta-GGA functional is correct for Ax0_cpks
        np.random.seed(0)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = dft.RKS(mol, xc="TPSS0").density_fit().run()
        mf_scf = RHDFTResp(mf)
        mf_scf.grids_cpks = mf_scf.scf.grids

        nocc, nvir, nmo = mf_scf.nocc, mf_scf.nvir, mf_scf.nmo
        so, sv = slice(0, nocc), slice(nocc, nmo)
        X = np.random.randn(3, nvir, nocc)
        ax_cpks = mf_scf.make_Ax0_Core(sv, so, sv, so)(X)
        ax_core = mf_scf.make_Ax0_Core_resp(sv, so, sv, so)(X)
        self.assertTrue("eri_cpks_vovo" in mf_scf.tensors)
        self.assertTrue(np.allclose(ax_cpks, ax_core))

    def test_no_density_fit(self):
        # test that RSH functional is correct for Ax0_cpks if no density fitting
        np.random.seed(0)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = dft.RKS(mol, xc="CAM-B3LYP").run()
        mf_scf = RHDFTResp(mf)
        mf_scf.grids_cpks = mf_scf.scf.grids

        nocc, nvir, nmo = mf_scf.nocc, mf_scf.nvir, mf_scf.nmo
        so, sv = slice(0, nocc), slice(nocc, nmo)
        X = np.random.randn(3, nvir, nocc)
        ax_cpks = mf_scf.make_Ax0_Core(sv, so, sv, so)(X)
        ax_core = mf_scf.make_Ax0_Core_resp(sv, so, sv, so)(X)
        self.assertTrue(np.allclose(ax_cpks, ax_core))

    def test_dipole_b3_cam(self):
        # test B3LYP -> CAM-B3LYP non-consistent functional dipole

        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()

        mf = dft.RKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit").run()
        mf_hdft = RHDFTResp(mf, xc="CAM-B3LYP")
        dip_anal = mf_hdft.make_dipole()

        REF = np.array([0.6039302487, 0., 0.7799863729])
        self.assertTrue(np.allclose(dip_anal, REF, atol=1e-5, rtol=1e-7))

        """
        # generation of numerical result
        
        def eng_with_dipole_field(t, h):
            mf_scf = dft.RKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_mp = RHDFTResp(mf_scf, xc="CAM-B3LYP").run()
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

    def test_dipole_b3_hf(self):
        # test B3LYP -> HF non-consistent functional dipole

        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()

        mf = dft.RKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit").run()
        mf_hdft = RHDFTResp(mf, xc="HF")
        dip_anal = mf_hdft.make_dipole()

        REF = np.array([0.6377739919, 0., 0.8236964397])
        self.assertTrue(np.allclose(dip_anal, REF, atol=1e-5, rtol=1e-7))

        """
        # generation of numerical result
        
        def eng_with_dipole_field(t, h):
            mf_scf = dft.RKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_mp = RHDFTResp(mf_scf, xc="HF").run()
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
