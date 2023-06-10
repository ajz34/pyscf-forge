import unittest
from pyscf.dh.response.hdft.uhdft import UHDFTResp
from pyscf import gto, dft
import numpy as np

α, β = 0, 1
αα, αβ, ββ = 0, 1, 2


class TestUHDFTResp(unittest.TestCase):

    def test_rsh_ax0_cpks(self):
        # test that RSH functional is correct for Ax0_cpks
        np.random.seed(0)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", spin=1, charge=1).build()
        mf = dft.UKS(mol, xc="CAM-B3LYP").density_fit().run()
        mf_scf = UHDFTResp(mf)
        mf_scf.grids_cpks = mf_scf.scf.grids

        nocc, nvir, nmo = mf_scf.nocc, mf_scf.nvir, mf_scf.nmo
        so, sv = mf_scf.mask_occ, mf_scf.mask_vir
        X = [np.random.randn(3, nvir[σ], nocc[σ]) for σ in (α, β)]
        ax_cpks = mf_scf.make_Ax0_cpks()(X)
        ax_core = mf_scf.make_Ax0_Core_resp(sv, so, sv, so)(X)
        self.assertTrue("eri_cpks_vovo" in mf_scf.tensors)
        for σ in (α, β):
            self.assertTrue(np.allclose(ax_cpks[σ], ax_core[σ]))

    def test_meta_ax0_cpks(self):
        # test that RSH functional is correct for Ax0_cpks
        np.random.seed(0)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", spin=1, charge=1).build()
        mf = dft.UKS(mol, xc="TPSS0").density_fit().run()
        mf_scf = UHDFTResp(mf)
        mf_scf.grids_cpks = mf_scf.scf.grids

        nocc, nvir, nmo = mf_scf.nocc, mf_scf.nvir, mf_scf.nmo
        so, sv = mf_scf.mask_occ, mf_scf.mask_vir
        X = [np.random.randn(3, nvir[σ], nocc[σ]) for σ in (α, β)]
        ax_cpks = mf_scf.make_Ax0_Core(sv, so, sv, so)(X)
        ax_core = mf_scf.make_Ax0_Core_resp(sv, so, sv, so)(X)
        self.assertTrue("eri_cpks_vovo" in mf_scf.tensors)
        for σ in (α, β):
            self.assertTrue(np.allclose(ax_cpks[σ], ax_core[σ]))

    def test_no_density_fit(self):
        # test that RSH functional is correct for Ax0_cpks
        np.random.seed(0)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", spin=1, charge=1).build()
        mf = dft.UKS(mol, xc="CAM-B3LYP").run()
        mf_scf = UHDFTResp(mf)
        mf_scf.grids_cpks = mf_scf.scf.grids

        nocc, nvir, nmo = mf_scf.nocc, mf_scf.nvir, mf_scf.nmo
        so, sv = mf_scf.mask_occ, mf_scf.mask_vir
        X = [np.random.randn(3, nvir[σ], nocc[σ]) for σ in (α, β)]
        ax_cpks = mf_scf.make_Ax0_Core(sv, so, sv, so)(X)
        ax_core = mf_scf.make_Ax0_Core_resp(sv, so, sv, so)(X)
        for σ in (α, β):
            self.assertTrue(np.allclose(ax_cpks[σ], ax_core[σ]))

    def test_dipole_b3_cam(self):
        # test B3LYP -> CAM-B3LYP non-consistent functional dipole

        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1).build()

        mf = dft.UKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit").run()
        mf_hdft = UHDFTResp(mf, xc="CAM-B3LYP")
        dip_anal = mf_hdft.make_dipole()

        REF = np.array([0.774919644, 0., 1.0008221174])
        self.assertTrue(np.allclose(dip_anal, REF, atol=1e-5, rtol=1e-7))

        """
        # generation of numerical result

        def eng_with_dipole_field(t, h):
            mf_scf = dft.UKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_mp = UHDFTResp(mf_scf, xc="CAM-B3LYP").run()
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

    def test_dipole_cam_b3(self):
        # test CAM-B3LYP -> B3LYP non-consistent functional dipole

        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1).build()

        mf = dft.UKS(mol, xc="CAM-B3LYP").density_fit("cc-pVDZ-jkfit").run()
        mf_hdft = UHDFTResp(mf, xc="B3LYPg")
        dip_anal = mf_hdft.make_dipole()

        REF = np.array([0.7679688808, 0., 0.9918449684])
        self.assertTrue(np.allclose(dip_anal, REF, atol=1e-5, rtol=1e-7))

        """
        # generation of numerical result

        def eng_with_dipole_field(t, h):
            mf_scf = dft.UKS(mol, xc="CAM-B3LYP").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_mp = UHDFTResp(mf_scf, xc="B3LYPg").run()
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
