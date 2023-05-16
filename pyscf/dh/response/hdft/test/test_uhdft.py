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
