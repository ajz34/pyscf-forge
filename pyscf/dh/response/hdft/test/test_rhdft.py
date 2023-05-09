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
        ax_cpks = mf_scf.Ax0_cpks()(X)
        ax_core = mf_scf.Ax0_Core_resp(sv, so, sv, so)(X)
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
        ax_cpks = mf_scf.Ax0_Core(sv, so, sv, so)(X)
        ax_core = mf_scf.Ax0_Core_resp(sv, so, sv, so)(X)
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
        ax_cpks = mf_scf.Ax0_Core(sv, so, sv, so)(X)
        ax_core = mf_scf.Ax0_Core_resp(sv, so, sv, so)(X)
        self.assertTrue(np.allclose(ax_cpks, ax_core))

