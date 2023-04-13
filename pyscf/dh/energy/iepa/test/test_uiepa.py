import unittest
from pyscf import gto, scf, mp, df, dft, dh
from pyscf.dh import RIEPAConv, UIEPAConv, RIEPARI, UIEPARI
import numpy as np


def get_mf_h2o_hf():
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVTZ").build()
    return scf.RHF(mol).run(conv_tol=1e-12), scf.UHF(mol).run(conv_tol=1e-12)


def get_mf_h2o_cation_hf():
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVTZ").build()
    return scf.UHF(mol).run(conv_tol=1e-12)


class TestEngUIEPA(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mf_h2o_hf_res, cls.mf_h2o_hf = get_mf_h2o_hf()
        cls.mf_h2o_cation_hf = get_mf_h2o_cation_hf()

    def test_eng_uiepa_conv_by_riepa(self):
        mf_s = self.mf_h2o_hf
        mf_s_res = self.mf_h2o_hf_res
        mf_dh = UIEPAConv(mf_s).run(iepa_schemes=["MP2", "MP2CR", "IEPA", "SIEPA"])
        mf_dh_res = RIEPAConv(mf_s_res).run(iepa_schemes=["MP2", "MP2CR", "IEPA", "SIEPA"])
        keys = [f"eng_corr_{scheme}" for scheme in ["MP2", "MP2CR", "IEPA", "SIEPA"]]
        results_omega_0 = mf_dh.results
        for key in keys:
            self.assertAlmostEqual(mf_dh.results[key], mf_dh_res.results[key], 8)

        mf_dh = UIEPAConv(mf_s).run(iepa_schemes=["MP2", "MP2CR", "IEPA", "SIEPA"], omega=0.7)
        mf_dh_res = RIEPAConv(mf_s_res).run(iepa_schemes=["MP2", "MP2CR", "IEPA", "SIEPA"], omega=0.7)
        keys = [f"eng_corr_{scheme}" for scheme in ["MP2", "MP2CR", "IEPA", "SIEPA"]]
        results_omega_0p7 = mf_dh.results
        for key in keys:
            self.assertAlmostEqual(mf_dh.results[key], mf_dh_res.results[key], 8)

        for key in keys:
            self.assertNotAlmostEqual(results_omega_0[key], results_omega_0p7[key], 3)

    def test_eng_uiepa_ri_by_riepa(self):
        mf_s = self.mf_h2o_hf
        mf_s_res = self.mf_h2o_hf_res
        mf_dh = UIEPARI(mf_s).run(iepa_schemes=["MP2", "MP2CR", "IEPA", "SIEPA"])
        mf_dh_res = RIEPARI(mf_s_res).run(iepa_schemes=["MP2", "MP2CR", "IEPA", "SIEPA"])
        keys = [f"eng_corr_{scheme}" for scheme in ["MP2", "MP2CR", "IEPA", "SIEPA"]]
        results_omega_0 = mf_dh.results
        for key in keys:
            self.assertAlmostEqual(mf_dh.results[key], mf_dh_res.results[key], 8)

        mf_dh = UIEPARI(mf_s).run(iepa_schemes=["MP2", "MP2CR", "IEPA", "SIEPA"], omega=0.7)
        mf_dh_res = RIEPARI(mf_s_res).run(iepa_schemes=["MP2", "MP2CR", "IEPA", "SIEPA"], omega=0.7)
        keys = [f"eng_corr_{scheme}" for scheme in ["MP2", "MP2CR", "IEPA", "SIEPA"]]
        results_omega_0p7 = mf_dh.results
        for key in keys:
            self.assertAlmostEqual(mf_dh.results[key], mf_dh_res.results[key], 8)

        for key in keys:
            self.assertNotAlmostEqual(results_omega_0[key], results_omega_0p7[key], 3)

    def test_eng_uiepa_coverage(self):
        # coverage only, not testing correctness
        mf_s = self.mf_h2o_cation_hf

        mf_dh = UIEPAConv(mf_s).run(iepa_schemes=["MP2", "MP2CR", "IEPA", "SIEPA", "DCPT2"], omega=0.7).run()
        mf_dh = UIEPARI(mf_s).run(iepa_schemes=["MP2", "MP2CR", "IEPA", "SIEPA", "DCPT2"], omega=0.7).run()

        eri = mf_s._eri.copy()
        mf_s._eri = None

        mf_dh = UIEPAConv(mf_s).run(iepa_schemes="MP2cr")
        mf_dh = UIEPARI(mf_s).run(iepa_schemes="MP2cr")

        with self.assertRaises(ValueError):
            UIEPAConv(mf_s).run(iepa_schemes="RPA")
            UIEPARI(mf_s).run(iepa_schemes="RPA")

        with self.assertRaises(NotImplementedError):
            UIEPAConv(mf_s).run(iepa_schemes="MP2cr2")
            UIEPARI(mf_s).run(iepa_schemes="MP2cr2")

        mf_s._eri = eri
