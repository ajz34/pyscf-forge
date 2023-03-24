import unittest
from pyscf import gto, scf, df, dft
from pyscf.dh import RRingCCDofDH


def get_mf_h2o_hf_mrcc():
    coord = """
    H
    O 1 R1
    H 2 R1 1 A
    """.replace("R1", "2.0").replace("A", "104.2458898548")

    mol = gto.Mole(atom=coord, basis="cc-pVTZ", unit="AU").build()
    return scf.RHF(mol).density_fit(auxbasis="cc-pVTZ-jkfit").run()


def get_mf_h2o_drpa75_mrcc():
    coord = """
    H
    O 1 R1
    H 2 R1 1 A
    """.replace("R1", "2.0").replace("A", "104.2458898548")

    mol = gto.Mole(atom=coord, basis="aug-cc-pVTZ", unit="AU").build()
    return dft.RKS(mol, xc="0.75*HF + 0.25*PBE, PBE").density_fit(auxbasis="aug-cc-pVTZ-jkfit").run()


class TestEngRRingCCD(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mf_h2o_hf_mrcc = get_mf_h2o_hf_mrcc()
        cls.mf_h2o_drpa75_mrcc = get_mf_h2o_drpa75_mrcc()

    def test_eng_rring_ccd_known(self):
        # mrcc: MINP_H2O_cc-pVTZ_dRPA
        mf_s = self.mf_h2o_hf_mrcc
        mol = mf_s.mol
        # cheat to get ri-eri and evaluate by conv
        mf_s._eri = df.DF(mol, auxbasis="cc-pVTZ-ri").get_ao_eri()
        mf_dh = RRingCCDofDH(mf_s)
        mf_dh.kernel(integral_scheme_ring_ccd="conv", frozen_rule="FreezeNobleGasCore")
        REF_MRCC = -0.312651518707
        self.assertAlmostEqual(mf_dh.params.results["eng_corr_RING_CCD"], REF_MRCC, 5)

    def test_eng_drpa75_known(self):
        # mrcc: MINP_H2O_aug-cc-pVTZ_dRPA75
        # a test case of how to evaluate doubly hybrid without handling xc code in package
        mf_s = self.mf_h2o_drpa75_mrcc
        mol = mf_s.mol
        # cheat to get ri-eri and evaluate by conv
        mf_s._eri = df.DF(mol, auxbasis="aug-cc-pVTZ-ri").get_ao_eri()
        mf_dh = RRingCCDofDH(mf_s)
        eng_low_rung = mf_dh.make_energy_dh(xc="0.75*HF + 0.25*PBE, ")
        results_ring_ccd = mf_dh.kernel(integral_scheme_ring_ccd="conv", frozen_rule="FreezeNobleGasCore")
        eng_rring_ccd = results_ring_ccd["eng_corr_RING_CCD"]
        eng_tot = eng_low_rung + eng_rring_ccd
        REF_MRCC = -76.377085365919
        self.assertAlmostEqual(eng_tot, REF_MRCC, 5)
