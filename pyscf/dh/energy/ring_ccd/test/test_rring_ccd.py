import unittest
from pyscf import gto, scf, df
from pyscf.dh import RRingCCDofDH


def get_mf_h2o_hf_mrcc():
    coord = """
    H
    O 1 R1
    H 2 R1 1 A
    """.replace("R1", "2.0").replace("A", "104.2458898548")

    mol = gto.Mole(atom=coord, basis="cc-pVTZ", unit="AU").build()
    return scf.RHF(mol).density_fit(auxbasis="cc-pVTZ-jkfit").run()


class TestEngRRingCCD(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mf_h2o_hf_mrcc = get_mf_h2o_hf_mrcc()

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
