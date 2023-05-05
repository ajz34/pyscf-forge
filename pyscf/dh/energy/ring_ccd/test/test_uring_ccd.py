import unittest
from pyscf import gto, scf, df, dft
from pyscf.dh import URingCCDConv


def get_mf_oh_hf_mrcc():
    coord = """
    H
    O 1 R1
    """.replace("R1", "2.0")

    mol = gto.Mole(atom=coord, basis="cc-pVTZ", unit="AU", spin=1).build()
    return scf.UHF(mol).density_fit(auxbasis="cc-pVTZ-jkfit").run()


def get_mf_oh_drpa75_mrcc():
    coord = """
    H
    O 1 R1
    """.replace("R1", "2.0")

    mol = gto.Mole(atom=coord, basis="aug-cc-pVTZ", unit="AU", spin=1).build()
    return dft.UKS(mol, xc="0.75*HF + 0.25*PBE, PBE").density_fit(auxbasis="aug-cc-pVTZ-jkfit").run()


class TestEngURingCCD(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mf_oh_hf_mrcc = get_mf_oh_hf_mrcc()
        cls.mf_oh_drpa75_mrcc = get_mf_oh_drpa75_mrcc()

    def test_eng_ring_ccd_known(self):
        # mrcc: MINP_OH_cc-pVTZ_dRPA
        mf_s = self.mf_oh_hf_mrcc
        mol = mf_s.mol
        # cheat to get ri-eri and evaluate by conv
        mf_s._eri = df.DF(mol, auxbasis="cc-pVTZ-ri").get_ao_eri()
        mf_dh = URingCCDConv(mf_s).run(frozen="FreezeNobleGasCore")
        REF_MRCC = -0.254148360718
        self.assertAlmostEqual(mf_dh.results["eng_corr_RING_CCD"], REF_MRCC, 5)

    def test_eng_drpa75_known(self):
        # mrcc: MINP_OH_aug-cc-pVTZ_dRPA75
        # a test case of how to evaluate doubly hybrid without handling xc code in package
        mf_s = self.mf_oh_drpa75_mrcc
        mol = mf_s.mol
        # cheat to get ri-eri and evaluate by conv
        mf_s._eri = df.DF(mol, auxbasis="aug-cc-pVTZ-ri").get_ao_eri()
        mf_dh = URingCCDConv(mf_s).run(frozen="FreezeNobleGasCore")
        mf_n = dft.RKS(mol).set(mo_coeff=mf_s.mo_coeff, mo_occ=mf_s.mo_occ, xc="0.75*HF + 0.25*PBE, ")
        eng_low_rung = mf_n.energy_tot(dm=mf_s.make_rdm1())
        eng_rring_ccd = mf_dh.results["eng_corr_RING_CCD"]
        eng_rring_ccd_os = mf_dh.results["eng_corr_RING_CCD_OS"]
        eng_rring_ccd_ss = mf_dh.results["eng_corr_RING_CCD_SS"]
        eng_tot = eng_low_rung + eng_rring_ccd
        eng_scs = eng_low_rung + 1.5 * eng_rring_ccd_os + 0.5 * eng_rring_ccd_ss
        REF_MRCC_dRPA75 = -75.683803588262
        REF_MRCC_SCS_dRPA75 = -75.678981497985
        self.assertAlmostEqual(eng_tot, REF_MRCC_dRPA75, 5)
        self.assertAlmostEqual(eng_scs, REF_MRCC_SCS_dRPA75, 5)

