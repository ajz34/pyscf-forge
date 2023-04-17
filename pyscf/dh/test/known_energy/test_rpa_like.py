import unittest
from pyscf import gto, dh, df


class TestRPALike(unittest.TestCase):
    def test_dRPA75(self):
        # reference: MRCC
        # MINP_H2O_aug-cc-pVTZ_dRPA75
        # SCF energy of xDH-like in MRCC seems to be tricky
        # REF_ESCF = -76.036966443411
        REF_ETOT = -76.377085365919

        coord = """
        H
        O 1 R1
        H 2 R1 1 A
        """.replace("R1", "2.0").replace("A", "104.2458898548")
        mol = gto.Mole(atom=coord, basis="aug-cc-pVTZ", unit="AU", verbose=0).build()

        mf = dh.DH(mol, xc="dRPA75").build_scf(route_scf="ri", auxbasis_jk="aug-cc-pVTZ-jkfit").build()
        # cheat eri evaluation, since currently RI-ring-CCD is not implemented
        with_df = df.DF(mol, auxbasis="aug-cc-pVTZ-ri").build()
        mf.scf._eri = with_df.get_eri()
        # run drpa evaluation
        mf.run(frozen="FreezeNobleGasCore")
        print(mf._scf.e_tot)
        print(mf.e_tot)
        # self.assertAlmostEqual(mf._scf.e_tot, REF_ESCF, places=5)
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)
