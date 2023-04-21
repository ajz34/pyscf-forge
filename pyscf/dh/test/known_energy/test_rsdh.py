import unittest
from pyscf import dh, gto


"""
Comparison value from
- MRCC 2022-03-18.
"""


class TestRSDH(unittest.TestCase):
    def test_RS_PBE_P86(self):
        # reference: MRCC
        # test case: MINP_H2O_cc-pVTZ_RKS_B2PLYP
        REF_ESCF = -76.219885498301
        REF_ETOT = -76.315858865489

        mol = gto.Mole(atom="""
        O     0.00000000    0.00000000   -0.12502304
        H     0.00000000    1.43266384    0.99210317
        H     0.00000000   -1.43266384    0.99210317
        """, basis="aug-cc-pVDZ", unit="AU").build()
        mf = dh.DH(mol, xc="RS-PBE-P86").run(frozen="FreezeNobleGasCore")
        self.assertAlmostEqual(mf._scf.e_tot, REF_ESCF, places=5)
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)