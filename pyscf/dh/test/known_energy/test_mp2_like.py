import unittest
from pyscf import dh, gto, df


"""
Comparison value from
- MRCC 2022-03-18.
- Q-Chem (development ver)
"""


class TestRMP2LikeDH(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        coord = """
        H
        O 1 R1
        H 2 R1 1 A
        """.replace("R1", "2.0").replace("A", "104.2458898548")

        mol = gto.Mole(atom=coord, basis="cc-pVTZ", unit="AU", verbose=0).build()
        cls.mol = mol

    def test_B2PLYP(self):
        # reference: MRCC
        # test case: MINP_H2O_cc-pVTZ_RKS_B2PLYP
        REF_ESCF = -76.305197382056
        REF_ETOT = -76.391961061470

        flags = {
            "integral_scheme_scf": "Conv",
            "frozen_rule": "FreezeNobleGasCore",
            "auxbasis_ri": "cc-pVTZ-ri"}
        mol = self.mol
        mf = dh.RDH(mol, xc="B2PLYP", flags=flags)
        mf.with_df = df.DF(mol, auxbasis="cc-pVTZ-ri").run()
        self.assertAlmostEqual(mf._scf.e_tot, REF_ESCF, places=5)
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_B2GPPLYP(self):
        # reference: MRCC
        # MINP_H2O_cc-pVTZ_DF-RKS_B2GPPLYP-D3
        # without DFT-D3
        REF_ESCF = -76.268047709113
        REF_ETOT = -76.378191035928

        flags = {
            "integral_scheme_scf": "RI-JK",
            "frozen_rule": "FreezeNobleGasCore",
            "auxbasis_jk": "cc-pVTZ-jkfit",
            "auxbasis_ri": "cc-pVTZ-ri"}
        mol = self.mol
        mf = dh.RDH(mol, xc="B2GPPLYP", flags=flags).run()
        self.assertAlmostEqual(mf._scf.e_tot, REF_ESCF, places=5)
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_DSDPBEP86(self):
        # reference: MRCC
        # MINP_H2O_cc-pVTZ_DF-RKS_DSDPBEP86-D3
        # without DFT-D3
        # TODO: MRCC may uses an older version of DSD-PBEP86 (10.1039/C1CP22592H).
        REF_ESCF = -76.186838177949
        REF_ETOT = -76.325115287231

        flags={
            "integral_scheme_scf": "RI-JK",
            "frozen_rule": "FreezeNobleGasCore",
            "auxbasis_jk": "cc-pVTZ-jkfit",
            "auxbasis_ri": "cc-pVTZ-ri"}
        mol = self.mol
        mf = dh.RDH(mol, xc="DSD-PBEP86-D3", flags=flags).run()
        print()
        print(mf._scf.e_tot)
        print(mf.e_tot)
        # self.assertAlmostEqual(mf.mf.e_tot, REF_ESCF, places=5)
        # self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_XYG3(self):
        # reference: MRCC
        # MINP_H2O_cc-pVTZ_XYG3
        REF_ETOT = -76.400701189006

        flags={
            "integral_scheme_scf": "RI-JK",
            "frozen_rule": "FreezeNobleGasCore",
            "auxbasis_jk": "cc-pVTZ-jkfit",
            "auxbasis_ri": "cc-pVTZ-ri"}
        mol = self.mol
        mf = dh.RDH(mol, xc="XYG3", flags=flags).run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_SCAN0_2(self):
        # reference: MRCC
        # MINP_H2O_cc-pVTZ_SCAN0-2_Libxc
        # TODO: SCAN seems to be very instable for different softwares.
        REF_ESCF = -76.204558509844
        REF_ETOT = -76.348414592594

        flags={
            "integral_scheme_scf": "RI-JK",
            "frozen_rule": "FreezeNobleGasCore",
            "auxbasis_jk": "cc-pVTZ-jkfit",
            "auxbasis_ri": "cc-pVTZ-ri"}
        mol = self.mol
        mf = dh.RDH(mol, xc="SCAN", flags=flags).run()
        print()
        print(mf._scf.e_tot)
        print(mf.e_tot)
        # self.assertAlmostEqual(mf.mf.e_tot, REF_ESCF, places=5)
        # self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_LRC_XYG3(self):
        # reference: Q-Chem
        REF_ESCF = -109.5673226196
        REF_ETOT = -109.5329015372
        mol = gto.Mole(
            atom="N 0 0 0.54777500; N 0 0 -0.54777500",
            basis="6-311+G(3df,2p)").build()
        params = dh.util.Params(flags={
            "integral_scheme_scf": "Conv",
            "integral_scheme": "Conv",
        })
        mf = dh.RDH(mol, xc="lrc-XYG3", params=params).run()
        self.assertAlmostEqual(mf._scf.e_tot, REF_ESCF, places=5)
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)


class TestUMP2LikeDH(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        coord = """
        H
        O 1 R1
        """.replace("R1", "2.0")

        mol = gto.Mole(atom=coord, basis="cc-pVTZ", unit="AU", spin=1, verbose=0).build()
        cls.mol = mol

    def test_B2PLYP(self):
        # reference: MRCC
        # test input
        #    basis=cc-pVTZ
        #    calc=B2PLYP
        #    dfbasis_scf=none
        #
        #    unit=bohr
        #    geom
        #    H
        #    O 1 R1
        #
        #    R1=2.00000000000
        REF_ESCF = -75.643314480794
        REF_ETOT = -75.708153526127

        flags = {
            "integral_scheme_scf": "Conv",
            "frozen_rule": "FreezeNobleGasCore",
            "auxbasis_ri": "cc-pVTZ-ri"}
        mol = self.mol
        mf = dh.UDH(mol, xc="B2PLYP", flags=flags).run()
        self.assertAlmostEqual(mf._scf.e_tot, REF_ESCF, places=5)
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_XYG3(self):
        # reference: MRCC
        # test input
        #    basis=cc-pVTZ
        #    calc=XYG3
        #
        #    unit=bohr
        #    geom
        #    H
        #    O 1 R1
        #
        #    R1=2.00000000000
        REF_ETOT = -75.715897359518

        flags = {
            "frozen_rule": "FreezeNobleGasCore",
            "auxbasis_jk": "cc-pVTZ-jkfit",
            "auxbasis_ri": "cc-pVTZ-ri"}
        mol = self.mol
        mf = dh.UDH(mol, xc="XYG3", flags=flags).run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)