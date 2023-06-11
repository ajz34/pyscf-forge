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
        # reference: MRCC 2022-03-18
        # test case: MINP_H2O_cc-pVTZ_RKS_B2PLYP
        """
        # RKS/B2PLYP calculation for H2O with the cc-pVTZ basis set, no density-fitting
        basis=cc-pVTZ
        calc=B2PLYP
        dfbasis_scf=none
        mem=500MB

        test=-76.391961060726

        unit=bohr
        geom
        H
        O 1 R1
        H 2 R1 1 A

        R1=2.00000000000
        A=104.2458898548
        """
        REF_ETOT = -76.391961060726

        mol = gto.Mole(atom="H; O 1 2.0; H 2 2.0 1 104.2458898548", basis="cc-pVTZ", unit="AU", verbose=0).build()
        mf = dh.DH(mol, xc="B2PLYP", route_scf="conv", frozen="FreezeNobleGasCore", auxbasis_ri="cc-pVTZ-ri").run()
        self.assertFalse(hasattr(mf.scf, "with_df"))
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)
        print(mf.e_tot - REF_ETOT)

    def test_B2PLYP_D3BJ(self):
        # reference: Gaussian 16, Rev B.01
        """
        #p B2PLYPD3(Full)/cc-pVDZ NoSymm Int(Grid=99590)

        [Title]

        0 1
        O
        H 1 0.94
        H 1 0.94 2 104.5
        """
        REF_ETOT = -0.76352886529203E+02

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ", verbose=0).build()
        mf = dh.DH(mol, xc="B2PLYP_D3BJ", route_scf="conv", route_mp2="conv").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)
        print(mf.e_tot - REF_ETOT)

    def test_B2GPPLYP_D3BJ(self):
        # reference: MRCC 2022-03-18
        # MINP_H2O_cc-pVTZ_DF-RKS_B2GPPLYP-D3
        """
        # DF-RKS/B2GPPLYP-D3 calculation for H2O with the cc-pVTZ basis set, D3 correction with BJ damping
        basis=cc-pVTZ
        calc=B2GPPLYP-D3
        mem=500MB

        test=-76.378332323817

        unit=bohr
        geom
        H
        O 1 R1
        H 2 R1 1 A

        R1=2.00000000000
        A=104.2458898548
        """
        REF_ETOT = -76.378332323817

        mol = gto.Mole(atom="H; O 1 2.0; H 2 2.0 1 104.2458898548", basis="cc-pVTZ", unit="AU", verbose=0).build()
        mf = dh.DH(mol, xc="B2GPPLYP-D3BJ", frozen="FreezeNobleGasCore").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_DSD_PBEP86_D3(self):
        # reference: QChem 5.1.1
        """
        $molecule
        0 1
        O
        H 1 0.94
        H 1 0.94 2 104.5
        $end

        $rem
        JOBTYPE   sp
        EXCHANGE  DSD-PBEP86-D3
        BASIS     6-31G
        SCF_CONVERGENCE 8
        XC_GRID 000099000590
        $end
        """
        REF_ETOT = -76.20159807
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = dh.DH(mol, xc="DSD-PBEP86-D3BJ", route_scf="conv", route_mp2="conv").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_XYG3(self):
        # reference: MRCC
        # MINP_H2O_cc-pVTZ_XYG3
        REF_ETOT = -76.400701189006

        mol = self.mol
        mf = dh.DH(mol, xc="XYG3") \
            .build_scf(route_scf="ri", auxbasis_jk="cc-pVTZ-jkfit") \
            .run(frozen="FreezeNobleGasCore", auxbasis_ri="cc-pVTZ-ri")
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_SCAN0_2(self):
        # reference: MRCC
        # MINP_H2O_cc-pVTZ_SCAN0-2_Libxc
        # TODO: SCAN seems to be very instable for different softwares.
        REF_ESCF = -76.204558509844
        REF_ETOT = -76.348414592594

        mol = self.mol
        mf = dh.DH(mol, xc="SCAN0-2") \
            .build_scf(route_scf="ri", auxbasis_jk="cc-pVTZ-jkfit") \
            .run(frozen="FreezeNobleGasCore", auxbasis_ri="cc-pVTZ-ri")
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
        mf = dh.DH(mol, xc="lrc-XYG3") \
            .build_scf(route_scf="conv") \
            .run(route_mp2="conv")
        self.assertAlmostEqual(mf._scf.e_tot, REF_ESCF, places=5)
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

