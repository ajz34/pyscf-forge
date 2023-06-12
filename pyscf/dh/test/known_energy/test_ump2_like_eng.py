import unittest
from pyscf import dh, gto, df


class TestUMP2LikeDH(unittest.TestCase):

    def test_B2PLYP(self):
        # reference: MRCC 2022-03-18
        """
        basis=cc-pVTZ
        calc=B2PLYP
        dfbasis_scf=none

        unit=bohr

        geom
        H
        O 1 R1

        R1=2.00000000000
        """
        REF_ETOT = -75.708153526127
        mol = gto.Mole(atom="H; O 1 2", spin=1, unit="AU", basis="cc-pVTZ", verbose=0).build()
        mf = dh.DH(mol, xc="B2PLYP", route_scf="conv", frozen="FreezeNobleGasCore", auxbasis_ri="cc-pVTZ-ri").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_B2PLYP_D3BJ(self):
        # reference: Gaussian 16, Rev B.01
        """
        #p B2PLYPD3(Full)/cc-pVDZ NoSymm Int(Grid=99590)

        [Title]

        0 2
        N
        H 1 0.94
        H 1 0.94 2 104.5
        """
        REF_ETOT = -0.55803873816579E+02
        mol = gto.Mole(atom="N; H 1 0.94; H 1 0.94 2 104.5", spin=1, basis="cc-pVDZ", verbose=0).build()
        mf = dh.DH(mol, xc="B2PLYP_D3BJ", route_scf="conv", route_mp2="conv").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_B2GPPLYP_D3BJ(self):
        # reference: MRCC 2022-03-18
        # MINP_H2O_cc-pVTZ_DF-RKS_B2GPPLYP-D3
        """
        basis=cc-pVTZ
        calc=B2GPPLYP-D3
        mem=500MB

        unit=bohr
        geom
        H
        N 1 R1
        H 2 R1 1 A

        R1=2.00000000000
        A=104.2458898548
        """
        REF_ETOT = -55.841686701788
        mol = gto.Mole(atom="H; N 1 2.0; H 2 2.0 1 104.2458898548",
                       basis="cc-pVTZ", unit="AU", spin=1, verbose=0).build()
        mf = dh.DH(mol, xc="B2GPPLYP-D3BJ", frozen="FreezeNobleGasCore").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_mPW2PLYP(self):
        # reference: Gaussian 16, Rev B.01
        """
        #p mPW2PLYP(Full)/cc-pVDZ NoSymm Int(Grid=99590)

        [Title]

        0 2
        N
        H 1 0.94
        H 1 0.94 2 104.5
        """
        REF_ETOT = -0.55803835672023E+02
        mol = gto.Mole(atom="N; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ", spin=1, verbose=0).build()
        mf = dh.DH(mol, xc="mPW2PLYP", route_scf="conv", route_mp2="conv").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_PBE0_DH(self):
        # reference: Gaussian 16, Rev B.01
        """
        #p PBE0DH(Full)/cc-pVDZ NoSymm Int(Grid=99590)

        [Title]

        0 2
        N
        H 1 0.94
        H 1 0.94 2 104.5
        """
        REF_ETOT = -0.55793787299991E+02
        mol = gto.Mole(atom="N; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ", spin=1, verbose=0).build()
        mf = dh.DH(mol, xc="PBE0-DH-Gaussian", route_scf="conv", route_mp2="conv").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_PBE_QIDH_GAUSSIAN(self):
        # reference: Gaussian 16, Rev B.01
        """
        #p PBEQIDH(Full)/cc-pVDZ NoSymm Int(Grid=99590)

        [Title]

        0 2
        N
        H 1 0.94
        H 1 0.94 2 104.5
        """
        REF_ETOT = -0.55777013027558E+02
        mol = gto.Mole(atom="N; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ", spin=1, verbose=0).build()
        mf = dh.DH(mol, xc="PBE-QIDH-Gaussian", route_scf="conv", route_mp2="conv").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_PBE_QIDH_QCHEM(self):
        # reference: QChem 5.1.1
        """
        $molecule
        0 2
        N
        H 1 0.94
        H 1 0.94 2 104.5
        $end

        $rem
        JOBTYPE   sp
        EXCHANGE  PBE-QIDH
        BASIS     cc-pVDZ
        SCF_CONVERGENCE 8
        XC_GRID 000099000590
        $end
        """
        REF_ETOT = -55.77713676
        mol = gto.Mole(atom="N; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ", spin=1, verbose=0).build()
        mf = dh.DH(mol, xc="PBE-QIDH-QChem", route_scf="conv", route_mp2="conv").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_LS1DH_PBE(self):
        # reference: QChem 5.1.1
        """
        $molecule
        0 2
        N
        H 1 0.94
        H 1 0.94 2 104.5
        $end

        $rem
        JOBTYPE   sp
        EXCHANGE  LS1DH-PBE
        BASIS     cc-pVDZ
        SCF_CONVERGENCE 8
        XC_GRID 000099000590
        $end
        """
        REF_ETOT = -55.76805985
        mol = gto.Mole(atom="N; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ", spin=1, verbose=0).build()
        mf = dh.DH(mol, xc="LS1DH-PBE", route_scf="conv", route_mp2="conv").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_PBE0_2(self):
        # reference: QChem 5.1.1
        """
        $molecule
        0 2
        N
        H 1 0.94
        H 1 0.94 2 104.5
        $end

        $rem
        JOBTYPE   sp
        EXCHANGE  PBE0-2
        BASIS     cc-pVDZ
        SCF_CONVERGENCE 8
        XC_GRID 000099000590
        $end
        """
        REF_ETOT = -55.75948452
        mol = gto.Mole(atom="N; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ", spin=1, verbose=0).build()
        mf = dh.DH(mol, xc="PBE0-2", route_scf="conv", route_mp2="conv").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_DSD_PBEP86_D3(self):
        # reference: QChem 5.1.1
        """
        $molecule
        0 2
        N
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
        REF_ETOT = -55.68350335
        mol = gto.Mole(atom="N; H 1 0.94; H 1 0.94 2 104.5", spin=1, basis="6-31G").build()
        mf = dh.DH(mol, xc="DSD-PBEP86-D3BJ", route_scf="conv", route_mp2="conv").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_DSD_PBEPBE_D3(self):
        # reference: QChem 5.1.1
        """
        $molecule
        0 2
        N
        H 1 0.94
        H 1 0.94 2 104.5
        $end

        $rem
        JOBTYPE   sp
        EXCHANGE  DSD-PBEPBE-D3
        BASIS     6-31G
        SCF_CONVERGENCE 8
        XC_GRID 000099000590
        $end
        """
        REF_ETOT = -55.68647430
        mol = gto.Mole(atom="N; H 1 0.94; H 1 0.94 2 104.5", spin=1, basis="6-31G").build()
        mf = dh.DH(mol, xc="DSD-PBEPBE-D3BJ", route_scf="conv", route_mp2="conv").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_DSD_PBEB95_D3(self):
        # reference: QChem 5.1.1
        """
        $molecule
        0 2
        N
        H 1 0.94
        H 1 0.94 2 104.5
        $end

        $rem
        JOBTYPE   sp
        EXCHANGE  DSD-PBEB95-D3
        BASIS     6-31G
        SCF_CONVERGENCE 8
        XC_GRID 000099000590
        $end
        """
        REF_ETOT = -55.69615797
        mol = gto.Mole(atom="N; H 1 0.94; H 1 0.94 2 104.5", spin=1, basis="6-31G").build()
        mf = dh.DH(mol, xc="DSD-PBEB95-D3BJ", route_scf="conv", route_mp2="conv").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_DSD_BLYP_D3(self):
        # reference: ORCA 5.0.4
        """
        ! DSD-BLYP/2013 D3BJ 6-31G NORI NoFrozenCore

        * gzmt 0 2
        N
        H 1 0.94
        H 1 0.94 2 104.5
        *
        """
        REF_ETOT = -55.716954594253
        mol = gto.Mole(atom="N; H 1 0.94; H 1 0.94 2 104.5", spin=1, basis="6-31G").build()
        mf = dh.DH(mol, xc="DSD-BLYP-D3BJ", route_scf="conv", route_mp2="conv").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_wB2PLYP(self):
        # reference: ORCA 5.0.4
        """
        ! wB2PLYP 6-31G NORI NoFrozenCore

        * gzmt 0 2
        N
        H 1 0.94
        H 1 0.94 2 104.5
        *
        """
        REF_ETOT = -55.695205388045
        mol = gto.Mole(atom="N; H 1 0.94; H 1 0.94 2 104.5", spin=1, basis="6-31G").build()
        mf = dh.DH(mol, xc="wB2PLYP", route_scf="conv", route_mp2="conv").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_wB2GPPLYP(self):
        # reference: ORCA 5.0.4
        """
        ! wB2GP-PLYP 6-31G NORI NoFrozenCore

        * gzmt 0 2
        N
        H 1 0.94
        H 1 0.94 2 104.5
        *
        """
        REF_ETOT = -55.691810769285
        mol = gto.Mole(atom="N; H 1 0.94; H 1 0.94 2 104.5", spin=1, basis="6-31G").build()
        mf = dh.DH(mol, xc="wB2GPPLYP", route_scf="conv", route_mp2="conv").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_RSX_0DH(self):
        # reference: ORCA 5.0.4
        """
        ! RSX-0DH 6-31G NORI NoFrozenCore

        * gzmt 0 2
        N
        H 1 0.94
        H 1 0.94 2 104.5
        *
        """
        REF_ETOT = -55.700962329751
        mol = gto.Mole(atom="N; H 1 0.94; H 1 0.94 2 104.5", spin=1, basis="6-31G").build()
        mf = dh.DH(mol, xc="RSX-0DH", route_scf="conv", route_mp2="conv").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_RSX_QIDH(self):
        # reference: ORCA 5.0.4
        """
        ! RSX-QIDH 6-31G NORI NoFrozenCore

        * gzmt 0 2
        N
        H 1 0.94
        H 1 0.94 2 104.5
        *
        """
        REF_ETOT = -55.689372372061
        mol = gto.Mole(atom="N; H 1 0.94; H 1 0.94 2 104.5", spin=1, basis="6-31G").build()
        mf = dh.DH(mol, xc="RSX-QIDH", route_scf="conv", route_mp2="conv").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_XYG3(self):
        # reference: MRCC 2022-03-18
        """
        basis=cc-pVTZ
        calc=XYG3

        unit=bohr
        geom
        H
        N 1 R1
        H 2 R1 1 A

        R1=2.00000000000
        A=104.2458898548
        """
        REF_ETOT = -55.863759071398

        mol = gto.Mole(atom="N; H 1 2; H 1 2 2 104.2458898548", spin=1, unit="AU", basis="cc-pVTZ").build()
        mf = dh.DH(mol, xc="XYG3", frozen="FreezeNobleGasCore").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_RS_PBE_P86(self):
        # reference: MRCC 2022-03-18
        # test case: MINP_NH2_aug-cc-pVDZ_RS-PBE-P86
        """
        # RS-DH DFT calculation for water using a PBE-P86-based functional, ansatz of Toulouse,
        # as well as the default lambda=0.5 and mu=0.7 parameters
        basis=aug-cc-pVDZ
        calc=SCF
        dft=RS-PBE-P86
        mem=1GB
        test=-5.576733720503658E+01

        unit=bohr
        geom=xyz
        3

        N     0.00000000    0.00000000   -0.12502304
        H     0.00000000    1.43266384    0.99210317
        H     0.00000000   -1.43266384    0.99210317
        """
        REF_ETOT = -5.576733720503658E+01
        mol = gto.Mole(
            atom="""
                N     0.00000000    0.00000000   -0.12502304
                H     0.00000000    1.43266384    0.99210317
                H     0.00000000   -1.43266384    0.99210317""",
            basis="aug-cc-pVDZ", spin=1, unit="AU").build()
        mf = dh.DH(mol, xc="RS-PBE-P86", frozen="FreezeNobleGasCore").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_RS_PBE_PBE(self):
        # reference: MRCC 2022-03-18
        """
        basis=aug-cc-pVDZ
        calc=SCF
        dft=RS-PBE-PBE

        unit=bohr
        geom=xyz
        3

        N     0.00000000    0.00000000   -0.12502304
        H     0.00000000    1.43266384    0.99210317
        H     0.00000000   -1.43266384    0.99210317
        """
        REF_ETOT = -55.753154450773
        mol = gto.Mole(
            atom="""
                N     0.00000000    0.00000000   -0.12502304
                H     0.00000000    1.43266384    0.99210317
                H     0.00000000   -1.43266384    0.99210317""",
            basis="aug-cc-pVDZ", spin=1, unit="AU").build()
        mf = dh.DH(mol, xc="RS-PBE-PBE", frozen="FreezeNobleGasCore").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_RS_B88_LYP(self):
        # reference: MRCC 2022-03-18
        """
        basis=aug-cc-pVDZ
        calc=SCF
        dft=RS-B88-LYP

        unit=bohr
        geom=xyz
        3

        N     0.00000000    0.00000000   -0.12502304
        H     0.00000000    1.43266384    0.99210317
        H     0.00000000   -1.43266384    0.99210317
        """
        REF_ETOT = -55.775322590985
        mol = gto.Mole(
            atom="""
                N     0.00000000    0.00000000   -0.12502304
                H     0.00000000    1.43266384    0.99210317
                H     0.00000000   -1.43266384    0.99210317""",
            basis="aug-cc-pVDZ", spin=1, unit="AU").build()
        mf = dh.DH(mol, xc="RS-B88-LYP", frozen="FreezeNobleGasCore").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_RS_PW91_PW91(self):
        # reference: MRCC 2022-03-18
        """
        basis=aug-cc-pVDZ
        calc=SCF
        dft=RS-PW91-PW91

        unit=bohr
        geom=xyz
        3

        N     0.00000000    0.00000000   -0.12502304
        H     0.00000000    1.43266384    0.99210317
        H     0.00000000   -1.43266384    0.99210317
        """
        REF_ETOT = -55.773193557166
        mol = gto.Mole(
            atom="""
                N     0.00000000    0.00000000   -0.12502304
                H     0.00000000    1.43266384    0.99210317
                H     0.00000000   -1.43266384    0.99210317""",
            basis="aug-cc-pVDZ", spin=1, unit="AU").build()
        mf = dh.DH(mol, xc="RS-PW91-PW91", frozen="FreezeNobleGasCore").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)
