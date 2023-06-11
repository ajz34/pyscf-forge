import unittest
from pyscf import dh, gto, df


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

        mol = self.mol
        mf = dh.DH(mol, xc="B2PLYP", route_scf="conv", frozen="FreezeNobleGasCore", auxbasis_ri="cc-pVTZ-ri").run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)
        print(mf.e_tot - REF_ETOT)

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
        print(mf.e_tot - REF_ETOT)

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

        mol = self.mol
        mf = dh.DH(mol, xc="XYG3") \
            .build_scf(route_scf="ri", auxbasis_jk="cc-pVTZ-jkfit") \
            .run(frozen="FreezeNobleGasCore", auxbasis_ri="cc-pVTZ-ri")
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)
