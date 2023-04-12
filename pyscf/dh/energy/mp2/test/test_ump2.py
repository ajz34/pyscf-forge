import unittest
from pyscf import gto, scf, mp, df
from pyscf.mp.dfump2_native import DFUMP2
from pyscf.dh.energy.mp2 import UMP2RI, UMP2Conv, UMP2ConvPySCF
import numpy as np


def get_mf_h2o_cation_hf():
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVTZ", spin=1, charge=1).build()
    return scf.UHF(mol).run()


def get_mf_h2o_cation_hf_complex():
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVTZ", spin=1, charge=1).build()

    hcore_1_B = - 1j * (
        + 0.5 * mol.intor('int1e_giao_irjxp', 3)
        + mol.intor('int1e_ignuc', 3)
        + mol.intor('int1e_igkin', 3))
    ovlp_1_B = - 1j * mol.intor("int1e_igovlp")
    eri_1_B = -1j * (
        + np.einsum("tuvkl -> tuvkl", mol.intor('int2e_ig1'))
        + np.einsum("tkluv -> tuvkl", mol.intor('int2e_ig1')))

    mf_s = scf.UHF(mol)
    dev_xyz_B = np.array([1e-2, 2e-2, -1e-2])

    def get_hcore(mol_=mol):
        hcore_total = np.asarray(scf.rhf.get_hcore(mol_), dtype=np.complex128)
        hcore_total += np.einsum("tuv, t -> uv", hcore_1_B, dev_xyz_B)
        return hcore_total

    def get_ovlp(mol_=mol):
        ovlp_total = np.asarray(scf.rhf.get_ovlp(mol_), dtype=np.complex128)
        ovlp_total += np.einsum("tuv, t -> uv", ovlp_1_B, dev_xyz_B)
        return ovlp_total

    mf_s.get_hcore = get_hcore
    mf_s.get_ovlp = get_ovlp
    mf_s._eri = mol.intor("int2e") + np.einsum("tuvkl, t -> uvkl", eri_1_B, dev_xyz_B)
    mf_s.run()

    auxmol = df.make_auxmol(mol, df.aug_etb(mol))
    int3c2e = df.incore.aux_e2(mol, auxmol, "int3c2e")
    int3c2e_ig1 = df.incore.aux_e2(mol, auxmol, "int3c2e_ig1")
    int2c2e = auxmol.intor("int2c2e")
    L = np.linalg.cholesky(int2c2e)
    int3c2e_cd = np.linalg.solve(L, int3c2e.reshape(mol.nao ** 2, -1).T).reshape(-1, mol.nao, mol.nao)
    int3c2e_ig1_cd = np.linalg.solve(L, int3c2e_ig1.reshape(3 * mol.nao ** 2, -1).T).reshape(-1, 3, mol.nao, mol.nao)
    int3c2e_2_cd = int3c2e_cd + 2 * np.einsum("Ptuv, t -> Puv", -1j * int3c2e_ig1_cd, dev_xyz_B, optimize=True)

    return mf_s, int3c2e_cd, int3c2e_2_cd


class TestEngUMP2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mf_h2o_hf = get_mf_h2o_cation_hf()
        mf_h2o_hf_complex, int3c2e_cd, int3c2e_2_cd = get_mf_h2o_cation_hf_complex()
        cls.mf_h2o_hf = mf_h2o_hf
        cls.mf_h2o_hf_complex = mf_h2o_hf_complex
        cls.int3c2e_cd = int3c2e_cd
        cls.int3c2e_2_cd = int3c2e_2_cd

    def test_eng_ump2_conv(self):
        mf_s = self.mf_h2o_hf
        mf_dh = UMP2Conv(mf_s).run()
        print(mf_dh.results)
        # reference value
        REF = -0.209174918573074
        REF_PYSCF = mp.MP2(mf_s).run().e_corr
        self.assertAlmostEqual(REF, REF_PYSCF, 8)
        self.assertAlmostEqual(mf_dh.results["eng_corr_MP2"], REF, 8)

    def test_eng_ump2_ri(self):
        mf_s = self.mf_h2o_hf
        mol = mf_s.mol
        mf_dh = UMP2RI(mf_s).run(with_df=df.DF(mol, auxbasis=df.aug_etb(mol)))
        print(mf_dh.results)
        # reference value
        REF = -0.20915836544854347
        REF_PYSCF = DFUMP2(mf_s, auxbasis=df.aug_etb(mol)).run().e_corr
        print(REF_PYSCF)
        self.assertAlmostEqual(REF, REF_PYSCF, 8)
        self.assertAlmostEqual(mf_dh.results["eng_corr_MP2"], REF, 8)

    def test_eng_ump2_conv_fc_1(self):
        mf_s = self.mf_h2o_hf
        mf_dh = UMP2Conv(mf_s).run(frozen="FreezeNobleGasCore", incore_t_oovv_mp2=True)
        print(mf_dh.results)
        # reference value
        REF = -0.195783018787701
        REF_PYSCF = mp.MP2(mf_s, frozen=[0]).run().e_corr
        self.assertAlmostEqual(REF, REF_PYSCF, 8)
        self.assertAlmostEqual(mf_dh.results["eng_corr_MP2"], REF, 8)
        self.assertTrue("t_oovv" in mf_dh.tensors)

    def test_eng_ump2_conv_fc_2(self):
        mf_s = self.mf_h2o_hf
        mf_dh = UMP2Conv(mf_s).run(frozen=[[0, 1], [0, 2]], incore_t_oovv_mp2=True)
        print(mf_dh.results)
        # reference value
        REF = -0.10559463994349409
        REF_PYSCF = mp.MP2(mf_s, frozen=[[0, 1], [0, 2]]).run().e_corr
        self.assertAlmostEqual(REF, REF_PYSCF, 8)
        self.assertAlmostEqual(mf_dh.results["eng_corr_MP2"], REF, 8)

    def test_eng_ump2_ri_fc_1(self):
        mf_s = self.mf_h2o_hf
        mol = mf_s.mol
        mf_dh = UMP2RI(mf_s).run(
            frozen="FreezeNobleGasCore", incore_t_oovv_mp2=True,
            with_df=df.DF(mol, df.aug_etb(mol)))
        print(mf_dh.results)
        # reference value
        REF = -0.19576647277102
        mf_s.with_df = df.DF(mol, auxbasis=df.aug_etb(mol))
        REF_PYSCF = DFUMP2(mf_s, frozen=1, auxbasis=df.aug_etb(mol)).run().e_corr
        self.assertAlmostEqual(REF, REF_PYSCF, 8)
        self.assertAlmostEqual(mf_dh.results["eng_corr_MP2"], REF, 8)

    def test_eng_ump2_ri_fc_2(self):
        mf_s = self.mf_h2o_hf
        mol = mf_s.mol
        mf_dh = UMP2RI(mf_s).run(
            frozen=[[0, 1], [0, 2]], incore_t_oovv_mp2=True,
            with_df=df.DF(mol, df.aug_etb(mol)))
        print(mf_dh.results)
        # generated reference value
        REF = -0.1055960581512246
        #     -0.10559463994349409 in test_eng_rmp2_conv_fc_2
        mf_s.with_df = df.DF(mol, auxbasis=df.aug_etb(mol))
        self.assertAlmostEqual(mf_dh.results["eng_corr_MP2"], REF, 8)

    def test_eng_ump2_conv_complex(self):
        mf_s = self.mf_h2o_hf_complex
        mf_dh = UMP2Conv(mf_s).run()
        REF = -0.209474427130422
        REF_PYSCF = mp.MP2(mf_s).run().e_corr
        self.assertAlmostEqual(REF, REF_PYSCF, 8)
        self.assertAlmostEqual(mf_dh.results["eng_corr_MP2"], REF, 8)

    def test_eng_ump2_ri_complex(self):
        mf_s = self.mf_h2o_hf_complex
        mol = mf_s.mol
        int3c2e_cd = self.int3c2e_cd
        int3c2e_2_cd = self.int3c2e_2_cd

        mf_dh = UMP2RI(mf_s)
        mf_dh.with_df = df.DF(mol, df.aug_etb(mol))
        mf_dh.with_df._cderi = int3c2e_cd
        mf_dh.with_df_2 = df.DF(mol, df.aug_etb(mol))
        mf_dh.with_df_2._cderi = int3c2e_2_cd
        mf_dh.run()

        # generated reference value
        REF = -0.20945698217515063
        #     -0.209474427130422 in test_eng_rmp2_conv_complex
        self.assertAlmostEqual(mf_dh.results["eng_corr_MP2"], REF, 8)

    def test_eng_ump2_coverage(self):
        # coverage only, not testing correctness
        mf_s = self.mf_h2o_hf
        eri = mf_s._eri.copy()

        mf_s._eri = None
        mf_dh = UMP2Conv(mf_s).run(frac_num=np.random.randn(2, mf_s.mo_occ.shape[-1])).run()
        mf_dh = UMP2RI(mf_s).run(frac_num=np.random.randn(2, mf_s.mo_occ.shape[-1])).run()
        mf_s._eri = eri
