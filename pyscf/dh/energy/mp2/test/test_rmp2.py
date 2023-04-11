import unittest
from pyscf import gto, scf, mp, df
from pyscf.dh.energy.mp2 import RMP2RI, RMP2Conv, RMP2ConvPySCF, RMP2RIPySCF
import numpy as np


def get_mf_h2o_hf():
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVTZ").build()
    return scf.RHF(mol).run()


def get_mf_h2o_hf_complex():
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVTZ").build()

    hcore_1_B = - 1j * (
        + 0.5 * mol.intor('int1e_giao_irjxp', 3)
        + mol.intor('int1e_ignuc', 3)
        + mol.intor('int1e_igkin', 3))
    ovlp_1_B = - 1j * mol.intor("int1e_igovlp")
    eri_1_B = -1j * (
        + np.einsum("tuvkl -> tuvkl", mol.intor('int2e_ig1'))
        + np.einsum("tkluv -> tuvkl", mol.intor('int2e_ig1')))

    mf_s = scf.RHF(mol)
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


class TestEngRMP2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mf_h2o_hf = get_mf_h2o_hf()
        mf_h2o_hf_complex, int3c2e_cd, int3c2e_2_cd = get_mf_h2o_hf_complex()
        cls.mf_h2o_hf = mf_h2o_hf
        cls.mf_h2o_hf_complex = mf_h2o_hf_complex
        cls.int3c2e_cd = int3c2e_cd
        cls.int3c2e_2_cd = int3c2e_2_cd

    def test_eng_rmp2_conv(self):
        mf_s = self.mf_h2o_hf
        mf_dh = RMP2Conv(mf_s)
        mf_dh.run()
        print(mf_dh.results)
        # reference value
        REF = -0.273944755130888
        REF_PYSCF = mp.MP2(mf_s).run().e_corr
        self.assertAlmostEqual(REF, REF_PYSCF, 8)
        self.assertAlmostEqual(mf_dh.results["eng_corr_MP2"], REF, 8)

    def test_eng_rmp2_ri(self):
        mf_s = self.mf_h2o_hf
        mol = mf_s.mol
        mf_dh = RMP2RI(mf_s).run(with_df=df.DF(mol, df.aug_etb(mol)))
        print(mf_dh.results)
        # reference value
        REF = -0.2739374133400274
        mf_s.with_df = df.DF(mol, auxbasis=df.aug_etb(mol))
        REF_PYSCF = mp.dfmp2.DFMP2(mf_s).run().e_corr
        print(REF_PYSCF)
        self.assertAlmostEqual(REF, REF_PYSCF, 8)
        self.assertAlmostEqual(mf_dh.results["eng_corr_MP2"], REF, 8)

    def test_eng_rmp2_conv_fc(self):
        mf_s = self.mf_h2o_hf
        mf_dh = RMP2Conv(mf_s).run(frozen="FreezeNobleGasCore", incore_t_oovv_mp2=True)
        mf_dh.run()
        print(mf_dh.results)
        # reference value
        REF = -0.2602324295391498
        REF_PYSCF = mp.MP2(mf_s, frozen=[0]).run().e_corr
        self.assertAlmostEqual(REF, REF_PYSCF, 8)
        self.assertAlmostEqual(mf_dh.results["eng_corr_MP2"], REF, 8)
        self.assertAlmostEqual(mf_dh.e_corr, REF, 8)
        self.assertTrue("t_oovv" in mf_dh.tensors)

    def test_eng_rmp2_conv_complex(self):
        mf_s = self.mf_h2o_hf_complex
        mf_dh = RMP2Conv(mf_s).run()
        REF = -0.27425584824874516
        REF_PYSCF = mp.MP2(mf_s).run().e_corr
        self.assertAlmostEqual(REF, REF_PYSCF, 8)
        self.assertAlmostEqual(mf_dh.results["eng_corr_MP2"], REF, 8)

    def test_eng_rmp2_ri_fc_1(self):
        mf_s = self.mf_h2o_hf
        mol = mf_s.mol
        mf_dh = RMP2RI(mf_s).run(frozen="FreezeNobleGasCore", with_df=df.DF(mol, auxbasis=df.aug_etb(mol)))
        print(mf_dh.results)
        # reference value
        REF = -0.2602250917785774
        mf_s.with_df = df.DF(mol, auxbasis=df.aug_etb(mol))
        REF_PYSCF = mp.dfmp2.DFMP2(mf_s, frozen=[0]).run().e_corr
        print(REF_PYSCF)
        self.assertAlmostEqual(REF, REF_PYSCF, 8)
        self.assertAlmostEqual(mf_dh.results["eng_corr_MP2"], REF, 8)

    def test_eng_rmp2_ri_fc_2(self):
        mf_s = self.mf_h2o_hf
        mol = mf_s.mol
        mf_dh = RMP2RI(mf_s).run(frozen=[0, 2], with_df=df.DF(mol, auxbasis=df.aug_etb(mol)))
        mf_dh.run()
        print(mf_dh.results)
        # reference value
        REF = -0.13839934020349923
        mf_s.with_df = df.DF(mol, auxbasis=df.aug_etb(mol))
        REF_PYSCF = mp.dfmp2.DFMP2(mf_s, frozen=[0, 2]).run().e_corr
        print(REF_PYSCF)
        self.assertAlmostEqual(REF, REF_PYSCF, 8)
        self.assertAlmostEqual(mf_dh.results["eng_corr_MP2"], REF, 8)

    def test_eng_rmp2_ri_complex(self):
        mf_s = self.mf_h2o_hf_complex
        mol = mf_s.mol
        int3c2e_cd = self.int3c2e_cd
        int3c2e_2_cd = self.int3c2e_2_cd

        mf_dh = RMP2RI(mf_s)
        mf_dh.incore_t_oovv_mp2 = True
        mf_dh.with_df = df.DF(mol, df.aug_etb(mol))
        mf_dh.with_df._cderi = int3c2e_cd
        mf_dh.with_df_2 = df.DF(mol, df.aug_etb(mol))
        mf_dh.with_df_2._cderi = int3c2e_2_cd
        mf_dh.run()

        # generated reference value
        REF = -0.27424683619063206
        #     -0.27425584824874516 in test_eng_rmp2_conv_complex
        self.assertAlmostEqual(mf_dh.results["eng_corr_MP2"], REF, 8)

    def test_eng_rmp2_conv_pyscf(self):
        mf_s = self.mf_h2o_hf
        mf_dh = RMP2ConvPySCF(mf_s).run(frozen=[0])
        REF = -0.2602324295391498
        self.assertAlmostEqual(mf_dh.results["eng_corr_MP2"], REF, 8)

    def test_eng_rmp2_ri_pyscf(self):
        mf_s = self.mf_h2o_hf
        mol = mf_s.mol
        mf_dh = RMP2RIPySCF(mf_s).run(frozen=[0], with_df=df.DF(mol, auxbasis=df.aug_etb(mol)))
        REF = -0.2602250917785774
        self.assertAlmostEqual(mf_dh.results["eng_corr_MP2"], REF, 8)

    def test_eng_rmp2_coverage(self):
        # coverage only, not testing correctness
        mf_s = self.mf_h2o_hf
        eri = mf_s._eri.copy()

        mf_s._eri = None
        mf_dh = RMP2Conv(mf_s).run(frac_num_mp2=np.random.randn(len(mf_s.mo_occ))).run()
        mf_dh = RMP2RI(mf_s).run(frac_num_mp2=np.random.randn(len(mf_s.mo_occ))).run()
        mf_s._eri = eri
