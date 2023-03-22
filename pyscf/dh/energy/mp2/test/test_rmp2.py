import unittest
from pyscf import gto, scf, mp, df
from pyscf.dh import RMP2ofDH
import numpy as np


mf_h2o_hf = NotImplemented
mf_h2o_hf_complex = NotImplemented


def setUpModule():
    global mf_h2o_hf, mf_h2o_hf_complex
    mf_h2o_hf = get_mf_h2o_hf()
    mf_h2o_hf_complex = get_mf_h2o_hf_complex()


def tearDownModule():
    global mf_h2o_hf_complex
    del mf_h2o_hf_complex


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
    return mf_s


class TestEngRMP2(unittest.TestCase):
    def test_eng_rmp2_conv(self):
        mf_s = mf_h2o_hf
        mf_dh = RMP2ofDH(mf_s)
        mf_dh.params.flags.update({"integral_scheme": "conv"})
        mf_dh.run()
        print(mf_dh.params.results)
        # reference value
        REF = -0.273944755130888
        REF_PYSCF = mp.MP2(mf_s).run().e_corr
        self.assertAlmostEqual(REF, REF_PYSCF, 8)
        self.assertAlmostEqual(mf_dh.params.results["eng_corr_MP2"], REF, 8)

    def test_eng_rmp2_ri(self):
        mf_s = mf_h2o_hf
        mol = mf_s.mol
        mf_dh = RMP2ofDH(mf_s).run()
        print(mf_dh.params.results)
        # reference value
        REF = -0.2739374133400274
        mf_s.with_df = df.DF(mol, auxbasis=df.aug_etb(mol))
        REF_PYSCF = mp.dfmp2.DFMP2(mf_s).run().e_corr
        print(REF_PYSCF)
        self.assertAlmostEqual(REF, REF_PYSCF, 8)
        self.assertAlmostEqual(mf_dh.params.results["eng_corr_MP2"], REF, 8)

    def test_eng_rmp2_conv_fc(self):
        mf_s = mf_h2o_hf
        mf_dh = RMP2ofDH(mf_s)
        mf_dh.params.flags.update({
            "integral_scheme": "conv",
            "frozen_rule": "FreezeNobleGasCore",
            "incore_t_ijab_mp2": True,
        })
        mf_dh.run()
        print(mf_dh.params.results)
        # reference value
        REF = -0.2602324295391498
        REF_PYSCF = mp.MP2(mf_s, frozen=[0]).run().e_corr
        self.assertAlmostEqual(REF, REF_PYSCF, 8)
        self.assertAlmostEqual(mf_dh.params.results["eng_corr_MP2"], REF, 8)
        self.assertTrue("t_ijab" in mf_dh.params.tensors)

    def test_eng_rmp2_conv_complex(self):
        mf_s = mf_h2o_hf_complex
        mf_dh = RMP2ofDH(mf_s)
        mf_dh.params.flags.update({"integral_scheme": "conv"})
        mf_dh.run()
        REF = -0.27425584824874516
        REF_PYSCF = mp.MP2(mf_s).run().e_corr
        self.assertAlmostEqual(REF, REF_PYSCF, 8)
        self.assertAlmostEqual(mf_dh.params.results["eng_corr_MP2"], REF, 8)

    def test_eng_rmp2_ri_fc_1(self):
        mf_s = mf_h2o_hf
        mol = mf_s.mol
        mf_dh = RMP2ofDH(mf_s)
        mf_dh.params.flags.update({"frozen_rule": "FreezeNobleGasCore"})
        mf_dh.run()
        print(mf_dh.params.results)
        # reference value
        REF = -0.2602250917785774
        mf_s.with_df = df.DF(mol, auxbasis=df.aug_etb(mol))
        REF_PYSCF = mp.dfmp2.DFMP2(mf_s, frozen=[0]).run().e_corr
        print(REF_PYSCF)
        self.assertAlmostEqual(REF, REF_PYSCF, 8)
        self.assertAlmostEqual(mf_dh.params.results["eng_corr_MP2"], REF, 8)

    def test_eng_rmp2_ri_fc_2(self):
        mf_s = mf_h2o_hf
        mol = mf_s.mol
        mf_dh = RMP2ofDH(mf_s)
        mf_dh.params.flags.update({"frozen_list": [0, 2]})
        mf_dh.run()
        print(mf_dh.params.results)
        # reference value
        REF = -0.13839934020349923
        mf_s.with_df = df.DF(mol, auxbasis=df.aug_etb(mol))
        REF_PYSCF = mp.dfmp2.DFMP2(mf_s, frozen=[0, 2]).run().e_corr
        print(REF_PYSCF)
        self.assertAlmostEqual(REF, REF_PYSCF, 8)
        self.assertAlmostEqual(mf_dh.params.results["eng_corr_MP2"], REF, 8)

    def test_eng_rmp2_ri_complex(self):
        mf_s = mf_h2o_hf_complex
        mol = mf_s.mol

        auxmol = df.make_auxmol(mol, df.aug_etb(mol))
        dev_xyz_B = np.array([1e-2, 2e-2, -1e-2])
        int3c2e = df.incore.aux_e2(mol, auxmol, "int3c2e")
        int3c2e_ig1 = df.incore.aux_e2(mol, auxmol, "int3c2e_ig1")
        int2c2e = auxmol.intor("int2c2e")
        L = np.linalg.cholesky(int2c2e)
        int3c2e_cd = np.linalg.solve(L, int3c2e.reshape(mol.nao**2, -1).T).reshape(-1, mol.nao, mol.nao)
        int3c2e_ig1_cd = np.linalg.solve(L, int3c2e_ig1.reshape(3 * mol.nao**2, -1).T).reshape(-1, 3, mol.nao, mol.nao)
        int3c2e_2_cd = int3c2e_cd + 2 * np.einsum("Ptuv, t -> Puv", -1j * int3c2e_ig1_cd, dev_xyz_B, optimize=True)

        mf_dh = RMP2ofDH(mf_s)
        mf_dh.params.flags.update({"incore_t_ijab_mp2": True})
        mf_dh.build()
        mf_dh.with_df()
        mf_dh.with_df = df.DF(mol, df.aug_etb(mol))
        mf_dh.with_df._cderi = int3c2e_cd
        mf_dh.with_df_2 = df.DF(mol, df.aug_etb(mol))
        mf_dh.with_df_2._cderi = int3c2e_2_cd
        mf_dh.run()

        # reference value
        REF = -0.27424683619063206
        #     -0.27425584824874516 in test_eng_rmp2_conv_complex
        self.assertAlmostEqual(mf_dh.params.results["eng_corr_MP2"], REF, 8)

    def test_eng_rmp2_coverage(self):
        # coverage only, not testing correctness
        mf_s = mf_h2o_hf
        eri = mf_s._eri.copy()
        mf_s._eri = None
        mf_dh = RMP2ofDH(mf_s)
        mf_dh.params.flags.update({
            "omega_list_mp2": [0, 0.7, -0.7],
            "frac_num_mp2": np.random.randn(len(mf_s.mo_occ))
        })
        mf_dh.run()
        mf_dh.run()
        mf_dh = RMP2ofDH(mf_s)
        mf_dh.params.flags.update({
            "integral_scheme_mp2": "conv",
            "omega_list_mp2": [0, 0,7, -0.7],
            "frac_num_mp2": np.random.randn(len(mf_s.mo_occ))
        })
        mf_dh.run()

        mf_s._eri = eri
