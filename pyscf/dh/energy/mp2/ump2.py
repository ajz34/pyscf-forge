r""" Unrestricted MP2.
"""

from pyscf.dh import util
from pyscf.dh.energy.mp2.rmp2 import RMP2ConvPySCF, RMP2Conv
from pyscf.dh.energy.mp2.rmp2ri import RMP2RI
from pyscf import ao2mo, lib, mp, __config__, df
import numpy as np

from pyscf.dh.util import pad_omega

CONFIG_incore_t_oovv_mp2 = getattr(__config__, 'incore_t_oovv_mp2', None)
""" Flag for MP2 amplitude tensor :math:`t_{ij}^{ab}` stored in memory or disk.

Parameters
----------
True
    Store tensor in memory.
False
    Store tensor in disk.
None
    Do not store tensor in either disk or memory.
"auto"
    Leave program to judge whether tensor locates.
(int)
    If tensor size exceeds this size (in MBytes), then store in disk.
"""


# region UMP2ConvPySCF

class UMP2ConvPySCF(mp.ump2.UMP2, RMP2ConvPySCF):
    """ Unrestricted MP2 class of doubly hybrid with conventional integral evaluated by PySCF. """
    pass

# endregion


# region UMP2Conv

def kernel_energy_ump2_conv_full_incore(
        mo_energy, mo_coeff, eri_or_mol, mo_occ,
        c_os, c_ss,
        t_oovv=None, frac_num=None, verbose=lib.logger.NOTE):
    """ Kernel of unrestricted MP2 energy by conventional method.

    Parameters
    ----------
    mo_energy : list[np.ndarray]
        Molecular orbital energy levels.
    mo_coeff : list[np.ndarray]
        Molecular coefficients.
    eri_or_mol : np.ndarray or gto.Mole
        ERI that is recognized by ``pyscf.ao2mo.general``.
    c_os : float
        Coefficient of oppo-spin contribution.
    c_ss : float
        Coefficient of same-spin contribution.

    t_oovv : list[np.ndarray]
        Store space for ``t_oovv``
    mo_occ : list[np.ndarray]
        Molecular orbitals occupation numbers.

    frac_num : list[np.ndarray]
        Fractional occupation number list.
    verbose : int
        Verbose level for PySCF.

    Notes
    -----
    For energy of fractional occupation system, computation method is chosen
    by eq (7) from Su2016 (10.1021/acs.jctc.6b00197).

    Since conventional integral of MP2 is computationally costly,
    purpose of this function should only be benchmark.

    In this program, we assume that frozen orbitals are paired. Thus,
    different size of alpha and beta MO number is not allowed.
    """
    log = lib.logger.new_logger(verbose=verbose)
    log.info("[INFO] Start unrestricted conventional MP2")
    log.warn("Conventional integral of MP2 is not recommended!\n"
             "Use density fitting approximation is recommended.")

    mask_occ = [mo_occ[s] != 0 for s in (0, 1)]
    mask_vir = [mo_occ[s] == 0 for s in (0, 1)]
    nocc = tuple([mask_occ[s].sum() for s in (0, 1)])
    nvir = tuple([mask_vir[s].sum() for s in (0, 1)])
    Co = [mo_coeff[s][:, :nocc[s]] for s in (0, 1)]
    Cv = [mo_coeff[s][:, nocc[s]:] for s in (0, 1)]
    eo = [mo_energy[s][:nocc[s]] for s in (0, 1)]
    ev = [mo_energy[s][nocc[s]:] for s in (0, 1)]

    if frac_num is not None:
        frac_occ = [frac_num[s][:nocc[s]] for s in (0, 1)]
        frac_vir = [frac_num[s][nocc[s]:] for s in (0, 1)]
    else:
        frac_occ = frac_vir = None

    # ERI conversion
    log.info("[INFO] Start ao2mo")
    g_iajb = [np.array([])] * 3
    for s0, s1, ss in zip((0, 0, 1), (0, 1, 1), (0, 1, 2)):
        g_iajb[ss] = ao2mo.general(eri_or_mol, (Co[s0], Cv[s0], Co[s1], Cv[s1])) \
                          .reshape(nocc[s0], nvir[s0], nocc[s1], nvir[s1])
        log.info(f"[INFO] Spin {s0, s1} ao2mo finished")

    # loops
    eng_spin = np.array([0, 0, 0], dtype=mo_coeff[0].dtype)
    for s0, s1, ss in zip((0, 0, 1), (0, 1, 1), (0, 1, 2)):
        for i in range(nocc[s0]):
            g_Iajb = g_iajb[ss][i]
            D_Ijab = eo[s0][i] + eo[s1][:, None, None] - ev[s0][None, :, None] - ev[s1][None, None, :]
            t_Ijab = lib.einsum("ajb, jab -> jab", g_Iajb, 1 / D_Ijab)
            if s0 == s1:
                t_Ijab -= lib.einsum("bja, jab -> jab", g_Iajb, 1 / D_Ijab)
            if t_oovv is not None:
                t_oovv[ss][i] = t_Ijab
            if frac_num is not None:
                n_Ijab = frac_occ[s0][i] * frac_occ[s1][:, None, None] \
                    * (1 - frac_vir[s0][None, :, None]) * (1 - frac_vir[s1][None, None, :])
                eng_spin[ss] += lib.einsum("jab, jab, jab, jab ->", n_Ijab, t_Ijab.conj(), t_Ijab, D_Ijab)
            else:
                eng_spin[ss] += lib.einsum("jab, jab, jab ->", t_Ijab.conj(), t_Ijab, D_Ijab)
    eng_spin[0] *= 0.25
    eng_spin[2] *= 0.25
    eng_spin = util.check_real(eng_spin)
    eng_mp2 = c_os * eng_spin[1] + c_ss * (eng_spin[0] + eng_spin[2])
    # finalize results
    results = dict()
    results["eng_corr_MP2_aa"] = eng_spin[0]
    results["eng_corr_MP2_ab"] = eng_spin[1]
    results["eng_corr_MP2_bb"] = eng_spin[2]
    results["eng_corr_MP2_OS"] = eng_spin[1]
    results["eng_corr_MP2_SS"] = eng_spin[0] + eng_spin[2]
    results["eng_corr_MP2"] = eng_mp2
    log.note(f"[RESULT] Energy corr MP2 of spin aa: {eng_spin[0]:18.10f}")
    log.note(f"[RESULT] Energy corr MP2 of spin ab: {eng_spin[1]:18.10f}")
    log.note(f"[RESULT] Energy corr MP2 of spin bb: {eng_spin[2]:18.10f}")
    log.note(f"[RESULT] Energy corr MP2 of total  : {eng_mp2    :18.10f}")
    return results


class UMP2Conv(RMP2Conv):
    """ Unrestricted MP2 class of doubly hybrid with conventional integral. """

    def driver_eng_mp2(self, **kwargs):
        mask = self.get_frozen_mask()
        nocc_act = tuple((mask & (self.mo_occ != 0)).sum(axis=-1))
        nvir_act = tuple((mask & (self.mo_occ == 0)).sum(axis=-1))
        mo_coeff_act = [self.mo_coeff[s][:, mask[s]] for s in (0, 1)]
        mo_energy_act = [self.mo_energy[s][mask[s]] for s in (0, 1)]
        mo_occ_act = [self.mo_occ[s][mask[s]] for s in (0, 1)]
        frac_num_act = self.frac_num
        if frac_num_act is not None:
            frac_num_act = [frac_num_act[s][mask[s]] for s in (0, 1)]
        # prepare t_oovv
        max_memory = self.max_memory - lib.current_memory()[0]
        incore_t_oovv = util.parse_incore_flag(
            self.incore_t_oovv_mp2, 3 * max(nocc_act) ** 2 * max(nvir_act) ** 2,
            max_memory, dtype=mo_coeff_act[0].dtype)
        if incore_t_oovv is None:
            t_oovv = None
        else:
            t_oovv = [np.zeros(0)] * 3  # IDE type cheat
            for s0, s1, ss, ssn in ((0, 0, 0, "aa"), (0, 1, 1, "ab"), (1, 1, 2, "bb")):
                t_oovv[ss] = util.allocate_array(
                    incore_t_oovv, shape=(nocc_act[s0], nocc_act[s1], nvir_act[s0], nvir_act[s1]),
                    mem_avail=max_memory,
                    h5file=self._tmpfile,
                    name=f"t_oovv_{ssn}",
                    dtype=mo_coeff_act[0].dtype)
            self.tensors["t_oovv"] = t_oovv
        # kernel
        with self.mol.with_range_coulomb(self.omega):
            eri_or_mol = self.scf._eri if self.omega == 0 else self.mol
            eri_or_mol = eri_or_mol if eri_or_mol is not None else self.mol
            results = self.kernel_energy_mp2(
                mo_energy_act, mo_coeff_act, eri_or_mol, mo_occ_act,
                self.c_os, self.c_ss,
                t_oovv=t_oovv, frac_num=frac_num_act, verbose=self.verbose, **kwargs)
        self.e_corr = results["eng_corr_MP2"]
        # pad omega
        results = {pad_omega(key, self.omega): val for (key, val) in results.items()}
        self.results.update(results)
        return results

    kernel_energy_mp2 = staticmethod(kernel_energy_ump2_conv_full_incore)
    kernel = driver_eng_mp2

# endregion


# region UMP2RI

def kernel_energy_ump2_ri_incore(
        mo_energy, cderi_uov, c_os, c_ss,
        t_oovv=None, frac_num=None, verbose=lib.logger.NOTE, max_memory=2000, cderi_uov_2=None):
    """ Kernel of unrestricted MP2 energy by RI integral.

    For RI approximation, ERI integral is set to be

    .. math::
        g_{ij}^{ab} &= (ia|jb) = Y_{ia, P} Y_{jb, P}

    Parameters
    ----------
    mo_energy : list[np.ndarray]
        Molecular orbital energy levels.
    cderi_uov : list[np.ndarray]
        Cholesky decomposed 3c2e ERI in MO basis (occ-vir part). Spin in (aa, bb).
    c_os : float
        Coefficient of oppo-spin contribution.
    c_ss : float
        Coefficient of same-spin contribution.

    t_oovv : list[np.ndarray]
        Store space for ``t_oovv``
    frac_num : list[np.ndarray]
        Fractional occupation number list.
    verbose : int
        Verbose level for PySCF.
    max_memory : float
        Allocatable memory in MB.
    cderi_uov_2 : list[np.ndarray]
        Another part of 3c2e ERI in MO basis (occ-vir part). This is mostly used in magnetic computations.

    Notes
    -----
    For energy of fractional occupation system, computation method is chosen
    by eq (7) from Su2016 (10.1021/acs.jctc.6b00197).

    Since conventional integral of MP2 is computationally costly,
    purpose of this function should only be benchmark.
    """
    log = lib.logger.new_logger(verbose=verbose)
    log.info("[INFO] Start unrestricted RI-MP2")

    nocc, nvir = np.array([0, 0]), np.array([0, 0])
    naux, nocc[0], nvir[0] = cderi_uov[0].shape
    naux, nocc[1], nvir[1] = cderi_uov[1].shape

    if frac_num is not None:
        frac_occ = [frac_num[s][:nocc[s]] for s in (0, 1)]
        frac_vir = [frac_num[s][nocc[s]:] for s in (0, 1)]
    else:
        frac_occ = frac_vir = None

    eo = [mo_energy[s][:nocc[s]] for s in (0, 1)]
    ev = [mo_energy[s][nocc[s]:] for s in (0, 1)]

    # loops
    eng_spin = np.array([0, 0, 0], dtype=cderi_uov[0].dtype)
    log.info("[INFO] Start RI-MP2 loop")
    nbatch = util.calc_batch_size(4 * max(nocc) * max(nvir) ** 2, max_memory, dtype=cderi_uov[0].dtype)
    for s0, s1, ss in zip((0, 0, 1), (0, 1, 1), (0, 1, 2)):
        log.info("[INFO] Starting spin {:}{:}".format(s0, s1))
        for sI in util.gen_batch(0, nocc[s0], nbatch):
            log.info("[INFO] MP2 loop i: [{:}, {:})".format(sI.start, sI.stop))
            if cderi_uov_2 is None:
                g_Iajb = lib.einsum("PIa, Pjb -> Iajb", cderi_uov[s0][:, sI], cderi_uov[s1])
            else:
                g_Iajb = 0.5 * lib.einsum("PIa, Pjb -> Iajb", cderi_uov[s0][:, sI], cderi_uov_2[s1])
                g_Iajb += 0.5 * lib.einsum("PIa, Pjb -> Iajb", cderi_uov_2[s0][:, sI], cderi_uov[s1])
            D_Ijab = (
                + eo[s0][sI, None, None, None] + eo[s1][None, :, None, None]
                - ev[s0][None, None, :, None] - ev[s1][None, None, None, :])
            t_Ijab = lib.einsum("Iajb, Ijab -> Ijab", g_Iajb, 1 / D_Ijab)
            if s0 == s1:
                t_Ijab -= lib.einsum("Ibja, Ijab -> Ijab", g_Iajb, 1 / D_Ijab)
            if t_oovv is not None:
                t_oovv[ss][sI] = t_Ijab
            if frac_num is not None:
                n_Ijab = frac_occ[s0][sI, None, None, None] * frac_occ[s1][None, :, None, None] \
                    * (1 - frac_vir[s0][None, None, :, None]) * (1 - frac_vir[s1][None, None, None, :])
                eng_spin[ss] += lib.einsum("Ijab, Ijab, Ijab, Ijab ->", n_Ijab, t_Ijab.conj(), t_Ijab, D_Ijab)
            else:
                eng_spin[ss] += lib.einsum("Ijab, Ijab, Ijab ->", t_Ijab.conj(), t_Ijab, D_Ijab)
    eng_spin[0] *= 0.25
    eng_spin[2] *= 0.25
    eng_spin = util.check_real(eng_spin)
    eng_mp2 = c_os * eng_spin[1] + c_ss * (eng_spin[0] + eng_spin[2])
    # finalize results
    results = dict()
    results["eng_corr_MP2_aa"] = eng_spin[0]
    results["eng_corr_MP2_ab"] = eng_spin[1]
    results["eng_corr_MP2_bb"] = eng_spin[2]
    results["eng_corr_MP2_OS"] = eng_spin[1]
    results["eng_corr_MP2_SS"] = eng_spin[0] + eng_spin[2]
    results["eng_corr_MP2"] = eng_mp2
    log.note(f"[RESULT] Energy corr MP2 of spin aa: {eng_spin[0]:18.10f}")
    log.note(f"[RESULT] Energy corr MP2 of spin ab: {eng_spin[1]:18.10f}")
    log.note(f"[RESULT] Energy corr MP2 of spin bb: {eng_spin[2]:18.10f}")
    log.note(f"[RESULT] Energy corr MP2 of total  : {eng_mp2    :18.10f}")
    return results


class UMP2RI(RMP2RI):
    """ Unrestricted MP2 class of doubly hybrid with RI integral. """

    def driver_eng_mp2(self, **kwargs):
        mask = self.get_frozen_mask()
        nocc_act = tuple((mask & (self.mo_occ != 0)).sum(axis=-1))
        nvir_act = tuple((mask & (self.mo_occ == 0)).sum(axis=-1))
        nact = tuple(mask.sum(axis=-1))
        mo_coeff_act = [self.mo_coeff[s][:, mask[s]] for s in (0, 1)]
        mo_energy_act = [self.mo_energy[s][mask[s]] for s in (0, 1)]
        frac_num_act = self.frac_num
        if frac_num_act is not None:
            frac_num_act = [frac_num_act[s][mask[s]] for s in (0, 1)]
        # prepare t_oovv
        max_memory = self.max_memory - lib.current_memory()[0]
        incore_t_oovv = util.parse_incore_flag(
            self.incore_t_oovv_mp2, 3 * max(nocc_act) ** 2 * max(nvir_act) ** 2,
            max_memory, dtype=mo_coeff_act[0].dtype)
        if incore_t_oovv is None:
            t_oovv = None
        else:
            t_oovv = [np.zeros(0)] * 3  # IDE type cheat
            for s0, s1, ss, ssn in ((0, 0, 0, "aa"), (0, 1, 1, "ab"), (1, 1, 2, "bb")):
                t_oovv[ss] = util.allocate_array(
                    incore_t_oovv, shape=(nocc_act[s0], nocc_act[s1], nvir_act[s0], nvir_act[s1]),
                    mem_avail=max_memory,
                    h5file=self._tmpfile,
                    name=f"t_oovv_{ssn}",
                    dtype=mo_coeff_act[0].dtype)
            self.tensors["t_oovv"] = t_oovv
        # generate cderi_uov
        omega = self.omega
        with_df = util.get_with_df_omega(self.with_df, omega)
        max_memory = self.max_memory - lib.current_memory()[0]
        cderi_uov = self.tensors.get("cderi_uov", None)
        if cderi_uov is None:
            cderi_uov = [np.zeros(0)] * 2  # IDE type cheat
            for s in (0, 1):
                cderi_uov[s] = util.get_cderi_mo(
                    with_df, mo_coeff_act[s], None, (0, nocc_act[s], nocc_act[s], nact[s]), max_memory)
            self.tensors["cderi_uov"] = cderi_uov
        # cderi_uov_2 is rarely called, so do not try to build omega for this special case
        cderi_uov_2 = None
        max_memory = self.max_memory - lib.current_memory()[0]
        if self.with_df_2 is not None:
            cderi_uov_2 = [np.zeros(0)] * 2  # IDE type cheat
            for s in (0, 1):
                cderi_uov_2[s] = util.get_cderi_mo(
                    self.with_df_2, mo_coeff_act[s], None, (0, nocc_act[s], nocc_act[s], nact[s]), max_memory)
        # kernel
        max_memory = self.max_memory - lib.current_memory()[0]
        results = self.kernel_energy_mp2(
            mo_energy_act, cderi_uov,
            self.c_os, self.c_ss,
            t_oovv=t_oovv,
            frac_num=frac_num_act,
            verbose=self.verbose,
            max_memory=max_memory,
            cderi_uov_2=cderi_uov_2
        )

        self.e_corr = results["eng_corr_MP2"]
        # pad omega
        results = {pad_omega(key, self.omega): val for (key, val) in results.items()}
        self.results.update(results)
        return results

    kernel_energy_mp2 = staticmethod(kernel_energy_ump2_ri_incore)
    kernel = driver_eng_mp2


# endregion


if __name__ == '__main__':

    def main_1():
        from pyscf import gto, scf
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf_scf = scf.UHF(mol).run()
        mf_mp = UMP2Conv(mf_scf, frozen=[1, 2]).run()
        print(mf_mp.e_tot)
        print(mf_mp.results)

    def main_2():
        from pyscf import gto, scf
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf_scf = scf.UHF(mol).run()
        mf_mp = UMP2RI(mf_scf, frozen=[1, 2]).run()
        print(mf_mp.e_tot)
        print(mf_mp.results)

    main_1()
    main_2()
