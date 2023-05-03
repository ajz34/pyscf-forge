import numpy as np
from pyscf import lib, scf, __config__
from pyscf.dh import util
from pyscf.dh.energy.mp2.rmp2 import MP2Base
from pyscf.dh.util import pad_omega
from pyscf.scf import cphf

CONFIG_incore_cderi_uaa_mp2 = getattr(__config__, "incore_cderi_uaa_mp2", "auto")
CONFIG_max_cycle_cpks = getattr(__config__, "max_cycle_cpks", 20)
CONFIG_tol_cpks = getattr(__config__, "tol_cpks", 1e-9)


def kernel_energy_rmp2_ri_incore(
        mo_energy, cderi_uov,
        c_os, c_ss,
        t_oovv=None, frac_num=None, verbose=lib.logger.NOTE, max_memory=2000, cderi_uov_2=None):
    """ Kernel of MP2 energy by RI integral.

    Parameters
    ----------
    mo_energy : np.ndarray
        Molecular orbital energy levels.
    cderi_uov : np.ndarray
        Cholesky decomposed 3c2e ERI in MO basis (occ-vir part).
    c_os : float
        Coefficient of oppo-spin contribution.
    c_ss : float
        Coefficient of same-spin contribution.

    t_oovv : np.ndarray
        Store space for ``t_oovv``
    frac_num : np.ndarray
        Fractional occupation number list.
    verbose : int
        Verbose level for PySCF.
    max_memory : float
        Allocatable memory in MB.
    cderi_uov_2 : np.ndarray
        Another part of 3c2e ERI in MO basis (occ-vir part). This is mostly used in magnetic computations.

    Notes
    -----

    For RI approximation, ERI integral is set to be

    .. math::
        g_{ij}^{ab} = (ia|jb) = Y_{ia, P} Y_{jb, P}

    For energy of fractional occupation system, computation method is chosen
    by eq (7) from Su2016 (10.1021/acs.jctc.6b00197).
    """
    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()
    log.info("[INFO] Start unrestricted RI-MP2")

    naux, nocc, nvir = cderi_uov.shape
    if frac_num is not None:
        frac_occ, frac_vir = frac_num[:nocc], frac_num[nocc:]
    else:
        frac_occ = frac_vir = None
    eo = mo_energy[:nocc]
    ev = mo_energy[nocc:]

    # loops
    log.info("[INFO] Start RI-MP2 loop")
    mem_avail = max_memory - lib.current_memory()[0]
    nbatch = util.calc_batch_size(4 * nocc * nvir ** 2, mem_avail, dtype=cderi_uov.dtype)
    eng_bi1 = eng_bi2 = 0
    for sI in util.gen_batch(0, nocc, nbatch):
        log.info(f"[INFO] MP2 loop i: {sI}")
        if cderi_uov_2 is None:
            g_Iajb = lib.einsum("PIa, Pjb -> Iajb", cderi_uov[:, sI], cderi_uov)
        else:
            g_Iajb = 0.5 * lib.einsum("PIa, Pjb -> Iajb", cderi_uov[:, sI], cderi_uov_2)
            g_Iajb += 0.5 * lib.einsum("PIa, Pjb -> Iajb", cderi_uov_2[:, sI], cderi_uov)
        D_Ijab = lib.direct_sum("i + j - a - b -> ijab", eo[sI], eo, ev, ev)
        t_Ijab = lib.einsum("Iajb, Ijab -> Ijab", g_Iajb, 1 / D_Ijab)
        if t_oovv is not None:
            t_oovv[sI] = t_Ijab
        if frac_num is not None:
            n_Ijab = lib.einsum("i, j, a, b -> ijab", frac_occ[sI], frac_occ, 1 - frac_vir, 1 - frac_vir)
            eng_bi1 += lib.einsum("Ijab, Ijab, Iajb ->", n_Ijab, t_Ijab.conj(), g_Iajb)
            eng_bi2 += lib.einsum("Ijab, Ijab, Ibja ->", n_Ijab, t_Ijab.conj(), g_Iajb)
        else:
            eng_bi1 += lib.einsum("Ijab, Iajb ->", t_Ijab.conj(), g_Iajb)
            eng_bi2 += lib.einsum("Ijab, Ibja ->", t_Ijab.conj(), g_Iajb)
    eng_bi1 = util.check_real(eng_bi1)
    eng_bi2 = util.check_real(eng_bi2)
    log.info("[INFO] MP2 energy computation finished.")
    # report
    eng_os = eng_bi1
    eng_ss = eng_bi1 - eng_bi2
    eng_mp2 = c_os * eng_os + c_ss * eng_ss
    # results
    results = dict()
    results["eng_corr_MP2_bi1"] = eng_bi1
    results["eng_corr_MP2_bi2"] = eng_bi2
    results["eng_corr_MP2_OS"] = eng_os
    results["eng_corr_MP2_SS"] = eng_ss
    results["eng_corr_MP2"] = eng_mp2
    log.note(f"[RESULT] Energy corr MP2 of same-spin: {eng_ss :18.10f}")
    log.note(f"[RESULT] Energy corr MP2 of oppo-spin: {eng_os :18.10f}")
    log.note(f"[RESULT] Energy corr MP2 of total:     {eng_mp2:18.10f}")
    log.timer("kernel_energy_rmp2_ri_incore", *time0)
    return results


def get_rdm1_corr(
        mo_energy, cderi_uov, t_oovv,
        c_os, c_ss,
        verbose=lib.logger.NOTE, max_memory=2000):
    r""" 1-RDM of MP2 correlation by MO basis.

    .. math::
        D_{jk}^\mathrm{(2)} &= 2 T_{ij}^{ab} t_{ik}^{ab} \\
        D_{ab}^\mathrm{(2)} &= 2 T_{ij}^{ac} t_{ij}^{bc}

    By definition of this program, 1-RDM is not response density.

    Parameters
    ----------
    mo_energy : np.ndarray
    cderi_uov : np.ndarray
    t_oovv : np.ndarray
    c_os : float
    c_ss : float
    verbose : int
    max_memory : int or float

    Returns
    -------
    dict[str, np.ndarray]
    """
    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    # dimension definition and check sanity
    naux, nocc, nvir = cderi_uov.shape
    nmo = nocc + nvir
    assert len(mo_energy) == nmo
    assert t_oovv.shape == (nocc, nocc, nvir, nvir)

    # prepare essential matrices
    so, sv = slice(0, nocc), slice(nocc, nmo)

    # allocate results
    rdm1_corr = np.zeros((nmo, nmo))

    def load_slice(slc):
        return slc, t_oovv[slc]

    mem_avail = max_memory - lib.current_memory()[0]
    nbatch = util.calc_batch_size(2 * nocc * nvir**2, mem_avail)
    for sI, t_Oovv in lib.map_with_prefetch(load_slice, util.gen_batch(0, nocc, nbatch)):
        log.debug(f"[INFO] Generation of MP2 rdm1 oo vv and G_uov, {sI} of total {nocc} occ orbitals")
        T_Oovv = util.restricted_biorthogonalize(t_Oovv, 1, c_os, c_ss)
        rdm1_corr[sv, sv] += 2 * lib.einsum("ijac, ijbc -> ab", T_Oovv, t_Oovv)
        rdm1_corr[so, so] -= 2 * lib.einsum("ijab, ikab -> jk", T_Oovv, t_Oovv)

    log.timer("get_rdm1_corr", *time0)
    tensors = {"rdm1_corr": rdm1_corr}
    return tensors


def get_G_uov(
        mo_energy, cderi_uov, t_oovv,
        c_os, c_ss,
        verbose=lib.logger.NOTE, max_memory=2000):
    r""" Get 3-index transformed MP2 amplitude.

    Evaluation of this 3-index amplitude is performed together with non-response MP2 correlation 1-RDM.

    .. math::
        \Gamma_{ia, P} = T_{ij}^{ab} Y_{jb, P}

    Parameters
    ----------
    mo_energy : np.ndarray
    cderi_uov : np.ndarray
    t_oovv : np.ndarray
    c_os : float
    c_ss : float
    verbose : int
    max_memory : int or float

    Returns
    -------
    dict[str, np.ndarray]
    """
    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    # dimension definition and check sanity
    naux, nocc, nvir = cderi_uov.shape
    nmo = nocc + nvir
    assert len(mo_energy) == nmo
    assert t_oovv.shape == (nocc, nocc, nvir, nvir)

    # prepare essential matrices
    so, sv = slice(0, nocc), slice(nocc, nmo)

    # allocate results
    rdm1_corr = np.zeros((nmo, nmo))
    G_uov = np.zeros((naux, nocc, nvir))

    def load_slice(slc):
        return slc, t_oovv[slc]

    mem_avail = max_memory - lib.current_memory()[0]
    nbatch = util.calc_batch_size(2 * nocc * nvir**2 + naux * nvir, mem_avail)
    for sI, t_Oovv in lib.map_with_prefetch(load_slice, util.gen_batch(0, nocc, nbatch)):
        log.debug(f"[INFO] Generation of MP2 rdm1 oo vv and G_uov, {sI} of total {nocc} occ orbitals")
        T_Oovv = util.restricted_biorthogonalize(t_Oovv, 1, c_os, c_ss)
        rdm1_corr[sv, sv] += 2 * lib.einsum("ijac, ijbc -> ab", T_Oovv, t_Oovv)
        rdm1_corr[so, so] -= 2 * lib.einsum("ijab, ikab -> jk", T_Oovv, t_Oovv)
        G_uov[:, sI] += lib.einsum("ijab, Pjb -> Pia", T_Oovv, cderi_uov)

    log.timer("get_G_uov", *time0)
    tensors = {
        "rdm1_corr": rdm1_corr,
        "G_uov": G_uov,
    }
    return tensors


def get_W_I(cderi_uov, cderi_uoo, G_uov, verbose=lib.logger.NOTE):
    r""" Part I of MO-energy-weighted density matrix.

    .. math::
        W_{ij} [\mathrm{I}] &= -2 \Gamma_{ia, P} Y_{ja, P} \\
        W_{ab} [\mathrm{I}] &= -2 \Gamma_{ia, P} Y_{ib, P} \\
        W_{ai} [\mathrm{I}] &= -4 \Gamma_{ja, P} Y_{ij, P}

    This energy-weighted density matrix is not symmetric by current implementation.

    Parameters
    ----------
    cderi_uov : np.ndarray
    cderi_uoo : np.ndarray
    G_uov : np.ndarray
    verbose : int

    Returns
    -------
    dict[str, np.ndarray]
    """

    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    # dimension definition and check sanity
    naux, nocc, nvir = cderi_uov.shape
    nmo = nocc + nvir
    assert cderi_uoo.shape == (naux, nocc, nocc)
    assert G_uov.shape == (naux, nocc, nvir)

    # prepare essential matrices and slices
    so, sv = slice(0, nocc), slice(nocc, nmo)

    W_I = np.zeros((nmo, nmo))
    W_I[so, so] = - 2 * lib.einsum("Pia, Pja -> ij", G_uov, cderi_uov)
    W_I[sv, sv] = - 2 * lib.einsum("Pia, Pib -> ab", G_uov, cderi_uov)
    W_I[sv, so] = - 4 * lib.einsum("Pja, Pij -> ai", G_uov, cderi_uoo)

    log.timer("get_W_I", *time0)
    tensors = {"W_I": W_I}
    return tensors


def get_lag_vo(
        G_uov, cderi_uaa, W_I, rdm1_corr, Ax0_Core,
        max_memory=2000, verbose=lib.logger.NOTE):
    r""" MP2 contribution to Lagrangian vir-occ block.

    .. math::
        L_{ai} = - 4 \Gamma_{ja, P} Y_{ij, P} + 4 \Gamma_{ib, P} Y_{ab, P} + A_{ai, pq} D_{pq}^\mathrm{(2)}

    where :math:`- 4 \Gamma_{ja, P} Y_{ij, P}` is evaluated by :math:`W_{ai} [\mathrm{I}]`.

    Some pratical meaning of lagrangian, is that for vir-occ block, it can be defined by MP2 energy derivative wrt
    MO coefficients:

    .. math::
        \mathscr{L}_{pq} &= C_{\mu p} \frac{\partial E^\mathrm{(2)}}{\partial C_{\mu q}} \\
        L_{ai} &= \mathscr{L}_{ai} - \mathscr{L}_{ia}

    Parameters
    ----------
    G_uov : np.ndarray
    cderi_uaa : np.ndarray
    W_I : np.ndarray
    rdm1_corr : np.ndarray
    Ax0_Core : callable
    max_memory : float
    verbose : int

    Returns
    -------
    dict[str, np.ndarray]
    """

    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    # dimension definition and check sanity
    naux, nocc, nvir = G_uov.shape
    nmo = nocc + nvir
    assert cderi_uaa.shape == (naux, nmo, nmo)
    if W_I is not None:
        assert W_I.shape == (nmo, nmo)

    # prepare essential matrices and slices
    so, sv, sa = slice(0, nocc), slice(nocc, nmo), slice(0, nmo)

    # generate lagrangian occ-vir block
    lag_vo = np.zeros((nvir, nocc))
    lag_vo += W_I[sv, so]
    lag_vo += Ax0_Core(sv, so, sa, sa)(rdm1_corr)
    mem_avail = max_memory - lib.current_memory()[0]
    nbatch = util.calc_batch_size(nvir ** 2 + nocc * nvir, mem_avail)
    for saux in util.gen_batch(0, naux, nbatch):
        lag_vo += 4 * lib.einsum("Pib, Pab -> ai", G_uov[saux], cderi_uaa[saux, sv, sv])

    log.timer("get_lag_vo", *time0)
    tensors = {"lag_vo": lag_vo}
    return tensors


def get_rdm1_corr_resp(
        rdm1_corr, lag_vo,
        mo_energy, mo_occ, Ax0_Core,
        max_cycle=20, tol=1e-9, verbose=lib.logger.NOTE):
    r""" 1-RDM of response MP2 correlation by MO basis.

    For other parts, response density matrix is the same to the usual 1-RDM.

    .. math::
        A'_{ai, bj} D_{bj}^\mathrm{(2)} &= L_{ai}

    Parameters
    ----------
    rdm1_corr : np.ndarray
    lag_vo : np.ndarray
    mo_energy : np.ndarray
    mo_occ : np.ndarray
    Ax0_Core : callable
    max_cycle : int
    tol : float
    verbose : int

    Returns
    -------
    dict[str, np.ndarray]

    See Also
    --------
    get_rdm1_corr
    """

    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    # dimension definition and check sanity
    nocc = (mo_occ > 0).sum()
    nvir = (mo_occ == 0).sum()
    nmo = nocc + nvir
    assert lag_vo.shape == (nvir, nocc)
    assert rdm1_corr.shape == (nmo, nmo)
    assert len(mo_energy) == nmo
    assert len(mo_occ) == nmo

    # prepare essential matrices and slices
    so, sv = slice(0, nocc), slice(nocc, nmo)

    # cp-ks evaluation
    rdm1_corr_resp = rdm1_corr.copy()
    rdm1_corr_resp_vo = cphf.solve(
        Ax0_Core(sv, so, sv, so), mo_energy, mo_occ, lag_vo,
        max_cycle=max_cycle, tol=tol)[0]
    rdm1_corr_resp[sv, so] = rdm1_corr_resp_vo

    log.timer("get_rdm1_corr_resp", *time0)
    tensors = {"rdm1_corr_resp": rdm1_corr_resp}
    return tensors


class RMP2RI(MP2Base):
    """ Restricted MP2 class of doubly hybrid with RI integral. """

    def __init__(self, mf, **kwargs):
        super().__init__(mf, **kwargs)
        self.with_df_2 = None
        self.incore_cderi_uaa = CONFIG_incore_cderi_uaa_mp2
        self.max_cycle_cpks = CONFIG_max_cycle_cpks
        self.tol_cpks = CONFIG_tol_cpks
        self._Ax0_Core = NotImplemented

    def make_cderi_uov(self):
        r""" Generate cholesky decomposed ERI :math:`Y_{ia, P}` (in memory, occ-vir block). """
        mask = self.get_frozen_mask()
        nocc_act = (mask & (self.mo_occ != 0)).sum()
        nvir_act = (mask & (self.mo_occ == 0)).sum()
        nact = nocc_act + nvir_act
        mo_coeff_act = self.mo_coeff[:, mask]
        omega = self.omega
        with_df = util.get_with_df_omega(self.with_df, omega)
        cderi_uov = util.get_cderi_mo(
            with_df, mo_coeff_act, None, (0, nocc_act, nocc_act, nact),
            max_memory=self.max_memory, verbose=self.verbose)

        tensors = {"cderi_uov": cderi_uov}
        self.tensors.update(tensors)
        return cderi_uov

    def make_cderi_uoo(self):
        r""" Generate cholesky decomposed ERI math:`Y_{ij, P}` (in memory, occ-occ block). """
        mask = self.get_frozen_mask()
        idx_occ = (self.mo_occ > 0) & mask
        mo_occ_act = self.mo_coeff[:, idx_occ]
        nocc_act = idx_occ.sum()
        omega = self.omega
        with_df = util.get_with_df_omega(self.with_df, omega)
        cderi_uoo = util.get_cderi_mo(
            with_df, mo_occ_act, None, (0, nocc_act, 0, nocc_act),
            max_memory=self.max_memory, verbose=self.verbose)

        tensors = {"cderi_uoo": cderi_uoo}
        self.tensors.update(tensors)
        return cderi_uoo

    def make_cderi_uaa(self):
        r""" Generate cholesky decomposed ERI math:`Y_{pq, P}` (all block, s1 symm, in memory/disk). """
        log = lib.logger.new_logger(verbose=self.verbose)

        # dimension and mask
        mask = self.get_frozen_mask()
        nocc_act = (mask & (self.mo_occ != 0)).sum()
        nvir_act = (mask & (self.mo_occ == 0)).sum()
        nact = nocc_act + nvir_act
        mo_coeff_act = self.mo_coeff[:, mask]

        # density fitting preparation
        omega = self.omega
        with_df = util.get_with_df_omega(self.with_df, omega)
        naux = with_df.get_naoaux()

        # array allocation
        mem_avail = self.max_memory - lib.current_memory()[0]
        incore_cderi_uaa = self.incore_cderi_uaa
        cderi_uaa = util.allocate_array(
            incore_cderi_uaa, (naux, nact, nact), mem_avail,
            h5file=self._tmpfile, name="cderi_uaa", zero_init=False)
        log.info(f"[INFO] Store type of cderi_uaa: {type(cderi_uaa)}")

        # generate array
        util.get_cderi_mo(with_df, mo_coeff_act, cderi_uaa, max_memory=self.max_memory)

        tensors = {"cderi_uaa": cderi_uaa}
        self.tensors.update(tensors)
        return cderi_uaa

    def driver_eng_mp2(self, **kwargs):
        r""" Driver of MP2 energy. """
        mask = self.get_frozen_mask()
        nocc_act = (mask & (self.mo_occ != 0)).sum()
        nvir_act = (mask & (self.mo_occ == 0)).sum()
        nact = nocc_act + nvir_act
        mo_coeff_act = self.mo_coeff[:, mask]
        mo_energy_act = self.mo_energy[mask]
        frac_num_act = self.frac_num
        if frac_num_act is not None:
            frac_num_act = frac_num_act[mask]
        # prepare t_oovv
        max_memory = self.max_memory - lib.current_memory()[0]
        t_oovv = util.allocate_array(
            self.incore_t_oovv_mp2, (nocc_act, nocc_act, nvir_act, nvir_act), max_memory,
            h5file=self._tmpfile, name="t_oovv_mp2", zero_init=False, chunk=(1, 1, nvir_act, nvir_act),
            dtype=mo_coeff_act.dtype)
        if t_oovv is not None:
            self.tensors["t_oovv"] = t_oovv
        # generate cderi_uov
        cderi_uov = self.tensors.get("cderi_uov", self.make_cderi_uov())
        # cderi_uov_2 is rarely called, so do not try to build omega for this special case
        cderi_uov_2 = None
        if self.with_df_2 is not None:
            cderi_uov_2 = util.get_cderi_mo(
                self.with_df_2, mo_coeff_act, None, (0, nocc_act, nocc_act, nact),
                max_memory=self.max_memory, verbose=self.verbose)
        # kernel
        results = self.kernel_energy_mp2(
            mo_energy_act, cderi_uov,
            self.c_os, self.c_ss,
            t_oovv=t_oovv,
            frac_num=frac_num_act,
            verbose=self.verbose,
            max_memory=self.max_memory,
            cderi_uov_2=cderi_uov_2,
            **kwargs)

        self.e_corr = results["eng_corr_MP2"]
        # pad omega
        results = {pad_omega(key, self.omega): val for (key, val) in results.items()}
        self.results.update(results)
        return results

    @property
    def Ax0_Core(self):
        r""" Fock response of underlying SCF object in MO basis. """
        if self._Ax0_Core is NotImplemented:
            restricted = isinstance(self.scf, scf.rhf.RHF)
            from pyscf.dh import RHDFT, UHDFT
            HDFT = RHDFT if restricted else UHDFT
            mf_scf = HDFT(self.scf)
            self._Ax0_Core = mf_scf.Ax0_Core
        return self._Ax0_Core

    @Ax0_Core.setter
    def Ax0_Core(self, Ax0_Core):
        self._Ax0_Core = Ax0_Core

    def make_rdm1_corr(self):
        r""" Generate 1-RDM (non-response) of MP2 correlation :math:`D_{pq}^{(2)}`. """
        log = lib.logger.new_logger(verbose=self.verbose)
        mask = self.get_frozen_mask()
        mo_energy = self.mo_energy[mask]
        cderi_uov = self.tensors["cderi_uov"]
        t_oovv = self.tensors.get("t_oovv", None)
        if t_oovv is None:
            log.warn("t_oovv for MP2 has not been stored. "
                     "To perform G_uov, t_oovv and MP2 energy is to be re-evaluated.")
            self.incore_t_oovv_mp2 = True
            self.driver_eng_mp2()
            t_oovv = self.tensors["t_oovv"]
        tensors = get_rdm1_corr(
            mo_energy, cderi_uov, t_oovv, self.c_ss, self.c_os,
            verbose=self.verbose, max_memory=self.max_memory)
        self.tensors.update(tensors)
        return tensors["rdm1_corr"]

    def make_G_uov(self):
        r""" Generate 3-index transformed MP2 amplitude :math:`\Gamma_{ia, P}`. """
        log = lib.logger.new_logger(verbose=self.verbose)
        mask = self.get_frozen_mask()
        mo_energy = self.mo_energy[mask]
        cderi_uov = self.tensors["cderi_uov"]
        t_oovv = self.tensors.get("t_oovv", None)
        if t_oovv is None:
            log.warn("t_oovv for MP2 has not been stored. "
                     "To perform G_uov, t_oovv and MP2 energy is to be re-evaluated.")
            self.incore_t_oovv_mp2 = True
            self.driver_eng_mp2()
            t_oovv = self.tensors["t_oovv"]
        tensors = get_G_uov(
            mo_energy, cderi_uov, t_oovv, self.c_ss, self.c_os,
            verbose=self.verbose, max_memory=self.max_memory)
        self.tensors.update(tensors)
        return tensors["G_uov"]

    def make_W_I(self):
        r""" Generate part I of MO-energy-weighted density matrix. :math:`W_{pq} [\mathrm{I}]`. """
        log = lib.logger.new_logger(verbose=self.verbose)

        # prepare tensors
        if any([key not in self.tensors for key in ["rdm1_corr", "G_uov", "t_oovv"]]):
            log.warn("Some tensors (rdm1_corr, G_uov, t_oovv) has not been generated. "
                     "Perform make_mp2_integrals first.")
            self.make_G_uov()
        cderi_uov = self.tensors["cderi_uov"]
        cderi_uoo = self.tensors.get("cderi_uoo", self.make_cderi_uoo())
        G_uov = self.tensors["G_uov"]

        # main function
        tensors = get_W_I(cderi_uov, cderi_uoo, G_uov, verbose=self.verbose)
        self.tensors.update(tensors)
        return tensors["W_I"]

    def make_lag_vo(self):
        r""" Generate MP2 contribution to Lagrangian vir-occ block :math:`L_{ai}`. """
        # prepare tensors
        W_I = self.tensors.get("W_I", self.make_W_I())
        cderi_uaa = self.tensors.get("cderi_uaa", self.make_cderi_uaa())
        G_uov = self.tensors["G_uov"]
        rdm1_corr = self.tensors["rdm1_corr"]
        Ax0_Core = self.Ax0_Core

        # main function
        tensors = get_lag_vo(
            G_uov, cderi_uaa, W_I, rdm1_corr, Ax0_Core,
            max_memory=self.max_memory, verbose=self.verbose)

        self.tensors.update(tensors)
        return tensors["lag_vo"]

    def make_rdm1_corr_resp(self):
        """ Generate 1-RDM (response) of MP2 correlation :math:`D_{pq}^{(2)}`. """
        # prepare input
        lag_vo = self.tensors.get("lag_vo", self.make_lag_vo())
        rdm1_corr = self.tensors["rdm1_corr"]
        mask = self.get_frozen_mask()
        mo_energy = self.mo_energy[mask]
        mo_occ = self.mo_occ[mask]
        Ax0_Core = self.Ax0_Core
        max_cycle = self.max_cycle_cpks
        tol = self.tol_cpks
        verbose = self.verbose

        # main function
        tensors = get_rdm1_corr_resp(
            rdm1_corr, lag_vo, mo_energy, mo_occ, Ax0_Core,
            max_cycle=max_cycle, tol=tol, verbose=verbose)

        self.tensors.update(tensors)
        return tensors["rdm1_corr_resp"]

    def make_rdm1(self, ao=False):
        r""" Generate 1-RDM (non-response) of MP2 :math:`D_{pq}^{MP2}` in MO or :math:`D_{\mu \nu}^{MP2}` in AO. """
        # prepare input
        rdm1_corr = self.tensors.get("rdm1_corr", self.make_rdm1_corr())

        rdm1 = np.diag(self.mo_occ)
        mask = self.get_frozen_mask()
        ix_act = np.ix_(mask, mask)
        rdm1[ix_act] += rdm1_corr
        self.tensors["rdm1"] = rdm1
        if ao:
            rdm1 = self.mo_coeff @ rdm1 @ self.mo_coeff.T
        return rdm1

    def make_rdm1_resp(self, ao=False):
        r""" Generate 1-RDM (response) of MP2 :math:`D_{pq}^{MP2}` in MO or :math:`D_{\mu \nu}^{MP2}` in AO. """
        # prepare input
        rdm1_corr_resp = self.tensors.get("rdm1_corr_resp", self.make_rdm1_corr_resp())

        rdm1 = np.diag(self.mo_occ)
        mask = self.get_frozen_mask()
        ix_act = np.ix_(mask, mask)
        rdm1[ix_act] += rdm1_corr_resp
        self.tensors["rdm1_resp"] = rdm1
        if ao:
            rdm1 = self.mo_coeff @ rdm1 @ self.mo_coeff.T
        return rdm1

    def make_dipole(self):
        # prepare input
        mol = self.mol
        rdm1_ao = self.make_rdm1_resp(ao=True)
        int1e_r = mol.intor("int1e_r")

        dip_elec = - np.einsum("uv, tuv -> t", rdm1_ao, int1e_r)
        dip_nuc = np.einsum("A, At -> t", mol.atom_charges(), mol.atom_coords())
        dip = dip_elec + dip_nuc
        self.tensors["dipole"] = dip
        return dip

    kernel_energy_mp2 = staticmethod(kernel_energy_rmp2_ri_incore)
    kernel = driver_eng_mp2


if __name__ == '__main__':

    def main_1():
        # test RMP2RI
        from pyscf import gto, scf
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf_scf = scf.RHF(mol).run()
        mf_mp = RMP2RI(mf_scf, frozen=[1, 2]).run()
        print(mf_mp.e_tot)
        print(mf_mp.results)

    def main_2():
        # test response rdm1 generation
        from pyscf import gto, scf
        np.set_printoptions(4, suppress=True, linewidth=150)
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf_scf = scf.RHF(mol).run()
        mf_mp = RMP2RI(mf_scf, frozen=[2]).run(incore_t_oovv_mp2=True)
        mf_mp.make_rdm1()
        for key, val in mf_mp.tensors.items():
            print(key, val.shape)
        print(mf_mp.tensors["rdm1"])

    def main_3():
        # test numerical dipole
        from pyscf import gto, scf, mp, df
        np.set_printoptions(5, suppress=True, linewidth=150)
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()

        def eng_with_dipole_field(t, h):
            mf_scf = scf.RHF(mol).density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_mp = RMP2RI(mf_scf).run()
            # mf_mp = mp.dfmp2.DFMP2(mf_scf)
            # mf_mp.with_df = df.DF(mol, df.make_auxbasis(mol, mp2fit=True))
            # mf_mp.run()
            return mf_mp.e_tot

        mf_scf = scf.RHF(mol).density_fit("cc-pVDZ-jkfit").run()
        mf_mp = RMP2RI(mf_scf).run()
        dip_anal = mf_mp.make_dipole()

        eng_array = np.zeros((2, 3))
        h = 1e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                eng_array[idx, t] = eng_with_dipole_field(t, h)
        dip_elec_num = - (eng_array[0] - eng_array[1]) / (2 * h)
        dip_nuc = np.einsum("A, At -> t", mol.atom_charges(), mol.atom_coords())
        dip_num = dip_elec_num + dip_nuc

        print(mf_scf.dip_moment(unit="AU"))
        print(dip_num)
        print(dip_anal)

    main_3()
