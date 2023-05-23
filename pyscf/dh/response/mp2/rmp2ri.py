""" RI-MP2 Response-Related Utilities. """

from pyscf.dh import RMP2RI
from pyscf.dh import util
from pyscf import scf, lib, __config__
from pyscf.dh.response import RespBase
import h5py
import numpy as np

CONFIG_incore_t_oovv_mp2 = getattr(__config__, "incore_t_oovv_mp2", "auto")
CONFIG_incore_cderi_uaa_mp2 = getattr(__config__, "incore_cderi_uaa_mp2", "auto")
CONFIG_max_cycle_cpks = getattr(__config__, "max_cycle_cpks", 20)
CONFIG_tol_cpks = getattr(__config__, "tol_cpks", 1e-9)


def get_mp2_integrals(
        cderi_uov, mo_occ, mo_energy, mask_act,
        c_os, c_ss, incore_t_oovv,
        verbose=lib.logger.NOTE, max_memory=2000, h5file=NotImplemented):
    r""" Get 3-index transformed MP2 amplitude.

    .. math::
        t_{ij}^{ab} &= Y_{ia, P} Y_{jb, P} / D_{ij}^{ab} \\
        T_{ij}^{ab} &= 2 c_\mathrm{os} t_{ij}^{ab} - c_\mathrm{ss} t_{ij}^{ba} \\
        \Gamma_{ia, P} &= T_{ij}^{ab} Y_{jb, P} \\

    Parameters
    ----------
    cderi_uov : np.ndarray
        Cholesky decomposed ERI in MO basis (occ-vir block) :math:`Y_{ia, P}`.
        dim: (naux, nocc_act, nvir_act).
    mo_occ : np.ndarray
        Occupation number.
        dim: (nmo, ).
    mo_energy : np.ndarray
        Molecular orbital energy.
        dim: (nmo, ).
    mask_act : np.ndarray
        Active molecular orbital mask.
        dim: (nmo, ); dtype: bool.

    c_os : float
        Oppo-spin coefficient.
    c_ss : float
        Same-spin coefficient.
    incore_t_oovv : bool or str or float or int
        Whether save :math:`t_{ij}^{ab}` into memory (True) or disk (False).

    verbose : int
        Print verbosity.
    max_memory : int or float
        Maximum memory available for molecule object.
    h5file : h5py.File
        HDF5 file, used when ``t_oovv`` is stored in disk.

    Returns
    -------
    dict[str, np.ndarray], dict[str, float]

        Tensors and results are returned. Tensors included:

        - `t_oovv`: MP2 amplitude :math:`t_{ij}^{ab}`. dim: (nocc_act, nocc_act, nvir_act, nvir_act).
        - `G_uov`: MP2 amplitude contracted by ERI :math:`\Gamma_{ia, P}`. dim: (naux, nocc_act, nvir_act).
        - `rdm1_corr`: Response density matrix of MP2 correlation contribution in MO (non-response)
            :math:`D_{pq}^{(2)}`. dim: (nmo, nmo).

        Results are the MP2 energy contributions (including oppo-spin and same-spin).
    """
    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    # dimension definition and check sanity
    mask_occ_act = (mo_occ > 0) & mask_act
    mask_vir_act = (mo_occ == 0) & mask_act
    nocc_act = mask_occ_act.sum()
    nvir_act = mask_vir_act.sum()
    nmo = len(mask_act)
    naux = cderi_uov.shape[0]
    assert len(mo_occ) == nmo
    assert (naux, nocc_act, nvir_act) == cderi_uov.shape
    assert incore_t_oovv is not None, "t_oovv must be stored for evaluation of MP2 response"

    # preparation
    eo = mo_energy[mask_occ_act]
    ev = mo_energy[mask_vir_act]
    D_ovv = lib.direct_sum("j - a - b -> jab", eo, ev, ev)

    block_oo_act = np.ix_(mask_occ_act, mask_occ_act)
    block_vv_act = np.ix_(mask_vir_act, mask_vir_act)

    # allocate results
    rdm1_corr = np.zeros((nmo, nmo))
    G_uov = np.zeros((naux, nocc_act, nvir_act))
    t_oovv = util.allocate_array(
        incore_t_oovv, (nocc_act, nocc_act, nvir_act, nvir_act), max_memory,
        h5file=h5file, name="t_oovv", zero_init=False, chunk=(1, 1, nvir_act, nvir_act))
    eng_bi1 = eng_bi2 = 0

    # for async write t_oovv
    def write_t_oovv(slc, buf):
        t_oovv[slc] = buf

    # prepare batch
    mem_avail = max_memory - lib.current_memory()[0]
    nbatch = util.calc_batch_size(4 * nocc_act * nvir_act**2 + naux * nvir_act, mem_avail)

    with lib.call_in_background(write_t_oovv) as async_write_t_oovv:
        for sI in util.gen_batch(0, nocc_act, nbatch):
            log.debug(f"[DEBUG] Loop in get_mp2_integrals, slice {sI} of {nocc_act} orbitals.")
            D_Oovv = lib.direct_sum("I + jab -> Ijab", eo[sI], D_ovv)
            g_Oovv = lib.einsum("PIa, Pjb -> Ijab", cderi_uov[:, sI], cderi_uov)
            t_Oovv = g_Oovv / D_Oovv
            async_write_t_oovv(sI, t_Oovv)
            T_Oovv = util.restricted_biorthogonalize(t_Oovv, 1, c_os, c_ss)
            eng_bi1 += lib.einsum("Ijab, Ijab ->", t_Oovv, g_Oovv)
            eng_bi2 += lib.einsum("Ijab, Ijba ->", t_Oovv, g_Oovv)
            rdm1_corr[block_vv_act] += 2 * lib.einsum("ijac, ijbc -> ab", T_Oovv, t_Oovv)
            rdm1_corr[block_oo_act] -= 2 * lib.einsum("ijab, ikab -> jk", T_Oovv, t_Oovv)
            G_uov[:, sI] += lib.einsum("ijab, Pjb -> Pia", T_Oovv, cderi_uov)

    # results
    eng_os = eng_bi1
    eng_ss = eng_bi1 - eng_bi2
    eng_mp2 = c_os * eng_os + c_ss * eng_ss
    results = dict()
    results["eng_corr_MP2_bi1"] = eng_bi1
    results["eng_corr_MP2_bi2"] = eng_bi2
    results["eng_corr_MP2_OS"] = eng_os
    results["eng_corr_MP2_SS"] = eng_ss
    results["eng_corr_MP2"] = eng_mp2
    log.note(f"[RESULT] Energy corr MP2 of same-spin: {eng_ss :18.10f}")
    log.note(f"[RESULT] Energy corr MP2 of oppo-spin: {eng_os :18.10f}")
    log.note(f"[RESULT] Energy corr MP2 of total:     {eng_mp2:18.10f}")

    # tensors
    tensors = {
        "t_oovv": t_oovv,
        "G_uov": G_uov,
        "rdm1_corr": rdm1_corr}

    log.timer("get_mp2_integrals", *time0)
    return tensors, results


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

    def load_cderi_Uaa(slc):
        return cderi_uaa[slc, sv, sv]

    mem_avail = max_memory - lib.current_memory()[0]
    nbatch = util.calc_batch_size(nvir ** 2 + nocc * nvir, mem_avail)
    batches = util.gen_batch(0, naux, nbatch)
    for saux, cderi_Uaa in zip(batches, lib.map_with_prefetch(load_cderi_Uaa, batches)):
        lag_vo += 4 * lib.einsum("Pib, Pab -> ai", G_uov[saux], cderi_Uaa)

    log.timer("get_lag_vo", *time0)
    tensors = {"lag_vo": lag_vo}
    return tensors


class RMP2RespRI(RMP2RI, RespBase):
    
    def __init__(self, *args, **kwargs):
        if self.frozen not in (None, []):
            raise NotImplementedError("Frozen orbitals is not implemented currently!")

        super().__init__(*args, **kwargs)
        self.c_os = kwargs.get("c_os", 1)
        self.c_ss = kwargs.get("c_ss", 1)
        self.incore_cderi_uaa = CONFIG_incore_cderi_uaa_mp2
        self.incore_t_oovv = kwargs.get("incore_t_oovv", CONFIG_incore_t_oovv_mp2)
        self.max_cycle_cpks = CONFIG_max_cycle_cpks
        self.tol_cpks = CONFIG_tol_cpks
        self._Ax0_Core = NotImplemented

    def make_cderi_uaa(self):
        """ Generate cholesky decomposed ERI (all block, full orbitals, s1 symm, in memory/disk). """
        if "cderi_uaa" in self.tensors:
            return self.tensors["cderi_uaa"]

        log = lib.logger.new_logger(verbose=self.verbose)

        # dimension and mask
        mo_coeff = self.mo_coeff
        nmo = self.nmo

        # density fitting preparation
        omega = self.omega
        with_df = util.get_with_df_omega(self.with_df, omega)
        naux = with_df.get_naoaux()

        # array allocation
        mem_avail = self.max_memory - lib.current_memory()[0]
        incore_cderi_uaa = self.incore_cderi_uaa
        cderi_uaa = util.allocate_array(
            incore_cderi_uaa, (naux, nmo, nmo), mem_avail,
            h5file=self._tmpfile, name="cderi_uaa", zero_init=False)
        log.info(f"[INFO] Store type of cderi_uaa: {type(cderi_uaa)}")

        # generate array
        util.get_cderi_mo(with_df, mo_coeff, cderi_uaa, max_memory=self.max_memory)

        tensors = {"cderi_uaa": cderi_uaa}
        self.tensors.update(tensors)
        return cderi_uaa

    def make_cderi_uov(self):
        """ Generate cholesky decomposed ERI (occ-vir block, active only, in memory). """
        if "cderi_uov" in self.tensors:
            return self.tensors["cderi_uov"]

        # dimension and mask
        mask_occ_act = self.mask_occ_act
        mask_vir_act = self.mask_vir_act

        # In response evaluation, ERI of all orbitals are generally required.
        # Thus, cderi_uaa is a must-evaluated tensor.
        # note that cderi_uaa may be stored by h5py, so indexing by list should be done in following way
        cderi_uaa = self.tensors.get("cderi_uaa", self.make_cderi_uaa())
        cderi_uov = cderi_uaa[:, mask_occ_act, :][:, :, mask_vir_act]

        tensors = {"cderi_uov": cderi_uov}
        self.tensors.update(tensors)
        return cderi_uov

    def make_cderi_uoo(self):
        """ Generate cholesky decomposed ERI (occ-occ block, in memory). """
        # todo: frozen core not applied
        if "cderi_uoo" in self.tensors:
            return self.tensors["cderi_uoo"]

        # dimension and mask
        mask_occ_act = self.mask_occ_act

        # In response evaluation, ERI of all orbitals are generally required.
        # Thus, cderi_uaa is a must-evaluated tensor.
        # note that cderi_uaa may be stored by h5py, so indexing by list should be done in following way
        cderi_uaa = self.tensors.get("cderi_uaa", self.make_cderi_uaa())
        cderi_uoo = cderi_uaa[:, mask_occ_act, :][:, :, mask_occ_act]

        tensors = {"cderi_uoo": cderi_uoo}
        self.tensors.update(tensors)
        return cderi_uoo

    def _make_mp2_integrals(self):
        cderi_uov = self.tensors.get("cderi_uov", self.make_cderi_uov())
        mo_occ = self.mo_occ
        mo_energy = self.mo_energy
        mask_act = self.get_frozen_mask()
        c_os = self.c_os
        c_ss = self.c_ss
        incore_t_oovv = self.incore_t_oovv
        verbose = self.verbose
        max_memory = self.max_memory
        h5file = self._tmpfile
        tensors, results = self.get_mp2_integrals(
            cderi_uov, mo_occ, mo_energy, mask_act,
            c_os, c_ss, incore_t_oovv,
            verbose=verbose, max_memory=max_memory, h5file=h5file)
        self.tensors.update(tensors)
        results = {util.pad_omega(key, self.omega): val for (key, val) in results.items()}
        util.update_results(self.results, results)

    def make_t_oovv(self):
        r""" MP2 amplitude :math:`t_{ij}^{ab}`.

        dim: (nocc_act, nocc_act, nvir_act, nvir_act)
        """
        if "t_oovv" in self.tensors:
            return self.tensors["t_oovv"]

        self._make_mp2_integrals()
        return self.tensors["t_oovv"]

    def make_G_uov(self):
        r""" MP2 amplitude contracted by ERI :math:`\Gamma_{ia, P}`.

        dim: (naux, nocc_act, nvir_act)
        """
        if "G_uov" in self.tensors:
            return self.tensors["G_uov"]

        self._make_mp2_integrals()
        return self.tensors["G_uov"]

    def make_rdm1_corr(self):
        r""" Response density matrix of MP2 correlation contribution in MO (non-response) :math:`D_{pq}^{(2)}`.

        dim: (nmo, nmo)
        """
        if "rdm1_corr" in self.tensors:
            return self.tensors["rdm1_corr"]

        self._make_mp2_integrals()
        return self.tensors["rdm1_corr"]

    def driver_eng_mp2(self, **kwargs):
        """ Driver of MP2 energy.

        For response computation of MP2, energy evaluation is accompanied with evaluation of MP2 integrals.
        """
        if "eng_corr_MP2" not in self.results:
            self._make_mp2_integrals()

        return self.results

    def make_W_I(self):
        r""" Part I of MO-energy-weighted density matrix W_{pq} [\mathrm{I}].

        dim: (nmo, nmo)
        """
        if "W_I" in self.tensors:
            return self.tensors["W_I"]

        # prepare tensors
        cderi_uov = self.tensors.get("cderi_uov", self.make_cderi_uov())
        cderi_uoo = self.tensors.get("cderi_uoo", self.make_cderi_uoo())
        G_uov = self.tensors.get("G_uov", self.make_G_uov())

        # main function
        tensors = self.get_W_I(cderi_uov, cderi_uoo, G_uov, verbose=self.verbose)
        self.tensors.update(tensors)
        return tensors["W_I"]

    def make_lag_vo(self):
        r""" Generate MP2 contribution to Lagrangian vir-occ block :math:`L_{ai}`. """
        if "lag_vo" in self.tensors:
            return self.tensors["lag_vo"]

        # prepare tensors
        W_I = self.tensors.get("W_I", self.make_W_I())
        cderi_uaa = self.tensors.get("cderi_uaa", self.make_cderi_uaa())
        G_uov = self.tensors["G_uov"]
        rdm1_corr = self.tensors["rdm1_corr"]
        Ax0_Core = self.Ax0_Core

        # main function
        tensors = self.get_lag_vo(
            G_uov, cderi_uaa, W_I, rdm1_corr, Ax0_Core,
            max_memory=self.max_memory, verbose=self.verbose)

        self.tensors.update(tensors)
        return tensors["lag_vo"]

    def make_rdm1_corr_resp(self):
        """ Generate 1-RDM (response) of MP2 correlation :math:`D_{pq}^{(2)}`. """
        if "rdm1_corr_resp" in self.tensors:
            return self.tensors["rdm1_corr_resp"]

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
        rdm1_corr_resp_vo = self.solve_cpks(lag_vo)
        rdm1_corr_resp = rdm1_corr.copy()
        mask_occ, mask_vir = self.mask_occ, self.mask_vir
        rdm1_corr_resp[np.ix_(mask_vir, mask_occ)] = rdm1_corr_resp_vo

        self.tensors["rdm1_corr_resp"] = rdm1_corr_resp
        return rdm1_corr_resp

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

    kernel = driver_eng_mp2
    get_mp2_integrals = staticmethod(get_mp2_integrals)
    get_W_I = staticmethod(get_W_I)
    get_lag_vo = staticmethod(get_lag_vo)


if __name__ == '__main__':
    def main_1():
        from pyscf import gto, scf
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = scf.RHF(mol).run()
        mf_mp2 = RMP2RespRI(mf)
        print(mf_mp2.make_dipole())

    main_1()
