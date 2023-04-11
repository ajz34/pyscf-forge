r""" Restricted MP2.

Notes
-----

General MP2 evaluation equations:

.. math::
    g_{ij}^{ab} &= (ia|jb) = (\mu \nu | \kappa \lambda) C_{\mu i} C_{\nu a} C_{\kappa j} C_{\lambda b} \\
    D_{ij}^{ab} &= \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b \\
    t_{ij}^{ab} &= g_{ij}^{ab} / D_{ij}^{ab} \\
    n_{ij}^{ab} &= n_i n_j (1 - n_a) (1 - n_b) \\
    E_\mathrm{OS} &= n_{ij}^{ab} t_{ij}^{ab} g_{ij}^{ab} \\
    E_\mathrm{SS} &= n_{ij}^{ab} t_{ij}^{ab} (g_{ij}^{ab} - g_{ij}^{ba}) \\
    E_\mathrm{corr,MP2} &= c_\mathrm{c} (c_\mathrm{OS} E_\mathrm{OS} + c_\mathrm{SS} E_\mathrm{SS}) \\

See also ``pyscf.mp.mp2.kernel``. Compared to original PySCF's version, we do not allow arbitary ``mo_occ``
or ``mo_coeff``. To modify these attributes, one may required to substitute something like
``mf_dh.scf.mo_coeff``, and clear all tensor intermediates before calling MP2 evaluation.

Several other options such as ``frozen`` in ``pyscf.mp.mp2.kernel``, we prefer to change ``frozen_list`` in flags
of parameters. One may also substitute these options in ``mf_dh.kernel`` function by kwargs.

This function does not make checks, such as SCF convergence.

This class does not perform MP2 amplitude iteration algorithm, and assuming Fock matrix is diagonal in
molecular-orbital representation.
"""

from pyscf.dh.energy import EngPostSCFBase
from pyscf.dh import util
from pyscf import ao2mo, lib, mp, __config__, df
import numpy as np

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


# region RMP2ConvPySCF

class RMP2ConvPySCF(mp.mp2.RMP2, EngPostSCFBase):
    """ Restricted MP2 class of doubly hybrid with conventional integral evaluated by PySCF. """
    def __init__(self, mf, *args, **kwargs):
        EngPostSCFBase.__init__(self, mf)
        super().__init__(mf, *args, **kwargs)

    @property
    def restricted(self):  # type: () -> bool
        return True

    def get_frozen_mask(self):  # type: () -> np.ndarray
        return EngPostSCFBase.get_frozen_mask(self)

    def kernel(self, *args, **kwargs):
        kernel_output = super().kernel(*args, **kwargs)
        self.results["eng_corr_MP2_OS"] = self.e_corr_os
        self.results["eng_corr_MP2_SS"] = self.e_corr_ss
        self.results["eng_corr_MP2"] = self.e_corr
        return kernel_output

# endregion


# region RMP2Conv

def kernel_energy_rmp2_conv_full_incore(
        mo_energy, mo_coeff, eri_or_mol, nocc, nvir,
        t_oovv=None, frac_num=None, verbose=lib.logger.NOTE, **_kwargs):
    """ Kernel of restricted MP2 energy by conventional method.

    Parameters
    ----------
    mo_energy : np.ndarray
        Molecular orbital energy levels.
    mo_coeff : np.ndarray
        Molecular coefficients.
    eri_or_mol : np.ndarray or gto.Mole
        ERI that is recognized by ``pyscf.ao2mo.general``.

    nocc : int
        Number of occupied orbitals.
    nvir : int
        Number of virtual orbitals.

    t_oovv : np.ndarray
        Store space for ``t_oovv``
    frac_num : np.ndarray
        Fractional occupation number list.
    verbose : int
        Verbose level for PySCF.

    Notes
    -----
    For energy of fractional occupation system, computation method is chosen
    by eq (7) from Su2016 (10.1021/acs.jctc.6b00197).

    Since conventional integral of MP2 is computationally costly,
    purpose of this function should only be benchmark.
    """
    log = lib.logger.new_logger(verbose=verbose)
    log.info("[INFO] Start restricted conventional MP2")
    log.warn("Conventional integral of MP2 is not recommended!\n"
             "Use density fitting approximation is recommended.")

    if frac_num is not None:
        frac_occ, frac_vir = frac_num[:nocc], frac_num[nocc:]
    else:
        frac_occ = frac_vir = None

    # ERI conversion
    Co = mo_coeff[:, :nocc]
    Cv = mo_coeff[:, nocc:]
    eo = mo_energy[:nocc]
    ev = mo_energy[nocc:]
    log.info("[INFO] Start ao2mo")
    g_iajb = ao2mo.general(eri_or_mol, (Co, Cv, Co, Cv)).reshape(nocc, nvir, nocc, nvir)

    # loops
    eng_bi1 = eng_bi2 = 0
    for i in range(nocc):
        log.info(f"[INFO] MP2 loop i: {i}")
        g_Iajb = g_iajb[i]
        D_Ijab = eo[i] + lib.direct_sum("j - a - b -> jab", eo, ev, ev)
        t_Ijab = lib.einsum("ajb, jab -> jab", g_Iajb, 1 / D_Ijab)
        if t_oovv is not None:
            t_oovv[i] = t_Ijab
        if frac_num is not None:
            n_Ijab = frac_occ[i] * lib.einsum("j, a, b -> jab", frac_occ, 1 - frac_vir, 1 - frac_vir)
            eng_bi1 += lib.einsum("jab, jab, ajb ->", n_Ijab, t_Ijab.conj(), g_Iajb)
            eng_bi2 += lib.einsum("jab, jab, bja ->", n_Ijab, t_Ijab.conj(), g_Iajb)
        else:
            eng_bi1 += lib.einsum("jab, ajb ->", t_Ijab.conj(), g_Iajb)
            eng_bi2 += lib.einsum("jab, bja ->", t_Ijab.conj(), g_Iajb)
    eng_bi1 = util.check_real(eng_bi1)
    eng_bi2 = util.check_real(eng_bi2)
    log.info("[INFO] MP2 energy computation finished.")
    # report
    eng_os = eng_bi1
    eng_ss = eng_bi1 - eng_bi2
    eng_mp2 = eng_os + eng_ss
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
    return results


class RMP2Conv(EngPostSCFBase):
    @property
    def restricted(self):  # type: () -> bool
        return True

    def __init__(self, mf, frozen=None, omega=0, **kwargs):
        super().__init__(mf)
        self.omega = omega
        self.incore_t_oovv_mp2 = CONFIG_incore_t_oovv_mp2
        self.frozen = frozen if frozen is not None else 0
        self.frac_num = None
        self.set(**kwargs)

    kernel_energy_rmp2_conv = staticmethod(kernel_energy_rmp2_conv_full_incore)

    def kernel(self, **kwargs):
        mask = self.get_frozen_mask()
        nOcc = (mask & (self.mo_occ != 0)).sum()
        nVir = (mask & (self.mo_occ == 0)).sum()
        mo_coeff_act = self.mo_coeff[:, mask]
        mo_energy_act = self.mo_energy[mask]
        frac_num_act = self.frac_num
        if frac_num_act is not None:
            frac_num_act = frac_num_act[mask]
        # prepare t_oovv
        max_memory = self.max_memory - lib.current_memory()[0]
        t_oovv = util.allocate_array(
            self.incore_t_oovv_mp2, (nOcc, nOcc, nVir, nVir), max_memory,
            h5file=self._tmpfile, name="t_oovv_mp2", zero_init=False, chunk=(1, 1, nVir, nVir),
            dtype=mo_coeff_act.dtype)
        if t_oovv is not None:
            self.tensors["t_oovv"] = t_oovv
        # kernel
        with self.mol.with_range_coulomb(self.omega):
            eri_or_mol = self.scf._eri if self.omega == 0 else self.mol
            eri_or_mol = eri_or_mol if eri_or_mol is not None else self.mol
            results = self.kernel_energy_rmp2_conv(
                mo_energy_act, mo_coeff_act, eri_or_mol, nOcc, nVir,
                t_oovv=t_oovv, frac_num=frac_num_act, verbose=self.verbose, **kwargs)
        self.e_corr = results["eng_corr_MP2"]
        self.results.update(results)
        return results

# endregion


# region RMP2RI

def kernel_energy_rmp2_ri_incore(
        mo_energy, cderi_uov,
        t_oovv=None, frac_num=None, verbose=lib.logger.NOTE, max_memory=2000, cderi_uov_2=None):
    """ Kernel of MP2 energy by RI integral.

    Parameters
    ----------
    mo_energy : np.ndarray
        Molecular orbital energy levels.
    cderi_uov : np.ndarray
        Cholesky decomposed 3c2e ERI in MO basis (occ-vir part).

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

    Since conventional integral of MP2 is computationally costly,
    purpose of this function should only be benchmark.
    """
    log = lib.logger.new_logger(verbose=verbose)
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
    nbatch = util.calc_batch_size(4 * nocc * nvir ** 2, max_memory, dtype=cderi_uov.dtype)
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
    eng_mp2 = eng_os + eng_ss
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
    return results


class RMP2RI(EngPostSCFBase):
    @property
    def restricted(self):  # type: () -> bool
        return True

    def __init__(self, mf, frozen=None, omega=0, with_df=None, **kwargs):
        super().__init__(mf)
        self.omega = omega
        self.incore_t_oovv_mp2 = CONFIG_incore_t_oovv_mp2
        self.frozen = frozen if frozen is not None else 0
        self.frac_num = None
        if with_df is None:
            with_df = getattr(self.scf, "with_df", None)
        if with_df is None:
            with_df = df.DF(self.mol, auxbasis=df.make_auxbasis(self.mol, mp2fit=True))
        self.with_df = with_df
        self.with_df_2 = None
        self.set(**kwargs)

    kernel_energy_rmp2_ri = staticmethod(kernel_energy_rmp2_ri_incore)

    def kernel(self, **kwargs):
        mask = self.get_frozen_mask()
        nOcc = (mask & (self.mo_occ != 0)).sum()
        nVir = (mask & (self.mo_occ == 0)).sum()
        nact = nOcc + nVir
        mo_coeff_act = self.mo_coeff[:, mask]
        mo_energy_act = self.mo_energy[mask]
        frac_num_act = self.frac_num
        if frac_num_act is not None:
            frac_num_act = frac_num_act[mask]
        # prepare t_oovv
        max_memory = self.max_memory - lib.current_memory()[0]
        t_oovv = util.allocate_array(
            self.incore_t_oovv_mp2, (nOcc, nOcc, nVir, nVir), max_memory,
            h5file=self._tmpfile, name="t_oovv_mp2", zero_init=False, chunk=(1, 1, nVir, nVir),
            dtype=mo_coeff_act.dtype)
        # generate cderi_uov
        omega = self.omega
        with_df = util.get_with_df_omega(self.with_df, omega)
        max_memory = self.max_memory - lib.current_memory()[0]
        cderi_uov = self.tensors.get(util.pad_omega("cderi_uov", omega), None)
        if cderi_uov is None:
            cderi_uov = util.get_cderi_mo(with_df, mo_coeff_act, None, (0, nOcc, nOcc, nact), max_memory)
            self.tensors[util.pad_omega("cderi_uov", omega)] = cderi_uov
        # cderi_uov_2 is rarely called, so do not try to build omega for this special case
        cderi_uov_2 = None
        max_memory = self.max_memory - lib.current_memory()[0]
        if self.with_df_2 is not None:
            cderi_uov_2 = util.get_cderi_mo(self.with_df_2, mo_coeff_act, None, (0, nOcc, nOcc, nact), max_memory)
        # kernel
        max_memory = self.max_memory - lib.current_memory()[0]
        results = self.kernel_energy_rmp2_ri(
            mo_energy_act, cderi_uov,
            t_oovv=t_oovv,
            frac_num=frac_num_act,
            verbose=self.verbose,
            max_memory=max_memory,
            cderi_uov_2=cderi_uov_2)
        
        self.e_corr = results["eng_corr_MP2"]
        self.results.update(results)
        return results

# endregion


def __main_1__():
    from pyscf import gto, scf
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
    mf_scf = scf.RHF(mol).run()
    mf_mp = RMP2ConvPySCF(mf_scf, frozen=[1, 2]).run()
    print(mf_mp.e_tot)
    print(mf_mp.results)


def __main_2__():
    from pyscf import gto, scf
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
    mf_scf = scf.RHF(mol).run()
    mf_mp = RMP2Conv(mf_scf, frozen=[1, 2]).run()
    print(mf_mp.e_tot)
    print(mf_mp.results)


def __main_3__():
    from pyscf import gto, scf
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
    mf_scf = scf.RHF(mol).run()
    mf_mp = RMP2RI(mf_scf, frozen=[1, 2]).run()
    print(mf_mp.e_tot)
    print(mf_mp.results)


if __name__ == '__main__':
    __main_1__()
    __main_2__()
    __main_3__()
