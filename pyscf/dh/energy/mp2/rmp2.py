from pyscf.dh.energy import RDHBase
from pyscf.dh import util
from pyscf import ao2mo, lib
import numpy as np


class RMP2ofDH(RDHBase):
    """ Restricted MP2 class of doubly hybrid. """

    def kernel(self, **kwargs):
        with self.params.temporary_flags(kwargs):
            results = driver_energy_rmp2(self)
        self.params.update_results(results)
        return results


def driver_energy_rmp2(mf_dh):
    """ Driver of restricted MP2 energy.

    Parameters
    ----------
    mf_dh : RMP2ofDH
        Restricted doubly hybrid object.

    Returns
    -------
    dict
    """
    mf_dh.build()
    mol = mf_dh.mol
    log = mf_dh.log
    results_summary = dict()
    # parse frozen orbitals
    mask_act = mf_dh.get_mask_act()
    nact, nOcc, nVir = mf_dh.nact, mf_dh.nOcc, mf_dh.nVir
    mo_coeff_act = mf_dh.mo_coeff_act
    mo_energy_act = mf_dh.mo_energy_act
    # other options
    frac_num = mf_dh.params.flags["frac_num_mp2"]
    frac_num_f = frac_num[mask_act] if frac_num is not None else None
    omega_list = mf_dh.params.flags["omega_list_mp2"]
    integral_scheme = mf_dh.params.flags["integral_scheme_mp2"]
    if integral_scheme is None:
        integral_scheme = mf_dh.params.flags["integral_scheme"]
    integral_scheme = integral_scheme.lower()
    for omega in omega_list:
        log.log(f"[INFO] omega in MP2 energy driver: {omega}")
        # prepare t_ijab space
        t_ijab_name = util.pad_omega("t_ijab", omega)
        params = mf_dh.params
        max_memory = mol.max_memory - lib.current_memory()[0]
        incore_t_ijab = util.parse_incore_flag(
            params.flags["incore_t_ijab_mp2"], nOcc**2 * nVir**2,
            max_memory, dtype=mo_coeff_act.dtype)
        if incore_t_ijab is None:
            t_ijab = None
        else:
            t_ijab = params.tensors.create(
                name=t_ijab_name, shape=(nOcc, nOcc, nVir, nVir),
                incore=incore_t_ijab, dtype=mo_coeff_act.dtype)
        # MP2 kernels
        if integral_scheme.startswith("conv"):
            # Conventional MP2
            eri_or_mol = mf_dh.scf._eri if omega == 0 else mol
            if eri_or_mol is None:
                eri_or_mol = mol
            with mol.with_range_coulomb(omega):
                results = kernel_energy_rmp2_conv_full_incore(
                    mo_energy_act, mo_coeff_act, eri_or_mol, nOcc, nVir,
                    t_ijab=t_ijab,
                    frac_num=frac_num_f,
                    verbose=mf_dh.verbose)
            if omega != 0:
                results = {util.pad_omega(key, omega): val for (key, val) in results.items()}
            results_summary.update(results)
        elif integral_scheme.startswith("ri"):
            # RI MP2
            with_df = util.get_with_df_omega(mf_dh.with_df, omega)
            Y_OV = params.tensors.get(util.pad_omega("Y_OV", omega), None)
            if Y_OV is None:
                Y_OV = params.tensors[util.pad_omega("Y_OV", omega)] = util.get_cderi_mo(
                    with_df, mo_coeff_act, None, (0, nOcc, nOcc, nact),
                    mol.max_memory - lib.current_memory()[0])
            # Y_OV_2 is rarely called, so do not try to build omega for this special case
            Y_OV_2 = None
            if mf_dh.with_df_2 is not None:
                Y_OV_2 = util.get_cderi_mo(
                    mf_dh.with_df_2, mo_coeff_act, None, (0, nOcc, nOcc, nact),
                    mol.max_memory - lib.current_memory()[0])
            results = kernel_energy_rmp2_ri(
                mo_energy_act, Y_OV,
                t_ijab=t_ijab,
                frac_num=frac_num_f,
                verbose=mf_dh.verbose,
                max_memory=mol.max_memory - lib.current_memory()[0],
                Y_OV_2=Y_OV_2)
            results = {util.pad_omega(key, omega): val for (key, val) in results.items()}
            results_summary.update(results)
        else:
            raise NotImplementedError("Not implemented currently!")
    return results_summary


def kernel_energy_rmp2_conv_full_incore(
        mo_energy, mo_coeff, eri_or_mol,
        nocc, nvir,
        t_ijab=None,
        frac_num=None, verbose=lib.logger.NOTE):
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

    t_ijab : np.ndarray
        Store space for ``t_ijab``
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
        D_Ijab = eo[i] + eo[:, None, None] - ev[None, :, None] - ev[None, None, :]
        t_Ijab = lib.einsum("ajb, jab -> jab", g_Iajb, 1 / D_Ijab)
        if t_ijab is not None:
            t_ijab[i] = t_Ijab
        if frac_num is not None:
            n_Ijab = frac_occ[i] * frac_occ[:, None, None] \
                * (1 - frac_vir[None, :, None]) * (1 - frac_vir[None, None, :])
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


def kernel_energy_rmp2_ri(
        mo_energy, Y_OV,
        t_ijab=None,
        frac_num=None, verbose=lib.logger.NOTE, max_memory=2000, Y_OV_2=None):
    """ Kernel of MP2 energy by RI integral.

    Parameters
    ----------
    mo_energy : np.ndarray
        Molecular orbital energy levels.
    Y_OV : np.ndarray
        Cholesky decomposed 3c2e ERI in MO basis (occ-vir part).

    t_ijab : np.ndarray
        Store space for ``t_ijab``
    frac_num : np.ndarray
        Fractional occupation number list.
    verbose : int
        Verbose level for PySCF.
    max_memory : float
        Allocatable memory in MB.
    Y_OV_2 : np.ndarray
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

    naux, nocc, nvir = Y_OV.shape
    if frac_num is not None:
        frac_occ, frac_vir = frac_num[:nocc], frac_num[nocc:]
    else:
        frac_occ = frac_vir = None
    eo = mo_energy[:nocc]
    ev = mo_energy[nocc:]

    # loops
    log.info("[INFO] Start RI-MP2 loop")
    nbatch = util.calc_batch_size(4 * nocc * nvir ** 2, max_memory, dtype=Y_OV.dtype)
    eng_bi1 = eng_bi2 = 0
    for sI in util.gen_batch(0, nocc, nbatch):
        log.info(f"[INFO] MP2 loop i: {sI}")
        if Y_OV_2 is None:
            g_Iajb = lib.einsum("PIa, Pjb -> Iajb", Y_OV[:, sI], Y_OV)
        else:
            g_Iajb = 0.5 * lib.einsum("PIa, Pjb -> Iajb", Y_OV[:, sI], Y_OV_2)
            g_Iajb += 0.5 * lib.einsum("PIa, Pjb -> Iajb", Y_OV_2[:, sI], Y_OV)
        D_Ijab = eo[sI, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
        t_Ijab = lib.einsum("Iajb, Ijab -> Ijab", g_Iajb, 1 / D_Ijab)
        if t_ijab is not None:
            t_ijab[sI] = t_Ijab
        if frac_num is not None:
            n_Ijab = frac_occ[sI, None, None, None] * frac_occ[None, :, None, None] \
                * (1 - frac_vir[None, None, :, None]) * (1 - frac_vir[None, None, None, :])
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


driver_energy_rmp2.__doc__ += \
    r"""
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
