r""" Restricted Ring-CCD.

Notes
-----

For restricted case, ring-CCD should give direct-RPA (dRPA) energy.

.. math::
    \mathbf{T} &= - \frac{1}{\mathbf{\Delta \varepsilon}} \odot (\mathbf{B} - 2 \mathbf{B T} - 2
    \mathbf{T B} + 4 \mathbf{T B T}) \\
    E^\mathrm{dRPA, OS} = E^\mathrm{dRPA, SS} &= - \mathrm{tr} (\mathbf{T B})

Equation here is similar but not the same to 10.1063/1.3043729.
The formula of Scuseria's article (eq 9) applies to spin-orbital and thus no coefficients 2 or 4.
More over, we evaluate :math:`B_{ia, jb} = (ia|jb)` to match result of direct-RPA (thus, we only evaluate
direct ring-CCD instead of full ring-CCD).
"""

from pyscf.dh.energy import EngPostSCFBase
from pyscf import ao2mo, lib, __config__
import numpy as np


CONFIG_tol_eng_ring_ccd = getattr(__config__, "tol_eng_ring_ccd", 1e-8)
CONFIG_tol_amp_ring_ccd = getattr(__config__, "tol_amp_ring_ccd", 1e-6)
CONFIG_max_cycle_ring_ccd = getattr(__config__, "max_cycle_ring_ccd", 64)


def kernel_energy_rring_ccd_conv(
        mo_energy, mo_coeff, eri_or_mol, mo_occ,
        tol_e=1e-8, tol_amp=1e-6, max_cycle=64,
        verbose=lib.logger.NOTE):
    """ dRPA evaluation by ring-CCD with conventional integral.

    Parameters
    ----------
    mo_energy : np.ndarray
        Molecular orbital energy levels.
    mo_coeff : np.ndarray
        Molecular coefficients.
    eri_or_mol : np.ndarray or gto.Mole
        ERI that is recognized by ``pyscf.ao2mo.general``.
    mo_occ : np.ndarray
        Molecular orbital occupation numbers.

    tol_e : float
        Threshold of ring-CCD energy difference while in DIIS update.
    tol_amp : float
        Threshold of L2 norm of ring-CCD amplitude while in DIIS update.
    max_cycle : int
        Maximum iteration of ring-CCD iteration.
    verbose : int
        Verbose level for PySCF.
    """
    log = lib.logger.new_logger(verbose=verbose)
    log.warn("Conventional integral of MP2 is not recommended!\n"
             "Use density fitting approximation is recommended.")
    log.info(f"[INFO] dRPA (ring-CCD) iteration, tol_e {tol_e:9.4e}, tol_amp {tol_amp:9.4e}, max_cycle {max_cycle:3d}")

    mask_occ = mo_occ != 0
    mask_vir = mo_occ == 0
    eo = mo_energy[mask_occ]
    ev = mo_energy[mask_vir]
    Co = mo_coeff[:, mask_occ]
    Cv = mo_coeff[:, mask_vir]
    nocc = mask_occ.sum()
    nvir = mask_vir.sum()

    log.info("[INFO] Start ao2mo")
    g_iajb = ao2mo.general(eri_or_mol, (Co, Cv, Co, Cv)).reshape(nocc, nvir, nocc, nvir)
    D_iajb = lib.direct_sum("i - a + j - b -> iajb", eo, ev, eo, ev)
    log.info("[INFO] Finish ao2mo")

    dim_ov = nocc * nvir
    B = g_iajb.reshape(dim_ov, dim_ov)
    D = D_iajb.reshape(dim_ov, dim_ov)

    def update_T(T, B, D):
        return - 1 / D * (B - 2 * B @ T - 2 * T @ B + 4 * T @ B @ T)

    # begin diis, start from third iteration, space of diis is 6
    # T_old = np.zeros_like(B)
    T_new = np.zeros_like(B)
    eng_os = eng_ss = eng_drpa = 0
    diis = lib.diis.DIIS()
    diis.space = 6
    converged = False
    for epoch in range(max_cycle):
        T_old, T_new = T_new, update_T(T_new, B, D)
        eng_old = eng_drpa
        if epoch > 3:
            T_new = diis.update(T_new)
        eng_os = eng_ss = - np.einsum("AB, AB ->", T_new, B)
        eng_drpa = eng_os + eng_ss
        err_amp = np.linalg.norm(T_new - T_old)
        err_e = abs(eng_drpa - eng_old)
        log.info(
            f"[INFO] dRPA (ring-CCD) energy in iter {epoch:3d}: {eng_drpa:20.12f}, "
            f"amplitude L2 error {err_amp:9.4e}, eng_err {err_e:9.4e}")
        if err_amp < tol_amp and err_e < tol_e:
            converged = True
            log.info("[INFO] dRPA (ring-CCD) amplitude converges.")
            break
    if not converged:
        log.warn(f"dRPA (ring-CCD) not converged in {max_cycle} iterations!")

    results = dict()
    results["eng_corr_RING_CCD_OS"] = eng_os
    results["eng_corr_RING_CCD_SS"] = eng_ss
    results["eng_corr_RING_CCD"] = eng_drpa
    results["converged_RING_CCD"] = converged
    log.info(f"[RESULT] Energy corr ring-CCD of same-spin: {eng_os  :18.10f}")
    log.info(f"[RESULT] Energy corr ring-CCD of oppo-spin: {eng_ss  :18.10f}")
    log.info(f"[RESULT] Energy corr ring-CCD of total    : {eng_drpa:18.10f}")
    return results


class RRingCCDConv(EngPostSCFBase):
    """ Restricted Ring-CCD class of doubly hybrid with conventional integral. """

    def __init__(self, mf, frozen=None, omega=0, **kwargs):
        super().__init__(mf)
        self.omega = omega
        self.frozen = frozen if frozen is not None else 0
        self.conv_tol = CONFIG_tol_eng_ring_ccd
        self.conv_tol_amp = CONFIG_tol_amp_ring_ccd
        self.max_cycle = CONFIG_max_cycle_ring_ccd
        self.set(**kwargs)

    def driver_eng_ring_ccd(self, **_kwargs):
        mask = self.get_frozen_mask()
        mol = self.mol
        mo_coeff_act = self.mo_coeff[:, mask]
        mo_energy_act = self.mo_energy[mask]
        mo_occ_act = self.mo_occ[mask]
        # eri generator
        eri_or_mol = self.scf._eri if self.omega == 0 else mol
        eri_or_mol = eri_or_mol if eri_or_mol is not None else mol
        with mol.with_range_coulomb(self.omega):
            results = kernel_energy_rring_ccd_conv(
                mo_energy=mo_energy_act,
                mo_coeff=mo_coeff_act,
                eri_or_mol=eri_or_mol,
                mo_occ=mo_occ_act,
                tol_e=self.conv_tol,
                tol_amp=self.conv_tol_amp,
                max_cycle=self.max_cycle,
                verbose=self.verbose
            )
        self.results.update(results)
        return results

    kernel_energy_ring_ccd = staticmethod(kernel_energy_rring_ccd_conv)
    kernel = driver_eng_ring_ccd
