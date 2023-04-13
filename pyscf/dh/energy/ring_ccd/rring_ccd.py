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

from pyscf.dh.energy import RDHBase
from pyscf.dh import util
from pyscf import ao2mo, lib
import numpy as np


class RRingCCDofDH(RDHBase):
    """ Restricted MP2 class of doubly hybrid. """

    def kernel(self, **kwargs):
        with self.params.temporary_flags(kwargs):
            results = driver_energy_rring_ccd(self)
        self.params.update_results(results)
        return results


def driver_energy_rring_ccd(mf_dh):
    """ Driver of restricted ring-CCD energy.

    Parameters
    ----------
    mf_dh : RDH
        Restricted doubly hybrid object.

    Returns
    -------
    RDH
    """
    mf_dh.build()
    mol = mf_dh.mol
    log = mf_dh.log
    mf_dh._flag_snapshot = mf_dh.params.flags.copy()
    results_summary = dict()
    integral_scheme = mf_dh.params.flags.get("integral_scheme_ring_ccd", mf_dh.params.flags["integral_scheme"]).lower()
    # parse frozen orbitals
    nact, nOcc, nVir = mf_dh.nact, mf_dh.nOcc, mf_dh.nVir
    mo_coeff_act = mf_dh.mo_coeff_act
    mo_energy_act = mf_dh.mo_energy_act
    # other options
    omega_list = mf_dh.params.flags["omega_list_ring_ccd"]
    tol_e = mf_dh.params.flags["tol_eng_ring_ccd"]
    tol_amp = mf_dh.params.flags["tol_amp_ring_ccd"]
    max_cycle = mf_dh.params.flags["max_cycle_ring_ccd"]
    for omega in omega_list:
        log.log(f"[INFO] omega in ring-CCD energy driver: {omega}")
        if integral_scheme.startswith("conv"):
            eri_or_mol = mf_dh.scf._eri if omega == 0 else mol
            if eri_or_mol is None:
                eri_or_mol = mol
            with mol.with_range_coulomb(omega):
                results = kernel_energy_rring_ccd_conv(
                    mo_energy_act, mo_coeff_act, eri_or_mol,
                    nOcc, nVir,
                    tol_e=tol_e, tol_amp=tol_amp, max_cycle=max_cycle,
                    verbose=mf_dh.verbose)
            results = {util.pad_omega(key, omega): val for (key, val) in results.items()}
            results_summary.update(results)
        else:
            raise NotImplementedError
    return results_summary


def kernel_energy_rring_ccd_conv(
        mo_energy, mo_coeff, eri_or_mol,
        nocc, nvir,
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

    nocc : int
        Number of occupied orbitals.
    nvir : int
        Number of virtual orbitals.

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

    Co = mo_coeff[:, :nocc]
    Cv = mo_coeff[:, nocc:]
    eo = mo_energy[:nocc]
    ev = mo_energy[nocc:]
    log.info("[INFO] Start ao2mo")
    g_iajb = ao2mo.general(eri_or_mol, (Co, Cv, Co, Cv)).reshape(nocc, nvir, nocc, nvir)
    D_iajb = eo[:, None, None, None] - ev[None, :, None, None] + eo[None, None, :, None] - ev[None, None, None, :]
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

